import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx
from multiprocessing import Manager
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import argparse
import os
from timeit import default_timer as timer
import pandas as pd
from statistics import mean, stdev
#from cs336_systems.parallel.bucket_ddp import BucketDDP
from cs336_systems.parallel.ddp import DDPBucketed
from cs336_systems.parallel.individual_ddp import IndividualOverlapDDP

# 1.所有硬件条件及变量相同情况下:针对xl model 对比测试nativeDDP、flatDDP、IndividualDDP、BucketDDP
# 2.所有硬件条件相同及变量相同情况下:针对xl model 测试BucketDDP 在不同bucket size下的性能

# Define the model sizes (from the table)
model_configs = {
    "small": {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {
        "size": "medium",
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"size": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"size": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

# Constants
world_size = 2
vocab_size = 10_000
context_length = 128
batch_size = 32
rope_theta = 10000.0
warmup_steps = 5
train_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"



def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

@nvtx.range("naive_ddp")
def naive_ddp(rank, world_size, model, train_x, train_y, optimizer, lossfn, 
            num_train_steps, num_warmup_steps, train_times, communicate_times, 
            device, model_type):
        print (f"start naive_ddp {rank}/{world_size} warmup:")
        for step in range(num_warmup_steps):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                optimizer.zero_grad()
                outputs = model(train_x)
                loss = lossfn(outputs.view(-1, vocab_size), train_y.view(-1))
                loss.backward()
                for param in model.parameters():
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
                optimizer.step()
                torch.cuda.synchronize()
        
        print (f"start naive_ddp {rank}/{world_size} train:")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        for step in range(num_train_steps):
            step_start_time = timer()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                with nvtx.range("naive_ddp_train"):
                    optimizer.zero_grad()
                    outputs = model(train_x)
                    loss = lossfn(outputs.view(-1, vocab_size), train_y.view(-1))
                    loss.backward()
                    torch.cuda.synchronize()

                with nvtx.range("naive_ddp_communicate"):
                    network_start_time = timer()
                    for param in model.parameters():
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
        
                torch.cuda.synchronize()
                communicate_times.append(timer() - network_start_time)

                with nvtx.range("naive_ddp_optimizer"):
                    optimizer.step()
                torch.cuda.synchronize()

            train_times.append(timer() - step_start_time)
        torch.cuda.memory._dump_snapshot(f"memory_naive_ddp_{model_type}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

# 梯度批量打包通讯
def chunked_all_reduce_gradients(model, chunk_size_mb=50):
    """分批处理梯度的all_reduce操作，避免内存溢出"""
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    if not params_with_grad:
        return
    
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    current_chunk = []
    current_size = 0
    
    for param in params_with_grad:
        param_size = param.grad.numel() * param.grad.element_size()
        
        # 如果当前参数会使chunk超过大小限制，先处理当前chunk
        if current_size + param_size > chunk_size_bytes and current_chunk:
            # 处理当前chunk
            flatten_grads = torch._utils._flatten_dense_tensors([p.grad for p in current_chunk])
            dist.all_reduce(flatten_grads, op=dist.ReduceOp.AVG, async_op=False)
            unflatten_grads = torch._utils._unflatten_dense_tensors(flatten_grads, [p.grad for p in current_chunk])
            
            # 更新梯度
            for p, new_grad in zip(current_chunk, unflatten_grads):
                p.grad = new_grad
            
            # 重置chunk
            current_chunk = []
            current_size = 0
        
        current_chunk.append(param)
        current_size += param_size
    
    # 处理最后一个chunk
    if current_chunk:
        flatten_grads = torch._utils._flatten_dense_tensors([p.grad for p in current_chunk])
        dist.all_reduce(flatten_grads, op=dist.ReduceOp.AVG, async_op=False)
        unflatten_grads = torch._utils._unflatten_dense_tensors(flatten_grads, [p.grad for p in current_chunk])
        
        for p, new_grad in zip(current_chunk, unflatten_grads):
            p.grad = new_grad

@nvtx.range("flat_ddp")
def flat_ddp(rank, world_size, model, train_x, train_y, optimizer, lossfn, 
            num_train_steps, num_warmup_steps, train_times, communicate_times, 
            device, model_type):
        print (f"start flat_ddp {rank}/{world_size} warmup:")
        for step in range(num_warmup_steps):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                optimizer.zero_grad()
                outputs = model(train_x)
                loss = lossfn(outputs.view(-1, vocab_size), train_y.view(-1))
                loss.backward()

                # 使用分批处理策略避免内存溢出
                chunked_all_reduce_gradients(model, chunk_size_mb=50)
                optimizer.step()
        
        print (f"start flat_ddp {rank}/{world_size} train:")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        for step in range(num_train_steps):
            step_start_time = timer()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                with nvtx.range("flat_ddp_train_step"):
                    optimizer.zero_grad()
                    outputs = model(train_x)
                    loss = lossfn(outputs.view(-1, vocab_size), train_y.view(-1))
                    loss.backward()
                torch.cuda.synchronize()

                with nvtx.range("flat_ddp_communicate"):
                    network_start_time = timer()
                    # 使用分批处理策略避免内存溢出
                    chunked_all_reduce_gradients(model, chunk_size_mb=50)
        
                torch.cuda.synchronize()
                communicate_times.append(timer() - network_start_time)

                with nvtx.range("flat_ddp_optimizer_params"):
                    optimizer.step()

            torch.cuda.synchronize()
            train_times.append(timer() - step_start_time)

        torch.cuda.memory._dump_snapshot(f"memory_flat_ddp_{model_type}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

@nvtx.range("individual_ddp")
def individual_ddp(rank, world_size, model, train_x, train_y, optimizer, lossfn, 
            num_train_steps, num_warmup_steps, train_times, communicate_times, device,model_type):
        print (f"start individual_ddp {rank}/{world_size} warmup:")
        model = IndividualOverlapDDP(model)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            for step in range(num_warmup_steps):
                optimizer.zero_grad()
                outputs = model(train_x)
                loss = lossfn(outputs.view(-1, vocab_size), train_y.view(-1))
                loss.backward()
                model.finish_gradient_synchronization()
                optimizer.step()
        
        print (f"start individual_ddp {rank}/{world_size} train:")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        for step in range(num_train_steps):
            step_start_time = timer()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                with nvtx.range("individual_ddp_train"):
                    optimizer.zero_grad()
                    outputs = model(train_x)
                    loss = lossfn(outputs.view(-1, vocab_size), train_y.view(-1))
                    loss.backward()
                torch.cuda.synchronize()

                with nvtx.range("individual_ddp_communicate"):
                    network_start_time = timer()
                    model.finish_gradient_synchronization()

                torch.cuda.synchronize()
                communicate_times.append(timer() - network_start_time)

                with nvtx.range("individual_ddp_optimizer_params"):
                    optimizer.step()

            torch.cuda.synchronize()
            train_times.append(timer() - step_start_time)

        torch.cuda.memory._dump_snapshot(f"memory_individual_ddp_{model_type}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

@nvtx.range("bucket_ddp")
def bucket_ddp(rank, world_size, model, train_x, train_y, optimizer, lossfn, 
            num_train_steps, num_warmup_steps, 
            train_times, communicate_times, bucket_size_mb, device,model_type):
        print (f"start bucket_ddp {rank}/{world_size} warmup:")
        model = DDPBucketed(model, bucket_size_mb)
        for step in range(num_warmup_steps):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                optimizer.zero_grad()
                outputs = model(train_x)
                loss = lossfn(outputs.view(-1, vocab_size), train_y.view(-1))
                loss.backward()
                model.finish_gradient_synchronization()
                optimizer.step()
        
        print (f"start bucket_ddp {rank}/{world_size} train:")
        #torch.cuda.memory._record_memory_history(max_entries=1000000)
        for step in range(num_train_steps):
            step_start_time = timer()
            with nvtx.range("bucket_ddp_train_step"):
                optimizer.zero_grad()
                outputs = model(train_x)
                loss = lossfn(outputs.view(-1, vocab_size), train_y.view(-1))
                loss.backward()
                torch.cuda.synchronize()

            with nvtx.range("bucket_ddp_communicate"):
                network_start_time = timer()
                model.finish_gradient_synchronization()
            torch.cuda.synchronize()
            communicate_times.append(timer() - network_start_time)

            with nvtx.range("bucket_ddp_optimizer_params"):
                optimizer.step()

            torch.cuda.synchronize()
            train_times.append(timer() - step_start_time)

        # torch.cuda.memory._dump_snapshot(f"memory_bucket_ddp_{model_type}.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None) 

"""
    model_config: vocab_size, context_length,
                 layers, d_model, num_heads, d_ff,rope_theta
"""
def benchmark(rank, world_size, 
                train_x, train_y, model_config, batch_size,
                num_train_steps, num_warmup_steps,
                ddp_type, bucket_size_mb, model_type, result_queue
                ):
    print (f"start benchmark {ddp_type}")
    setup(rank, world_size)
    device = f"cuda:{torch.cuda.current_device()}"
    transformer_model = BasicsTransformerLM(
        vocab_size = model_config["vocab_size"],
        context_length = model_config["context_length"],
        d_model = model_config["d_model"],
        num_layers = model_config["num_layers"],
        num_heads = model_config["num_heads"],
        d_ff = model_config["d_ff"],
        rope_theta = model_config["rope_theta"]
    ).to(device)

    transformer_model.train()

    local_batch_size = batch_size // world_size
    start_index = local_batch_size * rank
    end_index = min(start_index + local_batch_size, train_x.shape[0])

    train_x = train_x[start_index: end_index].to(device)
    train_y = train_y[start_index: end_index].to(device)

    optimizer = AdamW(transformer_model.parameters(), lr=0.001)
    lossfn = nn.CrossEntropyLoss()
    train_times = []
    communicate_times = []

    if ddp_type == "naive":
        naive_ddp(rank, world_size, transformer_model, train_x, train_y, optimizer, lossfn, 
            num_train_steps, num_warmup_steps, 
            train_times, communicate_times, 
            device, model_type)
    if ddp_type == "flat_ddp":
        flat_ddp(rank, world_size, transformer_model, train_x, train_y, optimizer, lossfn, 
        num_train_steps, num_warmup_steps, train_times, communicate_times, 
        device, model_type)
    if ddp_type == "individual_ddp":
        individual_ddp(rank, world_size, transformer_model, train_x, train_y, optimizer, lossfn, 
                num_train_steps, num_warmup_steps, 
                train_times, communicate_times, 
                device, model_type)
    if ddp_type == "bucketed_ddp":
        bucket_ddp(rank, world_size, transformer_model, train_x, train_y, optimizer, lossfn, 
                num_train_steps, num_warmup_steps, 
                train_times, communicate_times, bucket_size_mb, 
                device, model_type)

    train_time = torch.tensor(train_times, device=device)
    gathered_train_times = [torch.zeros_like(train_time) for _ in range(world_size)]
    dist.all_gather(gathered_train_times, train_time)

    communicate_time = torch.tensor(communicate_times, device=device)
    gathered_communicate_times = [torch.zeros_like(communicate_time) for _ in range(world_size)]
    dist.all_gather(gathered_communicate_times, communicate_time)

    if rank == 0:
        train_times = [x for one_rank in gathered_train_times for x in one_rank.cpu().tolist()]
        communicate_times = [x for one_rank in gathered_communicate_times for x in one_rank.cpu().tolist()]
        result_queue.put((train_times, communicate_times))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Transformer models.")
    parser.add_argument("--ddp_type", type=str, default="naive", choices=["naive", "flat_ddp", "individual_ddp", "bucketed_ddp"])
    parser.add_argument("--bucket_size_mb", type=float, default=100)
    parser.add_argument("--batch_size", type=int, default="16", choices=[16,32,64,128])
    parser.add_argument("--model_type", type=str, default="xl", choices=["medium", "large","xl","2.7B"])
    args = parser.parse_args()

    test_batch_size = args.batch_size if args.batch_size else batch_size

    print (f"ddp_type:{args.ddp_type}, batch_size:{test_batch_size}, "
            f"model_type:{args.model_type}")

    mp.set_start_method("spawn", force=True)
    manager = Manager()
    result_queue = manager.Queue()

    train_x = torch.randint(0, vocab_size, (test_batch_size, context_length))
    train_y = torch.randint(0, vocab_size, (test_batch_size, context_length))

    model_config = model_configs[args.model_type]
    model_config["vocab_size"] = vocab_size
    model_config["context_length"] = context_length
    model_config["rope_theta"] = rope_theta

    mp.spawn(benchmark, 
            args=(world_size, train_x, train_y, model_config, test_batch_size,
            train_steps, warmup_steps, args.ddp_type, 
            args.bucket_size_mb, args.model_type,
            result_queue), 
            nprocs=world_size,
            join=True)
    
    # Get results from Queue
    train_times, comms_times = result_queue.get()
    print(f"Average train_step time:{sum(train_times) * 1000 / len(train_times):.2f} ms.")
    print(f"Average communication time:{sum(comms_times) * 1000 / len(comms_times):.2f} ms.")
    #print(f"Average communication ratio:{sum(comms_times) / sum(train_times):.2%}")
    torch.distributed.destroy_process_group()
    print("finish benchmark.")
