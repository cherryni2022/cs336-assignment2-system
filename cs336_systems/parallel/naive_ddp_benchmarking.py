from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM
from multiprocessing import Manager
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as timer
import pandas as pd
from statistics import mean, stdev


"""
作业: (naive_ddp_benchmarking):
要求使用上面实现的朴素 DDP,对之前作业中的 XL 尺寸语言模型进行基准测试。
设置: 单节点,2 个 GPU。
测量: 测量每个训练步骤的总时间, 以及其中花在梯度通信上的时间比例。
"""
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

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_data_parallel(rank, world_size, train_x, train_y, 
                    num_steps, model_config, result_queue):
    setup(rank, world_size)

    torch.manual_seed(0)
    batch_size, _ = train_x.shape
    local_batch_size = batch_size // world_size

    start_index = rank * local_batch_size
    end_index = min(start_index + local_batch_size, train_x.shape[0])

    device = torch.cuda.current_device()
    model = BasicsTransformerLM(
                vocab_size=model_config["vocab_size"],
                context_length=model_config["context_length"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                rope_theta=model_config["rope_theta"],
            ).to(device)
    lossfn = torch.nn.CrossEntropyLoss()

    #model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    model.train()
    local_train_x = train_x[start_index:end_index].to(device)
    local_train_y = train_y[start_index:end_index].to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    train_time_list = []
    network_time_list = []
    network_ratio_list = []
    for _ in range(num_steps):
        train_time_start = timer()

        optimizer.zero_grad()
        out = model(local_train_x)
        loss = lossfn(out.view(-1, vocab_size), local_train_y.view(-1))
        loss.backward()
        network_time_start = timer()
        # 批量通信:
        # torch._utils._flatten_dense_tensors 
        # torch._utils._unflatten_dense_tensors
        for params in model.parameters():
            # dist.all_reduce(params.grad, op=dist.ReduceOp.SUM, async_op=False)
            # params.grad.data /= world_size
            dist.all_reduce(params.grad, op=dist.ReduceOp.AVG, async_op=False)
        network_time_end = timer()
        optimizer.step()
        torch.cuda.synchronize()

        time_end = timer()
        train_time_list.append(time_end - train_time_start)
        network_time_list.append(network_time_end - network_time_start)
        network_ratio_list.append((network_time_end - network_time_start) / (time_end - train_time_start))
        print(f"device:{rank}/{world_size} run step {_}, "
                f"train_time:{(time_end - train_time_start) * 1000:.2f}ms, "
                f"network_time:{(network_time_end - network_time_start) * 1000:.2f}ms, "
                f"network_ratio:{((network_time_end - network_time_start) / (time_end - train_time_start)):.2%}")

    avg_train_time = mean(train_time_list)
    avg_network_time = mean(network_time_list)
    avg_ratio = mean(network_ratio_list)
    result_queue.put((avg_train_time, avg_network_time, avg_ratio))

batch_size = 16
context_length = 256
world_size = 2
vocab_size = 10_000
rope_theta = 10000.0
warmup_steps = 5
timing_steps = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    args = parser.parse_args()

    test_world_size = args.world_size if args.world_size else world_size
    test_batch_size = args.batch_size if args.batch_size else batch_size
    test_context_length = args.context_length if args.context_length else context_length
    test_benchmark_steps = args.benchmark_steps if args.benchmark_steps else 10
    test_model_config = model_configs["xl"]

    num_steps = test_benchmark_steps
    num_procs = test_world_size

    test_model_config["rope_theta"] = rope_theta
    test_model_config["vocab_size"] = vocab_size
    test_model_config["context_length"] = test_context_length

    # Create random input data 
    device = "cuda" 
    train_x = torch.randint(
        0, vocab_size, (test_batch_size, test_context_length))
    train_y = torch.randint(
        0, vocab_size, (test_batch_size, test_context_length))

    mp.set_start_method('spawn', force=True)
    with Manager() as manager:
        result_queue = manager.Queue()
        print(f"start train_data_parallel. num_steps: {num_steps}")
        mp.spawn(train_data_parallel, 
                args=(num_procs, train_x, train_y, num_steps, result_queue), 
                nprocs=num_procs,
                join=True)
        
        print(f"start get results")

        csv_results = []
        all_results = []
        for i in range(num_procs):  # num_procs = world_size
            result = result_queue.get()
            all_results.append(result)
            print(f"Process {i} result:"
             f"avg_train_time={result[0]:.4f}s, "
             f"avg_network_time={result[1]:.4f}s, "
             f"avg_ratio={result[2]:.2%}")

        # 计算所有进程的平均值
        avg_train_times = mean([r[0] for r in all_results])
        avg_network_times = mean([r[1] for r in all_results])
        avg_ratios = mean([r[2] for r in all_results])
        print(f"end train_data_parallel =======================")

        
        csv_results.append({
            "world_size": num_procs,
            "batch_size": test_batch_size,
            "context_length": test_context_length,
            "benchmark_steps": test_benchmark_steps,
            "avg_train_time": avg_train_times,
            "avg_network_time": avg_network_times,
            "avg_ratio": avg_ratios,
        })
    
    df = pd.DataFrame(csv_results)
    print(df.to_markdown("naive_ddp_benchmarking.md", index=False))
