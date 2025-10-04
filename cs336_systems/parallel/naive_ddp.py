
from cs336_basics.optimizer import AdamW
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

class ToyModel(nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# data 被切分成 world_size 份
def train_data_parallel(rank, world_size, data, num_steps, result_queue):
    setup(rank, world_size)

    torch.manual_seed(0)
    batch_size, d_model = data.shape
    local_batch_size = batch_size // world_size

    start_index = rank * local_batch_size
    end_index = min(start_index + local_batch_size, data.shape[0])

    device = torch.cuda.current_device()
    model = ToyModel(d_model, d_model).to(device)
    model.train()

    local_data = data[start_index:end_index].to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)

    for _ in range(num_steps):
        optimizer.zero_grad()
        out = model(local_data)
        loss = out.mean()
        loss.backward()

        for params in model.parameters():
            # dist.all_reduce(params.grad, op=dist.ReduceOp.SUM, async_op=False)
            # params.grad.data /= world_size
            dist.all_reduce(params.grad, op=dist.ReduceOp.AVG, async_op=False)
            
        optimizer.step()

    if rank == 0:
        cpu_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        result_queue.put(cpu_state)

def train_single_process(data, num_steps):
    torch.manual_seed(0)
    batch_size, d_model = data.shape
    model = ToyModel(d_model, d_model).to("cuda")
    data = data.to("cuda")
    optimizer = AdamW(model.parameters(), lr=0.001)
    model.train()

    for _ in range(num_steps):
        optimizer.zero_grad()
        out = model(data)
        loss = out.mean()
        loss.backward()
        optimizer.step()
    
    return {k: v.detach().cpu() for k,v in model.state_dict().items()}


d_model = 128
batch_size = 100
world_size = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    args = parser.parse_args()

    test_world_size = args.world_size if args.world_size else world_size
    test_batch_size = args.batch_size if args.batch_size else batch_size
    test_d_model = args.d_model if args.d_model else d_model
    test_benchmark_steps = args.benchmark_steps if args.benchmark_steps else 10

    train_data = torch.randn(test_batch_size, test_d_model)

    num_steps = test_benchmark_steps
    num_procs = test_world_size

    print(f"start train_single_process. num_steps: {num_steps}")
    ref_state = train_single_process(train_data, num_steps)
    print(f"end train_single_process =======================")

    mp.set_start_method('spawn', force=True)
    with Manager() as manager:
        result_queue = manager.Queue()
        print(f"start train_data_parallel. num_steps: {num_steps}")
        mp.spawn(train_data_parallel, 
                args=(num_procs, train_data, num_steps, result_queue), 
                nprocs=num_procs,
                join=True)

        ddp_state = result_queue.get()
        print(f"end train_data_parallel =======================")

        # 检查单卡训练与分布式多卡训练后,每个参数是否一致
        print(f"check train parameters train_single_process and train_data_parallel")
        for key in ref_state.keys():
            assert torch.allclose(ref_state[key], ddp_state[key])
        
        print("train_single_process and train_data_parallel parameters are equal")
