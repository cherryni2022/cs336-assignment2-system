import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as timer
from statistics import mean, stdev
import pandas as pd

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def all_reduce_dist(rank, world_size,data_size_mb, backend, benchmark_trails, warmup_trails, result_queue):
    setup(rank, world_size, backend)
    # dtype: float32
    num_elements = data_size_mb * 1024 * 1024 // 4
    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")

    # warmup
    for i in range(warmup_trails):
        data = torch.randint(0, 10, (num_elements,), dtype=torch.float32, device=device)
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
    
    # benchmark
    bench_times = []
    for i in range(benchmark_trails):
        data = torch.randint(0, 10, (num_elements,), dtype=torch.float32, device=device)
        start = timer()
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        bench_times.append((timer() - start) * 1000)
    
    # calculate mean and std
    
    mean_time = mean(bench_times)
    std_time = stdev(bench_times)
    print(f"device {rank}/{world_size} run {benchmark_trails} trails, "
            f"data size is {data_size_mb} MB, "
            f"mean time is {mean_time:.2f} ms, std time is {std_time:.2f} ms")

    gather_stats = [None] * world_size
    dist.all_gather_object(gather_stats, mean_time)

    # 只在一个进程中输出统计结果
    if rank == 0:
        final_avg = sum([stat for stat in gather_stats]) / len(gather_stats)
        print (f"average all reduce time is {final_avg:.2f} ms")
        if result_queue is not None:
            result_queue.put(round(final_avg, 2))

backends = ["nccl", "gloo"]
world_sizes = [2, 4, 6]
data_size_mbs = [1,10,100,1024]
warmup_steps = 5
benchmark_steps = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--data_size_mb", type=int, default=100)
    parser.add_argument("--benchmark_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10)
    args = parser.parse_args()

    
    test_world_sizes = [args.world_size] if args.world_size else world_sizes
    test_data_size_mbs = [args.data_size_mb] if args.data_size_mb else data_size_mbs
    
    for backend in backends:
        results = []
        for world_size in test_world_sizes:
            for data_size_mb in test_data_size_mbs:
                print (f"Benchmark {backend} with {world_size} processes and {data_size_mb} MB data")
                ctx = mp.get_context('spawn')
                result_queue = ctx.Queue()
                
                mp.spawn(
                    all_reduce_dist,
                    args=(world_size, data_size_mb, backend, benchmark_steps, warmup_steps, result_queue),
                    nprocs=world_size,
                    join=True
                )

                avg_time, std_time = result_queue.get()

                results.append({
                    "Data size (MB)": data_size_mb,
                    "Number of Processes": world_size,
                    "Average Time (ms)": avg_time
                })
        
        df = pd.DataFrame(results)
        print(df.to_markdown(f"{backend}_all_reduce.md", index=False))
