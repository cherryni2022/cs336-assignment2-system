import timeit
import torch
import pandas as pd
from statistics import mean, stdev
import torch.cuda.nvtx as nvtx
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import rearrange, einsum
import einx
from cs336_basics.model import BasicsTransformerLM
import argparse

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
vocab_size = 10_000
context_length = [128, 256, 512, 1024]
batch_size = 8
rope_theta = 10000.0
warmup_steps = 5
timing_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# for 2.7B model
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

context_lengths = [128, 256, 512]

# 是否activate mixed-precision
use_mixed_precision = True

#torch.cuda.memory._record_memory_history(max_entries=1000000)
#torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
#torch.cuda.memory._record_memory_history(enabled=None)

def benchmark(model, x, y, mode, model_name,context_length):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = torch.nn.CrossEntropyLoss()

    @nvtx.range("only forward pass")
    def step_forward():
        with torch.no_grad():
            _ = model(x)

    @nvtx.range(f"forward_backward_{model_name}_{context_length}")
    def step_forward_backward():
        optimizer.zero_grad()
        with nvtx.range(f"forward_{model_name}_{context_length}"):
            out = model(x)
        with nvtx.range(f"loss_{model_name}_{context_length}"):
            loss = lossfn(out.view(-1, vocab_size), y.view(-1))
        with nvtx.range(f"backward_{model_name}_{context_length}"):
            loss.backward()
        with nvtx.range(f"optimizer_{model_name}_{context_length}"):
            optimizer.step()

    step_fn = step_forward if mode == "forward" else step_forward_backward

    # Warm-up
    for _ in range(warmup_steps):
        step_fn()

    # Timed steps
    times = []
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(timing_steps):
        start = timeit.default_timer()
        step_fn()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)

    torch.cuda.memory._dump_snapshot(f"memory_snapshot_{model_name}_{context_length}_{mode}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    return mean(times), stdev(times)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models.")
    parser.add_argument("--model_type", type=str, help="Model type (small, medium, large, xl, 2.7B)")
    parser.add_argument("--context_length", type=int, help="Sequence context length")
    return parser.parse_args()


def main():
    args = parse_args()

    results = []
    configs_to_run = []

    if args.model_type:
        # 检查model_type是否存在于model_configs中
        if args.model_type not in model_configs:
            raise ValueError(f"Model type '{args.model_type}' not found in model_configs")
        model_config = model_configs[args.model_type]
        configs_to_run = {args.model_type: model_configs[args.model_type]}
    else:
        configs_to_run = {"2.7B": 
                    {"size": "2.7B", 
                        "d_model": 2560, 
                        "d_ff": 10240, 
                        "num_layers": 32, 
                        "num_heads": 32
                    }
                }
    if args.context_length:
        context_length = args.context_length
        context_lens_run = [context_length]
    else:
        context_lens_run = context_lengths
    
    print("\nRunning the following configurations:")
    for cfg in configs_to_run:
        print(f"  - {cfg}")
    print()

    for config in configs_to_run.values():
        for context_length in context_lens_run:
            for mode in ["forward", "forward_backward"]:
                print(f"Running {config['size']} model [{mode}]...")

                model = BasicsTransformerLM(
                    vocab_size=vocab_size,
                    context_length=context_length,
                    d_model=config["d_model"],
                    num_layers=config["num_layers"],
                    num_heads=config["num_heads"],
                    d_ff=config["d_ff"],
                    rope_theta=rope_theta,
                ).to(device)

                # Create random input data  
                x = torch.randint(
                    0, vocab_size, (batch_size, context_length), device=device
                )
                y = torch.randint(
                    0, vocab_size, (batch_size, context_length), device=device
                )

                avg, std = benchmark(model, x, y, mode, config["size"], context_length)
                
                print(f"  - {config['size']} [{mode}]: Avg Time = {avg:.6f}s, Std Dev = {std:.6f}s")
                del model, x, y 
                torch.cuda.empty_cache()
                

                results.append(
                    {
                        "Size": config["size"],
                        "Mode": mode,
                        "d_model": config["d_model"],
                        "d_ff": config["d_ff"],
                        "num_layers": config["num_layers"],
                        "num_heads": config["num_heads"],
                        "Context Length": context_length,
                        "Avg Time (s)": round(avg, 6),
                        "Std Dev (s)": round(std, 6),
                        "Warmup Steps": warmup_steps,
                    }
                )


    # Output results
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

    # Save to file
    with open("benchmark_memory_results.md", "w") as f:
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
