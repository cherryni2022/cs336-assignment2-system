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
from cs336_basics.optimizer import AdamW, get_cosine_lr
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Define the model sizes (from the table)
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
context_length = 256
context_lengths = [128, 256, 512, 1024]
batch_size = 8
rope_theta = 10000.0
warmup_steps = 5
benchmark_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark(model, x, y, mode, model_type, context_length):
    logging.info(f"start run model: {model_type}, mode: {mode}, context_length: {context_length}")
    model.train()
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    max_lr = 1e-3
    min_lr = 1e-5
    warmup_iters = warmup_steps
    total_iters = warmup_steps + benchmark_steps
    global_step = 0
    lossfn = torch.nn.CrossEntropyLoss()

    #@nvtx.range(f"only forward_{model_type}_{context_length}")
    def step_forward():
        with torch.no_grad():
            start = timeit.default_timer()
            _ = model(x)
            torch.cuda.synchronize()
            forward_takes = (timeit.default_timer() - start) * 1000
            return {"forward_takes": forward_takes,
                    "total_takes": forward_takes,
                    }

    def step_forward_backward():
        optimizer.zero_grad()
        start = timeit.default_timer()
        out = model(x)
        torch.cuda.synchronize()
        forward_time = timeit.default_timer()
        forward_takes = (forward_time - start) * 1000
        
        loss = lossfn(out.view(-1, vocab_size), y.view(-1))
        torch.cuda.synchronize()
        loss_time = timeit.default_timer()
        loss_takes = (loss_time - forward_time) * 1000

        loss.backward()
        torch.cuda.synchronize()
        backward_time = timeit.default_timer()
        backward_takes = (backward_time - loss_time) * 1000

        optimizer.step()
        torch.cuda.synchronize()
        optimizer_time = timeit.default_timer()
        optimizer_takes = (optimizer_time - backward_time) * 1000
        total_takes = (optimizer_time - start) * 1000
        logging.info(f"mode {mode} total_takes: {total_takes} ms,forward_takes: {forward_takes} ms, loss_takes: {loss_takes} ms, backward_takes: {backward_takes} ms, optimizer_takes: {optimizer_takes} ms")
        return {
                "total_takes": total_takes,
                "forward_takes": forward_takes, 
                "loss_takes": loss_takes, 
                "backward_takes": backward_takes, 
                "optimizer_takes": optimizer_takes,
                }

    step_fn = step_forward if mode == "forward" else step_forward_backward

    # Warm-up
    logging.info(f"start warmup model: {model_type}, mode: {mode}, context_length: {context_length} ===============")
    #with nvtx.range(f"Warmup_{model_type}_{context_length}_{mode}"):
    for warmup_step in range(warmup_steps):
        lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr
        try:
            time_spend = step_fn()
        except torch.cuda.OutOfMemoryError as e:
            logging.error("warmup step_%s mode %s CUDA OOM: %s", warmup_step, mode, e)
            torch.cuda.empty_cache()
            raise
        logging.info("warmup step_%s mode %s time_spend: %s", warmup_step, mode, time_spend)
        global_step += 1

    # Timed steps
    times = []
    logging.info(f"start benchmark model: {model_type}, mode: {mode}, context_length: {context_length} ================")
    #with nvtx.range(f"Training_{model_type}_{context_length}_{mode}"):
    for step in range(benchmark_steps):
        # start = timeit.default_timer()
        lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr
        try:
            time_spend = step_fn()
        except torch.cuda.OutOfMemoryError as e:
            logging.error("train step_%s mode %s CUDA OOM: %s", step, mode, e)
            torch.cuda.empty_cache()
            raise
        torch.cuda.synchronize()
        global_step += 1
        # take_time = (timeit.default_timer() - start) * 1000
        times.append(time_spend)
        logging.info("train step_%s mode %s time_spend: %s", step, mode, time_spend)

    forward_time = []
    loss_time = []
    backward_time = []
    optimizer_time = []
    total_time = []
    for time_spend in times:
        forward_time.append(time_spend.get("forward_takes", 0))
        loss_time.append(time_spend.get("loss_takes", 0))
        backward_time.append(time_spend.get("backward_takes", 0))
        optimizer_time.append(time_spend.get("optimizer_takes", 0))
        total_time.append(time_spend.get("total_takes", 0))
    # Aggregate and return stats dict safely
    stats = {
        "mean_total": mean(total_time),
        "std_total": stdev(total_time),
        "mean_forward": mean(forward_time),
        "std_forward": stdev(forward_time),
        "mean_backward": mean(backward_time),
        "std_backward": stdev(backward_time),
        "mean_loss": mean(loss_time),
        "std_loss": stdev(loss_time),
        "mean_optimizer": mean(optimizer_time),
        "std_optimizer": stdev(optimizer_time),
    }
    logging.info(f"benchmark model_{model_type},mode_{mode},context_length_{context_length} mean_total: {stats['mean_total']} ms, std_total: {stats['std_total']} ms")
    return stats

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models.")
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--d_ff", type=int, help="Feedforward dimension")
    parser.add_argument("--num_layers", type=int, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    parser.add_argument("--all", action="store_true", help="Run all predefined configurations")
    parser.add_argument("--model_type", type=str, help="Model type (small, medium, large, xl, 2.7B)")
    parser.add_argument("--context_length", type=int, help="Sequence context length")
    parser.add_argument("--mode", type=str, help="Benchmark mode: forward or forward_backward")
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps")
    parser.add_argument("--benchmark_steps", type=int, help="Number of benchmark steps")
    return parser.parse_args()


def main():
    args = parse_args()

    global warmup_steps
    if args.warmup_steps:
        warmup_steps = args.warmup_steps
    
    global benchmark_steps
    if args.benchmark_steps:
        benchmark_steps = args.benchmark_steps
    
    global context_lengths
    if args.context_length:
        context_lengths = [args.context_length]


    configs_to_run = {}

    if args.all:
        configs_to_run = model_configs
    elif args.d_model and args.d_ff and args.num_layers and args.num_heads:
        configs_to_run = {
            "custom":{
            "size": "custom",
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        }}
    elif args.model_type:
        # 检查model_type是否存在于model_configs中
        if args.model_type not in model_configs:
            raise ValueError(f"Model type '{args.model_type}' not found in model_configs")
        model_config = model_configs[args.model_type]
        configs_to_run = {args.model_type: model_configs[args.model_type]}
    else:
        raise ValueError("Must specify either --all or all custom model hyperparameters.")
    
    if args.mode:
        modes = [args.mode]
    else:
        modes = ["forward", "forward_backward"]

    logging.info("\nRunning the following configurations:")
    for cfg in configs_to_run:
        logging.info(f"  - {cfg}")
    

    results = []

    for mode in modes:
        for model_type, config in configs_to_run.items():
            for context_length in context_lengths:
                logging.info(f"Running {config['size']} model, mode: [{mode}], context_length: {context_length}...")

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

                try:
                    stats = benchmark(model, x, y, mode, model_type, context_length)
                except torch.cuda.OutOfMemoryError as e:
                    logging.error("Benchmark OOM for model %s, mode %s, context_length %s: %s", config['size'], mode, context_length, e)
                    stats = {
                        "mean_total": float('nan'),
                        "std_total": float('nan'),
                        "mean_forward": float('nan'),
                        "std_forward": float('nan'),
                        "mean_backward": float('nan'),
                        "std_backward": float('nan'),
                        "mean_loss": float('nan'),
                        "std_loss": float('nan'),
                        "mean_optimizer": float('nan'),
                        "std_optimizer": float('nan'),
                    }
                
                logging.info(f"model {config['size']} [{mode}]: Avg Time = {stats['mean_total']:.6f}ms, Std Dev = {stats['std_total']:.6f}ms")
                del model, x, y 
                torch.cuda.empty_cache()
                

                results.append(
                    {
                        "model_size": config["size"],
                        "Mode": mode,
                        "d_model": config["d_model"],
                        "d_ff": config["d_ff"],
                        "num_layers": config["num_layers"],
                        "num_heads": config["num_heads"],
                        "Context Length": context_length,
                        "Avg Time (ms)": round(stats['mean_total'], 6),
                        "Std Dev (ms)": round(stats['std_total'], 6),
                        "Avg Forward (ms)": round(stats['mean_forward'], 6),
                        "Std Dev Forward (ms)": round(stats['std_forward'], 6),
                        "Avg Backward (ms)": round(stats['mean_backward'], 6),
                        "Std Dev Backward (ms)": round(stats['std_backward'], 6),
                        "Avg Loss (ms)": round(stats['mean_loss'], 6),
                        "Std Dev Loss (ms)": round(stats['std_loss'], 6),
                        "Avg Optimizer (ms)": round(stats['mean_optimizer'], 6),
                        "Std Dev Optimizer (ms)": round(stats['std_optimizer'], 6),
                        "Warmup Steps": warmup_steps,
                        "Benchmark Steps": benchmark_steps,
                    }
                )


    # Output results
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    save_file = f"benchmark_nvtx_results.md"
    # Save to file
    with open(save_file, "w") as f:
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
