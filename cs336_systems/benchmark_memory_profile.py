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
from einops import rearrange, reduce, einsum, repeat
from cs336_basics.model import BasicsTransformerLM,scaled_dot_product_attention
from cs336_basics.optimizer import AdamW, get_cosine_lr
from contextlib import nullcontext
import math
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

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
context_lengths = [128, 256, 512]
batch_size = 8
rope_theta = 10000.0
warmup_steps = 5
benchmark_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output
# Use module-level monkey patch so that CausalMultiHeadSelfAttention in cs336_basics.model
# calls our NVTX-annotated attention during forward.
# import cs336_basics.model as basics_model
# basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def benchmark(model, x, y, mode, model_type, context_length, mixed_precision=False):
    # scaled_dot_product_attention = annotated_scaled_dot_product_attention  # ineffective local alias; removed
    logging.info(f"start run model: {model_type}, mode: {mode}, context_length: {context_length}, mixed_precision: {mixed_precision}")
    model.train()
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    max_lr = 1e-3
    min_lr = 1e-5
    warmup_iters = warmup_steps
    total_iters = warmup_steps + benchmark_steps
    global_step = 0
    lossfn = torch.nn.CrossEntropyLoss()
    ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if mixed_precision
        else nullcontext()
    )

    def warmup_forward():
        with torch.no_grad():
            _ = model(x)
        
    def warmup_forward_backward():
        optimizer.zero_grad()
        out = model(x)
        loss = lossfn(out.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

    def train_forward():
        with torch.no_grad():
            _ = model(x)

    
    def train_forward_backward():
        optimizer.zero_grad()
        out = model(x)
        loss = lossfn(out.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

    step_fn = train_forward if mode == "forward" else train_forward_backward
    # 排除warmup的nvtx统计
    warmup_fn = warmup_forward if mode == "forward" else warmup_forward_backward
    # Warm-up
    logging.info(f"start warmup model: {model_type}, mode: {mode}, context_length: {context_length} ===============")
    for warmup_step in range(warmup_steps):
        lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr
        start = timeit.default_timer()
        try:
            with ctx:
                warmup_fn()
        except torch.cuda.OutOfMemoryError as e:
            logging.error("warmup step_%s mode %s CUDA OOM: %s", warmup_step, mode, e)
            torch.cuda.empty_cache()
            raise
        torch.cuda.synchronize()
        end = timeit.default_timer()
        logging.info("warmup step_%s mode %s time_spend: %s ms", warmup_step, mode, (end - start)*1000)
        global_step += 1

    # Timed steps
    times = []
    logging.info(f"start benchmark model: {model_type}, mode: {mode}, context_length: {context_length} ================")
    torch.cuda.memory._record_memory_history(max_entries=10000)
    for step in range(benchmark_steps):
        start = timeit.default_timer()
        lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr

        try:
            with ctx:
                step_fn()
        except torch.cuda.OutOfMemoryError as e:
            logging.error("train step_%s mode %s CUDA OOM: %s", step, mode, e)
            torch.cuda.empty_cache()
            raise
        
        torch.cuda.synchronize()
        global_step += 1
        take_time = (timeit.default_timer() - start) * 1000
        times.append(take_time)
        logging.info("train step_%s mode %s time_spend: %s ms", step, mode, take_time)
        
    is_mixed_precision_str = "bf16" if mixed_precision else "fp32"
    torch.cuda.memory._dump_snapshot(f"memory_snapshot_{model_type}_{context_length}_{mode}_{is_mixed_precision_str}.pickle")
    torch.cuda.memory._record_memory_history(enabled=False)

    # ========= 打印峰值显存信息 =========
    allocated = torch.cuda.memory_allocated() / 1024**2
    allocated_GB = torch.cuda.memory_allocated() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    max_allocated_GB = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**2
    reserved_GB = torch.cuda.memory_reserved() / 1024**3
    logging.info(f"benchmark model_{model_type},mode_{mode},context_length_{context_length}"
                 f"GPU Allocated: {allocated:.1f} MB ({allocated_GB:.3f} GB), "
                 f"Max Allocated: {max_allocated:.1f} MB ({max_allocated_GB:.3f} GB), "
                 f"Reserved: {reserved:.1f} MB ({reserved_GB:.3f} GB)")
    logging.info(f"benchmark model_{model_type},mode_{mode},context_length_{context_length} mean time: {mean(times)} ms, std time: {stdev(times)} ms")
    return mean(times), stdev(times)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models.")
    parser.add_argument("--model_type", type=str, default="2.7B", help="Model type (small, medium, large, xl, 2.7B)")
    parser.add_argument("--context_length", type=int, default=128, help="Sequence context length")
    parser.add_argument("--mode", type=str, default="forward_backward", help="Benchmark mode: forward or forward_backward")
    parser.add_argument("--mixed_precision", action="store_true", default=False, help="Enable mixed precision training")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--benchmark_steps", type=int, default=10, help="Number of benchmark steps")
    return parser.parse_args()


def main():
    args = parse_args()
    
    global context_lengths
    if args.context_length:
        context_name = f"context_{args.context_length}"
        context_lengths = [args.context_length]


    configs_to_run = {}
    model_name = "all"
    if args.model_type:
        # 检查model_type是否存在于model_configs中
        if args.model_type not in model_configs:
            raise ValueError(f"Model type '{args.model_type}' not found in model_configs")
        model_name = args.model_type
        configs_to_run = {args.model_type: model_configs[args.model_type]}
    else:
        raise ValueError("Must specify either --all or all custom model hyperparameters.")
    
    mode_name = "all"
    if args.mode:
        mode_name = args.mode
        modes = [args.mode]
    else:
        modes = ["forward", "forward_backward"]

    logging.info(f"\nRunning model {model_name}, mode {mode_name},"
                f" context length {context_name}, mixed precision {args.mixed_precision}:")
    
    for cfg in configs_to_run:
        logging.info(f"{cfg}")

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
                    avg, std = benchmark(model, x, y, mode, model_type, context_length, args.mixed_precision)
                except torch.cuda.OutOfMemoryError as e:
                    logging.error("Benchmark OOM for model %s, mode %s, context_length %s: %s", config['size'], mode, context_length, e)
                    avg = float('nan')
                    std = float('nan')

                logging.info(f"model {config['size']} [{mode}] [context length {context_length}]: Avg Time = {avg:.6f}ms, Std Dev = {std:.6f}ms")
                del model, x, y 
                torch.cuda.empty_cache()
                
                results.append(
                    {
                        "model_size": config["size"],
                        "Mode": mode,
                        "Context Length": context_length,
                        "Avg Time (ms)": round(avg, 6),
                        "Std Dev (ms)": round(std, 6),
                    }
                )


    # Output results
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
