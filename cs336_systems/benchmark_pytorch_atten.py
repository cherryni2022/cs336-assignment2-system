import timeit
import torch.cuda.nvtx as nvtx
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import rearrange, einsum
import einx
import math
import pandas as pd
from statistics import mean, stdev
from cs336_basics.model import BasicsTransformerLM
from contextlib import nullcontext
import argparse
from itertools import product
import logging

logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


batch_size = 8
d_model_params = [16, 32, 64, 128]
seq_len_params = [256, 1024, 4096, 8192, 16384]
# for test
# d_model_params = [128]
# seq_len_params = [256, 1024, 4096]
forward_times = 100
backward_times = 100
vocab_size = 10_000
rope_theta = 10000.0
warmup_steps = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

#@nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = torch.softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output

def create_random_input(batch_size, d_model, seq_len):
    # Create random input data  
    Q = torch.randn(
        (batch_size, seq_len, d_model), device=device
    )
    K = torch.randn(
        (batch_size, seq_len, d_model), device=device
    )
    V = torch.randn(
        (batch_size, seq_len, d_model), device=device
    )
    return Q,K,V


# a.在不同规模下对你的注意力实现进行基准测试。编写一个脚本，该脚本将：
# b.固定批量大小为 8，并且不使用多头注意力（即移除头的维度）。
# c.遍历头的嵌入维度 d_model 的 和序列长度的 的笛卡尔积。
# d.为相应的尺寸创建随机输入 Q, K, V。
# e.预热后, 执行100次forward, 100次backward,分别记录forward和backward 耗时和显存占用情况
# f.使用这些输入计时 100 次通过注意力的前向传播。
# g.测量在后向传播开始前使用了多少内存，并计时 100 次后向传播。
# h.确保进行了预热，并在每次前向/后向传播后调用 torch.cuda.synchronize()。

def benchmark_attention(d_model, context_length, mixed_precision=False):
    """
    对注意力机制进行基准测试
    warmup + 100 times forward pass,
    warmup + 100 times backward pass
    Args:
        Q, K, V: 输入张量
    Returns:
        (forward_mean_time, forward_std_time, forward_memory, 
         backward_mean_time, backward_std_time, backward_memory)
    """

    Q, K, V = create_random_input(batch_size, d_model, context_length)

    compiled_attention = torch.compile(scaled_dot_product_attention)
    ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if mixed_precision
        else nullcontext()
    )

    def step_forward():
        """执行前向传播"""
        output = compiled_attention(Q, K, V)
        return output
    
    def step_backward():
        """执行前向和后向传播"""
        output = compiled_attention(Q, K, V)
        # 创建一个简单的损失用于反向传播
        loss = output.sum()
        loss.backward()
        # 清零梯度
        Q.grad = None
        K.grad = None
        V.grad = None
        return output
    
    # 预热阶段
    logging.info(f"warmup forward pass...")
    try:
        for _ in range(warmup_steps):
            with ctx:
                step_forward()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
            logging.error("d_model=%s, context_length=%s, forward warmup CUDA OOM: %s", d_model, context_length, e)
            torch.cuda.empty_cache()
            raise
    
    # === forward pass ===
    logging.info(f"start run forward pass {forward_times} times...")
    
    torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
    initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    # forward timing
    forward_times_list = []
    try:
        for i in range(forward_times):
            start_time = timeit.default_timer()
            with ctx:
                step_forward()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            forward_times_list.append(end_time - start_time)
    except torch.cuda.OutOfMemoryError as e:
            logging.error("d_model=%s, context_length=%s, forward pass CUDA OOM: %s", d_model, context_length, e)
            torch.cuda.empty_cache()
            raise
    
    # record forward peak memory
    forward_peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    forward_memory_used = forward_peak_memory - initial_memory
    forward_avg_time = mean(forward_times_list)*1000
    forward_std_time = stdev(forward_times_list)*1000
    logging.info(f"[test case] forward pass d_model: {d_model}, context_length:{context_length}"
                 f", peak memory: {forward_peak_memory:.1f} MB, {forward_peak_memory/1024:.3f} GB"
                 f", initial_memory: {initial_memory:.1f} MB, {initial_memory/1024:.3f} GB"
                 f", memory used: {forward_memory_used:.1f} MB, {forward_memory_used/1024:.3f} GB"
                 f", avg_time: {forward_avg_time:.3f} ms"
                 f", std_time: {forward_std_time:.3f} ms")
    
    # === Backward pass ===
    # 确保张量需要梯度用于backward测试
    Q.requires_grad_(True)
    K.requires_grad_(True) 
    V.requires_grad_(True)
    
    # === Backward pass ===
    logging.info(f"warmup backward pass...")
    try:
        for _ in range(warmup_steps):
            with ctx:
                step_backward()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
            logging.error("d_model=%s, context_length=%s, backward warmup CUDA OOM: %s", d_model, context_length, e)
            torch.cuda.empty_cache()
            raise
    

    logging.info(f"start run backward pass {backward_times} times...")
    torch.cuda.reset_peak_memory_stats()
    initial_memory_backward = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    # Backward计时阶段
    backward_times_list = []
    try:
        for i in range(backward_times):
            start_time = timeit.default_timer()
            with ctx:
                step_backward()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            backward_times_list.append(end_time - start_time)
    except torch.cuda.OutOfMemoryError as e:
            logging.error("d_model=%s, context_length=%s, backward pass CUDA OOM: %s", d_model, context_length, e)
            torch.cuda.empty_cache()
            raise
    
    # 记录backward阶段的峰值内存使用
    backward_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    backward_memory_used = backward_peak_memory - initial_memory_backward
    backward_avg_time = mean(backward_times_list)*1000
    backward_std_time = stdev(backward_times_list)*1000
    logging.info(f"[test case] backward pass d_model: {d_model}, context_length:{context_length}"
                 f", peak memory: {backward_peak_memory:.1f} MB, {backward_peak_memory/1024:.3f} GB"
                 f", initial_memory: {initial_memory_backward:.1f} MB, {initial_memory_backward/1024:.3f} GB"
                 f", memory used: {backward_memory_used:.1f} MB, {backward_memory_used/1024:.3f} GB"
                 f", avg_time: {backward_avg_time:.3f} ms"
                 f", std_time: {backward_std_time:.3f} ms")

    return {
        "d_model": d_model,
        "seq_len": context_length,
        "forward_avg_time(ms)": round(forward_avg_time, 3),
        "backward_avg_time(ms)": round(backward_avg_time, 3),
        "forward_peak_memory(GB)": round(forward_peak_memory/1024, 3),
        "backward_peak_memory(GB)": round(backward_peak_memory/1024, 3),
        "forward_memory_used(GB)": round(forward_memory_used/1024, 3),
        "backward_memory_used(GB)": round(backward_memory_used/1024, 3),
        "status": "Success"
    }
    
def main():
    parser = argparse.ArgumentParser(description="Benchmark attention.")
    parser.add_argument("--mixed_precision", action="store_true", default=False, help="Enable mixed precision training")
    args = parser.parse_args()

    logging.info(f"start benchmark attention, d_models: {d_model_params},"
                f"context_length: {seq_len_params}, mixed_precision: {args.mixed_precision}")
    results = []

    for d_model, context_length in product(d_model_params, seq_len_params):
        logging.info(f"start benchmark case: d_model={d_model}, context_length={context_length}")
        
        # 执行基准测试（包含forward和backward）
        try:
            result = benchmark_attention(d_model, context_length)
            results.append(result)
        except RuntimeError as e:
            logging.error(f"RuntimeError: {e}")
            if "out of memory" in str(e).lower():
                mem_atten = batch_size * seq_len * seq_len * 4/(1024**3)
                mem_total = mem_atten + (3 * batch_size * seq_len * d_model * 4)/(1024**3)
                logging.info(f"benchmark case OOM d_model={d_model}, context_length={context_length}, "
                            f"required mem_atten:{mem_atten:.3f} GB, mem_total: {mem_total:.3f} GB")
                results.append(
                {
                    "d_model": d_model,
                    "seq_len": context_length,
                    "forward_avg_time(ms)": "OOM",
                    "backward_avg_time(ms)": "OOM",
                    "forward_peak_memory(GB)": "OOM",
                    "backward_peak_memory(GB)": "OOM",
                    "forward_memory_used(GB)": "OOM",
                    "backward_memory_used(GB)": "OOM",
                    "status": "OOM"
                }
                )
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        logging.info(f"finish test case: d_model={d_model}, context_length={context_length}")
    
    # 保存结果到CSV
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    save_file = f"attention_benchmark_results_{'bf16' if args.mixed_precision else 'fp32'}.md"
    with open(save_file, "w") as f:
        f.write(df.to_markdown(index=False))
    logging.info(f"finish benchmark attention, save result to {save_file}")
    print(df)

if __name__ == "__main__":
    main()