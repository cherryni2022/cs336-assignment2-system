import timeit
import torch.cuda.nvtx as nvtx
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import rearrange, einsum
import einx
import pandas as pd
from statistics import mean, stdev
from cs336_basics.model import BasicsTransformerLM
import argparse
from itertools import product


batch_size = 8
d_model_params = [16, 32, 64, 128]
seq_len_params = [256, 1024, 4096, 8192, 16384]
forward_times = 100
backward_times = 100
vocab_size = 10_000
test_context_lengths = [128, 256, 512, 1024]
rope_theta = 10000.0
warmup_steps = 5
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

def benchmark_attention(Q, K, V):
    """
    对注意力机制进行基准测试
    先执行100次forward传播，再执行100次backward传播
    Args:
        Q, K, V: 输入张量
    Returns:
        (forward_mean_time, forward_std_time, forward_memory, 
         backward_mean_time, backward_std_time, backward_memory)
    """
    
    # 确保张量需要梯度用于backward测试
    Q.requires_grad_(True)
    K.requires_grad_(True) 
    V.requires_grad_(True)
    
    def step_forward():
        """执行前向传播"""
        output = annotated_scaled_dot_product_attention(Q, K, V)
        if device == "cuda":
            torch.cuda.synchronize()  # 确保GPU操作完成
        return output
    
    def step_backward():
        """执行前向和后向传播"""
        output = annotated_scaled_dot_product_attention(Q, K, V)
        # 创建一个简单的损失用于反向传播
        loss = output.sum()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()  # 确保GPU操作完成
        # 清零梯度
        Q.grad = None
        K.grad = None
        V.grad = None
        return output
    
    # 预热阶段
    print(f"  预热阶段 ({warmup_steps} 步)...")
    for _ in range(warmup_steps):
        step_forward()
    
    # === Forward传播测试 ===
    print(f"  执行 {forward_times} 次 forward 传播...")
    
    # 清理缓存并记录初始内存
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    else:
        initial_memory = 0
    
    # Forward计时阶段
    forward_times_list = []
    for i in range(forward_times):
        start_time = timeit.default_timer()
        step_forward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        forward_times_list.append(end_time - start_time)
    
    # 记录forward阶段的峰值内存使用
    if device == "cuda":
        forward_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        forward_memory_used = forward_peak_memory - initial_memory
    else:
        forward_memory_used = 0
    
    # === Backward传播测试 ===
    print(f"  执行 {backward_times} 次 backward 传播...")
    
    # 清理缓存并重置峰值内存统计
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory_backward = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    else:
        initial_memory_backward = 0
    
    # Backward计时阶段
    backward_times_list = []
    for i in range(backward_times):
        start = timeit.default_timer()
        step_backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        backward_times_list.append(end_time - start_time)
    
    # 记录backward阶段的峰值内存使用
    if device == "cuda":
        backward_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        backward_memory_used = backward_peak_memory - initial_memory_backward
    else:
        backward_memory_used = 0
    
    return (mean(forward_times_list), stdev(forward_times_list), forward_memory_used,
            mean(backward_times_list), stdev(backward_times_list), backward_memory_used)

def main():
    """
    主函数：遍历不同的d_model和seq_len组合进行基准测试
    """
    print(f"开始注意力机制基准测试")
    print(f"设备: {device}")
    print(f"批量大小: {batch_size}")
    print(f"d_model参数: {d_model_params}")
    print(f"序列长度参数: {seq_len_params}")
    print("="*60)
    
    results = []
    
    for d_model, seq_len in product(d_model_params, seq_len_params):
        print(f"\n测试配置: d_model={d_model}, seq_len={seq_len}")
        
        # 创建随机输入
        Q, K, V = create_random_input(batch_size, d_model, seq_len)
        
        # 执行基准测试（包含forward和backward）
        print("  开始基准测试:")
        (forward_mean_time, forward_std_time, forward_memory,
         backward_mean_time, backward_std_time, backward_memory) = benchmark_attention(Q, K, V)
        
        # 保存结果
        result = {
            'd_model': d_model,
            'seq_len': seq_len,
            'forward_mean_time': forward_mean_time,
            'forward_std_time': forward_std_time,
            'forward_memory_mb': forward_memory,
            'backward_mean_time': backward_mean_time,
            'backward_std_time': backward_std_time,
            'backward_memory_mb': backward_memory
        }
        results.append(result)
        
        print(f"  Forward: {forward_mean_time:.6f}±{forward_std_time:.6f}s, Memory: {forward_memory:.2f}MB")
        print(f"  Backward: {backward_mean_time:.6f}±{backward_std_time:.6f}s, Memory: {backward_memory:.2f}MB")
    
    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv('attention_benchmark_results.csv', index=False)
    print(f"\n结果已保存到 attention_benchmark_results.csv")
    print(df)

if __name__ == "__main__":
    main()
    pass