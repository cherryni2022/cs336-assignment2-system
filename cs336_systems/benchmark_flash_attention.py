import torch
import torch.cuda.nvtx as nvtx
import triton
import pandas as pd
from typing import Tuple
import logging
import argparse
import math
from itertools import product

logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# torch._dynamo.config.verbose = True
# torch.autograd.set_detect_anomaly(True)
# torch._inductor.config.debug = True

#from cs336_systems.flash_atten.flash_attention import TritonFlashAttention2
from cs336_systems.flash_atten.flash_atten_triton import FlashAttentionTritonImpl
from cs336_basics.model import scaled_dot_product_attention

def generate_inputs(batch_size: int, context_length: int, d_model: int, dtype: torch.dtype, device: str = "cuda") -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = torch.randn(batch_size, context_length, d_model, device=device, dtype=dtype)
    k = torch.randn(batch_size, context_length, d_model, device=device, dtype=dtype)
    v = torch.randn(batch_size, context_length, d_model, device=device, dtype=dtype)
    do = torch.randn(batch_size, context_length, d_model, device=device, dtype=dtype)
    return q, k, v, do

def flash_attn_triton_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                        is_causal: bool = True, context_length=None, d_model=None, dtype=None) -> torch.Tensor:
    with nvtx.range(f"triton_fwd_pass_{context_length}_{d_model}_{dtype}"):
        out = FlashAttentionTritonImpl.apply(q, k, v, is_causal)
    return out

def flash_attn_bwd(q, k, v, is_causal, do, context_length=None, d_model=None, dtype=None):
    with nvtx.range(f"flash_attn_forward_{context_length}_{d_model}_{dtype}"):
        out = FlashAttentionTritonImpl.apply(q, k, v, is_causal)
    #out = flash_attn_triton_fwd(q, k, v, is_causal, context_length, d_model, dtype)
    with nvtx.range(f"flash_attn_backward_{context_length}_{d_model}_{dtype}"):
        out.backward(do)

def pytorch_regular_attn_fwd(q, k, v, is_causal=True, context_length=None, d_model=None, dtype=None):
    with nvtx.range(f"regular_attn_fwd_pass_{context_length}_{d_model}_{dtype}"):
        if is_causal:
            mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool), diagonal=1)
            return scaled_dot_product_attention(q, k, v, mask=mask)
        else:
            return scaled_dot_product_attention(q, k, v, mask=None)

def pytorch_regular_attn_bwd(q, k, v, is_causal, do, context_length=None, d_model=None, dtype=None):
    with nvtx.range(f"regular_attn_forward_{context_length}_{d_model}_{dtype}"):
        if is_causal:
            mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool), diagonal=1)
            out = scaled_dot_product_attention(q, k, v, mask=mask)
        else:
            out = scaled_dot_product_attention(q, k, v, mask=None)
        #out = pytorch_regular_attn_fwd(q, k, v, is_causal, context_length, d_model, dtype)
    with nvtx.range(f"regular_attn_backward_{context_length}_{d_model}_{dtype}"):
        out.backward(do)

context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
d_models = [16, 32, 64, 128]
dtypes = [torch.bfloat16, torch.float32]

def benchmark_flash_attn(context_lengths, d_models, dtypes):
    results_by_dtype = {dtype: [] for dtype in dtypes}

    for dtype in dtypes:
        for d_model, context_length in product(d_models, context_lengths):
        # for context_length in context_lengths:
        #     # if dtype == torch.float32 and context_length == 65536:
        #     #     logging.error(f"Skip context_length={context_length}, dtype={dtype} due to memory constraints")
        #     #     continue

        #     for d_model in d_models:
            # Skip configurations that would exceed GPU memory
            # if context_length * d_model * (4 if dtype == torch.float32 else 2) > 2**31:
            #     logging.error(f"Skip context_length={context_length}, d_model={d_model}, dtype={dtype} due to memory constraints"
            #                     f",context_length * d_model * 4(or 2) > 2^31")
            #     continue

            logging.info(f"start benchmarking dtype={dtype}, context_length={context_length}, d_model={d_model}")
            # These are not actually being used
            if context_length <= 2048:
                q_tile_size = 64
                k_tile_size = 64
            elif context_length <= 8192:
                q_tile_size = 32
                k_tile_size = 32
            else:
                q_tile_size = 16
                k_tile_size = 16

            # Benchmark regular pytorch attention
            q, k, v, do = generate_inputs(1, context_length, d_model, dtype)
            # 1.benchmark pytorch regular attention forward pass
            try:
                regular_attn_fwd_time = triton.testing.do_bench(lambda: pytorch_regular_attn_fwd(q, k, v, False, context_length, d_model, dtype))
                logging.info(
                    f"[dtype={dtype}, context_length={context_length}, d_model={d_model}]"
                    f", pytorch_regular_attn_fwd forward pass regular_attn_fwd_time: {regular_attn_fwd_time:.2f} ms")
            except Exception as e:
                logging.error(
                    f"[dtype={dtype}, context_length={context_length}, d_model={d_model}]"
                    f", Exception in pytorch_regular_attn_fwd: {e}")

            # 2.benchmark pytorch regular attention forward+backward pass
            try:
                q.requires_grad_(True)
                k.requires_grad_(True)
                v.requires_grad_(True)
                regular_attn_full_time = triton.testing.do_bench(lambda: (pytorch_regular_attn_bwd(q, k, v, False, do, context_length, d_model, dtype), torch.cuda.synchronize()))
                regular_attn_bwd_time = regular_attn_full_time - regular_attn_fwd_time
                logging.info(
                    f"[dtype={dtype}, context_length={context_length}, d_model={d_model}]"
                    f", pytorch_regular_attn_bwd full pass: "
                    f" regular_attn_fwd_time:{regular_attn_fwd_time:.2f} ms"
                    f", regular_attn_bwd_time:{regular_attn_bwd_time:.2f} ms"
                    f", regular_attn_full_time:{regular_attn_full_time:.2f} ms")
            except Exception as e:
                logging.error(
                    f"[dtype={dtype}, context_length={context_length}, d_model={d_model}]"
                    f", Exception in pytorch_regular_attn_bwd: {e}")
            


            # Benchmark partial Triton
            # 3.benchmark triton flash attention forward pass
            q, k, v, do = generate_inputs(1, context_length, d_model, dtype)
            try:
                triton_fwd_time = triton.testing.do_bench(lambda: flash_attn_triton_fwd(q, k, v, False, context_length, d_model, dtype))
                logging.info(
                    f"[dtype={dtype}, context_length={context_length}, d_model={d_model}]"
                    f", triton_flash_attn_fwd forward pass triton_fwd_time: {triton_fwd_time:.2f} ms")
            except Exception as e:
                logging.error(
                    f"benchmarking dtype={dtype}, context_length={context_length}, d_model={d_model}"
                    f", Exception in flash_attn_triton_fwd: {e}")
                continue
            # 4.benchmark triton flash attention forward+backward pass
            try:
                q.requires_grad_(True)
                k.requires_grad_(True)
                v.requires_grad_(True)
                #warmup for backward: compile 耗时
                for _ in range(3):
                    start_time = timeit.default_timer()
                    flash_attn_bwd(q, k, v, False, do, context_length, d_model, dtype)
                    end_time = timeit.default_timer()
                    logging.info(f"[dtype={dtype}, context_length={context_length}, d_model={d_model}]"
                        f", warmup flash_attn_bwd {_} time: {(end_time - start_time)*1000:.2f} ms")
                torch.cuda.synchronize()

                flash_attn_full_time = triton.testing.do_bench(lambda: (flash_attn_bwd(q, k, v, False, do, context_length, d_model, dtype), torch.cuda.synchronize()))
                flash_attn_bwd_time = flash_attn_full_time - triton_fwd_time
                logging.info(
                    f"[dtype={dtype}, context_length={context_length}, d_model={d_model}]"
                    f", flash_attn_bwd full pass "
                    f", triton_fwd_time:{triton_fwd_time:.2f} ms"
                    f", flash_attn_bwd_time:{flash_attn_bwd_time:.2f} ms"
                    f", flash_attn_full_time:{flash_attn_full_time:.2f} ms")
            except Exception as e:
                logging.error(
                    f"[dtype={dtype}, context_length={context_length}, d_model={d_model}]"
                    f", Exception in flash_attn_bwd full pass: {e}")
                continue
            
            results_by_dtype[dtype].append({
                'context_length': context_length,
                'd_model': d_model,
                'regular_attn_forward_ms': round(regular_attn_fwd_time, 2),
                'regular_attn_backward_ms': round(regular_attn_bwd_time, 2),
                'regular_attn_total_ms': round(regular_attn_full_time, 2),
                'flash_attn_triton_forward_ms': round(triton_fwd_time, 2),
                'flash_attn_backward_ms': round(flash_attn_bwd_time, 2),
                'triton_total_ms': round(flash_attn_full_time, 2),
            })

            torch.cuda.empty_cache()

    # Create DataFrames and convert to LaTeX tables
    latex_tables = {}
    for dtype, results in results_by_dtype.items():
        df = pd.DataFrame(results)
        dtype_str = str(dtype).split('.')[-1]
        save_file = f"flash_attn_benchmark_{dtype_str}.md"
        with open(save_file, "w") as f:
            f.write(df.to_markdown(index=False))

        latex_table = df.to_latex(index=False, float_format=lambda x: '{:.2f}'.format(x))
        latex_tables[dtype_str] = latex_table
        print("\nLaTeX Table:")
        print(latex_table)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark flashAttention.")
    #choices=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    parser.add_argument("--context_length", type=str, help="Sequence context length")
    parser.add_argument("--d_model", type=int, help="Model dimension", default="16", choices=[16, 32, 64, 128])
    parser.add_argument("--dtype", type=str, help="Data type", default="bfloat16", choices=["bfloat16", "float32"])
    return parser.parse_args()

def main():
    args = parse_args()
    if args.d_model:
            # 将输入的字符串按逗号分割并转换为整数列表
        test_d_models = [args.d_model]
    else:
        test_d_models = d_models

    if args.context_length:
        test_context_lengths = [int(x) for x in args.context_length.split(',')]
    else:
        test_context_lengths = context_lengths
    
    if args.dtype:
        test_dtypes = [torch.bfloat16 if args.dtype == "bfloat16" else torch.float32]
    else:
        test_dtypes = dtypes
    
    logging.info(f"Benchmark flash_atten for context_lengths: {test_context_lengths}"
                f", d_models: {test_d_models}"
                f", dtypes: {test_dtypes}")
    benchmark_flash_attn(test_context_lengths, test_d_models, test_dtypes)
    logging.info(f"finish benchmark flash_attention")

if __name__ == "__main__":
    main()