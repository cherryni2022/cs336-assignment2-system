#!/bin/bash
#small, medium, large, xl, 2.7B
model_type=2.7B
context_length=256
#mode=forward or forward_backward
mode=forward
#普通执行不用nsys profile启动, 只基于python time统计时间
# 1.指定model,context,mode
#uv run python cs336_systems/benchmark_basic.py --model_type $model_type --mode $mode --context_length $context_length > benchmark_${model_type}_${context_length}_${mode}.log 2>&1

# 2.测试mode=forward所有model size, context length
model_type=all
mode=forward
#uv run python cs336_systems/benchmark_basic.py --all --mode $mode > benchmark_${model_type}_all_${mode}.log 2>&1

# 3.测试mode=forward_backward 指定model size,context length
model_type=xl
mode=forward_backward
context_length=all
uv run python cs336_systems/benchmark_basic.py --model_type $model_type --mode $mode > benchmark_${model_type}_${context_length}_${mode}.log 2>&1
#nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result python cs336_systems/benchmark/benchmark.py --all
#nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result python cs336_systems/benchmark/benchmark.py --all
#nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result_nvtx_${model_type}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode forward --context_length $context_length
#测试ddp不同实现:naive,flat, individual, bucketed的性能对比
