#!/bin/bash
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"
#small, medium, large, xl, 2.7B
model_type=xl
context_length=128
#mode=forward_backward
mode=forward
#nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result python cs336_systems/benchmark/benchmark.py --all
#nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result python cs336_systems/benchmark/benchmark.py --all
nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result_nvtx_${model_type}_${mode}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode $mode --context_length $context_length > benchmark_${model_type}_${context_length}_${mode}_${current_time}.log 2>&1
#测试ddp不同实现:naive,flat, individual, bucketed的性能对比
