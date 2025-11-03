#!/bin/bash
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"

#model:small, medium, large, xl, 2.7B
#mode:forward, forward_backward
#context_length:128, 256, 512, 1024

# run指定model + model + context length
model_type=small
context_length=256
mode=forward_backward
#mode=forward
#nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result_nvtx_${model_type}_${mode}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode $mode --context_length $context_length > benchmark_nvtx_${model_type}_${context_length}_${mode}_${current_time}.log 2>&1
#加nvtx annotated_scaled_dot_product_attention
#nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result_nvtx_atten_${model_type}_${mode}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode $mode --context_length $context_length > benchmark_nvtx_atten_${model_type}_${context_length}_${mode}_${current_time}.log 2>&1
# 使用混合精度
nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result_nvtx_mixprecision_${model_type}_${mode}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode $mode --context_length $context_length --mixed_precision > benchmark_nvtx_mixprecision_${model_type}_${context_length}_${mode}_${current_time}.log 2>&1
#nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result python cs336_systems/benchmark/benchmark.py --all


# run指定model + mode, all context length
model_type=2.7B
#mode=forward
mode=forward_backward
context_length=all
#nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result_nvtx_${model_type}_${mode}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode $mode > benchmark_nvtx_${model_type}_all_${mode}_${current_time}.log 2>&1
# 使用混合精度
#nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result_nvtx_mixprecision_${model_type}_${mode}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode $mode --context_length $context_length --mixed_precision > benchmark_nvtx_mixprecision_${model_type}_${context_length}_${mode}_${current_time}.log 2>&1

