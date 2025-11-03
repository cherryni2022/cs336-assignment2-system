#!/bin/bash
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"

#model:small, medium, large, xl, 2.7B
#mode:forward, forward_backward
#context_length:128, 256, 512, 1024

# run指定model + model + context length
model_type=2.7B
context_length=128
#mode=forward
mode=forward_backward
# 非混合精度 fp32
python cs336_systems/benchmark_memory_profile.py --model_type $model_type --mode $mode --context_length $context_length > benchmark_memory_profile_fp32_${model_type}_${context_length}_${mode}_${current_time}.log 2>&1
# 混合精度 bf16
#python cs336_systems/benchmark_memory_profile.py --model_type $model_type --mode $mode --context_length $context_length --mixed_precision > benchmark_memory_profile_bf16_${model_type}_${context_length}_${mode}_${current_time}.log 2>&1


# run指定model + mode, all context length
model_type=2.7B
#mode=forward
mode=forward_backward
context_length=all
#nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result_nvtx_${model_type}_${mode}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode $mode > benchmark_nvtx_${model_type}_all_${mode}_${current_time}.log 2>&1
# 使用混合精度
#nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result_nvtx_mixprecision_${model_type}_${mode}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode $mode --context_length $context_length --mixed_precision > benchmark_nvtx_mixprecision_${model_type}_${context_length}_${mode}_${current_time}.log 2>&1

