#!/bin/bash
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"

context_length=65536
d_model=128
dtype="bfloat16"
#dtype="float32"
#python cs336_systems/benchmark_flash_attention.py --context_length $context_length --d_model $d_model --dtype $dtype > benchmark_flash_attn_${context_length}_${d_model}_${dtype}_${current_time}.log 2>&1
# run 所有context_length, d_model 组合
context_length=all
d_model=all
dtype="bfloat16"
python cs336_systems/benchmark_flash_attention.py --dtype $dtype > benchmark_flash_attn_${context_length}_${d_model}_${dtype}_${current_time}.log 2>&1
