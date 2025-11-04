#!/bin/bash
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"

context_length=128
d_model=128
dtype="bfloat16"
python cs336_systems/benchmark_flash_attention.py --context_length $context_length --d_model $d_model --dtype $dtype > benchmark_flash_attn_${context_length}_${d_model}_${dtype}_${current_time}.log 2>&1

