#!/bin/bash
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"

python cs336_systems/benchmark_pytorch_atten.py > benchmark_pytorch_attention_${current_time}.log 2>&1

