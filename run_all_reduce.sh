#!/bin/bash
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"
#试验选项：2,4,6
world_size=2
python cs336_systems/parallel/all_reduce_benchmark.py --world_size 2 > all_reduce_benchmark_${current_time}.log 2>&1
