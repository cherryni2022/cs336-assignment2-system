#!/bin/bash
#small, medium, large, xl, 2.7B
model_type=small
context_length=128
#nsys profile --trace=cuda,nvtx,osrt --python-backtrace=cuda --force-overwrite true -o result python cs336_systems/benchmark/benchmark.py --all
#nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result python cs336_systems/benchmark/benchmark.py --all
nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result_nvtx_${model_type}_${context_length} python cs336_systems/benchmark_nvtx.py --model_type $model_type --mode forward --context_length $context_length
#测试ddp不同实现:naive,flat, individual, bucketed的性能对比
