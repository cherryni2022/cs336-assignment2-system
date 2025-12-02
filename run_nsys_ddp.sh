#!/bin/bash
#测试ddp不同实现:naive,flat, individual, bucketed的性能对比
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $current_time"
#--------------- ddp 部分的测试 -------------------------
# individual ddp 单元测试
# uv run pytest tests/test_ddp_individual_parameters.py
# bucket ddp 单元测试
# uv run pytest tests/test_ddp.py

#xl model
model_type=xl
#naive,flat_dpp,individual_ddp, bucketed_ddp
#ddp_type=naive
ddp_type=flat_ddp
#ddp_type=individual_ddp

#python timer 统计train和通信耗时
#python cs336_systems/parallel/ddp_all_benchmark.py --ddp_type $ddp_type --model_type $model_type > benchmark_${ddp_type}_${model_type}_${current_time}.log 2>&1
#ddp_type=bucketed_ddp
#1MB,10MB,100MB,1000MB
#bucket_size_mb=1
#python cs336_systems/parallel/ddp_all_benchmark.py --ddp_type $ddp_type --model_type $model_type --bucket_size_mb $bucket_size_mb > benchmark_${ddp_type}_${model_type}_${current_time}.log 2>&1


#nsys profile
#python cs336_systems/parallel/ddp_nsys_benchmark.py --ddp_type $ddp_type --model_type $model_type > benchmark_${ddp_type}_${model_type}_${current_time}.log 2>&1
#naive,flat_dpp,individual_ddp, bucketed_ddp
# ddp_type=individual_ddp
# nsys profile \
#     --trace=cuda,nvtx,osrt \
#     --python-backtrace=cuda \
#     --force-overwrite true \
#     -o "nsys_${ddp_type}_${model_type}" \
#     python cs336_systems/parallel/ddp_nsys_benchmark.py \
#     --ddp_type $ddp_type \
#     --model_type $model_type > \
#     benchmark_${ddp_type}_${model_type}_${current_time}.log 2>&1

ddp_type=bucketed_ddp
#1MB,10MB,100MB,1000MB
bucket_size_mb=1000
nsys profile \
    --trace=cuda,nvtx,osrt \
    --python-backtrace=cuda \
    --force-overwrite true \
    -o "nsys_${ddp_type}_${model_type}_${bucket_size_mb}mb" \
    python cs336_systems/parallel/ddp_nsys_benchmark.py \
    --ddp_type $ddp_type \
    --model_type $model_type --bucket_size_mb $bucket_size_mb > \
    benchmark_${ddp_type}_${model_type}_${bucket_size_mb}mb_${current_time}.log 2>&1
