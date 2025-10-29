#!/bin/bash
#测试ddp不同实现:naive,flat, individual, bucketed的性能对比

#--------------- ddp 部分的测试 -------------------------
# individual ddp 单元测试
# uv run pytest tests/test_ddp_individual_parameters.py
# bucket ddp 单元测试
# uv run pytest tests/test_ddp.py

#small,medium,large,xl,2.7B
model_type=large
#naive,flat_dpp,individual_ddp, bucketed_ddp
ddp_type=naive
nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result_ddp_naive python cs336_systems/parallel/ddp_all_benchmark.py --ddp_type naive --model_type $model_type
#nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result_ddp_flat python cs336_systems/parallel/ddp_all_benchmark.py --ddp_type flat_ddp --model_type large
#nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result_ddp_individual python cs336_systems/parallel/ddp_all_benchmark.py --ddp_type individual_ddp --model_type large
#nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --python-backtrace=cuda --force-overwrite true -o result_ddp_bucket python cs336_systems/parallel/ddp_all_benchmark.py --ddp_type bucketed_ddp --bucket_size_mb 100 --model_type large
