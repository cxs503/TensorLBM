#!/bin/bash
# Launch all 4 SUBOFF cases on SDAA cards 0-3
cd /root/TensorLBM_dev
mkdir -p results_suboff_appendage

export PYTHONUNBUFFERED=1

nohup python suboff_appendage_worker.py 1 0 results_suboff_appendage/case1_with_sail.json > results_suboff_appendage/case1_with_sail.log 2>&1 &
echo "Case 1 (WITH_SAIL) launched on SDAA:0, PID=$!"

nohup python suboff_appendage_worker.py 2 1 results_suboff_appendage/case2_full.json > results_suboff_appendage/case2_full.log 2>&1 &
echo "Case 2 (FULL) launched on SDAA:1, PID=$!"

nohup python suboff_appendage_worker.py 3 2 results_suboff_appendage/case3_L160.json > results_suboff_appendage/case3_L160.log 2>&1 &
echo "Case 3 (L=160) launched on SDAA:2, PID=$!"

nohup python suboff_appendage_worker.py 4 3 results_suboff_appendage/case4_4L.json > results_suboff_appendage/case4_4L.log 2>&1 &
echo "Case 4 (4L domain) launched on SDAA:3, PID=$!"

echo "All 4 cases launched."
