#!/bin/bash
# SDAA:31 - Couette ny=8, 16, 32 (sequential)
cd /root/TensorLBM_dev
export PYTHONPATH=src

echo "=== SDAA:31 Couette ny=8 ==="
python bb_fix_retest_worker.py couette 8 31 results_bb_fix_retest/couette_ny8_sdaa31.json 2>&1

echo "=== SDAA:31 Couette ny=16 ==="
python bb_fix_retest_worker.py couette 16 31 results_bb_fix_retest/couette_ny16_sdaa31.json 2>&1

echo "=== SDAA:31 Couette ny=32 ==="
python bb_fix_retest_worker.py couette 32 31 results_bb_fix_retest/couette_ny32_sdaa31.json 2>&1

echo "=== SDAA:31 DONE ==="
