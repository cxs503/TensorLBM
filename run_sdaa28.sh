#!/bin/bash
# SDAA:28 - SUBOFF L=40 then L=160
cd /root/TensorLBM_dev
export PYTHONPATH=src

echo "=== SDAA:28 SUBOFF L=40 ==="
python bb_fix_retest_worker.py suboff 40 28 results_bb_fix_retest/suboff_L40_sdaa28.json 2>&1

echo "=== SDAA:28 SUBOFF L=160 ==="
python bb_fix_retest_worker.py suboff 160 28 results_bb_fix_retest/suboff_L160_sdaa28.json 2>&1

echo "=== SDAA:28 DONE ==="
