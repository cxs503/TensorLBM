#!/bin/bash
# SDAA:28 - SUBOFF grid convergence, formula='standard'
cd /root/TensorLBM_dev
export PYTHONPATH=src

echo "=== SDAA:28 SUBOFF L=40 standard ==="
python grid_conv_bbfix_worker.py suboff 40 standard 28 results_grid_conv_bbfix/suboff_L40_std_sdaa28.json 2>&1

echo "=== SDAA:28 SUBOFF L=80 standard ==="
python grid_conv_bbfix_worker.py suboff 80 standard 28 results_grid_conv_bbfix/suboff_L80_std_sdaa28.json 2>&1

echo "=== SDAA:28 SUBOFF L=160 standard ==="
python grid_conv_bbfix_worker.py suboff 160 standard 28 results_grid_conv_bbfix/suboff_L160_std_sdaa28.json 2>&1

echo "=== SDAA:28 DONE ==="
