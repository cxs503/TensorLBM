#!/bin/bash
# SDAA:29 - SUBOFF grid convergence, formula='lagrange'
cd /root/TensorLBM_dev
export PYTHONPATH=src

echo "=== SDAA:29 SUBOFF L=40 lagrange ==="
python grid_conv_bbfix_worker.py suboff 40 lagrange 29 results_grid_conv_bbfix/suboff_L40_lag_sdaa29.json 2>&1

echo "=== SDAA:29 SUBOFF L=80 lagrange ==="
python grid_conv_bbfix_worker.py suboff 80 lagrange 29 results_grid_conv_bbfix/suboff_L80_lag_sdaa29.json 2>&1

echo "=== SDAA:29 SUBOFF L=160 lagrange ==="
python grid_conv_bbfix_worker.py suboff 160 lagrange 29 results_grid_conv_bbfix/suboff_L160_lag_sdaa29.json 2>&1

echo "=== SDAA:29 DONE ==="
