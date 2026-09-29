#!/bin/bash
# SDAA:31 - Couette grid convergence with BB fix
cd /root/TensorLBM_dev
export PYTHONPATH=src

echo "=== SDAA:31 Couette ny=8 ==="
python grid_conv_bbfix_worker.py couette 8 31 results_grid_conv_bbfix/couette_ny8_sdaa31.json 2>&1

echo "=== SDAA:31 Couette ny=16 ==="
python grid_conv_bbfix_worker.py couette 16 31 results_grid_conv_bbfix/couette_ny16_sdaa31.json 2>&1

echo "=== SDAA:31 Couette ny=32 ==="
python grid_conv_bbfix_worker.py couette 32 31 results_grid_conv_bbfix/couette_ny32_sdaa31.json 2>&1

echo "=== SDAA:31 DONE ==="
