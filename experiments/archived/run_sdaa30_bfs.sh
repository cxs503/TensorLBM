#!/bin/bash
# SDAA:30 - BFS retest with Bug 28 fix + BB fix
cd /root/TensorLBM_dev
export PYTHONPATH=src

echo "=== SDAA:30 BFS BBfix ==="
python grid_conv_bbfix_worker.py bfs 30 results_grid_conv_bbfix/bfs_bbfix_sdaa30.json 2>&1

echo "=== SDAA:30 DONE ==="
