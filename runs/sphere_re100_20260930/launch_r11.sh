#!/usr/bin/env bash
set -u
cd /root/TensorLBM_feat2
export OMP_NUM_THREADS=2
export FT_TIMEOUT=86400
export PYTHONPATH=/root/TensorLBM_feat2/src:/root/TensorLBM_feat2/benchmarks
OUT=runs/sphere_re100_20260930
mkdir -p "$OUT"
setsid nohup python3 benchmarks/pending/sphere_re100/run_canonical.py \
  --radius 11 --width-over-r 16 --length-over-r 24 \
  --device sdaa:27 --output "$OUT/canonical_R11.json" \
  > "$OUT/canonical_R11.log" 2>&1 < /dev/null &
echo "launched canonical R=11 pid=$!"