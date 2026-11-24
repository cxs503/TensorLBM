#!/bin/bash
# W43 rework1 shard runner — SUBOFF bare-hull tau x grid matrix (18 shards).
# Usage: run_matrix_shard.sh <shard_index 0..17> <out_dir>
#   shard i -> tau = TAUS[i/3], N = GRIDS[i%3]  (6 tau x 3 grids)
# Every case: MRT D3Q19, half-way bounce-back, PRESSURE_FRICTION,
# 400 steps, force_sample_interval=5, eval_window=40 (last 200 steps).
set -euo pipefail
TAUS=(0.5005 0.505 0.510 0.515 0.520 0.530)
GRIDS=(48 64 96)
i=$1
OUT=$2
tau=${TAUS[$((i/3))]}
N=${GRIDS[$((i%3))]}
echo "[shard $i] tau=$tau N=$N -> $OUT"
python benchmarks/verified/suboff_tau_grid_w22/run_case.py \
    --tau "$tau" --resolution "$N" --steps 400 \
    --device cpu --threads 4 --out "$OUT"
