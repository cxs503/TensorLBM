#!/usr/bin/env bash
# Two-grid launch for the sphere Re=100 Ladd-MEM (surface-restricted) path,
# mirroring the cylinder_3d recipe, plus the canonical BFL/CV two grids.
set -u
cd /root/TensorLBM_feat2
export OMP_NUM_THREADS=2
export FT_TIMEOUT=86400
export PYTHONPATH=/root/TensorLBM_feat2/src:/root/TensorLBM_feat2/benchmarks
OUT=runs/sphere_re100_20260930
mkdir -p "$OUT"

# --- Ladd MEM surface-restricted, staircase sphere + half-way BB + far-field ---
launch_mem() {  # D dev steps
  local D=$1 dev=$2 steps=$3
  setsid nohup python3 benchmarks/pending/sphere_re100/run_mem_surface.py single "$D" \
    --out "$OUT/mem_D${D}.json" --save-field --steps "$steps" \
    --lateral 16 --up 4 --down 12 --u-in 0.06 \
    --device "sdaa:$dev" --sample 100 --compile-mode eager \
    > "$OUT/mem_D${D}.log" 2>&1 < /dev/null &
  echo "launched mem D=$D dev=$dev steps=$steps pid=$!"
}

# --- canonical BFL/CV library runner (Tensor->JSON crash fixed) ---
launch_can() {  # R dev
  local R=$1 dev=$2
  setsid nohup python3 benchmarks/pending/sphere_re100/run_canonical.py \
    --radius "$R" --width-over-r 16 --length-over-r 24 \
    --device "sdaa:$dev" --output "$OUT/canonical_R${R}.json" \
    > "$OUT/canonical_R${R}.log" 2>&1 < /dev/null &
  echo "launched canonical R=$R dev=$dev pid=$!"
}

launch_mem 12 24 7000
launch_mem 18 25 7000
launch_can 9 26
launch_can 12 27
echo "all launched"