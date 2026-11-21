#!/usr/bin/env bash
# Launch the cylinder_3d Re=40 40D-domain expansion runs in parallel.
# Each job: run.py single <D> --lateral <L> --nz 1 --compile-mode eager
set -u
cd /root/TensorLBM_feat2
export OMP_NUM_THREADS=2
export FT_TIMEOUT=86400
export PYTHONPATH=/root/TensorLBM_feat2/src
OUT=runs/cyl3d_40D
mkdir -p "$OUT"

launch() {
  local D="$1" tag="$2" steps="$3" lat="$4" dev="$5"
  setsid nohup python benchmarks/pending/cylinder_3d/run.py single "$D" \
    --out "$OUT/$tag.json" --save-field --steps "$steps" --nz 1 --lateral "$lat" \
    --device "$dev" --sample 100 --compile-mode eager \
    > "$OUT/$tag.log" 2>&1 < /dev/null &
  echo "launched $tag D=$D L=$lat steps=$steps dev=$dev pid=$!"
}

launch 20 D20_L40 40000 40 sdaa:3
launch 40 D40_L40 40000 40 sdaa:4
launch 20 D20_L32 30000 32 sdaa:5
launch 40 D40_L32 30000 32 sdaa:6
echo "all launched"