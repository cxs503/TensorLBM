#!/usr/bin/env bash
# Cheap 2D proxy scans (same mask + MEM convention as the 3D extruded case)
# to map Cd_mem vs blockage and vs resolution D.
set -u
cd /root/TensorLBM_feat2
export OMP_NUM_THREADS=2
export PYTHONPATH=/root/TensorLBM_feat2/src
OUT=runs/cyl3d_40D

run2d() {
  local D="$1" dom="$2" tag="$3" steps="$4" dev="$5"
  setsid nohup python benchmarks/pending/cylinder_3d/_ref2d_re40.py \
    --D "$D" --domain-D "$dom" --steps "$steps" --u-in 0.08 \
    --out "$OUT/$tag.json" --device "$dev" --sample 200 \
    > "$OUT/$tag.log" 2>&1 < /dev/null &
  echo "launched $tag D=$D dom=${dom}D steps=$steps dev=$dev pid=$!"
}

run2d 20 16 p2d_D20_dom16 30000 sdaa:0
run2d 20 40 p2d_D20_dom40 40000 sdaa:7
run2d 40 32 p2d_D40_dom32 30000 sdaa:2
echo "2d scans launched"