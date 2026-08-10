#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || $2 != GPU-* ]]; then
  echo "usage: $0 R9|R12|R15 PHYSICAL_GPU_UUID [RESULT_DIR]" >&2
  exit 2
fi
level=${1#R}
gpu=$2
result_dir=${3:-results/sphere_projected_pressure}
case "$level" in
  9)
    nx=216; ny=144; nz=144; steps=7200; warmup=4800; ramp=720
    sponge=18; cv=6; report=360; checkpoint_interval=720; statistics=2400
    projected_interval=30
    ;;
  12)
    nx=288; ny=192; nz=192; steps=9600; warmup=6400; ramp=960
    sponge=24; cv=8; report=480; checkpoint_interval=960; statistics=3200
    projected_interval=40
    ;;
  15)
    nx=360; ny=240; nz=240; steps=12000; warmup=8000; ramp=1200
    sponge=30; cv=10; report=600; checkpoint_interval=1200; statistics=4000
    projected_interval=50
    ;;
  *) exit 2 ;;
esac

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p "$result_dir"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$gpu
stem="$result_dir/sphere-v10-natural-kbc-projected-linear-r${level}-${steps}"
resume=()
if [[ -f "$stem.ckpt" ]]; then
  resume=(--resume)
fi

exec "$python" examples/sphere_bfl_cv_validate.py \
  --device cuda:0 --nx "$nx" --ny "$ny" --nz "$nz" --radius "$level" \
  --reynolds 100 --lattice-speed 0.06 \
  --collision-model natural_kbc_d3q19 --collision-chunk-cells 262144 \
  --compile-natural-kbc --steps "$steps" --warmup-steps "$warmup" \
  --ramp-steps "$ramp" --sponge-width "$sponge" --sponge-strength 0.2 \
  --cv-margin "$cv" --report-interval "$report" \
  --checkpoint-interval "$checkpoint_interval" --checkpoint "$stem.ckpt" \
  --statistics-window-steps "$statistics" \
  --minimum-statistics-convective-times 5 \
  --far-field-mode non_equilibrium_extrapolation \
  --projected-pressure-interval "$projected_interval" \
  --projected-pressure-reconstruction linear \
  --output "$stem.json" "${resume[@]}"
