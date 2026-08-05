#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || $1 != GPU-* ]]; then
  echo "usage: $0 PHYSICAL_GPU_UUID [RESULT_DIR]" >&2
  exit 2
fi
gpu=$1
result_dir=${2:-results/sphere_projected_pressure}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p "$result_dir"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$gpu
stem="$result_dir/sphere-v10-natural-kbc-projected-linear-r9-7200"
resume=()
if [[ -f "$stem.ckpt" ]]; then
  resume=(--resume)
fi

exec "$python" examples/sphere_bfl_cv_validate.py \
  --device cuda:0 --nx 216 --ny 144 --nz 144 --radius 9 \
  --reynolds 100 --lattice-speed 0.06 \
  --collision-model natural_kbc_d3q19 --collision-chunk-cells 262144 \
  --compile-natural-kbc --steps 7200 --warmup-steps 4800 \
  --ramp-steps 720 --sponge-width 18 --sponge-strength 0.2 \
  --cv-margin 6 --report-interval 360 --checkpoint-interval 720 \
  --checkpoint "$stem.ckpt" --statistics-window-steps 2400 \
  --minimum-statistics-convective-times 5 \
  --far-field-mode non_equilibrium_extrapolation \
  --projected-pressure-interval 30 \
  --projected-pressure-reconstruction linear \
  --output "$stem.json" "${resume[@]}"
