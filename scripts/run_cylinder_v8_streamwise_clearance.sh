#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "usage: $0 PHYSICAL_GPU_UUID" >&2
  exit 2
fi
gpu=$1
[[ $gpu == GPU-* ]] || exit 2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$gpu
result_dir=results/cylinder_collision_sensitivity
stem=$result_dir/cylinder-v8-planar-mach003-r9-streamwise-10d-20d-108000
resume=()
if [[ -f $stem.ckpt ]]; then
  resume=(--resume)
fi

# Relative to v7, only streamwise clearance changes: the cylinder-centre
# distances become 10D upstream and 20D downstream.  Lateral width, collision,
# resolution, Mach number and all convective time windows remain unchanged.
exec "$python" examples/cylinder_bfl_cv_validate.py \
  --device cuda:0 --nx 540 --ny 360 --nz 3 \
  --radius 9 --center-x-fraction 0.3333333333333333 \
  --reynolds 100 --lattice-speed 0.03 \
  --collision-model planar_cumulant_d2q9 \
  --steps 108000 --warmup-steps 63000 --ramp-steps 900 \
  --sponge-width 18 --sponge-strength 0.2 --cv-margin 6 \
  --report-interval 900 --checkpoint-interval 9000 \
  --checkpoint "$stem.ckpt" --statistics-window-steps 45000 \
  --minimum-shedding-cycles 8 \
  --far-field-mode non_equilibrium_extrapolation \
  --output "$stem.json" "${resume[@]}"
