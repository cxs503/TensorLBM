#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 R9|R12|R15 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
radius=${1#R}
gpu=$2
result_dir=${3:-results/sphere_grid_convergence}
wait_for_pid=${4:-}

case "$radius" in
  9)
    nx=216; cross=144; steps=7200; warmup=4800; ramp=720
    sponge=18; cv=6; report=360; checkpoint_interval=720; statistics=2400
    ;;
  12)
    nx=288; cross=192; steps=9600; warmup=6400; ramp=960
    sponge=24; cv=8; report=480; checkpoint_interval=960; statistics=3200
    ;;
  15)
    nx=360; cross=240; steps=12000; warmup=8000; ramp=1200
    sponge=30; cv=10; report=600; checkpoint_interval=1200; statistics=4000
    ;;
  *) usage ;;
esac

if [[ -n "$wait_for_pid" ]]; then
  [[ "$wait_for_pid" =~ ^[0-9]+$ ]] || usage
  while kill -0 "$wait_for_pid" 2>/dev/null; do
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p "$result_dir"
python=${TENSORLBM_PYTHON:-$root/.venv/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${TENSORLBM_PREFLIGHT_ONLY:-0} == 1 ]]; then
  exec "$python" -c 'import tensorlbm; print(tensorlbm.__file__)'
fi
stem="$result_dir/sphere-v3-equivalent-r${radius}-${steps}"
resume=()
if [[ -f "$stem.ckpt" ]]; then
  resume=(--resume)
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/sphere_bfl_cv_validate.py \
  --device cuda:0 --nx "$nx" --ny "$cross" --nz "$cross" \
  --radius "$radius" --reynolds 100 --lattice-speed 0.06 \
  --steps "$steps" --warmup-steps "$warmup" --ramp-steps "$ramp" \
  --sponge-width "$sponge" --sponge-strength 0.2 --cv-margin "$cv" \
  --report-interval "$report" --checkpoint-interval "$checkpoint_interval" \
  --checkpoint "$stem.ckpt" --statistics-window-steps "$statistics" \
  --minimum-statistics-convective-times 5 \
  --far-field-mode non_equilibrium_extrapolation \
  --output "$stem.json" "${resume[@]}"
