#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 R9|R12|R15 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
radius=${1#R}
gpu=$2
result_dir=${3:-results/cylinder_grid_convergence}
wait_for_pid=${4:-}

case "$radius" in
  9)
    domain=360; steps=54000; warmup=31500; ramp=450
    sponge=18; cv=6; report=450; checkpoint_interval=4500; statistics=22500
    ;;
  12)
    domain=480; steps=72000; warmup=42000; ramp=600
    sponge=24; cv=8; report=600; checkpoint_interval=6000; statistics=30000
    ;;
  15)
    domain=600; steps=90000; warmup=52500; ramp=750
    sponge=30; cv=10; report=750; checkpoint_interval=7500; statistics=37500
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
collision_model=${TENSORLBM_COLLISION_MODEL:-cumulant_d3q19_cs0}
case "$collision_model" in
  cumulant_d3q19_cs0) variant= ;;
  natural_kbc_d3q19) variant=-natural-kbc ;;
  *) echo "unsupported TENSORLBM_COLLISION_MODEL: $collision_model" >&2; exit 2 ;;
esac
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${TENSORLBM_PREFLIGHT_ONLY:-0} == 1 ]]; then
  exec "$python" -c 'import tensorlbm; print(tensorlbm.__file__)'
fi
stem="$result_dir/cylinder-v4${variant}-equivalent-r${radius}-${steps}"
resume=()
if [[ -f "$stem.ckpt" ]]; then
  resume=(--resume)
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/cylinder_bfl_cv_validate.py \
  --device cuda:0 --nx "$domain" --ny "$domain" --nz 3 \
  --radius "$radius" --center-x-fraction 0.30 \
  --reynolds 100 --lattice-speed 0.06 \
  --collision-model "$collision_model" \
  --steps "$steps" --warmup-steps "$warmup" --ramp-steps "$ramp" \
  --sponge-width "$sponge" --sponge-strength 0.2 --cv-margin "$cv" \
  --report-interval "$report" --checkpoint-interval "$checkpoint_interval" \
  --checkpoint "$stem.ckpt" --statistics-window-steps "$statistics" \
  --minimum-shedding-cycles 8 \
  --far-field-mode non_equilibrium_extrapolation \
  --output "$stem.json" "${resume[@]}"
