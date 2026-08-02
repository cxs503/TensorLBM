#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 R9|R12|R15|R18 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
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
  18)
    domain=720; steps=108000; warmup=63000; ramp=900
    sponge=36; cv=12; report=900; checkpoint_interval=9000; statistics=45000
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
if [[ -n ${TENSORLBM_PYTHON:-} ]]; then
  python=$TENSORLBM_PYTHON
elif [[ -x $root/.venv/bin/python ]]; then
  python=$root/.venv/bin/python
elif [[ -x /home/wxsc/anaconda3/envs/ftw-env/bin/python ]]; then
  python=/home/wxsc/anaconda3/envs/ftw-env/bin/python
elif command -v python3 >/dev/null 2>&1; then
  python=$(command -v python3)
else
  echo "no Python interpreter found; set TENSORLBM_PYTHON" >&2
  exit 127
fi
collision_model=${TENSORLBM_COLLISION_MODEL:-cumulant_d3q19_cs0}
collision_chunk_cells=${TENSORLBM_COLLISION_CHUNK_CELLS:-0}
[[ $collision_chunk_cells =~ ^[0-9]+$ ]] || {
  echo "TENSORLBM_COLLISION_CHUNK_CELLS must be non-negative" >&2
  exit 2
}
compile_natural_kbc=()
if [[ ${TENSORLBM_COMPILE_NATURAL_KBC:-0} == 1 ]]; then
  [[ $collision_model == natural_kbc_d3q19 ]] || {
    echo "TENSORLBM_COMPILE_NATURAL_KBC requires natural_kbc_d3q19" >&2
    exit 2
  }
  compile_natural_kbc=(--compile-natural-kbc)
elif [[ ${TENSORLBM_COMPILE_NATURAL_KBC:-0} != 0 ]]; then
  echo "TENSORLBM_COMPILE_NATURAL_KBC must be 0 or 1" >&2
  exit 2
fi
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
  --collision-chunk-cells "$collision_chunk_cells" \
  "${compile_natural_kbc[@]}" \
  --steps "$steps" --warmup-steps "$warmup" --ramp-steps "$ramp" \
  --sponge-width "$sponge" --sponge-strength 0.2 --cv-margin "$cv" \
  --report-interval "$report" --checkpoint-interval "$checkpoint_interval" \
  --checkpoint "$stem.ckpt" --statistics-window-steps "$statistics" \
  --minimum-shedding-cycles 8 \
  --far-field-mode non_equilibrium_extrapolation \
  --output "$stem.json" "${resume[@]}"
