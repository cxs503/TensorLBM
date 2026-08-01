#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 W20|W30|W40 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
width=${1#W}
gpu=$2
result_dir=${3:-results/cylinder_domain_convergence}
wait_for_pid=${4:-}

# Keep diameter resolution, streamwise clearance, time horizon, collision,
# and force observers fixed.  Only the lateral blockage ratio changes.
radius=9
diameter=$((2 * radius))
nx=360
case "$width" in
  20) ny=$((20 * diameter)) ;;
  30) ny=$((30 * diameter)) ;;
  40) ny=$((40 * diameter)) ;;
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
stem="$result_dir/cylinder-v4-domain-w${width}d-r9-54k"
resume=()
if [[ -f "$stem.ckpt" ]]; then
  resume=(--resume)
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/cylinder_bfl_cv_validate.py \
  --device cuda:0 --nx "$nx" --ny "$ny" --nz 3 \
  --radius "$radius" --center-x-fraction 0.30 \
  --reynolds 100 --lattice-speed 0.06 \
  --steps 54000 --warmup-steps 31500 --ramp-steps 450 \
  --sponge-width 18 --sponge-strength 0.2 --cv-margin 6 \
  --report-interval 450 --checkpoint-interval 4500 \
  --checkpoint "$stem.ckpt" --statistics-window-steps 22500 \
  --minimum-shedding-cycles 8 \
  --far-field-mode non_equilibrium_extrapolation \
  --output "$stem.json" "${resume[@]}"
