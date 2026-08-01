#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 L90|L120|L150 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  echo "       SUBOFF_HULL_TYPE=bare_hull|full (default: bare_hull)" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
level=${1#L}
gpu=$2
result_dir=${3:-results/amr_campaign_20260801}
wait_for_pid=${4:-}
hull_type=${SUBOFF_HULL_TYPE:-bare_hull}
if [[ "$hull_type" != bare_hull && "$hull_type" != full ]]; then
  echo "SUBOFF_HULL_TYPE must be bare_hull or full" >&2
  exit 2
fi

case "$level" in
  90)
    nx=450; cross=90; outer_wall=6; outer_wake=75
    surface=30; steps=12000; warmup=4500; report=375
    ramp=3000; statistics=7500; wall_diagnostic=60; sponge=18
    checkpoint_interval=750
    ;;
  120)
    nx=600; cross=120; outer_wall=8; outer_wake=100
    surface=40; steps=16000; warmup=6000; report=500
    ramp=4000; statistics=10000; wall_diagnostic=80; sponge=24
    checkpoint_interval=1000
    ;;
  150)
    nx=750; cross=150; outer_wall=10; outer_wake=125
    surface=50; steps=20000; warmup=7500; report=625
    ramp=5000; statistics=12500; wall_diagnostic=100; sponge=30
    checkpoint_interval=1250
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
elif command -v python3 >/dev/null 2>&1; then
  python=$(command -v python3)
else
  echo "no Python interpreter found; set TENSORLBM_PYTHON" >&2
  exit 127
fi
if [[ ! -x $python ]]; then
  echo "TensorLBM Python is not executable: $python" >&2
  exit 126
fi
export PYTHONPATH="$root/src:$root/examples${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${TENSORLBM_PREFLIGHT_ONLY:-0} == 1 ]]; then
  exec "$python" -c 'import tensorlbm; print(tensorlbm.__file__)'
fi

variant=
if [[ "$hull_type" == full ]]; then
  variant=-aff8
fi
checkpoint="$result_dir/suboff-nested-v3${variant}-equivalent-l${level}-${steps%000}k.ckpt"
output="$result_dir/suboff-nested-v3${variant}-equivalent-l${level}-${steps%000}k.json"
resume=()
if [[ -f "$checkpoint" ]]; then
  resume=(--resume)
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/suboff_nested_static_amr_smoke.py \
  --device cuda:0 --hull-type "$hull_type" --speed-knots 5.92 \
  --nx "$nx" --ny "$cross" --nz "$cross" --hull-length "$level" \
  --center-x-fraction 0.3 \
  --outer-wall-margin "$outer_wall" --outer-wake-cells "$outer_wake" \
  --inner-wall-margin 4 --inner-wake-cells 8 \
  --cv-margin 4 --aux-cv-margins 2,6 \
  --surface-force-interval "$surface" \
  --steps "$steps" --warmup-steps "$warmup" \
  --statistics-window-steps "$statistics" --ramp-steps "$ramp" \
  --report-interval "$report" --wall-diagnostic-interval "$wall_diagnostic" \
  --minimum-convective-times 8 \
  --minimum-statistics-convective-times 5 \
  --lattice-speed 0.06 --resolved-reynolds 100000 \
  --wall-law musker --stress-exchange-distance 1 \
  --sponge-width "$sponge" --sponge-strength 0.3 \
  --far-field-mode non_equilibrium_extrapolation \
  --memory-bytes-per-cell 742 \
  --checkpoint "$checkpoint" --checkpoint-interval "$checkpoint_interval" \
  --output "$output" "${resume[@]}"
