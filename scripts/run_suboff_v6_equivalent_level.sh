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
    nx=450; cross=90; wall=6; wake=75; cv=6; aux=3,9
    surface=30; steps=12000; warmup=4500; report=375; average=1125
    ramp=3000; exchange=2.109375; wall_diagnostic=60; sponge=18
    checkpoint_interval=750
    ;;
  120)
    nx=600; cross=120; wall=8; wake=100; cv=8; aux=4,12
    surface=40; steps=16000; warmup=6000; report=500; average=1500
    ramp=4000; exchange=2.8125; wall_diagnostic=80; sponge=24
    checkpoint_interval=1000
    ;;
  150)
    nx=750; cross=150; wall=10; wake=125; cv=10; aux=5,15
    surface=50; steps=20000; warmup=7500; report=625; average=1875
    ramp=5000; exchange=3.515625; wall_diagnostic=100; sponge=30
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
python=${TENSORLBM_PYTHON:-$root/.venv/bin/python}
variant=
if [[ "$hull_type" == full ]]; then
  variant=-aff8
fi
checkpoint="$result_dir/suboff-v6${variant}-equivalent-l${level}-${steps%000}k.ckpt"
output="$result_dir/suboff-v6${variant}-equivalent-l${level}-${steps%000}k.json"
resume=()
if [[ -f "$checkpoint" ]]; then
  resume=(--resume)
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/suboff_static_amr_resistance.py \
  --device cuda:0 --hull-type "$hull_type" --speed-knots 5.92 \
  --nx "$nx" --ny "$cross" --nz "$cross" --hull-length "$level" \
  --center-x-fraction 0.3 --wall-margin "$wall" --wake-cells "$wake" \
  --cv-margin "$cv" --aux-cv-margins "$aux" \
  --surface-force-interval "$surface" --steps "$steps" \
  --warmup-steps "$warmup" --report-interval "$report" \
  --average-window "$average" --ramp-steps "$ramp" \
  --lattice-speed 0.06 --resolved-reynolds 100000 \
  --collision-model cumulant_smagorinsky --wall-law musker \
  --stress-exchange-distance "$exchange" \
  --wall-diagnostic-interval "$wall_diagnostic" \
  --sponge-width "$sponge" --sponge-strength 0.3 \
  --far-field-mode non_equilibrium_extrapolation \
  --checkpoint "$checkpoint" --checkpoint-interval "$checkpoint_interval" \
  --output "$output" "${resume[@]}"
