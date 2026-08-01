#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 wale|vreman COEFFICIENT GENERATION PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
}

[[ $# -ge 4 && $# -le 6 ]] || usage
model=$1
coefficient=$2
generation=$3
gpu=$4
result_dir=${5:-results/amr_campaign_20260801}
wait_for_pid=${6:-}
[[ $generation =~ ^v[0-9]+$ ]] || usage
[[ $coefficient =~ ^(0|[0-9]+)(\.[0-9]+)?$ ]] || usage

case "$model" in
  wale)
    collision_model=cumulant_wale
    wale_cw=$coefficient
    vreman_cv=0.025
    ;;
  vreman)
    collision_model=cumulant_vreman
    wale_cw=0.5
    vreman_cv=$coefficient
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
export PYTHONPATH="$root/src:$root/examples${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${TENSORLBM_PREFLIGHT_ONLY:-0} == 1 ]]; then
  exec "$python" -c 'import tensorlbm; print(tensorlbm.__file__)'
fi
stem="$result_dir/suboff-nested-${generation}-l90-${model}-masked-2400"
resume=()
if [[ -f "$stem.ckpt" ]]; then
  resume=(--resume)
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/suboff_nested_static_amr_smoke.py \
  --device cuda:0 --hull-type bare_hull --speed-knots 5.92 \
  --nx 450 --ny 90 --nz 90 --hull-length 90 --center-x-fraction 0.3 \
  --outer-wall-margin 6 --outer-wake-cells 75 \
  --inner-wall-margin 8 --inner-wake-cells 12 \
  --cv-margin 8 --aux-cv-margins 4,12 --surface-force-interval 30 \
  --steps 2400 --warmup-steps 0 --statistics-window-steps 0 \
  --ramp-steps 3000 --report-interval 60 --wall-diagnostic-interval 60 \
  --health-interval 60 --maximum-health-speed 0.3 \
  --minimum-health-population 1e-8 \
  --maximum-positivity-limited-fraction 1e-6 \
  --maximum-reflux-applied-correction-fraction 1e-3 \
  --minimum-convective-times 8 \
  --minimum-target-reynolds-convective-times 7.5 \
  --minimum-statistics-convective-times 5 \
  --lattice-speed 0.06 --resolved-reynolds 100000 \
  --resolved-reynolds-start 5000 \
  --viscosity-ramp-start-step 300 --viscosity-ramp-end-step 600 \
  --collision-model "$collision_model" --cs-smag 0 \
  --wale-cw "$wale_cw" --vreman-cv "$vreman_cv" \
  --wall-law musker --stress-exchange-distance 1 \
  --sponge-width 18 --sponge-strength 0.3 \
  --far-field-mode non_equilibrium_extrapolation \
  --memory-bytes-per-cell 742 --regularize-restriction \
  --ghost-interpolation trilinear \
  --reflux-correction-stencil exterior_cells \
  --enforce-transfer-positivity \
  --interface-filter-width 0 --interface-filter-strength 0 \
  --checkpoint "$stem.ckpt" --checkpoint-interval 300 \
  --output "$stem.json" "${resume[@]}"
