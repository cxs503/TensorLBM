#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 || $1 != L90 ]]; then
  echo "usage: $0 L90 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
fi

gpu=$2
result_dir=${3:-results/amr_campaign_20260801}
wait_for_pid=${4:-}
if [[ -n $wait_for_pid ]]; then
  [[ $wait_for_pid =~ ^[0-9]+$ ]] || exit 2
  while kill -0 "$wait_for_pid" 2>/dev/null; do
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p "$result_dir"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src:$root/examples${PYTHONPATH:+:$PYTHONPATH}"
generation=${TENSORLBM_CAMPAIGN_GENERATION:-v17}
[[ $generation =~ ^v[0-9]+$ ]] || {
  echo "TENSORLBM_CAMPAIGN_GENERATION must match vN" >&2
  exit 2
}
compile_natural_kbc=()
if [[ ${TENSORLBM_COMPILE_NATURAL_KBC:-0} == 1 ]]; then
  compile_natural_kbc=(--compile-natural-kbc)
elif [[ ${TENSORLBM_COMPILE_NATURAL_KBC:-0} != 0 ]]; then
  echo "TENSORLBM_COMPILE_NATURAL_KBC must be 0 or 1" >&2
  exit 2
fi
stress_exchange_distance=${TENSORLBM_STRESS_EXCHANGE_DISTANCE:-1.0}
stem="$result_dir/suboff-nested-$generation-aff1-four-level-l90-chunked-allocation-r1"
if [[ -f $stem.json ]]; then
  echo "chunked allocation result already exists: $stem.json" >&2
  exit 0
fi
if [[ -f $stem.ckpt ]]; then
  echo "refusing ambiguous allocation-probe checkpoint: $stem.ckpt" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/suboff_nested_static_amr_smoke.py \
  --device cuda:0 --hull-type bare_hull --speed-knots 5.92 \
  --nx 450 --ny 90 --nz 90 --hull-length 90 \
  --center-x-fraction 0.3 \
  --outer-wall-margin 6 --outer-wake-cells 75 \
  --inner-wall-margin 8 --inner-wake-cells 12 \
  --deep-wall-margin 7 --deep-wake-cells 14 \
  --cv-margin 8 --aux-cv-margins 4,12 \
  --surface-force-interval 1 \
  --steps 1 --warmup-steps 0 --statistics-window-steps 1 \
  --ramp-steps 100 --report-interval 1 \
  --wall-diagnostic-interval 1 --health-interval 1 \
  --maximum-health-speed 0.3 \
  --maximum-reflux-applied-correction-fraction 0.001 \
  --minimum-convective-times 8 \
  --minimum-target-reynolds-convective-times 7.5 \
  --minimum-statistics-convective-times 5 \
  --lattice-speed 0.06 --resolved-reynolds 100000 \
  --resolved-reynolds-start 100000 \
  --viscosity-ramp-start-step 0 --viscosity-ramp-end-step 0 \
  --collision-model natural_kbc --collision-chunk-cells 262144 \
  "${compile_natural_kbc[@]}" \
  --wall-force-direction-chunk 4 \
  --low-memory-wall-macroscopic \
  --cs-smag 0 --wall-law musker \
  --stress-exchange-distance "$stress_exchange_distance" \
  --sponge-width 18 --sponge-strength 0.3 \
  --far-field-mode non_equilibrium_extrapolation \
  --memory-bytes-per-cell 1100 \
  --ghost-interpolation trilinear \
  --reflux-correction-stencil exterior_cells \
  --interface-filter-width 0 --interface-filter-strength 0 \
  --checkpoint "$stem.ckpt" --checkpoint-interval 1 \
  --output "$stem.json" \
  --regularize-restriction --enforce-transfer-positivity
