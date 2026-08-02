#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 L90 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
[[ $1 == L90 ]] || usage
gpu=$2
result_dir=${3:-results/amr_campaign_20260801}
wait_for_pid=${4:-}

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
export PYTHONPATH="$root/src:$root/examples${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${TENSORLBM_PREFLIGHT_ONLY:-0} == 1 ]]; then
  exec "$python" -c 'import tensorlbm; print(tensorlbm.__file__)'
fi

stem="$result_dir/suboff-nested-v16-aff8-four-level-l90-allocation-probe-r1"
if [[ -f "$stem.json" ]]; then
  echo "allocation probe result already exists: $stem.json" >&2
  exit 0
fi
if [[ -f "$stem.ckpt" ]]; then
  echo "refusing ambiguous allocation-probe checkpoint: $stem.ckpt" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/suboff_nested_static_amr_smoke.py \
  --device cuda:0 --hull-type full --speed-knots 5.92 \
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
  --collision-model natural_kbc --cs-smag 0 \
  --wall-law musker --stress-exchange-distance 1.0 \
  --sponge-width 18 --sponge-strength 0.3 \
  --far-field-mode non_equilibrium_extrapolation \
  --memory-bytes-per-cell 1100 \
  --ghost-interpolation trilinear \
  --reflux-correction-stencil exterior_cells \
  --interface-filter-width 0 --interface-filter-strength 0 \
  --checkpoint "$stem.ckpt" --checkpoint-interval 1 \
  --output "$stem.json" \
  --regularize-restriction --enforce-transfer-positivity
