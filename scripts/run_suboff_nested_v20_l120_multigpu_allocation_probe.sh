#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 GPU_UUID_LIST [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
fi

gpu_list=$1
result_dir=${2:-results/amr_campaign_20260801}
wait_for_pid=${3:-}
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
stem="$result_dir/suboff-nested-v20-aff1-four-level-l120-multigpu-allocation-r1"
if [[ -f $stem.json ]]; then
  echo "multi-GPU L120 allocation result already exists: $stem.json" >&2
  exit 0
fi
if [[ -f $stem.ckpt ]]; then
  echo "refusing ambiguous allocation-probe checkpoint: $stem.ckpt" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=$gpu_list
exec "$python" examples/suboff_nested_static_amr_smoke.py \
  --device cuda:0 --level-devices cuda:0,cuda:0,cuda:1,cuda:2 \
  --hull-type bare_hull --speed-knots 5.92 \
  --nx 600 --ny 120 --nz 120 --hull-length 120 \
  --center-x-fraction 0.3 \
  --outer-wall-margin 8 --outer-wake-cells 100 \
  --inner-wall-margin 11 --inner-wake-cells 16 \
  --deep-wall-margin 9 --deep-wake-cells 19 \
  --cv-margin 11 --aux-cv-margins 5,16 \
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
  --wall-force-direction-chunk 4 --low-memory-wall-macroscopic \
  --cs-smag 0 --wall-law musker \
  --stress-exchange-distance 1.3333333333333333 \
  --sponge-width 24 --sponge-strength 0.3 \
  --far-field-mode non_equilibrium_extrapolation \
  --memory-bytes-per-cell 900 \
  --ghost-interpolation trilinear \
  --reflux-correction-stencil exterior_cells \
  --interface-filter-width 0 --interface-filter-strength 0 \
  --checkpoint "$stem.ckpt" --checkpoint-interval 1 \
  --output "$stem.json" \
  --regularize-restriction --enforce-transfer-positivity
