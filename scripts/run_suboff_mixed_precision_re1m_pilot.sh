#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "usage: $0 PHYSICAL_GPU_UUID" >&2
  exit 2
fi
gpu=$1
[[ $gpu == GPU-* ]] || exit 2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$gpu
result_dir=results/amr_campaign_20260801

exec "$python" examples/suboff_nested_static_amr_smoke.py \
  --device cuda:0 --level-devices "" --hull-type bare_hull --speed-knots 5.92 \
  --nx 450 --ny 90 --nz 90 --hull-length 90 --center-x-fraction 0.3 \
  --outer-wall-margin 6 --outer-wake-cells 75 \
  --inner-wall-margin 8 --inner-wake-cells 12 \
  --deep-wall-margin 7 --deep-wake-cells 14 \
  --cv-margin 8 --aux-cv-margins 4,12 --surface-force-interval 30 \
  --steps 180 --warmup-steps 120 --statistics-window-steps 60 \
  --ramp-steps 120 --wall-normal-ramp-steps 120 --wall-shear-ramp-steps 120 \
  --report-interval 60 --wall-diagnostic-interval 30 --health-interval 30 \
  --maximum-health-speed 0.3 --maximum-reflux-applied-correction-fraction 0.001 \
  --minimum-convective-times 0.08 \
  --minimum-target-reynolds-convective-times 0.04 \
  --minimum-statistics-convective-times 0.04 --lattice-speed 0.06 \
  --resolved-reynolds 1000000 --resolved-reynolds-start 5000 \
  --viscosity-ramp-start-step 30 --viscosity-ramp-end-step 120 \
  --collision-model natural_kbc --collision-chunk-cells 262144 \
  --compile-natural-kbc --natural-kbc-compute-dtype float64 \
  --wall-force-direction-chunk 4 --low-memory-wall-macroscopic \
  --cs-smag 0 --wale-cw 0.5 --vreman-cv 0.025 \
  --wall-law musker --stress-exchange-distance 1.0 \
  --wall-exchange-distance-over-length-target 0.0013888888888888889 \
  --wall-model-y-plus-lower-bound 30 --wall-model-y-plus-upper-bound 1000 \
  --minimum-wall-model-y-plus-in-range-fraction 0.9 \
  --sponge-width 18 --sponge-strength 0.3 --sponge-inlet \
  --far-field-mode non_equilibrium_extrapolation --memory-bytes-per-cell 1300 \
  --ghost-interpolation trilinear --reflux-correction-stencil exterior_cells \
  --interface-filter-width 0 --interface-filter-strength 0 \
  --regularize-restriction --enforce-transfer-positivity \
  --output "$result_dir/suboff-nested-v39-re1m-mixed-fp64compute-l90-180.json"
