#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 V37_PID PHYSICAL_GPU_UUID" >&2
  exit 2
fi
v37_pid=$1
gpu=$2
[[ $v37_pid =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $gpu == GPU-* ]] || exit 2

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
while [[ -r /proc/$v37_pid/cmdline ]]; do
  command_line=$(tr '\0' ' ' < "/proc/$v37_pid/cmdline")
  if [[ $command_line != *suboff-nested-v37-re1m-l90-3k* ]]; then
    echo "PID $v37_pid no longer belongs to registered SUBOFF v37" >&2
    exit 3
  fi
  sleep 20
done

python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
result_dir=results/amr_campaign_20260801
v37=$result_dir/suboff-nested-v37-re1m-l90-3k.json
[[ -f $v37 ]] || { echo "missing SUBOFF v37 result" >&2; exit 4; }
"$python" - "$v37" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
acceptance = result["acceptance"]
required = (
    "population_health_target_met",
    "nested_control_volume_target_met",
)
if not all(bool(acceptance.get(field)) for field in required):
    raise SystemExit("SUBOFF v37 failed health gates; refusing Re=2M launch")
health = result["result"].get("population_health", [])
if not health or max(
    float(record["maximum_collision_limited_fraction"]) for record in health
) != 0.0:
    raise SystemExit("SUBOFF v37 used a collision limiter; refusing Re=2M launch")
PY

"$python" scripts/assess_suboff_resolved_reynolds_sensitivity.py \
  "$result_dir/suboff-nested-v23-equivalent-l90-3k.json" \
  "$result_dir/suboff-nested-v25-equivalent-l90-3k.json" \
  "$v37" \
  --output "$result_dir/suboff-v23-v25-v37-resolved-re-sensitivity-r1.json"

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/suboff_nested_static_amr_smoke.py \
  --device cuda:0 --level-devices "" --hull-type bare_hull --speed-knots 5.92 \
  --nx 450 --ny 90 --nz 90 --hull-length 90 --center-x-fraction 0.3 \
  --outer-wall-margin 6 --outer-wake-cells 75 \
  --inner-wall-margin 8 --inner-wake-cells 12 \
  --deep-wall-margin 7 --deep-wake-cells 14 \
  --cv-margin 8 --aux-cv-margins 4,12 --surface-force-interval 30 \
  --steps 3000 --warmup-steps 1500 --statistics-window-steps 1500 \
  --ramp-steps 1500 --wall-normal-ramp-steps 1500 --wall-shear-ramp-steps 1500 \
  --report-interval 375 --wall-diagnostic-interval 60 --health-interval 60 \
  --maximum-health-speed 0.3 --maximum-reflux-applied-correction-fraction 0.001 \
  --minimum-convective-times 2 --minimum-target-reynolds-convective-times 1 \
  --minimum-statistics-convective-times 1 --lattice-speed 0.06 \
  --resolved-reynolds 2000000 --resolved-reynolds-start 5000 \
  --viscosity-ramp-start-step 300 --viscosity-ramp-end-step 1500 \
  --collision-model natural_kbc --collision-chunk-cells 262144 \
  --compile-natural-kbc --wall-force-direction-chunk 4 \
  --low-memory-wall-macroscopic --cs-smag 0 --wale-cw 0.5 --vreman-cv 0.025 \
  --wall-law musker --stress-exchange-distance 1.0 \
  --wall-exchange-distance-over-length-target 0.0013888888888888889 \
  --wall-model-y-plus-lower-bound 30 --wall-model-y-plus-upper-bound 1000 \
  --minimum-wall-model-y-plus-in-range-fraction 0.9 \
  --sponge-width 18 --sponge-strength 0.3 --sponge-inlet \
  --far-field-mode non_equilibrium_extrapolation --memory-bytes-per-cell 1100 \
  --ghost-interpolation trilinear --reflux-correction-stencil exterior_cells \
  --interface-filter-width 0 --interface-filter-strength 0 \
  --regularize-restriction --enforce-transfer-positivity \
  --checkpoint "$result_dir/suboff-nested-v38-re2m-l90-3k.ckpt" \
  --checkpoint-interval 750 \
  --output "$result_dir/suboff-nested-v38-re2m-l90-3k.json"
