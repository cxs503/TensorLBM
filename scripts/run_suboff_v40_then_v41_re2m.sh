#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 V40_OWNER_PID PHYSICAL_GPU_UUID" >&2
  exit 2
fi
owner_pid=$1
gpu=$2
[[ $owner_pid =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $gpu == GPU-* ]] || exit 2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

while [[ -r /proc/$owner_pid/cmdline ]]; do
  command_line=$(tr '\0' ' ' < "/proc/$owner_pid/cmdline")
  if [[ $command_line != *run_suboff_v37_and_mixed_pilot_then_v40* \
     && $command_line != *suboff-nested-v40-re1m-mixed-fp64compute-l90-3k* ]]; then
    echo "PID $owner_pid no longer belongs to the registered SUBOFF v40 queue" >&2
    exit 3
  fi
  sleep 20
done

python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
result_dir=results/amr_campaign_20260801
v40=$result_dir/suboff-nested-v40-re1m-mixed-fp64compute-l90-3k.json
schedule=results/collision-viscosity-natural-kbc-suboff-re2m-mixed-exactweights-r2.json
[[ -f $v40 ]] || { echo "missing SUBOFF v40 result" >&2; exit 4; }
[[ -f $schedule ]] || { echo "missing Re2M viscosity schedule" >&2; exit 4; }

"$python" - "$v40" "$schedule" <<'PY'
import json
import math
import sys

flow = json.load(open(sys.argv[1], encoding="utf-8"))
schedule = json.load(open(sys.argv[2], encoding="utf-8"))
acceptance = flow["acceptance"]
result = flow["result"]
execution = result["collision_execution"]
if not result.get("finite"):
    raise SystemExit("SUBOFF v40 is non-finite; refusing Re2M launch")
if not all(acceptance.get(field) for field in (
    "population_health_target_met",
    "nested_control_volume_target_met",
    "target_reynolds_reached",
)):
    raise SystemExit("SUBOFF v40 failed a registered health gate")
if execution.get("compute_dtype") != "float64":
    raise SystemExit("SUBOFF v40 lacks float64 collision identity")
if float(result.get("maximum_positivity_limited_fraction", math.inf)) != 0.0:
    raise SystemExit("SUBOFF v40 used positivity limiting")
if schedule.get("schema") != "tensorlbm-collision-viscosity-schedule-v1":
    raise SystemExit("unsupported Re2M viscosity schedule")
if not schedule.get("acceptance", {}).get(
    "all_levels_recover_configured_viscosity"
):
    raise SystemExit("Re2M collision viscosity is not certified")
if float(schedule.get("acceptance", {}).get(
    "minimum_fitted_log_decay", 0.0
)) < 0.004:
    raise SystemExit("Re2M viscosity certificate lacks a resolved decay signal")
if schedule.get("dtype") != "float32" or schedule.get(
    "natural_kbc_compute_dtype"
) != "float64":
    raise SystemExit("Re2M viscosity certificate precision mismatch")
if schedule.get("d3q19_weight_precision_scheme") != (
    "rational_binary64_cast_to_runtime_dtype_v1"
):
    raise SystemExit("Re2M viscosity certificate weight precision mismatch")
expected = [0.5000081, 0.5000162, 0.5000324, 0.5000648]
if len(schedule.get("taus", [])) != len(expected) or any(
    not math.isclose(float(actual), target, rel_tol=0.0, abs_tol=1.0e-12)
    for actual, target in zip(schedule["taus"], expected, strict=True)
):
    raise SystemExit("Re2M viscosity certificate tau mismatch")
PY

exec scripts/run_suboff_mixed_precision_re2m_long.sh "$gpu"
