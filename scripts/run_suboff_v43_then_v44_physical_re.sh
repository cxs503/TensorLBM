#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 V43_OWNER_PID PHYSICAL_GPU_UUID" >&2
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
  if [[ $command_line != *run_suboff_v42_then_v43_re8m* \
     && $command_line != *suboff-nested-v43-re8m-mixed-fp64compute-l90-3k* ]]; then
    echo "PID $owner_pid no longer belongs to the registered SUBOFF v43 queue" >&2
    exit 3
  fi
  sleep 20
done

python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
result_dir=results/amr_campaign_20260801
v43=$result_dir/suboff-nested-v43-re8m-mixed-fp64compute-l90-3k.json
schedule=results/collision-viscosity-natural-kbc-suboff-re13p213m-mixed-exactweights-r1.json
[[ -f $v43 ]] || { echo "missing SUBOFF v43 result" >&2; exit 4; }
[[ -f $schedule ]] || { echo "missing physical-Re viscosity schedule" >&2; exit 4; }

"$python" - "$v43" "$schedule" <<'PY'
import json
import math
import sys

flow = json.load(open(sys.argv[1], encoding="utf-8"))
schedule = json.load(open(sys.argv[2], encoding="utf-8"))
acceptance = flow["acceptance"]
result = flow["result"]
execution = result["collision_execution"]
required = (
    result.get("finite") is True
    and all(acceptance.get(field) for field in (
        "population_health_target_met",
        "nested_control_volume_target_met",
        "target_reynolds_reached",
    ))
    and execution.get("compute_dtype") == "float64"
    and execution.get("d3q19_weight_precision_scheme")
    == "rational_binary64_cast_to_runtime_dtype_v1"
    and float(result.get("maximum_positivity_limited_fraction", math.inf)) == 0.0
)
if not required:
    raise SystemExit("SUBOFF v43 failed the physical-Re launch health gate")
if schedule.get("schema") != "tensorlbm-collision-viscosity-schedule-v1":
    raise SystemExit("unsupported physical-Re viscosity schedule")
if not schedule.get("acceptance", {}).get(
    "all_levels_recover_configured_viscosity"
):
    raise SystemExit("physical-Re collision viscosity is not certified")
if float(schedule.get("acceptance", {}).get(
    "minimum_fitted_log_decay", 0.0
)) < 0.004:
    raise SystemExit("physical-Re certificate lacks a resolved decay signal")
if schedule.get("dtype") != "float32" or schedule.get(
    "natural_kbc_compute_dtype"
) != "float64":
    raise SystemExit("physical-Re certificate precision mismatch")
if schedule.get("d3q19_weight_precision_scheme") != (
    "rational_binary64_cast_to_runtime_dtype_v1"
):
    raise SystemExit("physical-Re certificate weight precision mismatch")
expected = [
    0.5000012260298475,
    0.5000024520596952,
    0.5000049041193902,
    0.5000098082387806,
]
if len(schedule.get("taus", [])) != len(expected) or any(
    not math.isclose(float(actual), target, rel_tol=0.0, abs_tol=1.0e-12)
    for actual, target in zip(schedule["taus"], expected, strict=True)
):
    raise SystemExit("physical-Re viscosity certificate tau mismatch")
PY

exec scripts/run_suboff_mixed_precision_physical_re_long.sh "$gpu"
