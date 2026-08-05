#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ! $1 =~ ^[0-9]+$ || $2 != GPU-* ]]; then
  echo "usage: $0 FLAT_PLATE_ASSESSMENT_PID PHYSICAL_GPU_UUID [RESULT_DIR]" >&2
  exit 2
fi
assessment_pid=$1
gpu=$2
flat_result_dir=${3:-results/flat_plate_physical_re}

if [[ -r /proc/$assessment_pid/stat ]]; then
  owner_start_time=$(awk '{print $22}' "/proc/$assessment_pid/stat")
  while kill -0 "$assessment_pid" 2>/dev/null; do
    current_start_time=$(awk '{print $22}' "/proc/$assessment_pid/stat" 2>/dev/null || true)
    [[ $current_start_time == "$owner_start_time" ]] || break
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
assessment="$flat_result_dir/flat-plate-v6-physical-re13p213m-convergence.json"
certificate=results/collision-viscosity-natural-kbc-suboff-re13p213m-mixed-exactweights-r1.json

"$python" - "$assessment" "$certificate" <<'PY'
import json
import sys
from pathlib import Path

assessment = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if assessment.get("admitted") is not True:
    raise SystemExit("physical-Re flat-plate convergence was not admitted")
certificate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
viscosity_acceptance = certificate.get("acceptance", {})
if not (
    viscosity_acceptance.get("all_levels_recover_configured_viscosity") is True
    and viscosity_acceptance.get("configured_reynolds_sequence_admitted") is True
    and certificate.get("d3q19_weight_precision_scheme")
    == "rational_binary64_cast_to_runtime_dtype_v1"
):
    raise SystemExit("exact-weight physical-Re collision viscosity was not admitted")
PY

export TENSORLBM_SUBOFF_STEM=suboff-nested-v45-re13p213m-flatplate-exchange-l90-3k
export TENSORLBM_STRESS_EXCHANGE_DISTANCE=8.4375
export TENSORLBM_WALL_EXCHANGE_RATIO_TARGET=0.01171875
export TENSORLBM_WALL_Y_PLUS_UPPER_BOUND=10000
exec "$root/scripts/run_suboff_mixed_precision_long.sh" REPHYSICAL "$gpu"
