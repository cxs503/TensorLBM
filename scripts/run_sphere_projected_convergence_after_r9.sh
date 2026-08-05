#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ! $1 =~ ^[0-9]+$ || $2 != GPU-* ]]; then
  echo "usage: $0 R9_ASSESSMENT_PID PHYSICAL_GPU_UUID [RESULT_DIR]" >&2
  exit 2
fi
assessment_pid=$1
gpu=$2
result_dir=${3:-results/sphere_projected_pressure}
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
r9_assessment="$result_dir/sphere-v10-projected-linear-r9-assessment.json"
"$python" - "$r9_assessment" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("acceptance", {}).get("single_grid_candidate") is not True:
    raise SystemExit("R9 projected-pressure observer was not admitted")
PY

for specification in "R12 9600" "R15 12000"; do
  read -r level steps <<<"$specification"
  "$root/scripts/run_sphere_v10_projected_pressure_level.sh" \
    "$level" "$gpu" "$result_dir"
  radius=${level#R}
  stem="$result_dir/sphere-v10-natural-kbc-projected-linear-r${radius}-${steps}"
  "$python" scripts/assess_sphere_projected_pressure.py \
    "$stem.json" "$stem.ckpt" \
    --output "$result_dir/sphere-v10-projected-linear-r${radius}-assessment.json"
done
