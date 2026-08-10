#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "usage: $0 SUBOFF_V30_PID PHYSICAL_GPU_UUID SPHERE_RESULT_DIR" >&2
  exit 2
fi
suboff_pid=$1
gpu=$2
sphere_result_dir=$3
[[ $suboff_pid =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $gpu == GPU-* ]] || exit 2

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
while [[ -r /proc/$suboff_pid/cmdline ]]; do
  command_line=$(tr '\0' ' ' < "/proc/$suboff_pid/cmdline")
  if [[ $command_line != *suboff-nested-v30-equivalent-l90-12k* ]]; then
    echo "PID $suboff_pid no longer belongs to registered SUBOFF v30" >&2
    exit 3
  fi
  sleep 20
done

python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
result=results/amr_campaign_20260801/suboff-nested-v30-equivalent-l90-12k.json
[[ -f $result ]] || { echo "missing SUBOFF v30 result" >&2; exit 4; }
"$python" - "$result" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if not bool(result["result"].get("finite")):
    raise SystemExit("SUBOFF v30 result is non-finite; refusing queued launch")
if not bool(result["acceptance"].get("population_health_target_met")):
    raise SystemExit("SUBOFF v30 failed population health; refusing queued launch")
PY

exec scripts/run_sphere_v9_corrected_bfl_family_level.sh \
  R12 "$gpu" "$sphere_result_dir"
