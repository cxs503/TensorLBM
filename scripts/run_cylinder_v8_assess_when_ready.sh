#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "usage: $0 CYLINDER_V8_OWNER_PID" >&2
  exit 2
fi
owner_pid=$1
[[ $owner_pid =~ ^[1-9][0-9]*$ ]] || exit 2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

while [[ -r /proc/$owner_pid/cmdline ]]; do
  command_line=$(tr '\0' ' ' < "/proc/$owner_pid/cmdline")
  if [[ $command_line != *run_sphere_r9_then_cylinder_v8_streamwise* \
     && $command_line != *cylinder-v8-planar-mach003-r9-streamwise-10d-20d* ]]; then
    echo "PID $owner_pid no longer belongs to the registered cylinder v8 queue" >&2
    exit 3
  fi
  sleep 20
done

python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
result_dir=results/cylinder_collision_sensitivity
baseline=$result_dir/cylinder-v7-planar-mach003-r9-108000.json
candidate=$result_dir/cylinder-v8-planar-mach003-r9-streamwise-10d-20d-108000.json
[[ -f $baseline ]] || { echo "missing cylinder v7 result" >&2; exit 4; }
[[ -f $candidate ]] || { echo "missing cylinder v8 result" >&2; exit 4; }

exec "$python" scripts/assess_cylinder_streamwise_clearance_pair.py \
  "$baseline" "$candidate" \
  --output "$result_dir/cylinder-v7-v8-streamwise-clearance-assessment-r1.json"
