#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 WAIT_FOR_CHAIN_PID [RESULT_DIR]" >&2
  exit 2
fi
owner_pid=$1
result_dir=${2:-results/flat_plate_physical_re}

if [[ -r /proc/$owner_pid/stat ]]; then
  owner_start_time=$(awk '{print $22}' "/proc/$owner_pid/stat")
  while kill -0 "$owner_pid" 2>/dev/null; do
    current_start_time=$(awk '{print $22}' "/proc/$owner_pid/stat" 2>/dev/null || true)
    [[ $current_start_time == "$owner_start_time" ]] || break
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
inputs=(
  "$result_dir/flat-plate-v6-physical-re13p213m-l256-32000.json"
  "$result_dir/flat-plate-v6-physical-re13p213m-l384-48000.json"
  "$result_dir/flat-plate-v6-physical-re13p213m-l512-64000.json"
)
for input in "${inputs[@]}"; do
  [[ -f $input ]] || {
    echo "missing physical-Re flat-plate member: $input" >&2
    exit 1
  }
done
exec "$python" examples/flat_plate_convergence_assess.py \
  "${inputs[@]}" \
  --output "$result_dir/flat-plate-v6-physical-re13p213m-convergence.json"
