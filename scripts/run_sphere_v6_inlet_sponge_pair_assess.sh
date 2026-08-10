#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 BASE_RESULT_DIR BOUNDARY_RESULT_DIR [WAIT_FOR_PID]" >&2
  exit 2
fi
base_dir=$1
boundary_dir=$2
wait_for_pid=${3:-}
if [[ -n $wait_for_pid ]]; then
  [[ $wait_for_pid =~ ^[0-9]+$ ]] || exit 2
  while kill -0 "$wait_for_pid" 2>/dev/null; do
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$root/examples/sphere_inlet_sponge_sensitivity_assess.py" \
  "$boundary_dir/sphere-v6-natural-kbc-no-inlet-sponge-r9-7200.json" \
  "$base_dir/sphere-v4-natural-kbc-equivalent-r9-7200.json" \
  --output "$boundary_dir/sphere-v6-natural-kbc-r9-inlet-sponge-sensitivity.json"
