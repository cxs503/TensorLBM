#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 BASE_DIR DOMAIN_DIR BOUNDARY_DIR [WAIT_FOR_PID]" >&2
  exit 2
fi
base_dir=$1
domain_dir=$2
boundary_dir=$3
wait_for_pid=${4:-}
if [[ -n $wait_for_pid ]]; then
  [[ $wait_for_pid =~ ^[0-9]+$ ]] || exit 2
  while kill -0 "$wait_for_pid" 2>/dev/null; do
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$root/examples/sphere_boundary_factorial_assess.py" \
  "$base_dir/sphere-v4-natural-kbc-equivalent-r9-7200.json" \
  "$boundary_dir/sphere-v6-natural-kbc-no-inlet-sponge-r9-7200.json" \
  "$domain_dir/sphere-v5-natural-kbc-domain-w24-r9-7200.json" \
  "$boundary_dir/sphere-v8-natural-kbc-no-inlet-domain-w24-r9-7200.json" \
  --output "$boundary_dir/sphere-v8-natural-kbc-r9-domain-inlet-factorial.json"
