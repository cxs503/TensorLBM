#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 BASE_RESULT_DIR DOMAIN_RESULT_DIR [WAIT_FOR_PID]" >&2
  exit 2
fi
base_dir=$1
domain_dir=$2
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
exec "$python" "$root/examples/sphere_domain_convergence_assess.py" \
  "$base_dir/sphere-v4-natural-kbc-equivalent-r9-7200.json" \
  "$domain_dir/sphere-v5-natural-kbc-domain-w20-r9-7200.json" \
  "$domain_dir/sphere-v5-natural-kbc-domain-w24-r9-7200.json" \
  --output "$domain_dir/sphere-v5-natural-kbc-r9-w16-w20-w24-convergence.json"
