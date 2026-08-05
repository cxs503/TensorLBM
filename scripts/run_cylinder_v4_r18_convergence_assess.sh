#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RESULT_DIR [WAIT_FOR_PID]" >&2
  exit 2
fi

result_dir=$1
wait_for_pid=${2:-}
if [[ -n $wait_for_pid ]]; then
  [[ $wait_for_pid =~ ^[0-9]+$ ]] || exit 2
  while kill -0 "$wait_for_pid" 2>/dev/null; do
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n ${TENSORLBM_PYTHON:-} ]]; then
  python=$TENSORLBM_PYTHON
elif [[ -x $root/.venv/bin/python ]]; then
  python=$root/.venv/bin/python
elif [[ -x /home/wxsc/anaconda3/envs/ftw-env/bin/python ]]; then
  python=/home/wxsc/anaconda3/envs/ftw-env/bin/python
elif command -v python3 >/dev/null 2>&1; then
  python=$(command -v python3)
else
  echo "no Python interpreter found; set TENSORLBM_PYTHON" >&2
  exit 127
fi
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

inputs=(
  "$result_dir/cylinder-v4-equivalent-r9-54000.json"
  "$result_dir/cylinder-v4-equivalent-r12-72000.json"
  "$result_dir/cylinder-v4-equivalent-r15-90000.json"
  "$result_dir/cylinder-v4-equivalent-r18-108000.json"
)
for input in "${inputs[@]}"; do
  [[ -f $input ]] || {
    echo "missing cylinder convergence member: $input" >&2
    exit 2
  }
done
output="$result_dir/cylinder-v4-r9-r12-r15-r18-convergence.json"
exec "$python" examples/cylinder_grid_convergence_assess.py \
  "${inputs[@]}" --output "$output"
