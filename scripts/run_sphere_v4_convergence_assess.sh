#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RESULT_DIR [WAIT_PID_LIST]" >&2
  exit 2
fi

result_dir=$1
wait_pid_list=${2:-}
if [[ -n $wait_pid_list ]]; then
  IFS=, read -r -a wait_pids <<<"$wait_pid_list"
  for pid in "${wait_pids[@]}"; do
    [[ $pid =~ ^[0-9]+$ ]] || {
      echo "WAIT_PID_LIST must contain comma-separated integers" >&2
      exit 2
    }
  done
  while :; do
    active=0
    for pid in "${wait_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        active=1
      fi
    done
    (( active )) || break
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
  "$result_dir/sphere-v4-natural-kbc-equivalent-r9-7200.json"
  "$result_dir/sphere-v4-natural-kbc-equivalent-r12-9600.json"
  "$result_dir/sphere-v4-natural-kbc-equivalent-r15-12000.json"
)
for input in "${inputs[@]}"; do
  [[ -f $input ]] || {
    echo "missing sphere convergence member: $input" >&2
    exit 2
  }
done
output="$result_dir/sphere-v4-natural-kbc-r9-r12-r15-convergence.json"
exec "$python" examples/sphere_grid_convergence_assess.py \
  "${inputs[@]}" --output "$output"
