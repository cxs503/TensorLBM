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
    [[ $pid =~ ^[0-9]+$ ]] || exit 2
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
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
inputs=(
  "$result_dir/sphere-v6-natural-kbc-no-inlet-sponge-r9-7200.json"
  "$result_dir/sphere-v7-natural-kbc-no-inlet-equivalent-r12-9600.json"
  "$result_dir/sphere-v7-natural-kbc-no-inlet-equivalent-r15-12000.json"
)
for input in "${inputs[@]}"; do
  [[ -f $input ]] || {
    echo "missing no-inlet sphere convergence member: $input" >&2
    exit 2
  }
done
exec "$python" "$root/examples/sphere_grid_convergence_assess.py" \
  "${inputs[@]}" \
  --output "$result_dir/sphere-v7-natural-kbc-no-inlet-r9-r12-r15-convergence.json"
