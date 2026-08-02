#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 GPU_UUID_LIST OUTPUT_JSON [WAIT_PID_LIST]" >&2
  exit 2
fi

gpu_list=$1
output=$2
wait_pid_list=${3:-}
IFS=, read -r -a wait_pids <<< "$wait_pid_list"
for wait_pid in "${wait_pids[@]}"; do
  [[ -z $wait_pid ]] && continue
  [[ $wait_pid =~ ^[0-9]+$ ]] || {
    echo "wait PIDs must be comma-separated positive integers" >&2
    exit 2
  }
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 20
  done
done

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p "$(dirname "$output")"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src:$root/examples${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$gpu_list
exec "$python" examples/nested_amr_multidevice_validate.py \
  --devices cuda:0,cuda:1 \
  --steps "${TENSORLBM_STEPS:-12}" \
  --output "$output"
