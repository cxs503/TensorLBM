#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU RESULT_DIR L120_PID" >&2
  exit 2
fi

gpu=$1
result_dir=$2
l120_pid=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $result_dir != /* ]]; then
  result_dir="$root/$result_dir"
fi
[[ $l120_pid =~ ^[1-9][0-9]*$ ]] || exit 2
while kill -0 "$l120_pid" 2>/dev/null; do
  sleep 20
done

l120_result="$result_dir/suboff-nested-v20-aff1-four-level-l120-multigpu-allocation-r1.json"
if [[ ! -s $l120_result ]]; then
  echo "L120 allocation dependency ended without result: $l120_result" >&2
  exit 1
fi

exec "$root/scripts/run_suboff_nested_v21_aff8_bounded_allocation_probe.sh" \
  "$gpu" "$result_dir"
