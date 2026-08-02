#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU_UUID [WAIT_FOR_PID] [RESULT_DIR]" >&2
  exit 2
fi
gpu=$1
wait_for_pid=${2:-}
result_dir=${3:-results/flat_plate_physical_re}
[[ $gpu == GPU-* ]] || exit 2

if [[ -n $wait_for_pid ]]; then
  [[ $wait_for_pid =~ ^[0-9]+$ ]] || exit 2
  if [[ -r /proc/$wait_for_pid/stat ]]; then
    owner_start_time=$(awk '{print $22}' "/proc/$wait_for_pid/stat")
    while kill -0 "$wait_for_pid" 2>/dev/null; do
      current_start_time=$(awk '{print $22}' "/proc/$wait_for_pid/stat" 2>/dev/null || true)
      [[ $current_start_time == "$owner_start_time" ]] || break
      sleep 20
    done
  fi
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for level in L256 L384 L512; do
  "$root/scripts/run_flat_plate_physical_re_level.sh" \
    "$level" "$gpu" "$result_dir"
done
