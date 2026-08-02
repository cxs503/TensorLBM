#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ! $3 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 L256|L384|L512 PHYSICAL_GPU_UUID WAIT_FOR_PID [RESULT_DIR]" >&2
  exit 2
fi
level=$1
gpu=$2
owner_pid=$3
result_dir=${4:-results/flat_plate_physical_re}
[[ $gpu == GPU-* ]] || exit 2

if [[ -r /proc/$owner_pid/stat ]]; then
  owner_start_time=$(awk '{print $22}' "/proc/$owner_pid/stat")
  while kill -0 "$owner_pid" 2>/dev/null; do
    current_start_time=$(awk '{print $22}' "/proc/$owner_pid/stat" 2>/dev/null || true)
    [[ $current_start_time == "$owner_start_time" ]] || break
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_flat_plate_physical_re_level.sh" \
  "$level" "$gpu" "$result_dir"
