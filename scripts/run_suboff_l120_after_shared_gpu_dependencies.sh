#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU_UUID_LIST RESULT_DIR RE500K_PID FLAT_L384_PID SPHERE_R15_PID" >&2
  exit 2
fi

gpu_list=$1
result_dir=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $result_dir != /* ]]; then
  result_dir="$root/$result_dir"
fi
shift 2
for pid in "$@"; do
  [[ $pid =~ ^[1-9][0-9]*$ ]] || exit 2
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done
done

required=(
  "$result_dir/suboff-nested-v25-equivalent-l90-3k.json"
  "$result_dir/../flat_plate_grid_convergence/flat-plate-v5-equivalent-l384-48000.json"
  "$result_dir/../sphere_boundary_sensitivity/sphere-v7-natural-kbc-no-inlet-equivalent-r15-12000.json"
)
for artifact in "${required[@]}"; do
  if [[ ! -s $artifact ]]; then
    echo "shared-GPU dependency ended without result: $artifact" >&2
    exit 1
  fi
done

exec "$root/scripts/run_suboff_nested_v20_l120_multigpu_allocation_probe.sh" \
  "$gpu_list" "$result_dir"
