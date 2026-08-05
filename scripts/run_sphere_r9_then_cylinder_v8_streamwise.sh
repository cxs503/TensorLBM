#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 SPHERE_R9_OWNER_PID PHYSICAL_GPU_UUID" >&2
  exit 2
fi
owner_pid=$1
gpu=$2
[[ $owner_pid =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $gpu == GPU-* ]] || exit 2

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
while [[ -r /proc/$owner_pid/cmdline ]]; do
  command_line=$(tr '\0' ' ' < "/proc/$owner_pid/cmdline")
  if [[ $command_line != *run_cylinder_v7_then_sphere_v9_r9* \
     && $command_line != *sphere-v9-natural-kbc-corrected-bfl-no-inlet-r9* ]]; then
    echo "PID $owner_pid no longer belongs to the registered sphere R9 queue" >&2
    exit 3
  fi
  sleep 20
done

exec scripts/run_cylinder_v8_streamwise_clearance.sh "$gpu"
