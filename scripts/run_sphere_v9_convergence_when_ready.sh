#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
  echo "usage: $0 R9_QUEUE_PID R12_QUEUE_PID R15_QUEUE_PID RESULT_DIR" >&2
  exit 2
fi
pids=($1 $2 $3)
result_dir=$4
for pid in "${pids[@]}"; do
  [[ $pid =~ ^[1-9][0-9]*$ ]] || exit 2
done

while :; do
  active=0
  for pid in "${pids[@]}"; do
    [[ -r /proc/$pid/cmdline ]] || continue
    command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    if [[ $command_line != *sphere-v9-natural-kbc-corrected-bfl-no-inlet* \
       && $command_line != *run_cylinder_v7_then_sphere_v9_r9* \
       && $command_line != *run_suboff_v30_then_sphere_v9_r12* \
       && $command_line != *run_channel_r3_then_sphere_v9_r15* \
       && $command_line != *run_channel_then_mixed_pilot_then_sphere_r15* ]]; then
      echo "PID $pid is not an owned corrected-sphere queue/run" >&2
      exit 3
    fi
    active=1
  done
  (( active )) || break
  sleep 30
done

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
exec scripts/run_sphere_v9_corrected_bfl_convergence_assess.sh "$result_dir"
