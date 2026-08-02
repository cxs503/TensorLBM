#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "usage: $0 V37_PID MIXED_PILOT_PID PHYSICAL_GPU_UUID" >&2
  exit 2
fi
v37_pid=$1
pilot_pid=$2
gpu=$3
[[ $v37_pid =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $pilot_pid =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $gpu == GPU-* ]] || exit 2

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

wait_for_owned_pid() {
  local pid=$1
  local expected=$2
  while [[ -r /proc/$pid/cmdline ]]; do
    local command_line
    command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    if [[ $command_line != *"$expected"* ]]; then
      echo "PID $pid no longer belongs to registered $expected process" >&2
      exit 3
    fi
    sleep 20
  done
}

wait_for_owned_pid "$v37_pid" suboff-nested-v37-re1m-l90-3k
wait_for_owned_pid "$pilot_pid" suboff-nested-v39-re1m-mixed-fp64compute-l90-180

python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
result_dir=results/amr_campaign_20260801
pilot=$result_dir/suboff-nested-v39-re1m-mixed-fp64compute-l90-180.json
assessment=$result_dir/suboff-nested-v39-re1m-mixed-fp64compute-l90-180-runtime-assessment.json
precision=docs/evidence/suboff-re1m-collision-viscosity-precision-gate-r1.json
[[ -f $pilot ]] || { echo "missing mixed-precision pilot result" >&2; exit 4; }
[[ -f $precision ]] || { echo "missing mixed-precision viscosity evidence" >&2; exit 4; }

"$python" scripts/assess_suboff_mixed_precision_pilot.py \
  "$pilot" "$precision" --output "$assessment"

exec scripts/run_suboff_mixed_precision_re1m_long.sh "$gpu"
