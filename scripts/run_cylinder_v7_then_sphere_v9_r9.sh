#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "usage: $0 CYLINDER_V7_PID PHYSICAL_GPU_UUID SPHERE_RESULT_DIR" >&2
  exit 2
fi
cylinder_pid=$1
gpu=$2
sphere_result_dir=$3
[[ $cylinder_pid =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $gpu == GPU-* ]] || exit 2

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
while [[ -r /proc/$cylinder_pid/cmdline ]]; do
  command_line=$(tr '\0' ' ' < "/proc/$cylinder_pid/cmdline")
  if [[ $command_line != *cylinder-v7-planar-mach003-r9-108000* ]]; then
    echo "PID $cylinder_pid no longer belongs to registered cylinder v7" >&2
    exit 3
  fi
  sleep 20
done

python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
baseline=results/cylinder_collision_sensitivity/cylinder-v6-natural-kbc-mach003-r9-108000.json
candidate=results/cylinder_collision_sensitivity/cylinder-v7-planar-mach003-r9-108000.json
assessment=results/cylinder_collision_sensitivity/cylinder-v6-v7-planar-causal-assessment-r1.json
[[ -f $baseline && -f $candidate ]] || {
  echo "cylinder v6/v7 result pair is incomplete" >&2
  exit 4
}
"$python" scripts/assess_cylinder_causal_pair.py \
  "$baseline" "$candidate" \
  --intervention collision_model \
  --output "$assessment"

exec scripts/run_sphere_v9_corrected_bfl_family_level.sh \
  R9 "$gpu" "$sphere_result_dir"
