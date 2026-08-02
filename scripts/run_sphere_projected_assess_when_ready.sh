#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 SPHERE_PID [RESULT_DIR]" >&2
  exit 2
fi
owner_pid=$1
result_dir=${2:-results/sphere_projected_pressure}
if [[ -r /proc/$owner_pid/stat ]]; then
  owner_start_time=$(awk '{print $22}' "/proc/$owner_pid/stat")
  while kill -0 "$owner_pid" 2>/dev/null; do
    current_start_time=$(awk '{print $22}' "/proc/$owner_pid/stat" 2>/dev/null || true)
    [[ $current_start_time == "$owner_start_time" ]] || break
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
stem="$result_dir/sphere-v10-natural-kbc-projected-linear-r9-7200"
[[ -f "$stem.json" && -f "$stem.ckpt" ]] || exit 1
exec "$python" scripts/assess_sphere_projected_pressure.py \
  "$stem.json" "$stem.ckpt" \
  --output "$result_dir/sphere-v10-projected-linear-r9-assessment.json"
