#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "usage: $0 CHANNEL_PID PHYSICAL_GPU_UUID SPHERE_RESULT_DIR" >&2
  exit 2
fi
channel_pid=$1
gpu=$2
sphere_result_dir=$3
[[ $channel_pid =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $gpu == GPU-* ]] || exit 2

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
while [[ -r /proc/$channel_pid/cmdline ]]; do
  command_line=$(tr '\0' ' ' < "/proc/$channel_pid/cmdline")
  if [[ $command_line != *channel3d-retau180-cumulant-spectral-fullbox-r3-120k* ]]; then
    echo "PID $channel_pid no longer belongs to registered channel r3" >&2
    exit 3
  fi
  sleep 20
done

python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
channel_result=results/canonical_wall/channel3d-retau180-cumulant-spectral-fullbox-r3-120k.json
[[ -f $channel_result ]] || { echo "missing channel r3 result" >&2; exit 4; }
"$python" scripts/assess_wall_resolved_channel3d_dns.py \
  "$channel_result" results/reference/mkm-chan180.means \
  --dns-reynolds-stress results/reference/mkm-chan180.reystress \
  --output results/canonical_wall/channel3d-retau180-cumulant-spectral-fullbox-r3-dns-assessment.json

set +e
scripts/run_suboff_mixed_precision_re1m_pilot.sh "$gpu" \
  > results/amr_campaign_20260801/suboff-nested-v39-re1m-mixed-fp64compute-l90-180.wait.log \
  2>&1
pilot_status=$?
set -e
echo "mixed-precision SUBOFF pilot exit status: $pilot_status"

exec scripts/run_sphere_v9_corrected_bfl_family_level.sh \
  R15 "$gpu" "$sphere_result_dir"
