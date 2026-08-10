#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 GENERATION TARGET_RE L90 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
}

[[ $# -ge 4 && $# -le 6 ]] || usage
generation=$1
target_re=$2
level=$3
gpu=$4
result_dir=${5:-results/amr_campaign_20260801}
wait_for_pid=${6:-}
[[ $generation =~ ^v[0-9]+$ ]] || usage
[[ $target_re =~ ^[1-9][0-9]*$ ]] || usage
[[ $level == L90 ]] || usage

# Short causal sweep: enough to cross the requested collision Reynolds and
# observe its pressure/health response, never enough to claim converged drag.
export TENSORLBM_CAMPAIGN_GENERATION=$generation
export TENSORLBM_STEPS=3000
export TENSORLBM_WARMUP_STEPS=1500
export TENSORLBM_STATISTICS_WINDOW_STEPS=1500
export TENSORLBM_RAMP_STEPS=1500
export TENSORLBM_REPORT_INTERVAL=375
export TENSORLBM_CHECKPOINT_INTERVAL=375
export TENSORLBM_RESOLVED_REYNOLDS=$target_re
export TENSORLBM_RESOLVED_REYNOLDS_START=5000
export TENSORLBM_VISCOSITY_RAMP_START_STEP=300
export TENSORLBM_VISCOSITY_RAMP_END_STEP=1500
export TENSORLBM_HEALTH_INTERVAL=375
export TENSORLBM_SPONGE_INLET=1
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_WALL_FORCE_DIRECTION_CHUNK=4
export TENSORLBM_LOW_MEMORY_WALL_MACROSCOPIC=1
export TENSORLBM_COMPILE_NATURAL_KBC=1

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
arguments=("$level" "$gpu" "$result_dir")
if [[ -n $wait_for_pid ]]; then
  arguments+=("$wait_for_pid")
fi
exec "$root/scripts/run_suboff_nested_v12_four_level_l90.sh" \
  "${arguments[@]}"
