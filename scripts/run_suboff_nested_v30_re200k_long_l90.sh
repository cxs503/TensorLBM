#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 L90 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID ...]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 5 ]] || usage
[[ $1 == L90 ]] || usage
level=$1
gpu=$2
result_dir=${3:-results/amr_campaign_20260801}
if (( $# >= 4 )); then
  for wait_for_pid in "${@:4}"; do
    [[ $wait_for_pid =~ ^[0-9]+$ ]] || usage
    while kill -0 "$wait_for_pid" 2>/dev/null; do
      sleep 20
    done
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
seed="$result_dir/suboff-nested-v23-equivalent-l90-3k.ckpt"
if [[ ! -f $seed ]]; then
  echo "completed v23 continuation seed does not exist: $seed" >&2
  exit 2
fi

# Continue the admitted one-cell L720 exchange family for eight total
# convective times.  The physical-Re y+ prior is about 694 and the v23
# target-Re checkpoint measured a mean of 772.  The 1/720 exchange-height
# ratio, rather than the unrelated flat-plate 3/256 ratio, is held fixed for
# the subsequent L720/L960/L1200 SUBOFF grid family.
export TENSORLBM_CAMPAIGN_GENERATION=v30
export TENSORLBM_CONTINUE_FROM_CHECKPOINT=$seed
export TENSORLBM_STEPS=12000
export TENSORLBM_WARMUP_STEPS=4500
export TENSORLBM_STATISTICS_WINDOW_STEPS=7500
export TENSORLBM_RAMP_STEPS=1500
export TENSORLBM_WALL_NORMAL_RAMP_STEPS=1500
export TENSORLBM_WALL_SHEAR_RAMP_STEPS=1500
export TENSORLBM_REPORT_INTERVAL=375
export TENSORLBM_CHECKPOINT_INTERVAL=750
export TENSORLBM_RESOLVED_REYNOLDS=200000
export TENSORLBM_RESOLVED_REYNOLDS_START=5000
export TENSORLBM_VISCOSITY_RAMP_START_STEP=300
export TENSORLBM_VISCOSITY_RAMP_END_STEP=1500
export TENSORLBM_SPONGE_INLET=1
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_WALL_FORCE_DIRECTION_CHUNK=4
export TENSORLBM_LOW_MEMORY_WALL_MACROSCOPIC=1
export TENSORLBM_COMPILE_NATURAL_KBC=1
export TENSORLBM_STRESS_EXCHANGE_DISTANCE=1.0
export TENSORLBM_WALL_EXCHANGE_DISTANCE_OVER_LENGTH_TARGET=0.0013888888888888889

exec "$root/scripts/run_suboff_nested_v12_four_level_l90.sh" \
  "$level" "$gpu" "$result_dir"
