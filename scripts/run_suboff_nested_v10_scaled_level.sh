#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 L90|L120|L150 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
level=${1#L}

# The artificial refinement boundary, all force-observer boxes and the wall
# sampling point occupy identical physical locations across L90/L120/L150.
# Integer 3:4:5 scaling is mandatory for a defensible three-grid sequence.
case "$level" in
  90)
    inner_wall=9; inner_wake=12; cv=6; auxiliary=3,9
    exchange=2.109375
    ;;
  120)
    inner_wall=12; inner_wake=16; cv=8; auxiliary=4,12
    exchange=2.8125
    ;;
  150)
    inner_wall=15; inner_wake=20; cv=10; auxiliary=5,15
    exchange=3.515625
    ;;
  *) usage ;;
esac

export TENSORLBM_CAMPAIGN_GENERATION=${TENSORLBM_CAMPAIGN_GENERATION:-v10}
export TENSORLBM_INNER_WALL_MARGIN=${TENSORLBM_INNER_WALL_MARGIN:-$inner_wall}
export TENSORLBM_INNER_WAKE_CELLS=${TENSORLBM_INNER_WAKE_CELLS:-$inner_wake}
export TENSORLBM_CV_MARGIN=${TENSORLBM_CV_MARGIN:-$cv}
export TENSORLBM_AUX_CV_MARGINS=${TENSORLBM_AUX_CV_MARGINS:-$auxiliary}
export TENSORLBM_STRESS_EXCHANGE_DISTANCE=${TENSORLBM_STRESS_EXCHANGE_DISTANCE:-$exchange}
export TENSORLBM_REGULARIZE_PROLONGATION=${TENSORLBM_REGULARIZE_PROLONGATION:-1}
export TENSORLBM_INTERFACE_FILTER_WIDTH=${TENSORLBM_INTERFACE_FILTER_WIDTH:-2}
export TENSORLBM_INTERFACE_FILTER_STRENGTH=${TENSORLBM_INTERFACE_FILTER_STRENGTH:-1.0}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_suboff_nested_v4_continuation_level.sh" "$@"
