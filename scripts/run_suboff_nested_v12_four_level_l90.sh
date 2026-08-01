#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 L90 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
[[ $1 == L90 ]] || usage

# Four physical levels: L=90 -> 180 -> 360 -> 720.  The deepest block keeps
# the largest force CV (24 finest cells) plus its streaming-source guard away
# from the interface.  A one-cell exchange distance is the interpolation
# floor and has an a-priori y+ estimate of about 694 at L=720 (versus about
# 1527 measured at L=360 in v20).  This is a wall-distance/refinement pilot,
# not a member of the fixed exchange-height grid-convergence sequence.
export TENSORLBM_CAMPAIGN_GENERATION=${TENSORLBM_CAMPAIGN_GENERATION:-v12}
export TENSORLBM_INNER_WALL_MARGIN=${TENSORLBM_INNER_WALL_MARGIN:-8}
export TENSORLBM_INNER_WAKE_CELLS=${TENSORLBM_INNER_WAKE_CELLS:-12}
export TENSORLBM_DEEP_WALL_MARGIN=${TENSORLBM_DEEP_WALL_MARGIN:-13}
export TENSORLBM_DEEP_WAKE_CELLS=${TENSORLBM_DEEP_WAKE_CELLS:-26}
export TENSORLBM_CV_MARGIN=${TENSORLBM_CV_MARGIN:-16}
export TENSORLBM_AUX_CV_MARGINS=${TENSORLBM_AUX_CV_MARGINS:-8,24}
export TENSORLBM_STRESS_EXCHANGE_DISTANCE=${TENSORLBM_STRESS_EXCHANGE_DISTANCE:-1.0}
export TENSORLBM_MEMORY_BYTES_PER_CELL=${TENSORLBM_MEMORY_BYTES_PER_CELL:-943}
export TENSORLBM_REGULARIZE_RESTRICTION=${TENSORLBM_REGULARIZE_RESTRICTION:-1}
export TENSORLBM_REGULARIZE_PROLONGATION=${TENSORLBM_REGULARIZE_PROLONGATION:-0}
export TENSORLBM_GHOST_INTERPOLATION=${TENSORLBM_GHOST_INTERPOLATION:-trilinear}
export TENSORLBM_ENFORCE_TRANSFER_POSITIVITY=${TENSORLBM_ENFORCE_TRANSFER_POSITIVITY:-1}
export TENSORLBM_INTERFACE_FILTER_WIDTH=${TENSORLBM_INTERFACE_FILTER_WIDTH:-0}
export TENSORLBM_INTERFACE_FILTER_STRENGTH=${TENSORLBM_INTERFACE_FILTER_STRENGTH:-0}
export TENSORLBM_COLLISION_MODEL=${TENSORLBM_COLLISION_MODEL:-natural_kbc}
export TENSORLBM_CS_SMAG=${TENSORLBM_CS_SMAG:-0}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_suboff_nested_v4_continuation_level.sh" "$@"
