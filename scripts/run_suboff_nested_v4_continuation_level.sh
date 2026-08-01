#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 L90|L120|L150 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
level=${1#L}

case "$level" in
  90)
    start_re=5000
    viscosity_start=300
    viscosity_end=600
    health_interval=60
    ;;
  120)
    start_re=3000
    viscosity_start=400
    viscosity_end=800
    health_interval=80
    ;;
  150)
    start_re=2000
    viscosity_start=500
    viscosity_end=1000
    health_interval=100
    ;;
  *) usage ;;
esac

# Keep every stability mechanism explicit and overridable.  These defaults
# encode the audited continuation path; they do not alter the physical wall
# viscosity or the target Re=100000 used for admitted force statistics.
export TENSORLBM_CAMPAIGN_GENERATION=${TENSORLBM_CAMPAIGN_GENERATION:-v4}
export TENSORLBM_INNER_WALL_MARGIN=${TENSORLBM_INNER_WALL_MARGIN:-8}
export TENSORLBM_INNER_WAKE_CELLS=${TENSORLBM_INNER_WAKE_CELLS:-12}
export TENSORLBM_CV_MARGIN=${TENSORLBM_CV_MARGIN:-8}
export TENSORLBM_AUX_CV_MARGINS=${TENSORLBM_AUX_CV_MARGINS:-4,12}
export TENSORLBM_REGULARIZE_RESTRICTION=${TENSORLBM_REGULARIZE_RESTRICTION:-1}
export TENSORLBM_REGULARIZE_PROLONGATION=${TENSORLBM_REGULARIZE_PROLONGATION:-0}
export TENSORLBM_GHOST_INTERPOLATION=${TENSORLBM_GHOST_INTERPOLATION:-trilinear}
export TENSORLBM_ENFORCE_TRANSFER_POSITIVITY=${TENSORLBM_ENFORCE_TRANSFER_POSITIVITY:-1}
export TENSORLBM_RESOLVED_REYNOLDS_START=${TENSORLBM_RESOLVED_REYNOLDS_START:-$start_re}
export TENSORLBM_VISCOSITY_RAMP_START_STEP=${TENSORLBM_VISCOSITY_RAMP_START_STEP:-$viscosity_start}
export TENSORLBM_VISCOSITY_RAMP_END_STEP=${TENSORLBM_VISCOSITY_RAMP_END_STEP:-$viscosity_end}
export TENSORLBM_HEALTH_INTERVAL=${TENSORLBM_HEALTH_INTERVAL:-$health_interval}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_suboff_nested_v3_equivalent_level.sh" "$@"
