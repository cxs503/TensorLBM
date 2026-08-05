#!/usr/bin/env bash
set -euo pipefail

# v9 changes only the independent surface-pressure observer.  The flow solver,
# wall model, BFL force and control-volume force remain identical to v8.
export TENSORLBM_SUBOFF_AMR_GENERATION=${TENSORLBM_SUBOFF_AMR_GENERATION:-v9}
export TENSORLBM_PRESSURE_REFERENCE=${TENSORLBM_PRESSURE_REFERENCE:-inlet}
export TENSORLBM_SURFACE_PRESSURE_EXTRAPOLATION=${TENSORLBM_SURFACE_PRESSURE_EXTRAPOLATION:-quadratic}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_suboff_v8_equivalent_level.sh" "$@"
