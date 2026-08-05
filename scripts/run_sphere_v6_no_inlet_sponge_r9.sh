#!/usr/bin/env bash
set -euo pipefail

# Matched current-code A/B against sphere v4 R9.  The only physical-boundary
# switch is removal of x- equilibrium-difference sponge damping; bounded
# compiled Natural-KBC and the non-equilibrium far field remain unchanged.
export TENSORLBM_SPHERE_GENERATION=v6
export TENSORLBM_SPHERE_CAMPAIGN_LABEL=no-inlet-sponge
export TENSORLBM_COLLISION_MODEL=natural_kbc_d3q19
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_COMPILE_NATURAL_KBC=1
export TENSORLBM_SPONGE_INLET=0

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_sphere_v3_equivalent_level.sh" R9 "$@"
