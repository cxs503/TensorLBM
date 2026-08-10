#!/usr/bin/env bash
set -euo pipefail

# Fourth member of the 2x2 open-boundary experiment:
# transverse width W16/W24 times inlet sponge disabled/enabled.
export TENSORLBM_SPHERE_GENERATION=v8
export TENSORLBM_SPHERE_CAMPAIGN_LABEL=no-inlet-domain-w24
export TENSORLBM_SPHERE_NX=216
export TENSORLBM_SPHERE_CROSS=216
export TENSORLBM_COLLISION_MODEL=natural_kbc_d3q19
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_COMPILE_NATURAL_KBC=1
export TENSORLBM_SPONGE_INLET=0

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_sphere_v3_equivalent_level.sh" R9 "$@"
