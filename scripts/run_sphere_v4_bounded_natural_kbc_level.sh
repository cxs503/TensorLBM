#!/usr/bin/env bash
set -euo pipefail

# Corrected open boundary plus bounded, graph-reusing natural-KBC.  R9/R12/R15
# must all use this wrapper before they are assessed as one grid family.
export TENSORLBM_SPHERE_GENERATION=v4
export TENSORLBM_COLLISION_MODEL=natural_kbc_d3q19
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_COMPILE_NATURAL_KBC=1
export TENSORLBM_SPONGE_INLET=1

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_sphere_v3_equivalent_level.sh" "$@"
