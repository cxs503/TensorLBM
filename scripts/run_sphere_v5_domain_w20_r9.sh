#!/usr/bin/env bash
set -euo pipefail

# Memory-bounded transverse-domain pilot: hold the admitted R9 resolution,
# streamwise extent, collision, duration and open boundaries fixed while the
# transverse width grows from 16R to 20R.
export TENSORLBM_SPHERE_GENERATION=v5
export TENSORLBM_SPHERE_CAMPAIGN_LABEL=domain-w20
export TENSORLBM_SPHERE_NX=216
export TENSORLBM_SPHERE_CROSS=180
export TENSORLBM_COLLISION_MODEL=natural_kbc_d3q19
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_COMPILE_NATURAL_KBC=1
export TENSORLBM_SPONGE_INLET=1

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_sphere_v3_equivalent_level.sh" R9 "$@"
