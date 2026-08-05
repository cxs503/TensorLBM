#!/usr/bin/env bash
set -euo pipefail

# Hold R12 resolution, x-domain, collision, duration and open boundaries fixed;
# enlarge only the transverse width from 16R to 20R.  This isolates blockage
# and lateral-boundary sensitivity before attributing the remaining drag error
# to BFL geometry or the collision operator.
export TENSORLBM_SPHERE_GENERATION=v5
export TENSORLBM_SPHERE_CAMPAIGN_LABEL=domain-w20
export TENSORLBM_SPHERE_NX=288
export TENSORLBM_SPHERE_CROSS=240
export TENSORLBM_COLLISION_MODEL=natural_kbc_d3q19
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_COMPILE_NATURAL_KBC=1
export TENSORLBM_SPONGE_INLET=1

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_sphere_v3_equivalent_level.sh" R12 "$@"
