#!/usr/bin/env bash
set -euo pipefail

# Current-code no-inlet-sponge grid family.  This is intentionally distinct
# from v4 so the open-boundary choice is never mixed inside one provenance
# family.  R12 can run within the bounded-memory GPU1 envelope; R15 is queued
# only when a larger card is free.
export TENSORLBM_SPHERE_GENERATION=v7
export TENSORLBM_SPHERE_CAMPAIGN_LABEL=no-inlet-equivalent
export TENSORLBM_COLLISION_MODEL=natural_kbc_d3q19
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_COMPILE_NATURAL_KBC=1
export TENSORLBM_SPONGE_INLET=0

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_sphere_v3_equivalent_level.sh" "$@"
