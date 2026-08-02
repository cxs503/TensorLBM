#!/usr/bin/env bash
set -euo pipefail

# Long-time four-level production candidate.  The inlet sponge is mandatory
# after the matched v13/v14 A/B test isolated its root-grid stability role.
export TENSORLBM_CAMPAIGN_GENERATION=v24
export TENSORLBM_SPONGE_INLET=1
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_WALL_FORCE_DIRECTION_CHUNK=4
export TENSORLBM_LOW_MEMORY_WALL_MACROSCOPIC=1
export TENSORLBM_COMPILE_NATURAL_KBC=1

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_suboff_nested_v12_four_level_l90.sh" "$@"
