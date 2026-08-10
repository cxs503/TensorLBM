#!/usr/bin/env bash
set -euo pipefail

# Corrected BFL provenance family.  This deliberately uses a new generation
# and fresh checkpoints because pre-v9 diagonal link fractions were wrong.
export TENSORLBM_SPHERE_GENERATION=v9
export TENSORLBM_SPHERE_CAMPAIGN_LABEL=corrected-bfl-no-inlet
export TENSORLBM_COLLISION_MODEL=natural_kbc_d3q19
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_COMPILE_NATURAL_KBC=1
export TENSORLBM_SPONGE_INLET=0

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_sphere_v3_equivalent_level.sh" "$@"
