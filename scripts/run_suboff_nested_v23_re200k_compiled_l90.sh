#!/usr/bin/env bash
set -euo pipefail

# Corrected-boundary, bounded-memory and graph-reusing Re=200k causal pilot.
# The 3k duration is a stability/component diagnostic, not a converged drag
# claim.  Its explicit generation prevents accidental resume from the eager
# no-inlet-sponge v18 trajectory.
export TENSORLBM_CAMPAIGN_GENERATION=v23
export TENSORLBM_STEPS=3000
export TENSORLBM_WARMUP_STEPS=1500
export TENSORLBM_STATISTICS_WINDOW_STEPS=1500
export TENSORLBM_RAMP_STEPS=1500
export TENSORLBM_REPORT_INTERVAL=375
export TENSORLBM_CHECKPOINT_INTERVAL=375
export TENSORLBM_RESOLVED_REYNOLDS=200000
export TENSORLBM_RESOLVED_REYNOLDS_START=5000
export TENSORLBM_VISCOSITY_RAMP_START_STEP=300
export TENSORLBM_VISCOSITY_RAMP_END_STEP=1500
export TENSORLBM_HEALTH_INTERVAL=375
export TENSORLBM_SPONGE_INLET=1
export TENSORLBM_COLLISION_CHUNK_CELLS=262144
export TENSORLBM_WALL_FORCE_DIRECTION_CHUNK=4
export TENSORLBM_LOW_MEMORY_WALL_MACROSCOPIC=1
export TENSORLBM_COMPILE_NATURAL_KBC=1

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/scripts/run_suboff_nested_v12_four_level_l90.sh" "$@"
