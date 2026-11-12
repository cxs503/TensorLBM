#!/usr/bin/env bash
# Periodic-box confinement study of the 3-D fluid leg (GAP-LBM-T5-ICE-COUPLING round 2).
#
# Four grids at the same physical Reynolds number (nu = 0.03, u = 0.05) and different
# sphere-diameter / box-size ratios.  The reference is the Clift-Grace-Weber rigid-sphere
# correlation; the Stokes-limit Hasimoto periodic-array factor is reported alongside so the
# residual can be attributed to confinement instead of being hidden.
#
# Usage (also used verbatim by the cluster submission):
#   bash scripts/three_way_coupled3d_scan.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
for pair in "48 8" "64 10" "96 12" "64 16"; do
  set -- $pair
  n="$1"; d="$2"
  python3 scripts/three_way_coupled3d_run.py --fluid-only \
      --n "$n" --diameter "$d" --steps 3000 --measure 800 --tag "n${n}_d${d}"
done