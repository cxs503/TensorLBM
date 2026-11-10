#!/usr/bin/env bash
# Dam-break sensitivity + grid-refinement sweep (Martin & Moyce 1952 anchor).
# Every case writes runs/p351w1_n07/dambreak/<tag>.json; nothing is hard-coded.
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p runs/p351w1_n07/logs
for cells in 32 48 16; do
  for wall in no-slip free-slip; do
    python benchmarks/free_surface_dambreak_wigley/run_dambreak.py \
      --cells "$cells" --wall "$wall" --nu 1e-3 --tol 1e-6 \
      --out runs/p351w1_n07/dambreak \
      >> runs/p351w1_n07/logs/dambreak_sweep.log 2>&1
  done
done
echo "SWEEP_DONE" >> runs/p351w1_n07/logs/dambreak_sweep.log