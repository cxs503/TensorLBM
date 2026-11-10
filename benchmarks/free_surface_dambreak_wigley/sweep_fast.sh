#!/usr/bin/env bash
# fast coarse-grid cases (16 cells across the dam width) for the FIGURE + ladder
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p runs/p351w1_n07/logs
for wall in no-slip free-slip; do
  python benchmarks/free_surface_dambreak_wigley/run_dambreak.py \
    --cells 16 --wall "$wall" --nu 1e-3 --tol 1e-5 \
    --out runs/p351w1_n07/dambreak >> runs/p351w1_n07/logs/dambreak_fast.log 2>&1
done
echo FAST_DONE >> runs/p351w1_n07/logs/dambreak_fast.log
