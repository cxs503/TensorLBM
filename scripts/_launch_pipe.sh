#!/usr/bin/env bash
# SDAA launcher for benchmarks/verified/poiseuille_3d_pipe (m74 round-2).
#
# Runs the case's own driver on the 'sdaa' device with the ORIGINAL criteria
# (README: radial-averaged profile max rel err <= 3%, R=20 tier, 20000 steps
# = README lower bound) and writes the result under outputs/sdaa/poiseuille_3d_pipe/.
#
# Notes / bypasses recorded honestly:
#   * --compile-mode eager: Torch-SDAA 3.1.1a3 teco_inductor cannot compile the
#     full LBM step chain (known L5 defect), so route_step() would fall back to
#     eager anyway.  Nothing numerical changes; wall clock is saved.
#   * TRITON_INTERPRET=1 is NOT needed on this path (no inductor/triton trace).
#   * The agent shell has a 120 s per-command limit, so the run is launched with
#     nohup in the background and polled via the run log.
set -euo pipefail
cd "$(dirname "$0")/.."

MIN_STEPS="${MIN_STEPS:-20000}"   # README lower bound — do not lower
MAX_STEPS="${MAX_STEPS:-20000}"
R="${R:-20}"
OUTPUT_DIR="outputs/sdaa/poiseuille_3d_pipe"
LOG="$OUTPUT_DIR/sdaa_poiseuille_3d_pipe.log"
mkdir -p "$OUTPUT_DIR"

nohup python scripts/sdaa_sweep_poiseuille_3d_pipe.py \
  --R "$R" --min-steps "$MIN_STEPS" --max-steps "$MAX_STEPS" \
  --compile-mode eager \
  --out-json "$OUTPUT_DIR/sdaa_poiseuille_3d_pipe.json" \
  --case-out "$OUTPUT_DIR/sdaa_poiseuille_3d_pipe_case.json" \
  > "$LOG" 2>&1 &
echo "launched pid=$! log=$LOG"
echo "poll with:  tail -f $LOG ; ls -l $OUTPUT_DIR/sdaa_poiseuille_3d_pipe.json"
