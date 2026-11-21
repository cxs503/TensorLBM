#!/bin/bash
# W26 / m17569 — detached launcher for the three-grid fixed-Re GCI ladder.
# Absolute paths only.  Note the run history: the first attempt at these two grids
# (06:51) was lost when a housekeeping kill of an unrelated batch took the process
# group with it, so this launcher is explicit and the batch re-runs all grids that
# are not yet on disk.
set -u
REPO=/srv/tensor/agent/work/m17569
ART="$REPO/benchmarks/anchors/evidence/suboff_w26"
OUT="$REPO/.scratch/m17569/runs"
mkdir -p "$OUT"
exec python3 "$ART/run_batch.py" \
  --plan "$ART/plan_ladder2.json" \
  --out "$OUT" \
  --concurrency 2 \
  --threads 8
