#!/bin/bash
# W26 / m17569 — detached launcher for the (tau, N) negative-drag scan batch.
# Absolute paths only: an unset $VAR once put the output root at "/scan", so the
# launcher now hard-codes it and the batch writes inside the worktree.
set -u
REPO=/root/tp-agent/work/m17569
ART="$REPO/papers/artifacts/w26/lbm-suboff-tau"
OUT="$REPO/.scratch/m17569/scan"
mkdir -p "$OUT"
exec python3 "$ART/run_batch.py" \
  --plan "$ART/plan_scan.json" \
  --out "$OUT" \
  --concurrency 3 \
  --threads 5