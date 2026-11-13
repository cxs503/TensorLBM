#!/bin/bash
# W26 / m17569 — detached launcher for the stage-2 (tau -> 0.5) sign-scan refinement.
set -u
REPO=/root/tp-agent/work/m17569
ART="$REPO/papers/artifacts/w26/lbm-suboff-tau"
OUT="$REPO/.scratch/m17569/scan2"
mkdir -p "$OUT"
exec python3 "$ART/run_batch.py" --plan "$ART/plan_scan2.json" --out "$OUT" --concurrency 2 --threads 5
