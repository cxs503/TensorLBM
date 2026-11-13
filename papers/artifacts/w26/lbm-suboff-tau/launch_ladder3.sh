#!/bin/bash
# W26 / m17569 — detached launcher for the fast uniform-ratio fallback ladder {24,36,54}.
set -u
REPO=/root/tp-agent/work/m17569
ART="$REPO/papers/artifacts/w26/lbm-suboff-tau"
OUT="$REPO/.scratch/m17569/runs"
mkdir -p "$OUT"
exec python3 "$ART/run_batch.py" --plan "$ART/plan_ladder3.json" --out "$OUT" --concurrency 3 --threads 6
