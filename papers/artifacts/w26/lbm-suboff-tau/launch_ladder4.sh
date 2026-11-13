#!/bin/bash
# W26 / m17569 — detached launcher for the coarse uniform-ratio ladder {16,24,36}.
set -u
REPO=/root/tp-agent/work/m17569
ART="$REPO/papers/artifacts/w26/lbm-suboff-tau"
mkdir -p "$REPO/.scratch/m17569/runs"
exec python3 "$ART/run_batch.py" --plan "$ART/plan_ladder4.json" --out "$REPO/.scratch/m17569/runs" --concurrency 3 --threads 6
