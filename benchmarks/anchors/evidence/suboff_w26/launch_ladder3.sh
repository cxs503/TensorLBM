#!/bin/bash
# W26 / m17569 — detached launcher for the fast uniform-ratio fallback ladder {24,36,54}.
set -u
REPO=/srv/tensor/agent/work/m17569
ART="$REPO/benchmarks/anchors/evidence/suboff_w26"
OUT="$REPO/.scratch/m17569/runs"
mkdir -p "$OUT"
exec python3 "$ART/run_batch.py" --plan "$ART/plan_ladder3.json" --out "$OUT" --concurrency 3 --threads 6
