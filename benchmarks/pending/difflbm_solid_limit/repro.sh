#!/usr/bin/env bash
# One-command reproduction of GAP-DIFFLBM-SOLID-LIMIT.
#
#   bash benchmarks/pending/difflbm_solid_limit/repro.sh            # local (fast) path
#   bash benchmarks/pending/difflbm_solid_limit/repro.sh --full     # full campaign
#
# The campaign is 14 independent (arm, D, mask-scale) runs; each one is a
# separate process so it can be sharded on the cluster:
#   PYTHONPATH=src python benchmarks/pending/difflbm_solid_limit/campaign.py --index {i} \
#       --out artifacts/GAP-DIFFLBM-SOLID-LIMIT/runs --resolutions 20 32 \
#       --scales 1.0 0.75 0.5 --max-steps 50000 --min-steps 8000 --u-lb 0.1
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src
OUT=artifacts/GAP-DIFFLBM-SOLID-LIMIT
python -m pytest tests/test_volume_solid.py -q
if [[ "${1:-}" == "--full" ]]; then
  for i in $(seq 0 13); do
    python benchmarks/pending/difflbm_solid_limit/campaign.py --index "$i" --out "$OUT/runs" \
      --resolutions 20 32 --scales 1.0 0.75 0.5 --max-steps 50000 --min-steps 8000 \
      --avg-window 2000 --plateau-tol 1e-3 --u-lb 0.1
  done
else
  # smoke path: 600 steps, two arms, one resolution (not verdict-bearing)
  python benchmarks/pending/difflbm_solid_limit/run_case.py --arm hardwall --D 20 --scale 1.0 \
    --max-steps 600 --min-steps 600 --avg-window 300 --cv-every 20 --out "$OUT/smoke"
  python benchmarks/pending/difflbm_solid_limit/run_case.py --arm interface --D 20 --scale 0.5 \
    --max-steps 600 --min-steps 600 --avg-window 300 --cv-every 20 --out "$OUT/smoke"
fi
python benchmarks/pending/difflbm_solid_limit/summarize.py
