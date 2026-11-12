#!/usr/bin/env bash
# One-command reproduction of CASE-CYLINDER-DRAG-RE100 (Schäfer–Turek 2D-2, Re=100).
#
#   ./benchmarks/verified/cylinder/re100_dfg/repro.sh              # full ladder D=20/28/40
#   ./benchmarks/verified/cylinder/re100_dfg/repro.sh 20 6000     # one level, explicit step count
#   ./benchmarks/verified/cylinder/re100_dfg/repro.sh --quick     # smoke run (NOT a validation claim)
#
# Writes into benchmarks/verified/cylinder/re100_dfg/artifacts/ :
#   level_D<k>_<tag>.json          per-grid observables + provenance
#   level_D<k>_<tag>.series.npz    raw Cd/Cl/dp histories (re-analysis without a re-run)
#   result.json                    aggregated ref-vs-got table + per-metric verdict
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
THREADS="${OMP_NUM_THREADS:-8}"

if [ "${1:-}" = "--quick" ]; then
  python benchmarks/case_pool/cylinder_drag_re100/run_case.py \
    --grids 10 --steps 400 --avg-periods 1 --threads 2 --out /tmp/cyl_re100_quick
  exit 0
fi

if [ "$#" -ge 1 ]; then
  python benchmarks/verified/cylinder/re100_dfg/run.py \
    --levels "$1" --steps "${2:-20000}" --threads "$THREADS"
  exit 0
fi

# Full ladder.  Shedding period in lattice steps: T = D / (St * u_lb) = D / (0.3*0.0833) ≈ 40·D.
# 17 periods per level = ~5 periods of transient + 12 periods of averaging window.
# steps = period(T=40·D) × (transient 5 + averaging 12 + 2 margin) = 40·D·19.
# A shorter run is REJECTED by CaseConfig (m16827: the old 17-period run made the
# averaging window swallow the whole record, transient included).
for D in 20 28 40; do
  STEPS=$(( 40 * D * 19 ))
  echo ">>> D=$D cells, $STEPS steps"
  python benchmarks/verified/cylinder/re100_dfg/run.py --levels "$D" --steps "$STEPS" --threads "$THREADS"
done
python benchmarks/verified/cylinder/re100_dfg/run.py --aggregate-only --levels 20 28 40