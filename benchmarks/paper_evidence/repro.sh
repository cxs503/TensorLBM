#!/usr/bin/env bash
# One-command reproduction of the paper-evidence bundle
# (GAP-DIFFLBM-PAPER-EVIDENCE).
#
#   bash benchmarks/paper_evidence/repro.sh            # run every item + aggregate + gate
#   bash benchmarks/paper_evidence/repro.sh A1_poiseuille_2d L3_cylinder_array_sangani
#
# Each item is independent; on a cluster submit them as separate jobs (shards)
# with the same command, e.g.
#   run_compute submit cmd='bash benchmarks/paper_evidence/repro.sh L1_cavity_ghia_re100'
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONPATH="${PYTHONPATH:-src}:src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [ "$#" -gt 0 ]; then
  args=()
  for it in "$@"; do args+=(--item "$it"); done
  python benchmarks/paper_evidence/run_evidence.py "${args[@]}" --out-dir runs/paper_evidence
else
  python benchmarks/paper_evidence/run_evidence.py --all --out-dir runs/paper_evidence
fi

python benchmarks/paper_evidence/aggregate.py
python -m pytest tests/test_paper_evidence_bundle.py -q