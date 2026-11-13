#!/bin/sh
# auditor re-run of the 10 predeclared port points (m17726 deliverable, verbatim)
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/refs/src"
cd refs/rerun
for pw in -0.50 -0.35 0.00 0.35 0.50 0.70; do
  python3 run_xval.py port --psi-wall $pw --xi 3.0 > /tmp/rerun_$pw.log 2>&1
done
for pw in 0.00 0.35; do
  for xi in 2.0 4.0; do
    python3 run_xval.py port --psi-wall $pw --xi $xi > /tmp/rerun_${pw}_${xi}.log 2>&1
  done
done
echo ALL_DONE > /tmp/rerun_done
