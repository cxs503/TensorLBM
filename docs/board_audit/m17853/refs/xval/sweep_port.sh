#!/bin/sh
set -e
cd "$(dirname "$0")"
for pw in -0.50 -0.35 0.00 0.35 0.50 0.70; do
  python3 run_xval.py port --psi-wall $pw --xi 3.0
done
for pw in 0.00 0.35; do
  for xi in 2.0 4.0; do
    python3 run_xval.py port --psi-wall $pw --xi $xi --tag "port_psi${pw}_xi${xi}"
  done
done
echo SWEEP_DONE
