#!/usr/bin/env bash
# R5 measurement campaign (anchor XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC).
# Sequential on purpose: every arm merges into artifacts_r5/*.json, so parallel
# jobs would clobber each other's artifact file.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src
X=development/verification/regularised_bc_noneq_stress/run_xval_r5.py
python "$X" --arm invariant
python "$X" --arm ladder --nys 16,24,32,48
python "$X" --arm tau --taus 0.6,1.0,1.5
python "$X" --arm force --forces 1e-5,4e-5
python "$X" --arm osc --deltas 4,6
python "$X" --arm analyse