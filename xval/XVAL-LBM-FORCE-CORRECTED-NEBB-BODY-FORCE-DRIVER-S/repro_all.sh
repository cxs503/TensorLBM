#!/usr/bin/env bash
# One-command reproduction of anchor XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S.
#
#   bash xval/XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S/repro_all.sh
#
# Regenerates every committed run artifact of the anchor, then rebuilds the
# machine-checkable evidence bundle (evidence.json) in --strict mode, which
# re-derives every ref/got row from the stored raw profiles and exits non-zero
# if any declared impl file / entry / artifact / caller is missing or any
# primary benchmark row misses its pre-declared tolerance.
#
# Stages (all CPU, float64, no network):
#   1. XVAL harness          OSS-transcribed arm A vs in-repo arm B (D2Q9) + sensitivity/grid
#   2. package CLI           python -m tensorlbm.nebb_bodyforce (entry, writes cli_guo_ny10.json)
#   3. L3 core path          GPUCorePathSolver(D3Q19) with wall_bc='nebb' vs analytic
#   4. invariants            mass conservation / arm-to-arm / grid convergence / EDM-vs-Guo
#   5. startup transient     impulsive-start series solution check (transient, not just steady)
#   6. evidence --strict     rebuild evidence.json and fail loudly on any gap
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

NODE=XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S
ART="xval/${NODE}/artifacts"
STARTUP_ART="artifacts/xval_nebb_startup"
mkdir -p "$ART"

echo "== 1/9 XVAL harness (arm A vs arm B, D2Q9) =="
python3 benchmarks/xval/xval_vivsim_nebb_bodyforce.py --out "$ART"

echo "== 2/9 package CLI entry =="
python3 -m tensorlbm.nebb_bodyforce --ny 10 --scheme guo --out "$ART/cli_guo_ny10.json"

echo "== 3/9 L3: main core-path solver, wall_bc='nebb' =="
python3 "xval/${NODE}/run_corepath_l3.py" --ny 18 --out "$ART/corepath"

echo "== 4/9 invariants =="
python3 "xval/${NODE}/run_xval_r4_invariants.py"

echo "== 5/9 startup-transient series-solution check =="
python3 benchmarks/xval/xval_nebb_startup_transient.py --out "$STARTUP_ART"

echo "== 6/9 evidence bundle (--strict) =="
python3 benchmarks/xval/nebb_bodyforce_evidence.py --strict

echo "== 7/9 Round 7: single-tree A/B of the drive normalisation =="
python3 "xval/${NODE}/run_xval_r7_drive_width_ab.py" --out "$ART/r7"

echo "== 8/9 Round 7b: order of the residual error after the fix =="
python3 "xval/${NODE}/run_xval_r7b_order_refine.py" --out "$ART/r7"

echo "== 9/9 Round 7 evidence bundle (--strict, recomputes ref from closed form) =="
python3 benchmarks/xval/nebb_bodyforce_evidence_r7.py --strict
python3 "xval/${NODE}/make_report_r7.py"

echo "OK: artifacts under $ART and $STARTUP_ART"