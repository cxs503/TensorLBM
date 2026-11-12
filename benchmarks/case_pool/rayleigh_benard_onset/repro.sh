#!/usr/bin/env bash
# CASE-LBM-RB-ONSET-RA1708 — one-command reproduction.
#
#   bash benchmarks/case_pool/rayleigh_benard_onset/repro.sh [OUT_DIR]
#
# Produces: Ra bisection scan artifacts (summary.json + per-route JSON),
# the six figures and REPORT.md rendered from the artifacts.  No number in the
# report is typed by hand.
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT_DIR="${1:-runs/case_rayleigh_benard_onset}"

export PYTHONPATH=src
export RB_LBM_THREADS="${RB_LBM_THREADS:-1}"   # one thread per arm measured fastest

echo "== [1/3] production run (bisection scan ny=32/48/64 + cross / control arms) =="
python3 benchmarks/case_pool/rayleigh_benard_onset/run_case.py \
    --out-dir "$OUT_DIR" --parallel "${PARALLEL:-3}"

echo "== [2/3] figures (incl. supercritical field snapshot) =="
python3 benchmarks/case_pool/rayleigh_benard_onset/make_figures.py \
    --run-dir "$OUT_DIR" --asym

echo "== [3/3] REPORT.md from artifacts =="
python3 benchmarks/case_pool/rayleigh_benard_onset/make_report.py \
    --run-dir "$OUT_DIR" --out benchmarks/case_pool/rayleigh_benard_onset/REPORT.md

echo "done -> $OUT_DIR/summary.json , benchmarks/case_pool/rayleigh_benard_onset/REPORT.md"