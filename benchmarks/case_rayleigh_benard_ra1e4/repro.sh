#!/usr/bin/env bash
# CASE-RAYLEIGH-BENARD-RA1E4 一键复现（单机 CPU，约 2×4.5 min）
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/case_rayleigh_benard_ra1e4
mkdir -p "$OUT" figures/case_rayleigh_benard_ra1e4

# 主档：nx=32, tau=0.90（稳态）
python3 benchmarks/case_rayleigh_benard_ra1e4/run.py --nx 32 --tau 0.9 \
    --steps 60000 --report-every 2000 --out "$OUT/local_nx32_tau0.9.json"
# 网格趋势档：nx=24
python3 benchmarks/case_rayleigh_benard_ra1e4/run.py --nx 24 --tau 0.9 \
    --steps 40000 --report-every 2000 --out "$OUT/local_nx24_tau0.9.json"

python3 benchmarks/case_rayleigh_benard_ra1e4/make_figures.py
echo "OK: $OUT + figures/case_rayleigh_benard_ra1e4"