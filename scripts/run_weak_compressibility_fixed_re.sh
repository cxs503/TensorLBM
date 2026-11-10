#!/usr/bin/env bash
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
OUT=artifacts/IN130-N03
LOG=logs/in130n03
mkdir -p "$OUT/raw_re"
python scripts/run_weak_compressibility_campaign.py --task pressure --out "$OUT/re" \
  --no-hold-viscosity --re 20 --ny 97 --nx 288 --pressure-steps 20000 \
  > "$LOG/pressure_fixedre.log" 2>&1
echo "fixed_re done" >> "$LOG/status.log"
cp -f "$OUT/re/raw/"pressure_ma*.json "$OUT/raw_re/" 2>/dev/null
