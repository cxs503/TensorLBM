#!/usr/bin/env bash
# Sequential driver for the XVAL main runs.
# One process at a time on purpose: the arms are timed, and running them in
# parallel would make the host-load confounder worse, not better.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=artifacts/xval_maxregcount_pinned
mkdir -p "$OUT"
RUN=xval/XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING/run_arm.py

echo "[driver] ladder $(date -u +%H:%M:%S)"
python3 "$RUN" --mode ladder --out "$OUT/ladder.json" > "$OUT/ladder.log" 2>&1 || echo "[driver] ladder FAILED"

for k in 0 1 2 3 4; do
  echo "[driver] U launch $k $(date -u +%H:%M:%S)"
  python3 "$RUN" --mode tune --arm U --launch "$k" --out "$OUT/tune_U_$k.json" > "$OUT/tune_U_$k.log" 2>&1 || echo "[driver] U $k FAILED"
done

for k in 0 1 2 3 4; do
  echo "[driver] P launch $k $(date -u +%H:%M:%S)"
  python3 "$RUN" --mode tune --arm P --launch "$k" --out "$OUT/tune_P_$k.json" > "$OUT/tune_P_$k.log" 2>&1 || echo "[driver] P $k FAILED"
done
echo "[driver] done $(date -u +%H:%M:%S)"