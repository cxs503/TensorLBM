#!/usr/bin/env bash
# Grid-refinement ladder for XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI.
#
# Fixed aspect ratio L/H = 8 so the exact plane-Poiseuille profile is the same
# function of yt/H at every level; two extra levels re-run a resolution with
# twice the averaging time, which is the temporal-sensitivity control for the
# (non-converging pointwise) residual.
#
#   bash xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/run_ladder.sh
#
# Records land in artifacts/xval_zouhe/refine_*.json.
set -euo pipefail
cd "$(dirname "$0")/../.."          # TensorLBM repo root
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH=src
RUN="python xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/refinement.py"
OUT=artifacts/xval_zouhe
mkdir -p "$OUT"

#        H   nx   steps  burn-in  avg-window  tag
while read -r H NX STEPS BURN AVG TAG; do
  [ -z "${H:-}" ] && continue
  echo "=== $TAG: H=$H nx=$NX steps=$STEPS ==="
  $RUN --height "$H" --length "$NX" --steps "$STEPS" \
       --burn-in "$BURN" --avg-window "$AVG" --tol 1e-5 \
       --out "$OUT/refine_${TAG}.json"
done <<'LEVELS'
16  128    6000   3000    1500  H16_t6k
16  128   12000   6000    3000  H16_t12k
24  192   20000   8000    4000  H24_t20k
32  256   40000  16000    8000  H32_t40k
32  256   80000  32000   16000  H32_t80k
48  384   80000  32000   16000  H48_t80k
LEVELS
echo "ladder complete"