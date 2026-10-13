#!/bin/bash
# W11-A formal ladder driver: Re=40 first (D40->D80->D160), then Re=20.
# Frozen prereg protocol; halt on any failure (NaN abort / stop-loss / crash).
cd /nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady || exit 9
export TMPDIR=/nfs/wangxi/tmp
export CUDA_VISIBLE_DEVICES=6
PY=/nfs/wangxi/venvs/tensorlbm/bin/python
mkdir -p formal
for re in 40 20; do
  for D in 40 80 160; do
    echo "=== START re${re} D${D} $(date -Is) ==="
    $PY src/run_formal.py --re ${re} --D ${D} > logs/formal_re${re}_D${D}.log 2>&1
    rc=$?
    echo "=== END   re${re} D${D} rc=${rc} $(date -Is) ==="
    if [ $rc -ne 0 ]; then
      echo "HALT: re${re} D${D} exit ${rc} — stop-and-report per §7"
      exit $rc
    fi
  done
done
$PY src/build_ladder.py > logs/formal_ladder.log 2>&1
echo "ALL_DONE $(date -Is)"
