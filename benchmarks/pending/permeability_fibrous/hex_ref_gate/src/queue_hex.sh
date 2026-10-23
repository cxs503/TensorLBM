#!/bin/bash
# hex ladders (launch ONLY after sq gate passes; prereg §7 order).
cd /nfs/wangxi/runs/bm_widen_w9_20260929/hex_ref/src
PY=/nfs/wangxi/venvs/hexref/bin/python
G=$1
run () { $PY run_case.py "$@"; }
case $G in
a)
  for m in 2 3 4 6 8 11 14 18; do run hexA 0.70 $((56*m)) $((97*m)); done ;;
b)
  for m in 2 3 4 6 8 11 14 18; do run hexA 0.75 $((56*m)) $((97*m)); done ;;
s)  # aspect-slope ladder A'=26/15
  for m in 20 28; do run hexB 0.70 $((15*m)) $((26*m)); done
  for m in 20 28; do run hexB 0.75 $((15*m)) $((26*m)); done ;;
esac
echo QUEUE_HEX_${G}_ALL_DONE
