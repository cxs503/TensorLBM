#!/bin/bash
# square-ladder queues (prereg §5). Usage: queue_sq.sh <group a|b|c>
cd /nfs/wangxi/runs/bm_widen_w9_20260929/hex_ref/src
PY=/nfs/wangxi/venvs/hexref/bin/python
G=$1
run () { $PY run_case.py "$@"; }
case $G in
a)
  for N in 192 288 384 512 768 1024 1280; do run sq 0.30 $N $N; done
  for N in 192 288 384 512 768 1024 1280; do run sq 0.40 $N $N; done ;;
b)
  for N in 192 288 384 512 768 1024 1280; do run sq 0.50 $N $N; done
  for N in 192 288 384 512 768 1024 1280; do run sq 0.60 $N $N; done ;;
c)
  for N in 256 384 512 768 1024 1280 1536; do run sq 0.70 $N $N; done
  for N in 512 768 1024 1280 1536 1792;   do run sq 0.75 $N $N; done ;;
esac
echo QUEUE_SQ_${G}_ALL_DONE
