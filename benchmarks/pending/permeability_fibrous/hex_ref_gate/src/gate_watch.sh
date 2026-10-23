#!/bin/bash
# wait for the three sq queues, then extrapolate automatically
cd /nfs/wangxi/runs/bm_widen_w9_20260929/hex_ref
PY=/nfs/wangxi/venvs/hexref/bin/python
while true; do
  n=$(grep -l "QUEUE_SQ_._ALL_DONE" out/logs/queue_sq_*.log 2>/dev/null | wc -l)
  [ "$n" -ge 3 ] && break
  sleep 60
done
$PY src/extrapolate.py sq out/sq_gate.json > out/logs/gate_extrapolate.log 2>&1
echo GATE_READY >> out/logs/gate_extrapolate.log
