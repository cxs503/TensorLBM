#!/bin/bash
# Wait for the extension queue, then run the v2 three-estimator gate.
cd /nfs/wangxi/runs/bm_widen_w9_20260929/hex_ref
PY=/nfs/wangxi/venvs/hexref/bin/python
while true; do
  n=$(grep -c "QUEUE_SQ_EXT_ALL_DONE" out/logs/queue_sq_ext.log 2>/dev/null || echo 0)
  [ "$n" -ge 1 ] && break
  sleep 60
done
$PY src/extrapolate.py sq out/sq_gate2.json > out/logs/gate2_extrapolate.log 2>&1
echo GATE2_READY >> out/logs/gate2_extrapolate.log
