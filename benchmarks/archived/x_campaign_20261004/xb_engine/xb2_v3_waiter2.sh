#!/bin/bash
# XB-2 v3 relay waiter v2 (2026-10-05, controller swap).
# v1 (pid 2839758) deadlocked: while-loop pgrep "xb[1]_formal" permanently
# matched its own launcher bash 2839752 (cmdline carried "xb1_formal.log"
# in a tail).  Meanwhile XA-3 correctly took GPU6 at 13:39:30 (pid 2857406,
# own 3-min-idle rule).  v2 keys on xa3_run.py exit + 3 consecutive minutes
# GPU6<500MiB, then runs xb2 v3 with v1 semantics unchanged (snapshot ->
# NOTES append -> run -> marker).  Log: out/xb2_v3_formal.log (10:33 crash
# log out/xb2_formal.log stays verbatim evidence).
while pgrep -f "xa3_run[.]py" >/dev/null 2>&1; do sleep 60; done
c=0
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
  if [ -n "$m" ] && [ "$m" -lt 500 ]; then c=$((c+1)); else c=0; fi
  [ "$c" -ge 3 ] && break
  sleep 60
done
cd /nfs/wangxi/runs/x_campaign_20261004/xb_engine || exit 1
SNAP=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 6)
{
  echo "- xb2 v3 relay window (waiter v2, 2026-10-05 ~$(date +%H:%M)): "
  echo "  xa3_run exited + GPU6 idle 3 min; snapshot before v3 start: ${SNAP};"
  echo "  launching xb2_formal.py v3 (md5 below); v1 waiter 2839758 retired"
  echo "  (deadlocked on launcher 2839752 cmdline match, see NOTES above):"
  md5sum xb2_formal.py | sed 's/^/  /'
} >> NOTES.md
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 TMPDIR=/nfs/wangxi/tmp
/nfs/wangxi/venvs/tensorlbm/bin/python xb2_formal.py > out/xb2_v3_formal.log 2>&1
echo done > out/xb2_v3_done.marker
