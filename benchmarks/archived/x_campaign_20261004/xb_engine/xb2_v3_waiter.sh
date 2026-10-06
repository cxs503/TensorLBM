#!/bin/bash
# XB-2 v3 relay waiter (2026-10-05, controller instruction 2).
# Waits for the xb1_formal.py battery process (any xb[1]_formal python,
# currently pid 2832077) to exit, snapshots GPU6 into NOTES.md
# (snapshot-before-use), then runs xb2_formal.py v3.  Log goes to
# out/xb2_v3_formal.log so the 10:33 crash log out/xb2_formal.log is
# preserved verbatim as evidence.  Completion marker: out/xb2_v3_done.marker
# Race vs XA-3 queue watcher (pid 2835022): XA-3 requires 3 consecutive
# minutes of no-xb[12]-process + GPU6<500MiB; this waiter launches v3 within
# ~85s of xb1 exit, so XA-3 defers automatically (no conflict).
while pgrep -f "xb[1]_formal" >/dev/null 2>&1; do sleep 60; done
cd /nfs/wangxi/runs/x_campaign_20261004/xb_engine || exit 1
SNAP=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 6)
{
  echo "- xb2 v3 relay window (2026-10-05): xb1_formal exited; GPU6 snapshot"
  echo "  before v3 start: ${SNAP}; launching xb2_formal.py v3 (md5 below):"
  md5sum xb2_formal.py | sed 's/^/  /'
} >> NOTES.md
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 TMPDIR=/nfs/wangxi/tmp
/nfs/wangxi/venvs/tensorlbm/bin/python xb2_formal.py > out/xb2_v3_formal.log 2>&1
echo done > out/xb2_v3_done.marker
