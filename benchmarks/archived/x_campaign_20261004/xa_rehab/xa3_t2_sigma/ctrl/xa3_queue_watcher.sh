#!/bin/bash
# XA-3 GPU6 queue watcher (X-campaign time-share protocol).
# Waits for the XB chain shell (arg1) to exit, then claims GPU6 for XA-3:
#   3 consecutive 60s free checks (no xb[12]_formal.py proc AND gpu6 < 500 MiB),
#   snapshot-before-use, abort if window lost at claim. Append-only NOTES.
CHAIN="$1"
BASE=/nfs/wangxi/runs/x_campaign_20261004/xa_rehab
cd "$BASE/xa3_t2_sigma" || exit 9
LOG=ctrl/xa3_queue.log
log(){ echo "[xa3-watcher $(date -u +%FT%TZ)] $*" >> "$LOG"; }

log "start; waiting for XB chain pid $CHAIN to exit"
while kill -0 "$CHAIN" 2>/dev/null; do sleep 60; done
log "XB chain $CHAIN exited"

FREE=0; I=0
while [ $I -lt 720 ]; do
  I=$((I+1))
  NXB=$(pgrep -fc "xb[12]_formal.py" || true)
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 | tr -d ' ')
  if [ "${NXB:-1}" -eq 0 ] && [ "${USED:-99999}" -lt 500 ]; then FREE=$((FREE+1)); else FREE=0; fi
  [ $FREE -ge 3 ] && break
  sleep 60
done
if [ $FREE -lt 3 ]; then log "TIMEOUT: no free GPU6 window within 12h; NOT launched"; exit 1; fi

TS=$(date -u +%Y%m%dT%H%M%SZ)
nvidia-smi > "ctrl/gpu6_snapshot_pre_xa3_${TS}.txt" 2>&1
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 | tr -d ' ')
NXB=$(pgrep -fc "xb[12]_formal.py" || true)
if [ "${USED:-99999}" -ge 500 ] || [ "${NXB:-1}" -ne 0 ]; then
  log "WINDOW LOST at claim (gpu6=${USED}MiB xb_procs=${NXB}); NOT launched"
  exit 2
fi

cat >> "$BASE/NOTES.md" <<EOF
- XA-3 GPU6 window OPEN (queued behind XB chain $CHAIN, watcher-fired):
  snapshot=xa3_t2_sigma/ctrl/gpu6_snapshot_pre_xa3_${TS}.txt (gpu6 ${USED} MiB,
  xb_procs=${NXB}); launch xa3_run.py (CUDA_VISIBLE_DEVICES=6, OMP=16).
  Queue history: XA-3 released by controller 2026-10-05 02:5xZ; GPU6 found
  occupied by XB window (xb1_formal chain, ~6.4 GPU*h plan per xb NOTES);
  queued per time-share protocol, 3-min grace + snapshot-abort.
EOF

(setsid env CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=16 TMPDIR=/nfs/wangxi/tmp \
   /nfs/wangxi/venvs/tensorlbm/bin/python xa3_run.py > run.log 2>&1 </dev/null &)
sleep 5
P=$(pgrep -f "xa3_run\.py" | head -1)
log "LAUNCHED xa3_run.py pid=${P:-unknown} snapshot=ctrl/gpu6_snapshot_pre_xa3_${TS}.txt"
