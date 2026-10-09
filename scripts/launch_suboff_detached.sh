#!/bin/bash
# Launch suboff_re1000 (progress-instrumented) detached via setsid.
# Usage: launch.sh <DEVICE> <OUTDIR> <RESOLUTION> <STEPS> <FRICTION> <LOGFILE>
set -u
DEV="$1"; OUTDIR="$2"; RES="$3"; STEPS="$4"; FRICTION="$5"; LOG="$6"
export PYTHONPATH=/root/TensorLBM_feat2/src
export PROG_EVERY="${PROG_EVERY:-100}"
cd /root/TensorLBM_feat2
setsid nohup python scripts/_suboff_progress_run.py \
  --resolution "$RES" --steps "$STEPS" --device "$DEV" \
  --collision mrt --friction "$FRICTION" --out "$OUTDIR" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! dev=$DEV log=$LOG out=$OUTDIR res=$RES steps=$STEPS friction=$FRICTION"