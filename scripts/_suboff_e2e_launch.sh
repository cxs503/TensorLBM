#!/bin/bash
# Launch the SUBOFF Re=1000 end-to-end convergence runs (optimized SDAA code).
set -u
cd /root/TensorLBM_feat2
export PYTHONPATH=/root/TensorLBM_feat2/src
export FT_TIMEOUT=86400
export OMP_NUM_THREADS=8
# optimized SDAA path (defaults already ON in d8a205e; pin explicitly)
export TL_STREAM_MODE=cat
export TL_FUSE_NODYN_BB=1
export TL_ISFINITE_INTERVAL=50
# drop free-surface env leftovers (irrelevant to suboff, avoid stray overhead)
unset TL_FS_TOGAS_EPS TL_FS_DIAG_FIELD TL_FS_WALL_MODE TL_FS_APRIME TL_FS_LIQ_TO_IFACE 2>/dev/null || true

LOGDIR=/root/TensorLBM_feat2/runs/e2e_suboff_20260929
mkdir -p "$LOGDIR"

launch () {
  local name="$1"; shift
  setsid nohup python scripts/_suboff_e2e_monitor.py "$@" \
    > "$LOGDIR/${name}.log" 2>&1 &
  echo "launched $name pid=$!"
}

# A: L=48 standard baseline , dev0
launch L48_standard --resolution 48 --steps 20000 --device sdaa:0 \
  --collision mrt --friction standard \
  --out /root/TensorLBM_feat2/results_e2e_suboff_L48_standard

# B: L=48 mix50 , dev1
launch L48_mix50 --resolution 48 --steps 20000 --device sdaa:1 \
  --collision mrt --friction mix50 \
  --out /root/TensorLBM_feat2/results_e2e_suboff_L48_mix50

# C: L=80 mix50 , dev2
launch L80_mix50 --resolution 80 --steps 12000 --device sdaa:2 \
  --collision mrt --friction mix50 \
  --out /root/TensorLBM_feat2/results_e2e_suboff_L80_mix50

sleep 3
echo "--- running python procs ---"
ps -eo pid,etime,cmd | grep _suboff_e2e_monitor | grep -v grep