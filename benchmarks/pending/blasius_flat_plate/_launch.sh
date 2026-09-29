#!/bin/bash
set -u
cd /root/TensorLBM_feat2
export PYTHONPATH=/root/TensorLBM_feat2/src
export FT_TIMEOUT=86400
export OMP_NUM_THREADS=8
export TL_STREAM_MODE=cat
export TL_FUSE_NODYN_BB=1
export TL_ISFINITE_INTERVAL=50
LOGDIR=/root/TensorLBM_feat2/benchmarks/pending/blasius_flat_plate/runs
mkdir -p "$LOGDIR"
launch () {
  local name="$1"; shift
  setsid nohup python benchmarks/pending/blasius_flat_plate/_exp.py "$@" \
    > "$LOGDIR/${name}.log" 2>&1 &
  echo "launched $name pid=$!"
}
STEPS=${STEPS:-20000}
launch A_mid_zouhe   --grid mid400 --steps $STEPS --outlet zouhe      --device sdaa:0 --out "$LOGDIR/A.json"
launch B_mid_conv    --grid mid400 --steps $STEPS --outlet convective --device sdaa:1 --out "$LOGDIR/B.json"
launch C_bot_conv    --grid bot400 --steps $STEPS --outlet convective --device sdaa:2 --out "$LOGDIR/C.json"
launch D_bot_zouhe   --grid bot400 --steps $STEPS --outlet zouhe      --device sdaa:3 --out "$LOGDIR/D.json"
sleep 3
ps -eo pid,etime,cmd | grep _exp.py | grep -v grep