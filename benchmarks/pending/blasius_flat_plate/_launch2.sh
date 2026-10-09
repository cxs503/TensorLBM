#!/bin/bash
set -u
cd /root/TensorLBM_feat2
export PYTHONPATH=/root/TensorLBM_feat2/src
export FT_TIMEOUT=86400
export OMP_NUM_THREADS=8
export TL_STREAM_MODE=cat
export TL_FUSE_NODYN_BB=1
export TL_ISFINITE_INTERVAL=50
D=/root/TensorLBM_feat2/benchmarks/pending/blasius_flat_plate/runs
mkdir -p "$D"
STEPS=${STEPS:-40000}
# S1: bottom wall, CORRECTED BB, Zou-He outlet, 3 probes (resolution trend)
setsid nohup python benchmarks/pending/blasius_flat_plate/_study.py \
  --wall bottom --outlet zouhe --nx 1100 --ny 400 --le 20 --plate_len 1000 \
  --probes 200,470,920 --steps $STEPS --device sdaa:4 --out "$D/S1_bot_zouhe.json" \
  > "$D/S1.log" 2>&1 &
echo "S1 pid=$!"
# S2: bottom wall, corrected BB, convective outlet
setsid nohup python benchmarks/pending/blasius_flat_plate/_study.py \
  --wall bottom --outlet convective --nx 1100 --ny 400 --le 20 --plate_len 1000 \
  --probes 200,470,920 --steps $STEPS --device sdaa:6 --out "$D/S2_bot_conv.json" \
  > "$D/S2.log" 2>&1 &
echo "S2 pid=$!"
# S3: mid thin plate, corrected BB, Zou-He outlet
setsid nohup python benchmarks/pending/blasius_flat_plate/_study.py \
  --wall mid --outlet zouhe --nx 700 --ny 700 --le 20 --plate_len 600 \
  --probes 200,470 --steps $STEPS --device sdaa:5 --out "$D/S3_mid_zouhe.json" \
  > "$D/S3.log" 2>&1 &
echo "S3 pid=$!"
sleep 3
ps -eo pid,etime,cmd | grep _study.py | grep -v grep