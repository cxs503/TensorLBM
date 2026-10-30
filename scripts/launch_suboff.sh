#!/bin/bash
cd /root/tp-agent/work/m212
export PYTHONPATH=src:benchmarks
nohup python3 scripts/suboff_re1000_checkpoint_run.py \
  --resolution 48 --steps 3000 --device sdaa:0 \
  --collision mrt --friction mix50 --compile-mode eager \
  --checkpoint-interval 300 --wall-budget 900 \
  --out runs/suboff_re1000_ckpt_L48 \
  > runs/suboff_re1000_ckpt_L48.log 2>&1 &
echo $! > runs/suboff_re1000_ckpt_L48.pid
echo "LAUNCHED PID=$(cat runs/suboff_re1000_ckpt_L48.pid)"
