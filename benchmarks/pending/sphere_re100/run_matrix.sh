#!/usr/bin/env bash
# Domain / resolution matrix for the sphere Re=100 probe.
# Each config runs on its own SDAA device; stdout -> probe_runs/<tag>.log
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/root/TensorLBM_feat2/src:/root/TensorLBM_feat2/benchmarks:${PYTHONPATH:-}
export FT_TIMEOUT=86400
mkdir -p probe_runs

run() {  # tag dev diam steps pad covmargins
  local tag=$1 dev=$2 diam=$3 steps=$4 pad=$5 cm=$6
  echo "[launch] $tag dev=$dev diam=$diam steps=$steps pad=$pad"
  setsid nohup python3 probe.py \
      --diam "$diam" --steps "$steps" --pad "$pad" \
      --device "sdaa:$dev" --compile-mode eager \
      --monitor-steps 500 --cov-margins "$cm" \
      > "probe_runs/$tag.log" 2>&1 &
}

# --- domain-size sweep at fixed D (downstream 1.75D vs 4D vs 8D) ---
run base_D16 0 16 5000 "1.75,1.75,1,1,1,1"      "0.25,0.5,0.75,1.0"   # current口径 baseline
run down4_D16 1 16 5000 "2,4,2,2,2,2"             "0.25,0.5,0.75,1.0"
run down8_D16 2 16 5000 "2,8,2,2,2,2"             "0.25,0.5,0.75,1.0"
run down16_D16 3 16 5000 "2,16,4,4,4,4"           "0.25,0.5,0.75,1.0"
# --- resolution check ---
run down4_D20 4 20 5000 "2,4,2,2,2,2"             "0.25,0.5,0.75,1.0"
run down8_D20 5 20 5000 "2,8,2,2,2,2"             "0.25,0.5,0.75,1.0"

wait
echo "[done]"