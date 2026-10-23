#!/usr/bin/env bash
# Blockage probes at fixed D=12 (and D=10) to isolate the lateral-domain lever.
set -u
cd /root/TensorLBM_feat2
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=2
OUT=benchmarks/pending/sphere_re100/scan
mkdir -p "$OUT"
run_one () {
  local D=$1 DEV=$2 LAT=$3 STEPS=$4
  setsid nohup python benchmarks/pending/sphere_re100/run_mem_surface.py single "$D" \
    --device "sdaa:$DEV" --steps "$STEPS" --lateral "$LAT" --up 4 --down 12 --sample 100 \
    --compile-mode eager --out "$OUT/scan_D${D}_L${LAT}.json" \
    > "$OUT/scan_D${D}_L${LAT}.log" 2>&1 < /dev/null &
  echo "launched D=$D lat=$LAT steps=$STEPS on sdaa:$DEV pid=$!"
}
run_one 12 5 32 5000
run_one 10 6 32 5000
run_one 12 7 24 5000
echo "launched probes"