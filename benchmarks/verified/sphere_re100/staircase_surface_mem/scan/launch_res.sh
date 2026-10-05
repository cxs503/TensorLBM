#!/usr/bin/env bash
# Resolution trend at fixed lateral=16 (blockage shown irrelevant: L32 == L16 at step 1000).
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
run_one 14 5 16 4000
run_one 16 6 16 4000
echo "launched resolution points"