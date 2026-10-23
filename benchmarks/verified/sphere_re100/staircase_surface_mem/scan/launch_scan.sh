#!/usr/bin/env bash
# Launch the sphere surface-MEM D-scan in parallel on separate SDAA devices.
set -u
cd /root/TensorLBM_feat2
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=2
OUT=benchmarks/pending/sphere_re100/scan
mkdir -p "$OUT"
# args: D device lateral steps
run_one () {
  local D=$1 DEV=$2 LAT=$3 STEPS=$4
  setsid nohup python benchmarks/pending/sphere_re100/run_mem_surface.py single "$D" \
    --device "sdaa:$DEV" --steps "$STEPS" --lateral "$LAT" --up 4 --down 12 --sample 100 \
    --compile-mode eager --out "$OUT/scan_D${D}_L${LAT}.json" \
    > "$OUT/scan_D${D}_L${LAT}.log" 2>&1 < /dev/null &
  echo "launched D=$D lat=$LAT steps=$STEPS on sdaa:$DEV pid=$!"
}
# default scan: cheap trend at lateral 16
run_one 8  0 16 3000
run_one 10 1 16 3000
run_one 12 2 16 3000
run_one 16 3 16 3000
sleep 2
echo "--- launched, waiting for startup ---"