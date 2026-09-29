#!/bin/bash
# Launch all 4 fix benchmarks on SDAA 0-3
cd /root/TensorLBM_dev
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results_fix_benchmarks

echo "=== Launching 4 benchmarks on SDAA 0-3 at $(date) ==="

python fix_benchmarks_worker.py sphere_bfl_fix 0 results_fix_benchmarks/sphere_bfl_fix_sdaa0.json > log_fix_sphere_bfl_sdaa0.txt 2>&1 &
PID0=$!
echo "SDAA:0 sphere_bfl_fix PID=$PID0"

python fix_benchmarks_worker.py cyl_re3900_3d_rans 1 results_fix_benchmarks/cyl_re3900_3d_rans_sdaa1.json > log_fix_cyl_re3900_sdaa1.txt 2>&1 &
PID1=$!
echo "SDAA:1 cyl_re3900_3d_rans PID=$PID1"

python fix_benchmarks_worker.py naca_re1000_grad 2 results_fix_benchmarks/naca_re1000_grad_sdaa2.json > log_fix_naca_re1000_sdaa2.txt 2>&1 &
PID2=$!
echo "SDAA:2 naca_re1000_grad PID=$PID2"

python fix_benchmarks_worker.py cyl_re200_bfl 3 results_fix_benchmarks/cyl_re200_bfl_sdaa3.json > log_fix_cyl_re200_bfl_sdaa3.txt 2>&1 &
PID3=$!
echo "SDAA:3 cyl_re200_bfl PID=$PID3"

echo "All 4 launched. PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Waiting for all to complete..."

wait $PID0; R0=$?
wait $PID1; R1=$?
wait $PID2; R2=$?
wait $PID3; R3=$?

echo "=== All done at $(date) ==="
echo "SDAA:0 sphere_bfl_fix exit=$R0"
echo "SDAA:1 cyl_re3900_3d_rans exit=$R1"
echo "SDAA:2 naca_re1000_grad exit=$R2"
echo "SDAA:3 cyl_re200_bfl exit=$R3"
