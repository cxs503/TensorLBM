#!/bin/bash
# Launch 4 fix benchmarks on SDAA cards 4-7
cd /root/TensorLBM_dev
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results_fix2_benchmarks

echo "=== Launching 4 benchmarks on SDAA 4-7 at $(date) ==="

# SDAA:4 — Cylinder Re=3900 3D+RANS 20% blockage (ny=240)
nohup python fix2_benchmarks_worker.py cyl_re3900_20pct 4 results_fix2_benchmarks/cyl_re3900_20pct_sdaa4.json > log_fix2_cyl_re3900_sdaa4.txt 2>&1 &
PID4=$!
echo "SDAA:4 cyl_re3900_20pct PID=$PID4"

# SDAA:5 — NACA 0012 Re=1000 from_naca
nohup python fix2_benchmarks_worker.py naca_re1000_from_naca 5 results_fix2_benchmarks/naca_re1000_from_naca_sdaa5.json > log_fix2_naca_re1000_sdaa5.txt 2>&1 &
PID5=$!
echo "SDAA:5 naca_re1000_from_naca PID=$PID5"

# SDAA:6 — BFL sphere friction 2nd-order Lagrange
nohup python fix2_benchmarks_worker.py sphere_bfl_lagrange 6 results_fix2_benchmarks/sphere_bfl_lagrange_sdaa6.json > log_fix2_sphere_bfl_lagrange_sdaa6.txt 2>&1 &
PID6=$!
echo "SDAA:6 sphere_bfl_lagrange PID=$PID6"

# SDAA:7 — Channel Re_tau=180 RANS 20000 steps
nohup python fix2_benchmarks_worker.py channel_retau180_long 7 results_fix2_benchmarks/channel_retau180_long_sdaa7.json > log_fix2_channel_retau180_long_sdaa7.txt 2>&1 &
PID7=$!
echo "SDAA:7 channel_retau180_long PID=$PID7"

echo "All 4 launched. PIDs: $PID4 $PID5 $PID6 $PID7"
echo "Waiting for all to complete..."

wait $PID4; R4=$?
wait $PID5; R5=$?
wait $PID6; R6=$?
wait $PID7; R7=$?

echo "=== All done at $(date) ==="
echo "SDAA:4 cyl_re3900_20pct exit=$R4"
echo "SDAA:5 naca_re1000_from_naca exit=$R5"
echo "SDAA:6 sphere_bfl_lagrange exit=$R6"
echo "SDAA:7 channel_retau180_long exit=$R7"
