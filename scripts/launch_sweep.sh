#!/bin/bash
# Launch SDAA sweep in background
cd "$(dirname "$0")/.." || exit 1
mkdir -p artifacts || exit 1
nohup python development/verification/sdaa-hardware/sweep-drivers/sdaa_sweep_cavity_natural_convection.py > artifacts/sdaa_sweep.log 2>&1 &
echo $! > artifacts/sdaa_sweep.pid
echo "Launched PID $(cat artifacts/sdaa_sweep.pid)"
