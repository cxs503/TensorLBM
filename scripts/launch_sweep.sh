#!/bin/bash
# Launch SDAA sweep in background
cd /root/tp-agent/work/m51
nohup python scripts/sdaa_sweep_cavity_natural_convection.py > artifacts/sdaa_sweep.log 2>&1 &
echo $! > artifacts/sdaa_sweep.pid
echo "Launched PID $(cat artifacts/sdaa_sweep.pid)"
