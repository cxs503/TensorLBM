#!/bin/bash
cd /root/TensorLBM_dev
for chord in 50 100 200 400; do
    did=$((chord / 50 - 1))
    PYTHONPATH=src nohup python naca_grid_conv_worker.py $chord $did > /tmp/naca_conv_chord${chord}.log 2>&1 &
    echo "Launched chord=$chord on SDAA:$did"
done
echo "All launched"
wait
echo "ALL DONE"
