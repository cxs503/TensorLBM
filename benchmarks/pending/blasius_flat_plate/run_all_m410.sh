#!/bin/bash
cd /root/tp-agent/work/m410

echo "=== Run 1: plate400_y1600 ===" 
python benchmarks/pending/blasius_flat_plate/run_m410_study.py --cases plate400_y1600 --collision trt --steps 5000 --device sdaa:0 --out /tmp/m410_y1600.json 2>&1 | tee /tmp/m410_run1.log

echo "=== Run 2: plate400_y3200 ==="
python benchmarks/pending/blasius_flat_plate/run_m410_study.py --cases plate400_y3200 --collision trt --steps 5000 --device sdaa:0 --out /tmp/m410_y3200.json 2>&1 | tee /tmp/m410_run2.log

echo "=== Run 3: plate400_bot_y800 ==="
python benchmarks/pending/blasius_flat_plate/run_m410_study.py --cases plate400_bot_y800 --collision trt --steps 5000 --device sdaa:0 --out /tmp/m410_bot800.json 2>&1 | tee /tmp/m410_run3.log

echo "=== Run 4: plate400_bot_y1600 ==="
python benchmarks/pending/blasius_flat_plate/run_m410_study.py --cases plate400_bot_y1600 --collision trt --steps 5000 --device sdaa:0 --out /tmp/m410_bot1600.json 2>&1 | tee /tmp/m410_run4.log

echo "=== Run 5: plate400_bot_y3200 ==="
python benchmarks/pending/blasius_flat_plate/run_m410_study.py --cases plate400_bot_y3200 --collision trt --steps 5000 --device sdaa:0 --out /tmp/m410_bot3200.json 2>&1 | tee /tmp/m410_run5.log

echo "=== Run 6: plate400_y1600 extrapolation ==="
python benchmarks/pending/blasius_flat_plate/run_m410_study.py --cases plate400_y1600 --collision trt --steps 5000 --device sdaa:0 --top-bc extrapolation --out /tmp/m410_y1600_extrap.json 2>&1 | tee /tmp/m410_run6.log

echo "=== All runs complete ==="
