#!/bin/bash
# Launch SDAA verification jobs in background
cd /root/tp-agent/work/m448/benchmarks/verified/thermal_cavity_3d

# Ra=1e3 ladder
nohup python run.py --ra 1e3 --n 32 --steps 100000 --device sdaa --tag sdaa_ra1e3_n32 > /tmp/sdaa_ra1e3_n32.log 2>&1 &
echo "Started Ra1e3 N32: PID $!"

nohup python run.py --ra 1e3 --n 48 --steps 150000 --device sdaa --tag sdaa_ra1e3_n48 > /tmp/sdaa_ra1e3_n48.log 2>&1 &
echo "Started Ra1e3 N48: PID $!"

nohup python run.py --ra 1e3 --n 64 --steps 200000 --device sdaa --tag sdaa_ra1e3_n64 > /tmp/sdaa_ra1e3_n64.log 2>&1 &
echo "Started Ra1e3 N64: PID $!"

# Ra=1e4 ladder
nohup python run.py --ra 1e4 --n 32 --steps 200000 --device sdaa --tag sdaa_ra1e4_n32 > /tmp/sdaa_ra1e4_n32.log 2>&1 &
echo "Started Ra1e4 N32: PID $!"

nohup python run.py --ra 1e4 --n 48 --steps 300000 --device sdaa --tag sdaa_ra1e4_n48 > /tmp/sdaa_ra1e4_n48.log 2>&1 &
echo "Started Ra1e4 N48: PID $!"

nohup python run.py --ra 1e4 --n 64 --steps 400000 --device sdaa --tag sdaa_ra1e4_n64 > /tmp/sdaa_ra1e4_n64.log 2>&1 &
echo "Started Ra1e4 N64: PID $!"

# Ra=1e5 ladder
nohup python run.py --ra 1e5 --n 64 --steps 400000 --device sdaa --tag sdaa_ra1e5_n64 > /tmp/sdaa_ra1e5_n64.log 2>&1 &
echo "Started Ra1e5 N64: PID $!"

nohup python run.py --ra 1e5 --n 96 --steps 600000 --device sdaa --tag sdaa_ra1e5_n96 > /tmp/sdaa_ra1e5_n96.log 2>&1 &
echo "Started Ra1e5 N96: PID $!"

nohup python run.py --ra 1e5 --n 128 --steps 800000 --device sdaa --tag sdaa_ra1e5_n128 > /tmp/sdaa_ra1e5_n128.log 2>&1 &
echo "Started Ra1e5 N128: PID $!"

echo "All jobs launched. Check /tmp/sdaa_*.log for progress."
