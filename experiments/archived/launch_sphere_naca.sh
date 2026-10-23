#!/bin/bash
# Launch sphere 3D grid convergence (Re=100) + NACA large domain in parallel.
# SDAA 12: Sphere D=20 (120³)
# SDAA 13: Sphere D=40 (180³)
# SDAA 14: Sphere D=60 (240³)
# SDAA 15: NACA 0012 12L domain (1200x400x4)
cd /root/TensorLBM_dev
export PYTHONPATH=src

mkdir -p results_sphere_gridconv_re100 results_naca_large_domain

# Sphere grid convergence (3 sizes in parallel)
PYTHONPATH=src nohup python sphere_gridconv_re100_worker.py 20 12 results_sphere_gridconv_re100/D20_sdaa12.json > /tmp/sphere_D20_sdaa12.log 2>&1 &
echo "Launched Sphere D=20 (120³) on SDAA:12, PID=$!"

PYTHONPATH=src nohup python sphere_gridconv_re100_worker.py 40 13 results_sphere_gridconv_re100/D40_sdaa13.json > /tmp/sphere_D40_sdaa13.log 2>&1 &
echo "Launched Sphere D=40 (180³) on SDAA:13, PID=$!"

PYTHONPATH=src nohup python sphere_gridconv_re100_worker.py 60 14 results_sphere_gridconv_re100/D60_sdaa14.json > /tmp/sphere_D60_sdaa14.log 2>&1 &
echo "Launched Sphere D=60 (240³) on SDAA:14, PID=$!"

# NACA 0012 large domain (12 chord)
PYTHONPATH=src nohup python naca_large_domain_worker.py 15 results_naca_large_domain/naca_12L_sdaa15.json > /tmp/naca_12L_sdaa15.log 2>&1 &
echo "Launched NACA 0012 12L (1200x400x4) on SDAA:15, PID=$!"

echo "All 4 jobs launched. Waiting..."
wait
echo "ALL DONE"
