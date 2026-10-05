#!/bin/bash
# Batch 3: synthetic Blasius inflow (virtual LE upstream of inlet), plate from inlet.
cd "$(dirname "$0")/.."
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=4
CM="--nx 440 --le 0 --plate_len 400 --U 0.05 --nu 0.01 --steps 30000 --avg 300 --collision bgk --wall bottom --outlet zouhe --top specular"

run() { local name=$1 dev=$2; shift 2
  setsid nohup python3 _mech.py $CM --device sdaa:$dev --out exp_mech/$name.json "$@" \
      > exp_mech/$name.log 2>&1 &
  echo "launched $name on sdaa:$dev pid $!"; }

run syn_x0_20  12 --x0 20  --ny 400 --probes 100,200,300
run syn_x0_50  13 --x0 50  --ny 400 --probes 100,200,300
run syn_x0_100 14 --x0 100 --ny 400 --probes 100,200,300
run syn_x0_50_y1200 15 --x0 50 --ny 1200 --probes 100,200,300
echo launched
wait
echo ALLDONE3