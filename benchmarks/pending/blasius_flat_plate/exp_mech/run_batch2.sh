#!/bin/bash
# Second mechanism batch: top BC, longer entry, taller-domain-tall.
cd "$(dirname "$0")/.."
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=4
CM="--nx 440 --le 20 --plate_len 400 --U 0.05 --nu 0.01 --steps 30000 --avg 300 --collision bgk --wall bottom"

run() { local name=$1 dev=$2; shift 2
  setsid nohup python3 _mech.py $CM --device sdaa:$dev --out exp_mech/$name.json "$@" \
      > exp_mech/$name.log 2>&1 &
  echo "launched $name on sdaa:$dev pid $!"; }

run top_free_400 8  --ny 400 --probes 100,200,300 --outlet zouhe --top freestream
run top_neum_400 9  --ny 400 --probes 100,200,300 --outlet zouhe --top neumann
run tall_6000    10 --ny 6000 --probes 100,200,300 --outlet zouhe --top specular
run le100        11 --nx 560 --le 100 --plate_len 440 --probes 200,300,400 --outlet zouhe --top specular --ny 600
echo "launched all"
wait
echo ALLDONE2