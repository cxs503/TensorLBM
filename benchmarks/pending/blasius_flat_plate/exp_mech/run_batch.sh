#!/bin/bash
# Mechanism experiment batch (run from the case dir; writes into exp_mech/).
cd "$(dirname "$0")/.."   # case dir (contains _mech.py)
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=4
mkdir -p exp_mech
rm -rf exp_mech/exp_mech
COMMON="--nx 440 --le 20 --plate_len 400 --probes 100,200,300 --U 0.05 --nu 0.01 --steps 30000 --avg 300 --collision bgk"

run() { # name gpu extra...
  local name=$1 dev=$2; shift 2
  setsid nohup python3 _mech.py $COMMON --device sdaa:$dev --out exp_mech/$name.json "$@" \
      > exp_mech/$name.log 2>&1 &
  echo "launched $name on sdaa:$dev pid $!"
}

run base_ny400   2 --ny 400  --wall bottom --outlet zouhe      --top specular
run base_ny1200  3 --ny 1200 --wall bottom --outlet zouhe      --top specular
run base_ny3000  4 --ny 3000 --wall bottom --outlet zouhe      --top specular
run noplate_1200 5 --ny 1200 --wall none   --outlet zouhe      --top specular
run conv_ny1200  6 --ny 1200 --wall bottom --outlet convective --top specular
run nscbc_ny1200 7 --ny 1200 --wall bottom --outlet nscbc      --top specular
wait
echo ALLDONE