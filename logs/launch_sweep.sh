#!/bin/bash
cd /root/tp-agent/work/m5490
export PYTHONPATH=src WT_THREADS=4 OMP_NUM_THREADS=4
for c in kh0.7_eps0.02 kh0.7_eps0.08 kh1.5_eps0.02 kh1.5_eps0.08 kh3.0_eps0.02 kh3.0_eps0.08 kh1.5_eps0.08_noreflect kh1.5_eps0.08_dx2; do
  python -u -m tensorlbm.wave_tank.campaign --run $c > logs/wt_$c.log 2>&1 &
  sleep 3
done
wait
echo DONE_ALL > logs/wt_sweep.done
