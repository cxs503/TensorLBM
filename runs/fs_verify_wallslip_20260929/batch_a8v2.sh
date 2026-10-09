#!/bin/bash
# a=8 sweep v2: ALL env inline per job (no export -f). WALL_SLIP=1 + SHELL_MODE=fill + halfway/TOGAS/APRIME
cd /root/TensorLBM_feat2
OUT=runs/fs_verify_wallslip_20260929
mkdir -p $OUT
CMDENV="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1 TL_FS_WALL_SLIP=1 TL_FS_SHELL_MODE=fill"

> /tmp/cmds_a8.txt
for abb in legacy sumeq eq; do
  for rg in 0.95 0.96 0.97 0.975 0.98 0.99; do
    tag="a8v2_${abb}_rg${rg}"
    echo "env PYTHONPATH=src OMP_NUM_THREADS=2 $CMDENV TL_FS_ABB_MODE=$abb python benchmarks/pending/dam_break_3d_mm/run.py --a 8 --g 1e-4 --steps 400 --interval 40 --rho_gas $rg --outdir $OUT > $OUT/${tag}.log 2>&1" >> /tmp/cmds_a8.txt
  done
done
cat /tmp/cmds_a8.txt | xargs -P 4 -d '\n' -I CMD bash -c 'CMD'
echo "ALL DONE a8v2"