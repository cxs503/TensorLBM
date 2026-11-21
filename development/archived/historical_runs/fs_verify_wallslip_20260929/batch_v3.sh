#!/bin/bash
# v3: FULLY controlled env. Unset ALL leaked TL_FS_* then set explicitly.
cd /root/TensorLBM_feat2
OUT=runs/fs_verify_wallslip_20260929
mkdir -p $OUT
UNSET="-u TL_FS_DIAG_FIELD -u TL_FS_LIQ_TO_IFACE -u TL_FS_TOGAS_EPS -u TL_FS_WALL_MODE -u TL_FS_APRIME -u TL_FS_ABB_MODE -u TL_FS_WALL_SLIP -u TL_FS_SHELL_MODE -u TL_FS_BIRTH_MODE -u TL_FS_GAS_CHANNEL -u TL_FS_BIRTH_RHO"
FIXED="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1 TL_FS_WALL_SLIP=1 TL_FS_SHELL_MODE=fill"

> /tmp/cmds_v3.txt
for l2i in 0 1; do
  for abb in legacy sumeq eq; do
    for rg in 0.95 0.96 0.97 0.975 0.98 0.99; do
      tag="v3_l2i${l2i}_${abb}_rg${rg}"
      extra=""
      [ "$l2i" = "1" ] && extra="TL_FS_LIQ_TO_IFACE=1"
      echo "env $UNSET PYTHONPATH=/root/TensorLBM_feat2/src OMP_NUM_THREADS=2 $FIXED TL_FS_ABB_MODE=$abb $extra python benchmarks/pending/dam_break_3d_mm/run.py --a 8 --g 1e-4 --steps 400 --interval 20 --rho_gas $rg --outdir $OUT > $OUT/${tag}.log 2>&1" >> /tmp/cmds_v3.txt
    done
  done
done
wc -l /tmp/cmds_v3.txt
cat /tmp/cmds_v3.txt | xargs -P 4 -d '\n' -I CMD bash -c 'CMD'
echo "ALL DONE v3"