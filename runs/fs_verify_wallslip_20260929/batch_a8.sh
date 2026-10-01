#!/bin/bash
# a=8 sweep: WALL_SLIP=1 + SHELL_MODE=fill + halfway/TOGAS/APRIME, scan rho_gas x ABB mode
cd /root/TensorLBM_feat2
export PYTHONPATH=src
export OMP_NUM_THREADS=2
export TL_FS_WALL_MODE=halfway
export TL_FS_TOGAS_EPS=1e-6
export TL_FS_APRIME=1
export TL_FS_WALL_SLIP=1
export TL_FS_SHELL_MODE=fill

OUT=runs/fs_verify_wallslip_20260929
mkdir -p $OUT

run_one () {
  local abb=$1 rg=$2
  local tag="a8_${abb}_rg${rg}"
  TL_FS_ABB_MODE=$abb python benchmarks/pending/dam_break_3d_mm/run.py \
    --a 8 --g 1e-4 --steps 400 --interval 40 --rho_gas $rg \
    --outdir $OUT > $OUT/${tag}.log 2>&1
}
export -f run_one
export OUT

# build job list
i=0
for abb in legacy sumeq eq; do
  for rg in 0.95 0.96 0.97 0.975 0.98 0.99; do
    echo "$abb $rg"
  done
done > /tmp/jobs_a8.txt

cat /tmp/jobs_a8.txt | xargs -P 4 -n 2 bash -c 'run_one "$0" "$1"'
echo "ALL DONE"