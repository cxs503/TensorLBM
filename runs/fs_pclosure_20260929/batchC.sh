#!/bin/bash
# Batch C: LIQ_TO_IFACE demotion vs rho_gas (legacy gas channel), a=8, 220 steps.
set -u
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=4
cd /root/TensorLBM_feat2
OUT=runs/fs_pclosure_20260929
COMMON="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1"

run () {
  tag=$1; rg=$2; shift 2
  env "$@" python3 runs/fs_pclosure_20260929/probe_pclosure.py \
    --a 8 --g 1e-4 --steps 220 --interval 20 --rho_gas "$rg" \
    --outdir "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
}
runfill () {
  tag=$1; rg=$2; shift 2
  env "$@" python3 runs/fs_pclosure_20260929/probe_fill.py \
    --a 8 --g 1e-4 --steps 220 --interval 20 --rho_gas "$rg" \
    --tag "$tag" > "$OUT/$tag.fill.log" 2>&1 &
}

for rg in 1.0 0.99 0.975 0.95; do
  run  "C_l2i_rg${rg}"  "$rg" $COMMON TL_FS_LIQ_TO_IFACE=1
  run  "C_l2i_sumeq_rg${rg}" "$rg" $COMMON TL_FS_LIQ_TO_IFACE=1 TL_FS_ABB_MODE=sumeq
done
runfill "C_l2i_rg0.95" 0.95 $COMMON TL_FS_LIQ_TO_IFACE=1
runfill "C_l2i_rg0.975" 0.975 $COMMON TL_FS_LIQ_TO_IFACE=1
wait
echo "ALL DONE batchC"
cd "$OUT"
for f in C_*.log; do echo -n "$f : "; grep -E "T=1.000" "$f" | tail -1; done