#!/bin/bash
# LIQUID->INTERFACE demotion diagnostic, a=8, g=1e-4, 400 steps (T=2).
set -u
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=8
cd /root/TensorLBM_feat2
OUT=runs/fs_pclosure_20260929
COMMON="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1"

run () {
  tag=$1; rg=$2; shift 2
  env "$@" python3 runs/fs_pclosure_20260929/probe_pclosure.py \
    --a 8 --g 1e-4 --steps 400 --interval 20 --rho_gas "$rg" \
    --outdir "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
}

run l2i_rg1.0 1.0 $COMMON TL_FS_LIQ_TO_IFACE=1
run l2i_rg0.995 0.995 $COMMON TL_FS_LIQ_TO_IFACE=1
run l2i_rg0.99 0.99 $COMMON TL_FS_LIQ_TO_IFACE=1
run l2i_gas_rg1.0 1.0 $COMMON TL_FS_LIQ_TO_IFACE=1 TL_FS_GAS_CHANNEL=1
run l2i_gas_rg0.995 0.995 $COMMON TL_FS_LIQ_TO_IFACE=1 TL_FS_GAS_CHANNEL=1
wait
echo DONE
cd "$OUT"
for f in l2i_*.log; do echo -n "$f : "; grep "st=" "$f" | tail -1; done