#!/bin/bash
# Batch D: longer (T=2) trajectories probing whether H ever drops; wall/ABB combos.
set -u
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=4
cd /root/TensorLBM_feat2
OUT=runs/fs_pclosure_20260929
COMMON="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1"

run () {
  tag=$1; rg=$2; shift 2
  env "$@" python3 runs/fs_pclosure_20260929/probe_pclosure.py \
    --a 8 --g 1e-4 --steps 400 --interval 20 --rho_gas "$rg" \
    --outdir "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
}
run D_l2i_rg0.95 0.95 $COMMON TL_FS_LIQ_TO_IFACE=1
run D_l2i_rg0.90 0.90 $COMMON TL_FS_LIQ_TO_IFACE=1
run D_l2i_rg0.85 0.85 $COMMON TL_FS_LIQ_TO_IFACE=1
run D_l2i_hywall_rg0.95 0.95 TL_FS_WALL_MODE=halfway_hydro TL_FS_WALL_HYDRO_COEF=1 TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1 TL_FS_LIQ_TO_IFACE=1
run D_sumeq_rg0.975 0.975 $COMMON TL_FS_ABB_MODE=sumeq
run D_legacy_rg0.95 0.95 $COMMON
wait
echo "ALL DONE batchD"
cd "$OUT"
for f in D_*.log; do echo "== $f"; grep -E "st=" "$f" | tail -3; done