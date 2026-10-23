#!/bin/bash
# Batch F: confirm conserv mass-conservation vs rho_gas; legacy wall stickiness.
set -u
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=4
cd /root/TensorLBM_feat2
OUT=runs/fs_pclosure_20260929
run () {
  tag=$1; rg=$2; shift 2
  env "$@" python3 runs/fs_pclosure_20260929/probe_pclosure.py \
    --a 8 --g 1e-4 --steps 400 --interval 40 --rho_gas "$rg" \
    --outdir "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
}
run F_conserv_rg0.99 0.99 TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1 TL_FS_GAS_CHANNEL=conserv
run F_conserv_rg0.90 0.90 TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1 TL_FS_GAS_CHANNEL=conserv
run F_conserv_rg0.975 0.975 TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1 TL_FS_GAS_CHANNEL=conserv
run F_legacywall_l2i_rg0.95 0.95 TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1 TL_FS_LIQ_TO_IFACE=1
run F_l2i_rel0.001_rg0.95 0.95 TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1 TL_FS_LIQ_TO_IFACE=1 TL_FS_LIQ_TO_IFACE_REL=0.001
wait
echo "ALL DONE batchF"
cd "$OUT"
for f in F_*.log; do echo "== $f"; grep "st=" "$f" | tail -2; done