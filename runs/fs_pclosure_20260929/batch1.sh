#!/bin/bash
# Batch free-surface pressure-closure probes (a=8, g=1e-4, 620 steps).
set -u
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=8
cd /root/TensorLBM_feat2
OUT=runs/fs_pclosure_20260929
mkdir -p "$OUT"

run () {
  tag=$1; rg=$2; shift 2
  env "$@" python3 runs/fs_pclosure_20260929/probe_pclosure.py \
    --a 8 --g 1e-4 --steps 620 --interval 40 --rho_gas "$rg" \
    --outdir "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
}

# common fixed levers
COMMON="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1"

run base_rg1.0_legacy 1.0 $COMMON
run hydro_rg1.0_c1 1.0 $COMMON TL_FS_ABB_MODE=hydro TL_FS_HYDRO_COEF=1.0
run hydrofill_rg1.0_c1 1.0 $COMMON TL_FS_ABB_MODE=hydrofill TL_FS_HYDRO_COEF=1.0
run hydrofill_rg1.0_c0 1.0 $COMMON TL_FS_ABB_MODE=hydrofill TL_FS_HYDRO_COEF=0.0
run hydrofill_rg0.1_c0 0.1 $COMMON TL_FS_ABB_MODE=hydrofill TL_FS_HYDRO_COEF=0.0
run hydrofill_rg0.1_c1 0.1 $COMMON TL_FS_ABB_MODE=hydrofill TL_FS_HYDRO_COEF=1.0
wait
echo "ALL DONE"
for f in "$OUT"/*.log; do echo "== $f"; tail -3 "$f"; done