#!/bin/bash
# hydrostatic ABB reference coefficient scan at rho_gas=1.0, T=1 (200 steps).
set -u
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=8
cd /root/TensorLBM_feat2
OUT=runs/fs_pclosure_20260929
COMMON="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1"

run () {
  tag=$1; shift 1
  env "$@" python3 runs/fs_pclosure_20260929/probe_pclosure.py \
    --a 8 --g 1e-4 --steps 200 --interval 40 --rho_gas 1.0 \
    --outdir "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
}

for c in -4 -2 -1 0.5 1 2 4; do
  run "hycoef${c}" $COMMON TL_FS_ABB_MODE=hydro TL_FS_HYDRO_COEF=$c
done
run hyfill_c1 $COMMON TL_FS_ABB_MODE=hydrofill TL_FS_HYDRO_COEF=1.0
run hywall_c0  $COMMON TL_FS_WALL_MODE=halfway_hydro TL_FS_WALL_HYDRO_COEF=0.0
run hywall_c1  $COMMON TL_FS_WALL_MODE=halfway_hydro TL_FS_WALL_HYDRO_COEF=1.0
run hywall_c2  $COMMON TL_FS_WALL_MODE=halfway_hydro TL_FS_WALL_HYDRO_COEF=2.0
wait
echo DONE
cd "$OUT"
for f in hycoef*.log hyfill*.log hywall*.log; do echo -n "$f : "; grep "st=" "$f" | tail -1; done