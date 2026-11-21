#!/bin/bash
# Gas mass-channel diagnostic: does re-enabling the free-surface mass flux
# unfreeze H and speed the front?  a=8, g=1e-4, 400 steps (T=2).
set -u
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=8
cd /root/TensorLBM_feat2
OUT=runs/fs_pclosure_20260929
COMMON="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1"
GASCH="TL_FS_GAS_CHANNEL=1"

run () {
  tag=$1; rg=$2; shift 2
  env "$@" python3 runs/fs_pclosure_20260929/probe_pclosure.py \
    --a 8 --g 1e-4 --steps 400 --interval 20 --rho_gas "$rg" \
    --outdir "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
}

run gasch_rg1.0 1.0 $COMMON $GASCH
run gasch_rg1.0_ap0 1.0 TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 $GASCH
run gasch_rg0.999 0.999 $COMMON $GASCH
run gasch_rg0.997 0.997 $COMMON $GASCH
run gasch_rg0.995 0.995 $COMMON $GASCH
wait
echo DONE
cd "$OUT"
for f in gasch_*.log; do echo "== $f"; grep "st=" "$f" | tail -1; done