#!/bin/bash
# rho_gas scan (a=8, g=1e-4, 400 steps -> T=2) + velocity diagnostics.
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

for rg in 1.0 0.999 0.997 0.995 0.99 0.98 0.95; do
  run "scan_rg${rg}_legacy" "$rg" $COMMON
done
wait
echo "ALL DONE"
for f in "$OUT"/scan_rg*.log; do echo "== $f"; grep -E "^  st= *(340|360|380|400) " "$f"; done