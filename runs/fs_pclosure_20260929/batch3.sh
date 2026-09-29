#!/bin/bash
# tau (viscosity) sensitivity at T=1 (200 steps), a=8, g=1e-4, rho_gas=1.0.
set -u
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=8
cd /root/TensorLBM_feat2
OUT=runs/fs_pclosure_20260929
COMMON="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1"

run () {
  tag=$1; rg=$2; tau=$3; shift 3
  env "$@" python3 runs/fs_pclosure_20260929/probe_pclosure.py \
    --a 8 --g 1e-4 --steps 200 --interval 20 --rho_gas "$rg" --tau "$tau" \
    --outdir "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
}

for tau in 0.55 0.6 0.7 0.9 1.0; do
  run "tau${tau}_rg1.0" 1.0 $tau $COMMON
done
wait
echo DONE
cd "$OUT"
for f in tau*_rg1.0.log; do echo "== $f"; tail -1 "$f"; done