#!/bin/bash
# Batch A: ABB mode (legacy/eq/eqref/sumeq) x rho_gas, a=8, g=1e-4, 220 steps (T~1.1).
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

for mode in legacy eq eqref sumeq eqrefflip; do
  for rg in 1.0 0.99 0.975 0.95 0.1; do
    if [ "$mode" = legacy ]; then
      run "A_${mode}_rg${rg}" "$rg" $COMMON
    else
      run "A_${mode}_rg${rg}" "$rg" $COMMON TL_FS_ABB_MODE=$mode
    fi
  done
done
wait
echo "ALL DONE batchA"
cd "$OUT"
for f in A_*.log; do echo -n "$f : "; grep -E "T=1.000" "$f" | tail -1; done