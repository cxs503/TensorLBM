#!/bin/bash
# Batch B: conservative gas channel modes x rho_gas, a=8, g=1e-4, 220 steps.
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

for gc in conserv redist; do
  for rg in 1.0 0.99 0.975 0.95 0.1; do
    run "B_${gc}_rg${rg}" "$rg" $COMMON TL_FS_GAS_CHANNEL=$gc
  done
done
# redist + ABB sumeq + l2i combos
run "B_redist_sumeq_rg0.975" 0.975 $COMMON TL_FS_GAS_CHANNEL=redist TL_FS_ABB_MODE=sumeq
run "B_redist_l2i_rg0.975" 0.975 $COMMON TL_FS_GAS_CHANNEL=redist TL_FS_LIQ_TO_IFACE=1
run "B_conserv_l2i_rg0.975" 0.975 $COMMON TL_FS_GAS_CHANNEL=conserv TL_FS_LIQ_TO_IFACE=1
run "B_redist_l2i_rg0.99" 0.99 $COMMON TL_FS_GAS_CHANNEL=redist TL_FS_LIQ_TO_IFACE=1
wait
echo "ALL DONE batchB"
cd "$OUT"
for f in B_*.log; do echo -n "$f : "; grep -E "T=1.000" "$f" | tail -1; done