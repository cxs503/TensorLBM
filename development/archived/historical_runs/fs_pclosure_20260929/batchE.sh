#!/bin/bash
# Batch E: a=16 verification (g=1e-4, 400 steps ~ T=1) + longer conservation check.
set -u
export PYTHONPATH=/root/TensorLBM_feat2/src
export OMP_NUM_THREADS=8
cd /root/TensorLBM_feat2
OUT=runs/fs_pclosure_20260929
COMMON="TL_FS_WALL_MODE=halfway TL_FS_TOGAS_EPS=1e-6 TL_FS_APRIME=1"

run () {
  a=$1; tag=$2; rg=$3; steps=$4; shift 4
  env "$@" python3 runs/fs_pclosure_20260929/probe_pclosure.py \
    --a "$a" --g 1e-4 --steps "$steps" --interval 40 --rho_gas "$rg" \
    --outdir "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
}

# a=16 T=1 (400 steps)
run 16 E16_base_rg1.0       1.0  400 $COMMON
run 16 E16_sumeq_rg0.975    0.975 400 $COMMON TL_FS_ABB_MODE=sumeq
run 16 E16_conserv_rg1.0    1.0  400 $COMMON TL_FS_GAS_CHANNEL=conserv
run 16 E16_conserv_l2i_rg0.975 0.975 400 $COMMON TL_FS_GAS_CHANNEL=conserv TL_FS_LIQ_TO_IFACE=1
run 16 E16_l2i_rg0.95       0.95 400 $COMMON TL_FS_LIQ_TO_IFACE=1
run 16 E16_naive_rg1.0      1.0  400 $COMMON TL_FS_GAS_CHANNEL=1
# a=8 long conservation check for conservative channel (800 steps ~ T=4)
run 8  E8_conserv_rg1.0_800  1.0  800 $COMMON TL_FS_GAS_CHANNEL=conserv
run 8  E8_base_rg1.0_800     1.0  800 $COMMON
wait
echo "ALL DONE batchE"
cd "$OUT"
for f in E16_*.log E8_*.log; do echo "== $f"; grep -E "T=1.000|T=4.000" "$f" | tail -2; done