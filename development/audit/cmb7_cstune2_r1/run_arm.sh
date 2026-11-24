#!/bin/bash
# DE-X-CMB7-CSTUNE2-R1: z-axis completion arms (z=10, z=12), Cs=0.075.
#
# Mechanism probe (local 40-step wrap) showed the hydrostatic correction IS
# applied to all interface cells; z_ref=18 vs 20 differ by a uniform 3.07e-5
# rho_g shift (below the depth gauge resolution).  These two arms complete the
# z-axis at the engagement threshold (cavity bottom ~ z=12).
#
# Base: RR rr_tau=0.5003877 + SMAG(C_s=0.075) + HYDRO(z_ref), cells=8, steps=10000.
#
# Usage: bash development/audit/cmb7_cstune2_r1/run_arm.sh <i>   (i = 1..2)
set -uo pipefail
i="${1:-0}"
case "$i" in
  1) arm=cs0075_z10; cs=0.075; z=10;;
  2) arm=cs0075_z12; cs=0.075; z=12;;
  *) echo "bad shard index $i" >&2; exit 2;;
esac

echo "ARM=$arm C_s=$cs z_ref=$z rr_tau=0.5003877 cells=8 steps=10000 node=$(hostname) pwd=$PWD"
mkdir -p "runs/cmb7r1/$arm"

WE_RR_TAU=0.5003877 TL_FS_CS="$cs" TL_FS_HYDRO_Z_REF="$z" PYTHONPATH="$PWD/src:$PWD" \
python3 benchmarks/water_entry/single_drop/free_surface_probe.py \
  --cells 8 --speed 0.06 --steps 10000 --gap 0.18 \
  --capacity --simple-exchange --release-speed 1.53 --pool-depth 3.0 --u-limit 0.15 \
  --out "runs/cmb7r1/$arm/result.json" > "runs/cmb7r1/$arm/stdout.log" 2>&1
rc=$?
echo "ARM=$arm rc=$rc"
tail -3 "runs/cmb7r1/$arm/stdout.log"
exit 0
