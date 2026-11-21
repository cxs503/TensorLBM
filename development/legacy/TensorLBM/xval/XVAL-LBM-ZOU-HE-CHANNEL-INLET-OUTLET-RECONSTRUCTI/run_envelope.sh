#!/usr/bin/env bash
# Validity-envelope scan for XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI.
#
# REPORT.md validates the Zou-He channel closure at *one* point of parameter
# space (drho = 4e-3, tau = 0.65) and REFINEMENT.md refines H at that same point.
# Nothing in the record says how far the 0.35 % claim travels in the two
# parameters that the closure actually feels:
#
#   drho  -- the imposed density difference.  Zou-He pressure ends prescribe
#            rho, so drho sets both the Mach number (compressibility error) and
#            the strength of the acoustic transient;
#   tau   -- the relaxation time.  Half-way bounce-back puts the wall at the
#            mid-link only for tau -> 1/2; the wall-position error is the
#            classic tau-dependent term of this discretisation.
#
# This script fills that grid: one case.py run per cell (each ~35 s, 4 at a
# time).  Records land in artifacts/xval_zouhe/envelope/ and are aggregated by
# make_envelope_report.py -- no number in ENVELOPE.md is typed by hand.
#
#   bash xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/run_envelope.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=artifacts/xval_zouhe/envelope
mkdir -p "$OUT"

CASES=()
for D in 4.0e-4 1.0e-3 2.0e-3 4.0e-3 8.0e-3 1.6e-2; do
  CASES+=("0.65 $D ma_drho_${D}")
done
for T in 0.55 0.60 0.80 1.00 1.50; do
  CASES+=("$T 4.0e-3 tau_${T}")
done

printf '%s\n' "${CASES[@]}" | OUT="$OUT" xargs -P 4 -n 3 bash -c '
  TAU=$1; DRHO=$2; TAG=$3
  [ -s "$OUT/$TAG.json" ] && exit 0
  PYTHONPATH=src python3 xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/case.py \
      --arm pressure-pressure --height 16 --length 128 \
      --tau "$TAU" --drho "$DRHO" \
      --steps 8000 --burn-in 2000 --avg-window 500 --ramp 0 \
      --out "$OUT/$TAG.json" >/dev/null || { echo "FAILED $TAG"; exit 1; }
  echo "done $TAG"
' _

echo "records in $OUT:"
ls -1 "$OUT"