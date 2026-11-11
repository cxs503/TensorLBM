#!/usr/bin/env bash
# Round-2 matrix for GAP-LBM-MCMP-POISEUILLE (m14689).
#
# Two families, one driver:
#
#   records/r2_*      acceptance records (run_sweep.py)   -> 5-ledger criteria
#   diag/sp_*         spurious-current time series         -> criterion (3)
#
# Configurations are the round-1 repair baseline (u_eq=viscous, force_scheme=guo,
# G12=-2.5) crossed with the SC wall treatment:
#
#   dry    -- historical: solid cells zeroed in the interaction sum.  The
#             artificial wall force is O(|G12| rho^2) ~ 0.09, i.e. ~1750x the
#             body force G_x=5e-5, so the wall pulls one component to itself,
#             drags the interface and drives a persistent transverse current.
#   mirror -- neutral wetting: solid cells filled with the mirrored nearest-fluid
#             density (tensorlbm.multiphase._mirror_wall_fill).  No net wall
#             force on a uniform field.
#
# Usage:
#   bash cases/mcmp_stratified_poiseuille/run_r2_matrix.sh <cell|all|list>
#
# Every cell is `python3 ... ; echo RC=$?` with PYTHONPATH=src, run from the
# repository root, and writes a self-describing JSON (config + runtime + series).
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="${PYTHONPATH:-src}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REC=artifacts/mcmp_stratified_poiseuille/records
DIAG=artifacts/mcmp_stratified_poiseuille/diag
mkdir -p "$REC" "$DIAG"
SW=cases/mcmp_stratified_poiseuille/run_sweep.py
SD=cases/mcmp_stratified_poiseuille/spurious_diagnostic.py

# ---- acceptance records (G12=-2.5, viscous+guo), dry vs mirror -------------
cell_record() {  # name ny steps wall_mode
  python3 "$SW" --ny "$2" --steps "$3" --g12 -2.5 --u-eq viscous \
    --force-scheme guo --wall-mode "$4" --out "$REC/$1.json"
}

# ---- spurious-current time series -----------------------------------------
cell_diag() {  # name ny steps gx g12 wall_mode
  python3 "$SD" --ny "$2" --steps "$3" --gx "$4" --g12 "$5" --wall-mode "$6" \
    --sample-every 200 --out "$DIAG/$1.json"
}

run_cell() {
  case "$1" in
    # acceptance records, mirror wall (candidate repair)
    r2_ny25_mirror) cell_record r2_ny25_mirror 25 20000 mirror ;;
    r2_ny33_mirror) cell_record r2_ny33_mirror 33 30000 mirror ;;
    r2_ny49_mirror) cell_record r2_ny49_mirror 49 45000 mirror ;;
    # acceptance records, dry wall (round-1 baseline, same step counts)
    r2_ny25_dry)    cell_record r2_ny25_dry    25 20000 dry ;;
    r2_ny33_dry)    cell_record r2_ny33_dry    33 30000 dry ;;
    r2_ny49_dry)    cell_record r2_ny49_dry    49 45000 dry ;;
    # spurious current: static control (G_x=0) = parasitic-current floor
    sp_static_ny25_mirror) cell_diag sp_gx0_ny25_g25_mirror 25 40000 0    -2.5 mirror ;;
    sp_static_ny33_mirror) cell_diag sp_gx0_ny33_g25_mirror 33 40000 0    -2.5 mirror ;;
    sp_static_ny49_mirror) cell_diag sp_gx0_ny49_g25_mirror 49 40000 0    -2.5 mirror ;;
    sp_static_ny25_dry)    cell_diag sp_gx0_ny25_g25_dry    25 40000 0    -2.5 dry ;;
    sp_static_ny49_dry)    cell_diag sp_gx0_ny49_g25_dry    49 40000 0    -2.5 dry ;;
    # spurious current: driven case, same step count as the record
    sp_driven_ny25_mirror) cell_diag sp_gx5e-5_ny25_g25_mirror 25 20000 5e-5 -2.5 mirror ;;
    sp_driven_ny49_mirror) cell_diag sp_gx5e-5_ny49_g25_mirror 49 45000 5e-5 -2.5 mirror ;;
    # weaker coupling (wider interface → smaller parasitic current), mirror wall
    sp_static_ny25_g20)    cell_diag sp_gx0_ny25_g20_mirror    25 40000 0    -2.0 mirror ;;
    sp_static_ny49_g20)    cell_diag sp_gx0_ny49_g20_mirror    49 40000 0    -2.0 mirror ;;
    sp_driven_ny25_g20)    cell_diag sp_gx5e-5_ny25_g20_mirror 25 20000 5e-5 -2.0 mirror ;;
    # coupling scan at fixed wall: how fast does the parasitic current fall?
    sp_static_ny25_g22)    cell_diag sp_gx0_ny25_g22_mirror    25 40000 0    -2.2 mirror ;;
    sp_static_ny49_g22)    cell_diag sp_gx0_ny49_g22_mirror    49 40000 0    -2.2 mirror ;;
    sp_static_ny49_long)   cell_diag sp_gx0_ny49_g25_mirror_long 49 80000 0  -2.5 mirror ;;
    # diagnostic: |G12|=1.5 -- does the mixture stay stratified at all?
    sp_static_ny25_g15)    cell_diag sp_gx0_ny25_g15_mirror    25 20000 0    -1.5 mirror ;;
    *) echo "unknown cell: $1" >&2; return 2 ;;
  esac
}

ALL="r2_ny25_mirror r2_ny33_mirror r2_ny49_mirror r2_ny25_dry r2_ny33_dry r2_ny49_dry \
sp_static_ny25_mirror sp_static_ny33_mirror sp_static_ny49_mirror sp_static_ny25_dry \
sp_static_ny49_dry sp_driven_ny25_mirror sp_driven_ny49_mirror sp_static_ny25_g20 \
sp_static_ny49_g20 sp_driven_ny25_g20 sp_static_ny25_g15 sp_static_ny25_g22 \
sp_static_ny49_g22 sp_static_ny49_long"

case "${1:-list}" in
  list) echo "$ALL" | tr ' ' '\n' ;;
  all)  for c in $ALL; do echo "== $c"; run_cell "$c" || echo "RC=$?"; done ;;
  *)    run_cell "$1"; echo "RC=$?" ;;
esac