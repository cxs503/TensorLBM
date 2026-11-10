#!/usr/bin/env bash
# Production driver for the stand-off resolvability study.
#
#   bash scripts/run_resolvability_ladder.sh full   # Ma=3 four-grid ladder + negative controls
#   bash scripts/run_resolvability_ladder.sh scan   # Ma=2/4/5 three-grid ladders + temporal check
#
# Every grid is one independent process writing one self-describing JSON record
# into runs/, so a job can be re-run grid by grid and the summary is rebuilt from
# the records (see `run_resolvability_study.py collect`).
set -u
MODE="${1:-full}"
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs

run_grid() {  # mach nx ny t_end nprocs label out
  python scripts/run_resolvability_study.py grid \
    --mach "$1" --nx "$2" --ny "$3" --t-end "$4" --backend "${6:-parallel}" \
    --nprocs "$5" --label "$7" --out "$8"
}

case "$MODE" in
  full)
    # finest grid first: it is the wall-clock bottleneck
    run_grid 3 1200 400 2.0 40 parallel G4_finest runs/ma3_G4.json
    run_grid 3  600 200 2.0 24 parallel G3_fine   runs/ma3_G3.json
    run_grid 3  300 100 2.0 12 parallel G2_medium runs/ma3_G2.json
    run_grid 3  150  50 2.0  8 parallel G1_coarse runs/ma3_G1.json
    run_grid 3   21   7 2.0  1 serial   N2_toocoarse runs/ma3_N2.json
    run_grid 3   15   5 2.0  1 serial   N1_toocoarse runs/ma3_N1.json
    ;;
  scan)
    # Mach trend + budget law.  Two grids per Mach (G1, G2): the observed order
    # comes from the Ma=3 four-grid ladder, the per-Mach estimate is the
    # formal-2-grid Richardson value and is flagged as such in the summary.
    for M in 2 4 5; do
      run_grid "$M" 300 100 2.0 12 parallel G2_medium "runs/ma${M}_G2.json"
      run_grid "$M" 150  50 2.0  8 parallel G1_coarse "runs/ma${M}_G1.json"
    done
    # temporal sensitivity of the QoI (same grid, longer physical time)
    run_grid 3 300 100 3.0 12 parallel G2_temporal runs/ma3_G2_t3.json
    ;;
  *)
    echo "unknown mode $MODE" >&2
    exit 2
    ;;
esac
echo "LADDER_DONE mode=$MODE"