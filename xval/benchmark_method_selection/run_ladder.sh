#!/usr/bin/env bash
# XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE — resolution ladder, LOCAL fallback driver.
# The cluster job (run_ladder_cluster.sh) is the intended path; this local driver exists so the
# ladder can also be produced on a loaded box: one arm at a time, wall-clock-bounded chunks that
# resume from runs/<arm>_D<D>/state.npz. Order = cheapest rung first, so partial progress is
# always useful (A3/A5 are failure probes, then the BGK arm, then the MRT arms).
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=xval/benchmark_method_selection
mkdir -p "$ROOT/logs"

run_until_done() {           # arm, D, threads
  local a="$1" d="$2" th="$3"
  local log="$ROOT/logs/${a}_D${d}.log"
  for _ in $(seq 1 60); do
    if [ -f "$ROOT/runs/${a}_D${d}/result.json" ] || [ -f "$ROOT/runs/${a}_D${d}/fail.json" ]; then
      echo "[ladder-local] ${a}_D${d} done $(date -u +%FT%TZ)" >> "$ROOT/logs/ladder_master.log"; return 0
    fi
    python "$ROOT/run_arm.py" --arm "$a" --D "$d" --threads "$th" --time-budget 300 >> "$log" 2>&1
  done
  echo "[ladder-local] ${a}_D${d} gave up" >> "$ROOT/logs/ladder_master.log"
}

for spec in "A3 14 2" "A5 14 2" "A2 14 6" "A4 14 6" "A1 14 6" "A2 20 6" "A4 20 8" "A1 20 8"; do
  # shellcheck disable=SC2086
  set -- $spec
  run_until_done "$1" "$2" "$3"
done
echo "[ladder-local] all done $(date -u +%FT%TZ)" >> "$ROOT/logs/ladder_master.log"