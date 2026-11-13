#!/usr/bin/env bash
# XVAL-LBM-ROUND2-METHOD-SELECTION — resolution ladder driver.
#
# Runs the predeclared (predeclared_round2.json) arm x resolution matrix in
# parallel *inside one job*, so a single submission covers one rung:
#
#   bash xval/benchmark_method_selection/run_round2.sh 10 A6 A2rep
#   bash xval/benchmark_method_selection/run_round2.sh 14 A1 A2 A4 A6 A3 A5
#   bash xval/benchmark_method_selection/run_round2.sh 20 A1 A2 A4 A6
#
# Each arm writes runs/<arm>_D<D>/{result.json|fail.json,trace.npz} and a log in
# logs/.  A2rep is the determinism re-run of A2 at the same D (predeclared
# determinism_rule) written to its own directory so it cannot overwrite A2.
set -u
cd "$(dirname "$0")/../.." || exit 1
D="${1:?usage: run_round2.sh D arm [arm ...]}"
shift
THREADS="${THREADS:-8}"
mkdir -p xval/benchmark_method_selection/logs
rc=0
pids=()
for a in "$@"; do
  arm="$a"; out=""
  if [ "$a" = "A2rep" ]; then arm="A2"; fi
  if [ "$a" = "A2rep" ]; then out="xval/benchmark_method_selection/runs/A2rep_D${D}"; fi
  if [ -n "$out" ]; then
    python3 xval/benchmark_method_selection/run_arm.py --arm "$arm" --D "$D" \
      --threads "$THREADS" --out "$out" --no-resume \
      > "xval/benchmark_method_selection/logs/${a}_D${D}.log" 2>&1 &
  else
    python3 xval/benchmark_method_selection/run_arm.py --arm "$arm" --D "$D" \
      --threads "$THREADS" \
      > "xval/benchmark_method_selection/logs/${a}_D${D}.log" 2>&1 &
  fi
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "round2 rung D=${D} arms=$* rc=${rc}"
exit "$rc"