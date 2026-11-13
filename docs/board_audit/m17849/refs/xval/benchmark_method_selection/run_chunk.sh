#!/usr/bin/env bash
# One wall-clock-bounded chunk of the D=10 matrix (the container kills
# long-lived background processes, so each arm resumes from state.npz).
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p xval/benchmark_method_selection/logs
BUDGET="${1:-80}"
for a in A1 A2 A3 A4 A5; do
  python xval/benchmark_method_selection/run_arm.py --arm "$a" --D 10 --threads 4 \
     --time-budget "$BUDGET" > "xval/benchmark_method_selection/logs/${a}_D10.log" 2>&1 &
done
wait
for a in A1 A2 A3 A4 A5; do
  tail -n 1 "xval/benchmark_method_selection/logs/${a}_D10.log" 2>/dev/null | head -c 160; echo " <= $a"
done
