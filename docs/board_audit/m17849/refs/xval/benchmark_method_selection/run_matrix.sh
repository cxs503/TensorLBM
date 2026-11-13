#!/usr/bin/env bash
# XVAL-LBM-BENCHMARK-METHOD-SELECTION full matrix (round 2).
# Detached batch launch: 5 arms at D=10 + 3 arms at D=14.
# Every arm writes runs/<arm>_D<D>/{result.json|fail.json,trace.npz}.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=xval/benchmark_method_selection/runs
mkdir -p "$OUT" xval/benchmark_method_selection/logs
for a in A1 A2 A3 A4 A5; do
  python xval/benchmark_method_selection/run_arm.py --arm "$a" --D 10 --threads 4 \
     > "xval/benchmark_method_selection/logs/${a}_D10.log" 2>&1 &
done
for a in A1 A2 A4; do
  python xval/benchmark_method_selection/run_arm.py --arm "$a" --D 14 --threads 8 \
     > "xval/benchmark_method_selection/logs/${a}_D14.log" 2>&1 &
done
wait
echo "matrix complete"