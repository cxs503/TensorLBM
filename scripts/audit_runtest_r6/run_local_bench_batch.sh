#!/bin/bash
# AUDIT-RUNTEST-6 item 2/4 — local detached driver for the cheap verified benchmarks.
# Each case: raw log + result JSON into runs/audit_runtest_r6/, plus a per-case
# rc/elapsed line appended to runs/audit_runtest_r6/bench_batch_index.txt.
# Heavy cases (sphere Re=100, 20k-step convergence ladder) are NOT here: they go
# to the 702 cluster (see runs/audit_runtest_r6/CLUSTER_JOBS.md).
set -u
cd "$(dirname "$0")/../.." || exit 1
R=runs/audit_runtest_r6
mkdir -p "$R"
IDX=$R/bench_batch_index.txt
: > "$IDX"
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

run_case () {   # name, timeout_s, cmd...
  local name=$1 tmo=$2; shift 2
  local t0=$(date +%s)
  timeout "$tmo" "$@" > "$R/bench_${name}.log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "$name rc=$rc elapsed_s=$dt cmd=$*" >> "$IDX"
  echo "$name rc=$rc elapsed_s=$dt" >> "$R/bench_batch_progress.txt"
}

run_case taylor_green_2d     1500 python benchmarks/verified/taylor_green_2d/run.py --out "$R/tg2d.json"
run_case shear_wave_decay    1500 python benchmarks/verified/shear_wave_decay/run.py --out "$R/shear_wave.json"
run_case poiseuille_3d_pipe  2400 python benchmarks/verified/poiseuille_3d_pipe/run.py 20 "$R/pipe_R20.json"
run_case mms_convergence_2d  1500 python benchmarks/verified/mms_convergence_2d/run.py --grids 16 32 48 --out "$R/mms_conv.json"
echo "BATCH_DONE" >> "$R/bench_batch_progress.txt"