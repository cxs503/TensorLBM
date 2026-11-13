#!/usr/bin/env bash
# F1 probe: delivery repro.sh is NOT idempotent (no cleanup of --out).
set -euo pipefail
WT=${1:?usage: repro_probe.sh <TensorLBM worktree at 56c30f13>}
cd "$WT"
PYTHONPATH=src python3 benchmarks/xval_job_checkpoint/run_xval.py --out artifacts/rerun_fresh   # 1st: OK
PYTHONPATH=src python3 benchmarks/xval_job_checkpoint/run_xval.py --out artifacts/rerun_fresh   # 2nd: RuntimeError
