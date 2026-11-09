#!/usr/bin/env bash
# Orchestrate the frozen AMR negative-drag re-run: all arms in parallel, then
# assemble the machine-written artifact (benchmarks/verified/suboff_sail_resistance/
# amr_negative_drag_repro.json).
#
#   THREADS=10 ./run_amr_negative_drag_repro.sh
#
# Nothing in here redefines a criterion: the arm table, the criteria and the
# tolerances live in run_amr_negative_drag_repro.py and were frozen before the
# first arm ran.  Exit code 0 means "the artifact was written", NOT that the
# physics passed — pass/fail lives inside the JSON.
set -uo pipefail
cd "$(dirname "$0")"
REPO="$(cd ../../.. && pwd)"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"
THREADS="${THREADS:-10}"
LOG_DIR="$REPO/runs/amr_negative_drag_repro/logs"
mkdir -p "$LOG_DIR"

ARMS=$($PY -c "import run_amr_negative_drag_repro as R; print(' '.join(a['arm'] for a in R.arm_definitions()))")
echo "[runner] arms: $ARMS"
echo "[runner] threads per arm: $THREADS; repo: $REPO"

pids=()
for arm in $ARMS; do
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    "$PY" -u run_amr_negative_drag_repro.py --arm "$arm" --threads "$THREADS" --quiet \
    > "$LOG_DIR/$arm.log" 2>&1 &
  pids+=("$!:${arm}")
  echo "[runner] launched $arm (pid ${pids[-1]%%:*})"
done

fail=0
for entry in "${pids[@]}"; do
  pid="${entry%%:*}"
  arm="${entry##*:}"
  if wait "$pid"; then
    echo "[runner] arm done: $arm"
  else
    echo "[runner] ARM FAILED (see $LOG_DIR/$arm.log): $arm"
    fail=1
  fi
done

"$PY" -u run_amr_negative_drag_repro.py --assemble --git-commit "$(git -C "$REPO" rev-parse --short HEAD)" \
  > "$LOG_DIR/assemble.log" 2>&1
echo "[runner] assemble exit=$? (log: $LOG_DIR/assemble.log)"
cat "$LOG_DIR/assemble.log"
echo "[runner] artifact: benchmarks/verified/suboff_sail_resistance/amr_negative_drag_repro.json"
exit "$fail"