#!/usr/bin/env bash
# Campaign node p387w2-n04: case matrix (one background process per shard).
#
# Each shard writes one JSON record under artifacts/p387w2_n04/cases/; the
# assembly step merges them into the acceptance report:
#
#   bash scripts/p387w2_n04_sphere_force/run_matrix.sh 4
#   python scripts/p387w2_n04_sphere_force/run_validation.py assemble
#
# Wave 1 (this file): D = 8/10/12, box 5D lateral & streamwise, u = 0.10
# (Ma = 0.17, compressibility error O(Ma^2) ~ 3 %), float32 for the sweep and
# float64 for the fidelity re-run.
set -u
THREADS="${1:-4}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/artifacts/p387w2_n04/cases"
LOG="$ROOT/runs/p387w2_n04"
mkdir -p "$OUT" "$LOG"
RUN="python3 -u $ROOT/scripts/p387w2_n04_sphere_force/run_validation.py"

launch () {  # launch <shard-name> <args...>
  local name="$1"; shift
  nohup setsid $RUN "$@" --out "$OUT" --threads "$THREADS" \
      > "$LOG/$name.log" 2>&1 < /dev/null &
  echo "launched $name"
}

SPEED=0.10

# --- Layer 1: drag over the Re range -------------------------------------
launch drag_re1     drag --re 1     --diameter 8  --lateral 5 --streamwise 5 --speed $SPEED \
    --max-steps 12000 --min-steps 4000 --steady-tol 2e-3
launch drag_re10    drag --re 10    --diameter 8  --lateral 5 --streamwise 5 --speed $SPEED \
    --max-steps 16000 --min-steps 5000 --steady-tol 2e-3
launch drag_re50    drag --re 50    --diameter 8  --lateral 5 --streamwise 5 --speed $SPEED \
    --max-steps 20000 --min-steps 6000 --steady-tol 3e-3
launch drag_re100   drag --re 100   --diameter 10 --lateral 5 --streamwise 5 --speed $SPEED \
    --max-steps 24000 --min-steps 8000 --steady-tol 3e-3
launch drag_re300   drag --re 300   --diameter 10 --lateral 5 --streamwise 5 --speed $SPEED \
    --collision trt --no-guo --max-steps 26000 --min-steps 8000 --steady-tol 1e-2
launch drag_re1000  drag --re 1000  --diameter 12 --lateral 5 --streamwise 5 --speed $SPEED \
    --collision trt --no-guo --max-steps 30000 --min-steps 10000 --steady-tol 3e-2

# --- Layer 2: grid convergence at Re = 100 (same box in diameters) --------
launch conv_d6      drag --re 100 --diameter 6  --lateral 5 --streamwise 5 --speed $SPEED \
    --max-steps 20000 --min-steps 6000 --steady-tol 3e-3
launch conv_d8      drag --re 100 --diameter 8  --lateral 5 --streamwise 5 --speed $SPEED \
    --max-steps 20000 --min-steps 6000 --steady-tol 3e-3
launch conv_d12     drag --re 100 --diameter 12 --lateral 5 --streamwise 5 --speed $SPEED \
    --max-steps 24000 --min-steps 8000 --steady-tol 3e-3
# blockage / domain sensitivity at Re = 100, D = 8
launch dom_d8_8D    drag --re 100 --diameter 8 --lateral 8 --streamwise 8 --speed $SPEED \
    --max-steps 20000 --min-steps 6000 --steady-tol 3e-3

# --- Layer 3: rotating-sphere torque --------------------------------------
launch torque_reo05 torque --re-omega 0.5 --diameter 8 --box 6 --speed 0.05 \
    --max-steps 20000 --min-steps 6000 --steady-tol 3e-3
launch torque_reo1  torque --re-omega 1   --diameter 8 --box 6 --speed 0.05 \
    --max-steps 20000 --min-steps 6000 --steady-tol 3e-3
launch torque_reo2  torque --re-omega 2   --diameter 8 --box 6 --speed 0.05 \
    --max-steps 20000 --min-steps 6000 --steady-tol 3e-3
launch torque_reo5  torque --re-omega 5   --diameter 8 --box 6 --speed 0.05 \
    --collision trt --no-guo --max-steps 20000 --min-steps 6000 --steady-tol 1e-2

# --- negative controls / analytic anchors (no solver) --------------------
$RUN nc --out "$OUT" > "$LOG/nc.log" 2>&1 || true
echo "controls: $(tail -1 "$LOG/nc.log")"