#!/usr/bin/env bash
# Launch the P399W2-N04 case matrix in detached sessions (local fallback when the
# cluster queue is long).  Each case is a separate process writing its own JSON.
#
# Re = 100 cylinder: D = 16, periodic box 15 D x 10 D, u_mean = 0.06
#   (tau = 0.5288), seeded perturbation, 16000 steps with 6000 transient.
# Stokes regime:    D = 12, u_mean = 0.08 (nu = 9.6, tau = 29.3) so that the
#   viscous diffusion time L^2/nu is crossed ~5x — the low-Re runs of the first
#   sweep were not converged (see docs §6).
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p runs/p399w2_n04/logs artifacts/p399w2_n04
L=runs/p399w2_n04/logs
C="--re 100 --diameter 16 --box 15 10 --steps 16000 --transient 6000 --u-target 0.06 --sample-every 20"
PY="env PYTHONPATH=src P399_NUM_THREADS=4 python scripts/p399w2_n04_curved_force/run_case.py"

run() {  # run <label> <args...>
  local label="$1"; shift
  if [ -f "artifacts/p399w2_n04/case_${label}.json" ]; then return; fi
  setsid $PY --label "$label" "$@" > "$L/${label}.log" 2>&1 < /dev/null &
  disown || true
}

# --- Re = 100 cylinder: marker sweep, scheme sweep, mass-closure study -----
run re100_quad_M64      --markers 64  --scheme quadrature --normal-passes 1 --save-fields $C
run re100_quad_M32      --markers 32  --scheme quadrature --normal-passes 1 $C
run re100_quad_M128     --markers 128 --scheme quadrature --normal-passes 1 $C
run re100_raw_M64       --markers 64  --scheme raw        --normal-passes 1 --save-fields $C
run re100_raw_M32       --markers 32  --scheme raw        --normal-passes 1 $C
run re100_quad_M64_norm0 --markers 64 --scheme quadrature --normal-passes 0 $C
run re100_raw_M128      --markers 128 --scheme raw        --normal-passes 1 $C

# --- Stokes regime: Hasimoto square array, two cell sizes ------------------
run hasimoto_L10D --re 0.1 --diameter 12 --box 10 10 --markers 48 --steps 16000 \
    --transient 8000 --u-target 0.08 --sample-every 40 --stokes-check
run hasimoto_L14D --re 0.1 --diameter 12 --box 14 14 --markers 48 --steps 16000 \
    --transient 8000 --u-target 0.08 --sample-every 40 --stokes-check

sleep 1
echo "launched: $(ls -1 $L | wc -l) logs, running: $(pgrep -fc 'run_cas[e].py' || echo 0)"