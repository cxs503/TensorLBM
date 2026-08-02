#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "usage: $0 RE4M_CERTIFICATE_PID" >&2
  exit 2
fi
previous_pid=$1
[[ $previous_pid =~ ^[1-9][0-9]*$ ]] || exit 2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

while [[ -r /proc/$previous_pid/cmdline ]]; do
  command_line=$(tr '\0' ' ' < "/proc/$previous_pid/cmdline")
  if [[ $command_line != *collision-viscosity-natural-kbc-suboff-re4m-mixed-exactweights-r2* ]]; then
    echo "PID $previous_pid no longer belongs to the Re4M certificate" >&2
    exit 3
  fi
  sleep 10
done

python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

"$python" scripts/assess_collision_viscosity_schedule.py \
  --collision-model natural_kbc \
  --taus 0.500002025,0.50000405,0.5000081,0.5000162 \
  --wavelength-cells 16 --transverse-cells 3 --amplitude 0.02 \
  --steps 48000 --fit-start-step 4000 \
  --maximum-relative-error-pct 5 --minimum-fitted-log-decay 0.004 \
  --device cpu --dtype float32 --natural-kbc-compute-dtype float64 \
  --output results/collision-viscosity-natural-kbc-suboff-re8m-mixed-exactweights-r1.json

# The target follows the experiment's 5.92 kn, 4.356 m, nu=1.004e-6
# Reynolds number (13,213,381.413), rather than a rounded display value.
exec "$python" scripts/assess_collision_viscosity_schedule.py \
  --collision-model natural_kbc \
  --taus 0.5000012260298475,0.5000024520596952,0.5000049041193902,0.5000098082387806 \
  --wavelength-cells 16 --transverse-cells 3 --amplitude 0.02 \
  --steps 80000 --fit-start-step 6000 \
  --maximum-relative-error-pct 5 --minimum-fitted-log-decay 0.004 \
  --device cpu --dtype float32 --natural-kbc-compute-dtype float64 \
  --output results/collision-viscosity-natural-kbc-suboff-re13p213m-mixed-exactweights-r1.json
