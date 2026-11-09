#!/usr/bin/env bash
# Shard dispatcher for the wall-model LES benchmark suite (cluster runs).
#   shard 1/3 -> channel WMLES, equilibrium closure
#   shard 2/3 -> channel WMLES, non-equilibrium (ODE) closure
#   shard 3/3 -> flat-plate ITTC-1957 benchmark + sensitivity sweep
set -euo pipefail
i="$1"; n="${2:-3}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
case "$i" in
  1) python benchmarks/wall_model_les/channel_wmles.py --case eq  --steps 4000 --averaging-start 2000 ;;
  2) python benchmarks/wall_model_les/channel_wmles.py --case neq --steps 4000 --averaging-start 2000 ;;
  3) python benchmarks/wall_model_les/flat_plate.py --out artifacts/wall_model_les ;;
  *) echo "unknown shard $i of $n"; exit 2 ;;
esac
