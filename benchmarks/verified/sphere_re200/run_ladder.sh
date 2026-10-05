#!/usr/bin/env bash
# sphere_re200 ladder driver — BFL interpolated-wall per-link momentum ledger,
# big domain (lat3.0/up3.0/down4.0, blockage 1.60%), two grids D=40/D=30.
# Re = u_lb*D/nu_lb = 200  ->  tau = 0.5 + 3*u_lb*D/200.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=/root/TensorLBM_feat2/src
export W8A_DEV=${W8A_DEV:-sdaa:0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# T1: D=40 (top grid)
python run.py re200_D40_12k.json --kernel sparse --D 40 --steps 12000 \
    --lat 3.0 --up 3.0 --down 4.0 --tau 0.53 --ulb 0.05 --sample 50 --cv_tail 2500

# T2: D=30 (bottom grid)
python run.py re200_D30_12k.json --kernel sparse --D 30 --steps 12000 \
    --lat 3.0 --up 3.0 --down 4.0 --tau 0.5225 --ulb 0.05 --sample 50 --cv_tail 2500

echo "ALL_DONE"