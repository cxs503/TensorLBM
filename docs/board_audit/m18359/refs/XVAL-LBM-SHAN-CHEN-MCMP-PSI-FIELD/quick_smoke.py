#!/usr/bin/env python3
"""Quick smoke test for XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD.

Runs a single coupling level with two radii to verify the pipeline works.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.mcmp_verification import (
    estimate_coexistence,
    simulate_static_droplet,
    measure_laplace,
)

torch.set_num_threads(1)

# Quick test: single G level, two radii
G_12 = -1.2
N = 64
RADII = [12.0, 20.0]
TAU = 1.0
STEPS = 3000

print(f"=== Quick XVAL smoke test: G_12={G_12}, n={N} ===")

# Coexistence
print("Running coexistence...")
t0 = time.time()
coex = estimate_coexistence(G_12=G_12, n=N, geometry="planar", steps=STEPS)
print(f"  rho1_liq={coex['rho1_liq']:.4f}, rho1_gas={coex['rho1_gas']:.4f}")
print(f"  ratio={coex['rho1_liq']/coex['rho1_gas']:.2f}, elapsed={time.time()-t0:.1f}s")

# Palabos reference for G=1.2
pal_rho_liq = 1.820
pal_rho_gas = 0.200
pal_sigma = 0.045

rho_liq_err = abs(coex["rho1_liq"] - pal_rho_liq) / pal_rho_liq
rho_gas_err = abs(coex["rho1_gas"] - pal_rho_gas) / pal_rho_gas
print(f"  vs Palabos: rho_liq err={rho_liq_err:.2%}, rho_gas err={rho_gas_err:.2%}")

# Droplets
for R in RADII:
    print(f"  Running droplet R={R}...")
    t0 = time.time()
    run = simulate_static_droplet(
        G_12=G_12, n=N, radius_init=R,
        rho1_in=coex["rho1_liq"], rho2_in=coex["rho2_liq"],
        rho1_out=coex["rho1_gas"], rho2_out=coex["rho2_gas"],
        steps=STEPS, tau1=TAU, tau2=TAU,
        stop_tol=1e-4, check_every=250,
    )
    case = measure_laplace(run, dims=2)
    print(f"    R={case.get('radius', 'n/a'):.2f}, dp={case.get('dp', 'n/a'):.5f}, sigma={case.get('sigma', 'n/a'):.4f}, umax={case.get('u_max', 'n/a'):.5f}")
    print(f"    elapsed={time.time()-t0:.1f}s")

print("\nSmoke test complete.")
