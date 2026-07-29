#!/usr/bin/env python
"""BB Bug Fix Verification: old BB vs corrected BB on Couette flow.

Demonstrates the critical bounce-back bug fix:
- OLD BB: ``bounce_back_cells_3d(f, solid)`` — uses post-collision f
- CORRECTED BB: ``bounce_back_cells_3d(f, solid, f_pre=f_pre)`` — uses pre-collision f

The old BB gives ~16.7% error in u_max for Couette flow.
The corrected BB gives 0.00% error.

Runs both variants on the same Couette setup and compares.

Usage:
    PYTHONPATH=src python teaching/06_bb_bug_fix.py [--device sdaa:14]

Expected output:
    OLD BB:      u_max_err = 16.67%
    CORRECTED BB: u_max_err = 0.00%
"""
from __future__ import annotations

import argparse
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d


def run_couette(use_f_pre: bool, ny=16, nx=32, nz=4, u_top=0.01, tau=1.0,
                n_steps=2000, device="sdaa:14"):
    dev = torch.device(device)
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    rho0 = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=dev)

    rho_top = torch.ones(nz, nx, device=dev)
    feq_top = equilibrium3d(rho_top,
                            torch.full((nz, nx), u_top, device=dev),
                            torch.zeros(nz, nx, device=dev),
                            torch.zeros(nz, nx, device=dev), device=dev)

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        if use_f_pre:
            f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        else:
            f = bounce_back_cells_3d(f, solid)  # OLD: no f_pre
        f[:, :, -1, :] = feq_top[:, :, :]
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
    y_int = torch.arange(1, ny - 1, device=dev, dtype=torch.float32)
    u_exact = u_top * (y_int - 0.5) / (ny - 1 - 0.5)
    u_num = u_profile.cpu().numpy()
    u_ex = u_exact.cpu().numpy()
    max_err = max(abs(a - b) for a, b in zip(u_num, u_ex)) / abs(u_top)
    return max_err * 100


def run(device="sdaa:14"):
    print("=" * 60)
    print("BB Bug Fix Verification: Old BB vs Corrected BB")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    print("Running OLD BB (post-collision f)...")
    err_old = run_couette(use_f_pre=False, device=device)
    print(f"  OLD BB u_max error: {err_old:.2f}%")

    print("Running CORRECTED BB (pre-collision f_pre)...")
    err_new = run_couette(use_f_pre=True, device=device)
    print(f"  CORRECTED BB u_max error: {err_new:.2f}%")

    print()
    print(f"Improvement: {err_old:.2f}% → {err_new:.2f}%")
    if err_new < 0.01:
        print("PASS: Corrected BB gives exact Couette (0.00% error)")
    else:
        print(f"CHECK: Corrected BB error = {err_new:.2f}%")

    if err_old > 5.0:
        print(f"CONFIRMED: Old BB has significant error ({err_old:.2f}%)")

    return {"old_bb_err": err_old, "corrected_bb_err": err_new}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BB Bug Fix Verification")
    parser.add_argument("--device", default="sdaa:14", help="Device")
    args = parser.parse_args()
    run(device=args.device)
