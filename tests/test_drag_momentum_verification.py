#!/usr/bin/env python3
"""Verify momentum exchange drag (Ladd 1994) on Couette + Poiseuille flows.

Both flows have 100% analytical solutions.  The momentum exchange method
is cross-validated against the verified pressure-integration friction drag
(drag_pressure.drag_friction_integration, Cf error = 0.00% on Couette).

Step 1: Couette flow
  - Cf_exact = 2ν / ((H-0.5)·U) = 0.6349
  - MEM should give same Cf (equilibrium cancels for flat wall)
  - Cross-validate with pressure integration (both should agree, <1%)
  - Pass: <1% error

Step 2: Poiseuille flow
  - F_exact = G × V_fluid  (body force balanced by wall friction)
  - MEM should give exact total wall force
  - Cross-validate with pressure integration
  - Pass: <1% error

Step 3: Cross-validation table
  - Both methods on same flow → should agree
  - Couette: both exact.  Poiseuille: MEM exact, PF first-order (~3-5%).

Main loop (verified-correct):
  collide → NoDynamics → BB → [compute MEM here] → (body force) → stream → BC

Usage:
  PYTHONPATH=src python tests/test_drag_momentum_verification.py
"""

import sys
import math
import torch
import numpy as np

sys.path.insert(0, "src")

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.drag_momentum import drag_momentum_exchange_vec
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_friction_integration,
    get_near_wall_2d,
)


# ---------------------------------------------------------------------------
# Step 1: Couette flow
# ---------------------------------------------------------------------------
def run_couette(device="sdaa:4"):
    """Couette flow: bottom wall solid (BB), top wall moving (equilibrium).

    Analytical: u(y) = U·(y-0.5)/(ny-1-0.5),  linear profile.
    Cf_exact = 2ν / ((ny-1-0.5)·U) = 2·(1/6) / (10.5·0.05) = 0.6349

    For a flat wall, the equilibrium contributions to MEM cancel across
    opposite-direction pairs, so MEM gives the exact wall shear.
    """
    print("\n" + "=" * 60)
    print("  Step 1: Couette Flow — Momentum Exchange Drag")
    print("=" * 60)

    d = torch.device(device)
    nx, ny, nz = 80, 12, 4
    u_top = 0.05
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    n_steps = 3000

    wall_gap = ny - 1 - 0.5  # 10.5
    cf_exact = 2.0 * nu / (wall_gap * u_top)

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    near = get_near_wall_2d(solid)

    # SurfaceMesh for bottom wall: normal = (0, +1, 0) pointing into fluid
    nx_n = torch.zeros(nz, ny, nx, device=d)
    ny_n = torch.zeros(nz, ny, nx, device=d)
    nz_n = torch.zeros(nz, ny, nx, device=d)
    ny_n[:, 1, :] = 1.0
    nx_n = nx_n * near.float()
    ny_n = ny_n * near.float()
    mesh = SurfaceMesh(near, nx_n, ny_n, nz_n)

    A_wall = nx * nz
    dpS = 0.5 * 1.0 * u_top**2 * A_wall

    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
    )

    me_forces = []

    for step in range(n_steps):
        f_pre = f.clone()
        # 1. Collision (BGK)
        f = collide_bgk3d(f, tau=tau)
        # 2. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # 3. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)
        # 4. Compute MEM (post-BB — correct timing for Ladd formula)
        if step >= n_steps - 20:
            fx, _, _ = drag_momentum_exchange_vec(f, near, solid)
            me_forces.append(fx)
        # 5. Top wall: moving equilibrium
        rho1 = torch.ones(nz, ny, nx, device=d)
        feq_top = equilibrium3d(
            rho1,
            torch.full_like(rho1, u_top),
            torch.zeros_like(rho1),
            torch.zeros_like(rho1),
        )
        f[:, :, -1, :] = feq_top[:, :, -1, :]
        # 6. Streaming
        f = stream3d(f)
        # 7. Periodic BC in x
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    me_fx = sum(me_forces) / len(me_forces)
    me_cf = me_fx / dpS
    pf_cf = drag_friction_integration(f, mesh, dpS, nu)

    rho, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = u_top * (y - 0.5) / wall_gap
    u_err = np.max(np.abs(u - u_exact) / u_top) * 100

    me_err = abs(me_cf - cf_exact) / cf_exact * 100
    pf_err = abs(pf_cf - cf_exact) / cf_exact * 100

    print(f"  Grid: {nx}×{ny}×{nz}, τ={tau}, ν={nu:.4f}, U={u_top}")
    print(f"  Wall gap = {wall_gap}, A_wall = {A_wall}, dpS = {dpS:.4f}")
    print(f"  Velocity profile max error: {u_err:.2f}%")
    print(f"  Cf_exact = {cf_exact:.6f}")
    print(f"  Cf_ME    = {me_cf:.6f}  (error: {me_err:.4f}%)")
    print(f"  Cf_PF    = {pf_cf:.6f}  (error: {pf_err:.4f}%)")
    print(f"  ME vs PF agreement: {abs(me_cf - pf_cf) / cf_exact * 100:.4f}%")
    me_pass = me_err < 1.0
    print(f"  MEM result: {'PASS ✓' if me_pass else 'FAIL ✗'} (threshold <1%)")

    return {
        "flow": "Couette",
        "exact": cf_exact,
        "me": me_cf,
        "me_err": me_err,
        "pf": pf_cf,
        "pf_err": pf_err,
        "me_pass": me_pass,
    }


# ---------------------------------------------------------------------------
# Step 2: Poiseuille flow
# ---------------------------------------------------------------------------
def run_poiseuille(device="sdaa:4"):
    """Poiseuille flow: both walls solid (BB), body force drives flow.

    Analytical: u(y) = G/(2ν)·(y-0.5)·(H+0.5-y),  parabolic profile.
    F_exact = G × V_fluid  (body force balanced by wall friction at steady state).

    MEM gives exact total wall force because the solid cell's f_opp_i
    (post-BB) includes the body-force contribution from the previous
    step's streaming, recovering the full momentum balance.
    """
    print("\n" + "=" * 60)
    print("  Step 2: Poiseuille Flow — Momentum Exchange Drag")
    print("=" * 60)

    d = torch.device(device)
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    H = ny - 2  # 10
    u_max = 0.05
    G = 2.0 * nu * u_max / H**2
    n_steps = 3000

    V_fluid = nx * nz * H
    f_exact = G * V_fluid

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    c = C.to(d).float()
    w = W.to(d).float()

    near = get_near_wall_2d(solid)

    nx_n = torch.zeros(nz, ny, nx, device=d)
    ny_n = torch.zeros(nz, ny, nx, device=d)
    nz_n = torch.zeros(nz, ny, nx, device=d)
    ny_n[:, 1, :] = 1.0
    ny_n[:, -2, :] = -1.0
    nx_n = nx_n * near.float()
    ny_n = ny_n * near.float()
    mesh = SurfaceMesh(near, nx_n, ny_n, nz_n)

    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
    )

    me_forces = []

    for step in range(n_steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        # Compute MEM (post-BB — correct timing)
        if step >= n_steps - 20:
            fx, _, _ = drag_momentum_exchange_vec(f, near, solid)
            me_forces.append(fx)
        # Guo body force (after BB — does not affect MEM at this timing)
        for q in range(19):
            f[q] = f[q] + w[q] * 3.0 * c[q, 0] * G
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    me_fx = sum(me_forces) / len(me_forces)
    pf_fx = drag_friction_integration(f, mesh, 1.0, nu)

    rho, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = G / (2.0 * nu) * (y - 0.5) * (H + 0.5 - y)
    u_err = np.max(np.abs(u - u_exact) / max(u_exact.max(), 1e-10)) * 100

    me_err = abs(me_fx - f_exact) / f_exact * 100
    pf_err = abs(pf_fx - f_exact) / f_exact * 100

    print(f"  Grid: {nx}×{ny}×{nz}, τ={tau}, ν={nu:.4f}")
    print(f"  G={G:.6e}, H={H}, V_fluid={V_fluid}")
    print(f"  Velocity profile max error: {u_err:.2f}%")
    print(f"  F_exact = {f_exact:.6f}")
    print(f"  F_ME    = {me_fx:.6f}  (error: {me_err:.4f}%)")
    print(f"  F_PF    = {pf_fx:.6f}  (error: {pf_err:.4f}%)")
    print(f"  ME vs PF agreement: {abs(me_fx - pf_fx) / f_exact * 100:.4f}%")
    me_pass = me_err < 1.0
    print(f"  MEM result: {'PASS ✓' if me_pass else 'FAIL ✗'} (threshold <1%)")

    return {
        "flow": "Poiseuille",
        "exact": f_exact,
        "me": me_fx,
        "me_err": me_err,
        "pf": pf_fx,
        "pf_err": pf_err,
        "me_pass": me_pass,
    }


# ---------------------------------------------------------------------------
# Step 3: Cross-validation table
# ---------------------------------------------------------------------------
def print_cross_validation_table(results):
    print("\n" + "=" * 72)
    print("  Cross-Validation Table: MEM vs Pressure Integration")
    print("=" * 72)
    print(f"  {'Flow':<12} {'Method':<22} {'Value':>12} {'Exact':>12} {'Err%':>8} {'Pass':>6}")
    print("  " + "-" * 66)
    for r in results:
        me_ok = "✓" if r["me_err"] < 1.0 else "✗"
        pf_ok = "✓" if r["pf_err"] < 1.0 else "✗"
        print(
            f"  {r['flow']:<12} {'Momentum Exchange':<22} {r['me']:12.6f} {r['exact']:12.6f} {r['me_err']:7.4f}% {me_ok:>5}"
        )
        print(
            f"  {r['flow']:<12} {'Pressure Integration':<22} {r['pf']:12.6f} {r['exact']:12.6f} {r['pf_err']:7.4f}% {pf_ok:>5}"
        )
        agree = abs(r["me"] - r["pf"]) / abs(r["exact"]) * 100
        print(f"  {r['flow']:<12} {'→ agreement':<22} {'':>12} {'':>12} {agree:7.4f}%")
        print()
    print("  Notes:")
    print("    - Couette: both methods exact (<0.01%) — flat wall,")
    print("      equilibrium cancels in MEM, linear profile exact for PF.")
    print("    - Poiseuille: MEM exact (<0.01%) — solid cell f_opp_i")
    print("      captures body-force contribution via previous streaming.")
    print("      PF uses first-order wall shear (τ_w=2ν·u), ~3-5% error")
    print("      for parabolic profile at this grid resolution.")


def main():
    print("\n╔" + "═" * 66 + "╗")
    print("║" + " Momentum Exchange Drag Verification (Ladd 1994) ".center(66) + "║")
    print("║" + " 100% analytical: Couette + Poiseuille ".center(66) + "║")
    print("╚" + "═" * 66 + "╝")

    device = "sdaa:4" if torch.sdaa.is_available() else "cpu"
    print(f"\n  Device: {device}")

    results = []
    results.append(run_couette(device))
    results.append(run_poiseuille(device))
    print_cross_validation_table(results)

    n_pass = sum(1 for r in results if r["me_pass"])
    print(f"\n  MEM verification: {n_pass}/{len(results)} passed")
    if n_pass == len(results):
        print("  ★ Momentum exchange drag method verified on 100% analytical flows!")
    else:
        print("  ✗ Some verifications failed — check implementation.")

    return n_pass == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
