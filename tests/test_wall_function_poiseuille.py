#!/usr/bin/env python3
"""Wall function Poiseuille verification — 100% analytical solution.

Tests the wall_function_3d module with the "gradient" wall law on
Poiseuille channel flow.  The gradient law uses τ_w = ν·u/y_val (direct
velocity gradient, no log-law assumption), which should give exact results
at low Re with τ=1.0.

Physics:
  u(y) = G/(2ν)·(y-0.5)·(H+0.5-y)   [halfway bounce-back wall at y=0.5]

Approaches tested:
  A. Bounce-back + wall function + Guo body force  (wall fn on top of BB)
  B. Wall function only (equilibrium BC for solid) + Guo body force
  C. Bounce-back only + Guo body force  (reference)

Usage: PYTHONPATH=src python tests/test_wall_function_poiseuille.py
"""

from __future__ import annotations
import sys
import math
import torch
import numpy as np

sys.path.insert(0, "src")

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.ibm import ibm_apply_body_force_3d


def run_poiseuille_bounceback(device, n_steps=3000):
    """Reference: bounce-back only + Guo body force."""
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    H = ny - 2  # 10
    u_max = 0.05
    G = 2 * nu * u_max / H**2

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    c = C.to(device).float()
    w = W.to(device).float()

    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
    )

    for step in range(n_steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        # Guo body force
        for q in range(19):
            f[q] = f[q] + w[q] * 3 * c[q, 0] * G
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    rho, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = G / (2 * nu) * (y - 0.5) * (H + 0.5 - y)
    max_err = np.max(np.abs(u - u_exact) / max(u_exact.max(), 1e-10)) * 100
    return u, u_exact, max_err, G, nu, H


def run_poiseuille_wallfn_with_bb(device, n_steps=3000, y_val=0.5):
    """Approach A: bounce-back + wall function + Guo body force."""
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    H = ny - 2  # 10
    u_max = 0.05
    G = 2 * nu * u_max / H**2

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    c = C.to(device).float()
    w = W.to(device).float()

    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
    )

    for step in range(n_steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        # Wall function (gradient law) — adds body force to near-wall fluid cells
        f, drag_f, drag_p = wall_function_3d(f, solid, nu, y_val=y_val, wall_law="gradient")
        # Guo body force (driving force)
        for q in range(19):
            f[q] = f[q] + w[q] * 3 * c[q, 0] * G
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    rho, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = G / (2 * nu) * (y - 0.5) * (H + 0.5 - y)
    max_err = np.max(np.abs(u - u_exact) / max(u_exact.max(), 1e-10)) * 100
    return u, u_exact, max_err, G, nu, H


def run_poiseuille_wallfn_only(device, n_steps=3000, y_val=0.5):
    """Approach B: wall function only (equilibrium BC for solid) + Guo body force.

    Instead of bounce-back, solid cells are set to equilibrium (rho=1, u=0)
    after streaming.  This enforces no-slip at the solid cell itself.
    The wall function provides the wall shear stress.
    """
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    H = ny - 2  # 10
    u_max = 0.05
    G = 2 * nu * u_max / H**2

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    c = C.to(device).float()
    w = W.to(device).float()

    # Pre-compute equilibrium for solid cells (rho=1, u=0)
    feq_solid = equilibrium3d(
        torch.ones(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
    )

    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
        torch.zeros(nz, ny, nx, device=device),
    )

    for step in range(n_steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        # NoDynamics: restore solid cells to pre-collision state
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # NO bounce-back — wall function provides wall shear
        f, drag_f, drag_p = wall_function_3d(f, solid, nu, y_val=y_val, wall_law="gradient")
        # Guo body force (driving force)
        for q in range(19):
            f[q] = f[q] + w[q] * 3 * c[q, 0] * G
        f = stream3d(f)
        # Set solid cells to equilibrium (rho=1, u=0) — enforces no-slip
        for q in range(19):
            f[q] = torch.where(sm[q], feq_solid[q], f[q])
        # Periodic BC in x
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    rho, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = G / (2 * nu) * (y - 0.5) * (H + 0.5 - y)
    max_err = np.max(np.abs(u - u_exact) / max(u_exact.max(), 1e-10)) * 100
    return u, u_exact, max_err, G, nu, H


def run_poiseuille_wallfn_only_y1(device, n_steps=3000):
    """Approach B2: wall function only, y_val=1.0 (wall at solid cell y=0)."""
    return run_poiseuille_wallfn_only(device, n_steps=n_steps, y_val=1.0)


def main():
    device = "sdaa:0" if torch.sdaa.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"SDAA device count: {torch.sdaa.device_count()}")
    print()

    # ── Reference: bounce-back only ──
    print("=" * 60)
    print("  Reference: Bounce-back only + Guo body force")
    print("=" * 60)
    u_ref, u_exact, err_ref, G, nu, H = run_poiseuille_bounceback(device)
    print(f"  ν={nu:.4f}, G={G:.6e}, H={H}")
    print(f"  u_num  ={np.round(u_ref, 6)}")
    print(f"  u_exact={np.round(u_exact, 6)}")
    print(f"  max_err={err_ref:.4f}%")
    print(f"  Result: {'PASS ✓' if err_ref < 1.0 else 'FAIL ✗'} (threshold <1%)")
    print()

    # ── Approach A: bounce-back + wall function ──
    print("=" * 60)
    print("  Approach A: Bounce-back + Wall function (gradient) + Guo force")
    print("=" * 60)
    u_a, _, err_a, _, _, _ = run_poiseuille_wallfn_with_bb(device, y_val=0.5)
    print(f"  u_num  ={np.round(u_a, 6)}")
    print(f"  u_exact={np.round(u_exact, 6)}")
    print(f"  max_err={err_a:.4f}%")
    print(f"  Result: {'PASS ✓' if err_a < 1.0 else 'FAIL ✗'} (threshold <1%)")
    print()

    # ── Approach B: wall function only (equilibrium BC), y_val=0.5 ──
    print("=" * 60)
    print("  Approach B: Wall function only (equilibrium BC), y_val=0.5")
    print("=" * 60)
    u_b, _, err_b, _, _, _ = run_poiseuille_wallfn_only(device, y_val=0.5)
    print(f"  u_num  ={np.round(u_b, 6)}")
    print(f"  u_exact={np.round(u_exact, 6)}")
    print(f"  max_err={err_b:.4f}%")
    print(f"  Result: {'PASS ✓' if err_b < 1.0 else 'FAIL ✗'} (threshold <1%)")
    print()

    # ── Approach B2: wall function only (equilibrium BC), y_val=1.0 ──
    print("=" * 60)
    print("  Approach B2: Wall function only (equilibrium BC), y_val=1.0")
    print("=" * 60)
    u_b2, _, err_b2, _, _, _ = run_poiseuille_wallfn_only_y1(device)
    print(f"  u_num  ={np.round(u_b2, 6)}")
    print(f"  u_exact={np.round(u_exact, 6)}")
    print(f"  max_err={err_b2:.4f}%")
    print(f"  Result: {'PASS ✓' if err_b2 < 1.0 else 'FAIL ✗'} (threshold <1%)")
    print()

    # ── Summary ──
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    results = [
        ("Reference (bounce-back)", err_ref),
        ("A: BB + wall fn (y=0.5)", err_a),
        ("B: wall fn only (y=0.5)", err_b),
        ("B2: wall fn only (y=1.0)", err_b2),
    ]
    for name, err in results:
        status = "PASS ✓" if err < 1.0 else "FAIL ✗"
        print(f"  {name:30s}: {err:8.4f}%  {status}")


if __name__ == "__main__":
    main()
