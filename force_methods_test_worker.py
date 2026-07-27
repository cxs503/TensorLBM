"""Force computation method comparison: 5 methods on 6 benchmark cases.

Tests all force computation methods (MEM standard/galilean/BFL, stress
integration, pressure integration, virtual work, IB direct forcing) on:
  1. Couette flow (analytical Cf known)
  2. Poiseuille flow (analytical Cf known)
  3. Cylinder Re=200 (Cd ≈ 1.33)
  4. Sphere Re=100 (Cd ≈ 1.09)
  5. SUBOFF Re=1000 (Ct ≈ 0.003)
  6. Bounce-back bug fix: old vs corrected MEM

Usage:
  PYTHONPATH=src python force_methods_test_worker.py --case couette --device sdaa:16
  PYTHONPATH=src python force_methods_test_worker.py --case poiseuille --device sdaa:17
  PYTHONPATH=src python force_methods_test_worker.py --case cylinder --device sdaa:18
  PYTHONPATH=src python force_methods_test_worker.py --case sphere --device sdaa:20
  PYTHONPATH=src python force_methods_test_worker.py --case suboff --device sdaa:21
  PYTHONPATH=src python force_methods_test_worker.py --case bb_bug --device sdaa:23
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch
import torch_sdaa  # noqa: F401

from tensorlbm.force_methods import (
    force_momentum_exchange,
    force_stress_integration,
    force_pressure_integration,
    force_virtual_work,
    force_immersed_boundary,
    compare_force_methods,
)
from tensorlbm.d2q9 import equilibrium as eq2d, macroscopic as macro2d, C as C2D, W as W2D, OPPOSITE as OPP2D
from tensorlbm.d3q19 import equilibrium3d as eq3d, macroscopic3d as macro3d, C as C3D, OPPOSITE as OPP3D


# ---------------------------------------------------------------------------
# 2D LBM solver helpers
# ---------------------------------------------------------------------------

def stream2d(f: torch.Tensor) -> torch.Tensor:
    c = C2D.to(f.device)
    f_out = torch.empty_like(f)
    for q in range(9):
        f_out[q] = torch.roll(f[q], shifts=(int(c[q, 1].item()), int(c[q, 0].item())), dims=(0, 1))
    return f_out


def collide_bgk2d(f: torch.Tensor, tau: float) -> torch.Tensor:
    rho, ux, uy = macro2d(f)
    feq = eq2d(rho, ux, uy)
    return f - (f - feq) / tau


def collide_bgk2d_force(f: torch.Tensor, tau: float, fx_field: torch.Tensor) -> torch.Tensor:
    """BGK collision with Guo body force (2D)."""
    rho, ux, uy = macro2d(f)
    feq = eq2d(rho, ux, uy)
    f_post = f - (f - feq) / tau
    # Guo forcing
    c = C2D.to(f.device).float()
    w = W2D.to(f.device).float()
    cs2 = 1.0 / 3.0
    for q in range(9):
        cu = c[q, 0] * ux + c[q, 1] * uy
        # Guo force term: (1 - 1/(2tau)) * w_q * (c_q - u)/cs2 + c_q c_q u / cs4
        factor = (1.0 - 0.5 / tau) * w[q]
        force_q = factor * (
            (c[q, 0] - ux) / cs2 + (c[q, 0] * cu + c[q, 1] * 0) / (cs2 * cs2)
        ) * fx_field
        f_post[q] = f_post[q] + force_q
    return f_post


def bounce_back_solid2d(f: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
    opp = OPP2D.to(f.device)
    f_out = f.clone()
    for q in range(9):
        q_opp = int(opp[q].item())
        f_out[q][solid] = f[q_opp][solid]
    return f_out


# ---------------------------------------------------------------------------
# 3D LBM solver helpers
# ---------------------------------------------------------------------------

def stream3d(f: torch.Tensor) -> torch.Tensor:
    c = C3D.to(f.device)
    f_out = torch.empty_like(f)
    for q in range(19):
        f_out[q] = torch.roll(
            f[q],
            shifts=(int(c[q, 2].item()), int(c[q, 1].item()), int(c[q, 0].item())),
            dims=(0, 1, 2),
        )
    return f_out


def collide_bgk3d(f: torch.Tensor, tau: float) -> torch.Tensor:
    rho, ux, uy, uz = macro3d(f)
    feq = eq3d(rho, ux, uy, uz)
    return f - (f - feq) / tau


def bounce_back_solid3d(f: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
    opp = OPP3D.to(f.device)
    f_out = f.clone()
    for q in range(19):
        q_opp = int(opp[q].item())
        f_out[q][solid] = f[q_opp][solid]
    return f_out


# ---------------------------------------------------------------------------
# Near-wall mask helpers
# ---------------------------------------------------------------------------

def near_wall_2d(solid: torch.Tensor) -> torch.Tensor:
    fluid = ~solid
    near = torch.zeros_like(fluid)
    near[1:, :] |= solid[:-1, :]
    near[:-1, :] |= solid[1:, :]
    near[:, 1:] |= solid[:, :-1]
    near[:, :-1] |= solid[:, 1:]
    return near & fluid


def near_wall_3d(solid: torch.Tensor) -> torch.Tensor:
    fluid = ~solid
    near = torch.zeros_like(fluid)
    near[1:, :, :] |= solid[:-1, :, :]
    near[:-1, :, :] |= solid[1:, :, :]
    near[:, 1:, :] |= solid[:, :-1, :]
    near[:, :-1, :] |= solid[:, 1:, :]
    near[:, :, 1:] |= solid[:, :, :-1]
    near[:, :, :-1] |= solid[:, :, 1:]
    return near & fluid


# ---------------------------------------------------------------------------
# Test Case 1: Couette Flow (body-force driven, stationary walls)
# ---------------------------------------------------------------------------

def run_couette(device: str = "sdaa:16", ny: int = 32, n_steps: int = 2000):
    """Couette flow driven by moving top wall via bounce-back.

    Uses half-way bounce-back with moving wall correction.
    Analytical: u(y) = u_wall * y / H, tau_w = nu * u_wall / H
    Force on bottom wall: F_x = tau_w * L (one wall only)
    """
    dev = torch.device(device)
    nx = ny * 4
    u_wall = 0.05
    nu_lat = 0.02
    tau = 3.0 * nu_lat + 0.5
    H = ny - 2

    # Separate solid masks for top and bottom walls
    solid_all = torch.zeros(ny, nx, dtype=torch.bool, device=dev)
    solid_all[0, :] = True    # bottom wall (stationary)
    solid_all[-1, :] = True   # top wall (moving)
    solid_bottom = torch.zeros_like(solid_all)
    solid_bottom[0, :] = True
    solid_top = torch.zeros_like(solid_all)
    solid_top[-1, :] = True

    # Initialize with linear profile
    rho = torch.ones(ny, nx, device=dev)
    ux = torch.zeros(ny, nx, device=dev)
    for j in range(1, ny - 1):
        ux[j, :] = u_wall * (j - 0.5) / (ny - 2)
    ux[solid_all] = 0
    uy = torch.zeros_like(ux)
    f = eq2d(rho, ux, uy)

    near = near_wall_2d(solid_all)
    near_bottom = near_wall_2d(solid_bottom)
    c = C2D.to(dev).float()
    w = W2D.to(dev).float()
    opp = OPP2D.to(dev)
    cs2 = 1.0 / 3.0

    force_history = {m: [] for m in ["mem_standard", "mem_galilean", "stress", "pressure", "virtual_work", "ib"]}

    for step in range(1, n_steps + 1):
        # Collision
        f = collide_bgk2d(f, tau)

        # Streaming
        f = stream2d(f)

        # Compute forces BEFORE bounce-back (post-stream, pre-BB)
        # MEM requires post-stream solid cell populations
        if step % 100 == 0 or step == n_steps:
            result = compare_force_methods(f, solid_bottom, near_bottom, nu=nu_lat, tau=tau)
            for method, forces in result.results.items():
                val = forces["fx"]
                if math.isfinite(val):
                    force_history[method].append(val)

        # Bounce-back on solid (stationary bottom + top)
        f = bounce_back_solid2d(f, solid_all)

        # Moving top wall correction
        rho_top = f[:, -1, :].sum(dim=0)
        for q in range(9):
            if c[q, 1] < 0:
                f[q, -1, :] = f[q, -1, :] - 2.0 * rho_top * w[q] * c[q, 1] * u_wall / cs2

        # Periodic in x
        f[:, :, 0] = f[:, :, -2]
        f[:, :, -1] = f[:, :, 1]

    # Analytical force on bottom wall
    tau_w_analytical = nu_lat * u_wall / H
    F_analytical = tau_w_analytical * nx

    print(f"\n{'='*70}")
    print(f"COUETTE FLOW (bottom wall force): ny={ny}, u_wall={u_wall}, nu={nu_lat}, tau={tau:.4f}")
    print(f"{'='*70}")
    print(f"Analytical: tau_w = {tau_w_analytical:.6f}, F_x = {F_analytical:.6f}")
    print(f"           Cf = {tau_w_analytical / (0.5 * 1.0 * u_wall**2):.6f}")
    print(f"\n{'Method':<25s} {'F_x (mean)':>12s} {'F_x (last)':>12s} {'Error%':>10s}")
    print("-" * 65)
    for method in force_history:
        vals = force_history[method]
        if vals:
            mean_f = sum(vals) / len(vals)
            last_f = vals[-1]
            err = abs(mean_f - F_analytical) / (abs(F_analytical) + 1e-12) * 100
            print(f"{method:<25s} {mean_f:>12.6f} {last_f:>12.6f} {err:>9.2f}%")
        else:
            print(f"{method:<25s} {'N/A':>12s} {'N/A':>12s} {'N/A':>10s}")

    return {
        "case": "couette",
        "ny": ny, "u_wall": u_wall, "nu": nu_lat, "tau": tau,
        "F_analytical": F_analytical,
        "Cf_analytical": tau_w_analytical / (0.5 * 1.0 * u_wall**2),
        "forces": {m: vals[-1] if vals else 0 for m, vals in force_history.items()},
        "forces_mean": {m: sum(vals)/len(vals) if vals else 0 for m, vals in force_history.items()},
    }


# ---------------------------------------------------------------------------
# Test Case 2: Poiseuille Flow (body-force driven)
# ---------------------------------------------------------------------------

def run_poiseuille(device: str = "sdaa:17", ny: int = 32, n_steps: int = 3000):
    """Poiseuille flow driven by body force.

    Analytical: u(y) = u_max * (1 - (2y/H-1)^2)
    tau_w = 4 * nu * u_max / H
    """
    dev = torch.device(device)
    nx = ny * 4
    u_max = 0.05
    nu_lat = 0.02
    tau = 3.0 * nu_lat + 0.5
    H = ny - 2

    solid = torch.zeros(ny, nx, dtype=torch.bool, device=dev)
    solid[0, :] = True
    solid[-1, :] = True

    # Body force to drive Poiseuille flow
    # For parabolic: dp/dx = 8 * nu * u_max / H^2
    dp_dx = 8.0 * nu_lat * u_max / (H * H)
    g_x = dp_dx  # body force per unit mass

    rho = torch.ones(ny, nx, device=dev)
    ux = torch.zeros(ny, nx, device=dev)
    uy = torch.zeros_like(ux)
    f = eq2d(rho, ux, uy)

    near = near_wall_2d(solid)
    c = C2D.to(dev).float()
    w = W2D.to(dev).float()
    cs2 = 1.0 / 3.0

    force_history = {m: [] for m in ["mem_standard", "mem_galilean", "stress", "pressure", "virtual_work", "ib"]}

    for step in range(1, n_steps + 1):
        # Collision with body force (simplified Guo)
        rho, ux, uy = macro2d(f)
        feq = eq2d(rho, ux, uy)
        f = f - (f - feq) / tau
        # Add body force (simplified: just add to f based on w_q * c_qx)
        for q in range(9):
            f[q] = f[q] + (1.0 - 0.5 / tau) * w[q] * 3.0 * c[q, 0] * g_x * rho / rho.clamp(min=1e-12)

        # Streaming
        f = stream2d(f)

        # Compute forces BEFORE bounce-back
        if step % 200 == 0 or step == n_steps:
            result = compare_force_methods(f, solid, near, nu=nu_lat, tau=tau)
            for method, forces in result.results.items():
                val = forces["fx"]
                if math.isfinite(val):
                    force_history[method].append(val)

        # Bounce-back
        f = bounce_back_solid2d(f, solid)

        # Periodic in x
        f[:, :, 0] = f[:, :, -2]
        f[:, :, -1] = f[:, :, 1]

    tau_w_analytical = 4.0 * nu_lat * u_max / H
    # Force on BOTH walls (top + bottom): F_total = 2 * tau_w * L
    F_analytical = 2.0 * tau_w_analytical * nx

    print(f"\n{'='*70}")
    print(f"POISEUILLE FLOW (both walls): ny={ny}, u_max={u_max}, nu={nu_lat}, tau={tau:.4f}")
    print(f"{'='*70}")
    print(f"Analytical: tau_w = {tau_w_analytical:.6f}, F_x (both walls) = {F_analytical:.6f}")
    u_b = 2.0 / 3.0 * u_max
    print(f"           Cf = {tau_w_analytical / (0.5 * 1.0 * u_b**2):.6f}")
    print(f"\n{'Method':<25s} {'F_x (mean)':>12s} {'F_x (last)':>12s} {'Error%':>10s}")
    print("-" * 65)
    for method in force_history:
        vals = force_history[method]
        if vals:
            mean_f = sum(vals) / len(vals)
            last_f = vals[-1]
            err = abs(mean_f - F_analytical) / (abs(F_analytical) + 1e-12) * 100
            print(f"{method:<25s} {mean_f:>12.6f} {last_f:>12.6f} {err:>9.2f}%")
        else:
            print(f"{method:<25s} {'N/A':>12s} {'N/A':>12s} {'N/A':>10s}")

    return {
        "case": "poiseuille",
        "ny": ny, "u_max": u_max, "nu": nu_lat, "tau": tau,
        "F_analytical": F_analytical,
        "Cf_analytical": tau_w_analytical / (0.5 * 1.0 * u_b**2),
        "forces": {m: vals[-1] if vals else 0 for m, vals in force_history.items()},
        "forces_mean": {m: sum(vals)/len(vals) if vals else 0 for m, vals in force_history.items()},
    }


# ---------------------------------------------------------------------------
# Test Case 3: Cylinder Re=200
# ---------------------------------------------------------------------------

def run_cylinder(device: str = "sdaa:18", nx: int = 200, ny: int = 100, n_steps: int = 5000):
    """Flow past a cylinder at Re=200.

    Reference: Cd ≈ 1.33, Cl_rms ≈ 0.18, St ≈ 0.196
    """
    dev = torch.device(device)
    D = 20
    cx, cy = nx // 4, ny // 2
    u_in = 0.1
    nu_lat = u_in * D / 200.0
    tau = 3.0 * nu_lat + 0.5

    yy, xx = torch.meshgrid(
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (D / 2) ** 2

    rho = torch.ones(ny, nx, device=dev)
    ux = torch.full((ny, nx), u_in, device=dev)
    ux[solid] = 0
    uy = torch.zeros_like(ux)
    f = eq2d(rho, ux, uy)

    near = near_wall_2d(solid)
    c = C2D.to(dev).float()
    w = W2D.to(dev).float()
    opp = OPP2D.to(dev)
    cs2 = 1.0 / 3.0

    force_history = {m: [] for m in ["mem_standard", "mem_galilean", "stress", "pressure", "virtual_work", "ib"]}
    cl_history = {m: [] for m in force_history}

    for step in range(1, n_steps + 1):
        f = collide_bgk2d(f, tau)
        f = stream2d(f)

        # Compute forces BEFORE bounce-back
        if step > 1000 and step % 50 == 0:
            result = compare_force_methods(f, solid, near, nu=nu_lat, tau=tau)
            for method, forces in result.results.items():
                if math.isfinite(forces["fx"]):
                    force_history[method].append(forces["fx"])
                if math.isfinite(forces["fy"]):
                    cl_history[method].append(forces["fy"])

        f = bounce_back_solid2d(f, solid)

        # Inlet: Zou-He velocity BC
        rho_in = f[:, :, 0].sum(dim=0)
        f[1, :, 0] = f[3, :, 0] + 2.0 / 3.0 * u_in * rho_in
        f[5, :, 0] = f[7, :, 0] + 0.5 * (f[4, :, 0] - f[2, :, 0]) + 1.0 / 6.0 * u_in * rho_in
        f[8, :, 0] = f[6, :, 0] + 0.5 * (f[2, :, 0] - f[4, :, 0]) + 1.0 / 6.0 * u_in * rho_in

        # Outlet: zero-gradient
        for q in [3, 6, 7]:
            f[q, :, -1] = f[q, :, -2]

        # Top/bottom: free-slip
        f[2, 0, :] = f[4, 0, :]
        f[5, 0, :] = f[8, 0, :]
        f[6, 0, :] = f[7, 0, :]
        f[4, -1, :] = f[2, -1, :]
        f[8, -1, :] = f[5, -1, :]
        f[7, -1, :] = f[6, -1, :]

    Cd_ref = 1.33
    Cl_rms_ref = 0.18
    dyn_p = 0.5 * 1.0 * u_in * u_in * D

    print(f"\n{'='*70}")
    print(f"CYLINDER Re=200: D={D}, nx={nx}, ny={ny}, u_in={u_in}, tau={tau:.4f}")
    print(f"{'='*70}")
    print(f"Reference: Cd = {Cd_ref}, Cl_rms = {Cl_rms_ref}")
    print(f"dyn_p*D = {dyn_p:.6f}")
    print(f"\n{'Method':<25s} {'Cd (mean)':>12s} {'Cl_rms':>12s} {'Cd_err%':>10s}")
    print("-" * 65)
    for method in force_history:
        vals = force_history[method]
        cl_vals = cl_history[method]
        if vals:
            cd_mean = sum(vals) / len(vals) / dyn_p
            if len(cl_vals) > 2:
                cl_mean = sum(cl_vals) / len(cl_vals)
                cl_rms = math.sqrt(sum((v - cl_mean) ** 2 for v in cl_vals) / len(cl_vals)) / dyn_p
            else:
                cl_rms = 0.0
            err = abs(cd_mean - Cd_ref) / Cd_ref * 100
            print(f"{method:<25s} {cd_mean:>12.4f} {cl_rms:>12.4f} {err:>9.1f}%")
        else:
            print(f"{method:<25s} {'N/A':>12s} {'N/A':>12s} {'N/A':>10s}")

    return {
        "case": "cylinder_re200", "D": D, "nx": nx, "ny": ny, "Re": 200,
        "u_in": u_in, "tau": tau, "Cd_ref": Cd_ref, "Cl_rms_ref": Cl_rms_ref,
        "forces": {m: sum(vals)/len(vals) if vals else 0 for m, vals in force_history.items()},
    }


# ---------------------------------------------------------------------------
# Test Case 4: Sphere Re=100
# ---------------------------------------------------------------------------

def run_sphere(device: str = "sdaa:20", nx: int = 64, n_steps: int = 3000):
    """Flow past a sphere at Re=100. Reference: Cd ≈ 1.09."""
    dev = torch.device(device)
    D = 12
    nz = ny = nx
    cx, cy, cz = nx // 4, ny // 2, nz // 2
    u_in = 0.05
    nu_lat = u_in * D / 100.0
    tau = 3.0 * nu_lat + 0.5

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= (D / 2) ** 2

    rho = torch.ones(nz, ny, nx, device=dev)
    ux = torch.full((nz, ny, nx), u_in, device=dev)
    ux[solid] = 0
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    f = eq3d(rho, ux, uy, uz)

    near = near_wall_3d(solid)

    force_history = {m: [] for m in ["mem_standard", "mem_galilean", "stress", "pressure", "virtual_work", "ib"]}

    for step in range(1, n_steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)

        # Compute forces BEFORE bounce-back
        if step > 500 and step % 50 == 0:
            result = compare_force_methods(f, solid, near, nu=nu_lat, tau=tau)
            for method, forces in result.results.items():
                if math.isfinite(forces["fx"]):
                    force_history[method].append(forces["fx"])

        f = bounce_back_solid3d(f, solid)

        # Inlet: equilibrium
        rho_in = torch.ones(nz, ny, 1, device=dev)
        ux_in = torch.full((nz, ny, 1), u_in, device=dev)
        uy_in = torch.zeros(nz, ny, 1, device=dev)
        uz_in = torch.zeros(nz, ny, 1, device=dev)
        f[:, :, :, 0] = eq3d(rho_in, ux_in, uy_in, uz_in).squeeze(-1)

        # Outlet: zero gradient
        f[:, :, :, -1] = f[:, :, :, -2]

    Cd_ref = 1.09
    A_ref = math.pi * (D / 2) ** 2
    dyn_p = 0.5 * 1.0 * u_in * u_in * A_ref

    print(f"\n{'='*70}")
    print(f"SPHERE Re=100: D={D}, nx={nx}, u_in={u_in}, tau={tau:.4f}")
    print(f"{'='*70}")
    print(f"Reference: Cd = {Cd_ref}")
    print(f"dyn_p*A = {dyn_p:.6f}")
    print(f"\n{'Method':<25s} {'Cd (mean)':>12s} {'Cd_err%':>10s}")
    print("-" * 55)
    for method in force_history:
        vals = force_history[method]
        if vals:
            cd_mean = sum(vals) / len(vals) / dyn_p
            err = abs(cd_mean - Cd_ref) / Cd_ref * 100
            print(f"{method:<25s} {cd_mean:>12.4f} {err:>9.1f}%")
        else:
            print(f"{method:<25s} {'N/A':>12s} {'N/A':>10s}")

    return {
        "case": "sphere_re100", "D": D, "nx": nx, "Re": 100,
        "u_in": u_in, "tau": tau, "Cd_ref": Cd_ref,
        "forces": {m: sum(vals)/len(vals) if vals else 0 for m, vals in force_history.items()},
    }


# ---------------------------------------------------------------------------
# Test Case 5: SUBOFF Re=1000 (simplified ellipsoid)
# ---------------------------------------------------------------------------

def run_suboff(device: str = "sdaa:21", nx: int = 96, n_steps: int = 3000):
    """Simplified SUBOFF (ellipsoid) at Re=1000. Reference: Ct ≈ 0.003."""
    dev = torch.device(device)
    ny = nx // 2
    nz = ny
    L = nx * 0.4
    D = ny * 0.15
    cx, cy, cz = nx // 3, ny // 2, nz // 2
    u_in = 0.05
    nu_lat = u_in * L / 1000.0
    tau = 3.0 * nu_lat + 0.5

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = ((xx - cx) / (L / 2)) ** 2 + ((yy - cy) / (D / 2)) ** 2 + ((zz - cz) / (D / 2)) ** 2 <= 1.0

    rho = torch.ones(nz, ny, nx, device=dev)
    ux = torch.full((nz, ny, nx), u_in, device=dev)
    ux[solid] = 0
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    f = eq3d(rho, ux, uy, uz)

    near = near_wall_3d(solid)

    force_history = {m: [] for m in ["mem_standard", "mem_galilean", "stress", "pressure", "virtual_work", "ib"]}

    for step in range(1, n_steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)

        # Compute forces BEFORE bounce-back
        if step > 500 and step % 50 == 0:
            result = compare_force_methods(f, solid, near, nu=nu_lat, tau=tau)
            for method, forces in result.results.items():
                if math.isfinite(forces["fx"]):
                    force_history[method].append(forces["fx"])

        f = bounce_back_solid3d(f, solid)

        # Inlet
        rho_in = torch.ones(nz, ny, 1, device=dev)
        ux_in = torch.full((nz, ny, 1), u_in, device=dev)
        uy_in = torch.zeros(nz, ny, 1, device=dev)
        uz_in = torch.zeros(nz, ny, 1, device=dev)
        f[:, :, :, 0] = eq3d(rho_in, ux_in, uy_in, uz_in).squeeze(-1)
        f[:, :, :, -1] = f[:, :, :, -2]

    S_wet = float(near.sum().item()) * 6.0
    dyn_p_S = 0.5 * 1.0 * u_in * u_in * S_wet
    Ct_ref = 0.003

    print(f"\n{'='*70}")
    print(f"SUBOFF (ellipsoid) Re=1000: L={L:.0f}, D={D:.0f}, nx={nx}, tau={tau:.4f}")
    print(f"{'='*70}")
    print(f"Reference: Ct ≈ {Ct_ref}")
    print(f"S_wet ≈ {S_wet:.0f}, dyn_p*S = {dyn_p_S:.6f}")
    print(f"\n{'Method':<25s} {'Ct (mean)':>12s}")
    print("-" * 40)
    for method in force_history:
        vals = force_history[method]
        if vals:
            ct_mean = sum(vals) / len(vals) / (dyn_p_S + 1e-12)
            print(f"{method:<25s} {ct_mean:>12.6f}")
        else:
            print(f"{method:<25s} {'N/A':>12s}")

    return {
        "case": "suboff_re1000", "L": L, "D": D, "nx": nx, "Re": 1000,
        "u_in": u_in, "tau": tau, "Ct_ref": Ct_ref,
        "forces": {m: sum(vals)/len(vals) if vals else 0 for m, vals in force_history.items()},
    }


# ---------------------------------------------------------------------------
# Test Case 6: Bounce-Back Bug Fix (old vs corrected MEM)
# ---------------------------------------------------------------------------

def run_bb_bug_test(device: str = "sdaa:23", ny: int = 32, n_steps: int = 1000):
    """Compare old (buggy) MEM vs corrected MEM on Couette flow.

    Old MEM: F = 2 * sum(f_solid * c) — counts ALL solid links (including
    internal solid-solid links), which overestimates the force.

    Corrected MEM: F = sum over fluid→solid links of (f_i + f_ī) * c —
    only counts boundary links, giving the correct force.
    """
    dev = torch.device(device)
    nx = ny * 4
    u_wall = 0.05
    nu_lat = 0.02
    tau = 3.0 * nu_lat + 0.5
    H = ny - 2

    solid_all = torch.zeros(ny, nx, dtype=torch.bool, device=dev)
    solid_all[0, :] = True
    solid_all[-1, :] = True
    solid_bottom = torch.zeros_like(solid_all)
    solid_bottom[0, :] = True

    rho = torch.ones(ny, nx, device=dev)
    ux = torch.zeros(ny, nx, device=dev)
    for j in range(1, ny - 1):
        ux[j, :] = u_wall * (j - 0.5) / (ny - 2)
    ux[solid_all] = 0
    uy = torch.zeros_like(ux)
    f = eq2d(rho, ux, uy)

    near = near_wall_2d(solid_all)
    near_bottom = near_wall_2d(solid_bottom)
    c = C2D.to(dev).float()
    w = W2D.to(dev).float()
    opp = OPP2D.to(dev)
    cs2 = 1.0 / 3.0

    old_me_history = []
    corrected_me_history = []

    for step in range(1, n_steps + 1):
        f = collide_bgk2d(f, tau)
        f = stream2d(f)

        # OLD (buggy) MEM: F = 2 * sum(f_solid * c) — counts ALL solid links
        mask_4d = solid_all.unsqueeze(0)
        f_solid_old = f * mask_4d.to(f.dtype)
        fx_old = 2.0 * float((f_solid_old * c[:, 0].view(9, 1, 1)).sum().item())
        if math.isfinite(fx_old):
            old_me_history.append(fx_old)

        # CORRECTED MEM: only fluid→solid links on bottom wall
        corrected = force_momentum_exchange(f, solid_bottom, near_bottom, method="standard")
        if math.isfinite(corrected["fx"]):
            corrected_me_history.append(corrected["fx"])

        # Bounce-back
        f = bounce_back_solid2d(f, solid_all)

        # Moving top wall
        rho_top = f[:, -1, :].sum(dim=0)
        for q in range(9):
            if c[q, 1] < 0:
                f[q, -1, :] = f[q, -1, :] - 2.0 * rho_top * w[q] * c[q, 1] * u_wall / cs2

        # Periodic in x
        f[:, :, 0] = f[:, :, -2]
        f[:, :, -1] = f[:, :, 1]

    tau_w_analytical = nu_lat * u_wall / H
    F_analytical = tau_w_analytical * nx

    n_avg = min(200, len(old_me_history), len(corrected_me_history))
    old_mean = sum(old_me_history[-n_avg:]) / n_avg if n_avg > 0 else 0
    corr_mean = sum(corrected_me_history[-n_avg:]) / n_avg if n_avg > 0 else 0

    print(f"\n{'='*70}")
    print(f"BOUNCE-BACK BUG FIX TEST: Couette, ny={ny}")
    print(f"{'='*70}")
    print(f"Analytical F_x = {F_analytical:.6f}")
    print(f"\n{'Method':<30s} {'F_x (mean)':>12s} {'Error%':>10s} {'Ratio':>10s}")
    print("-" * 65)
    err_old = abs(old_mean - F_analytical) / abs(F_analytical) * 100
    err_corr = abs(corr_mean - F_analytical) / abs(F_analytical) * 100
    ratio = old_mean / (corr_mean + 1e-12)
    print(f"{'OLD (buggy) MEM':<30s} {old_mean:>12.6f} {err_old:>9.2f}% {ratio:>10.3f}")
    print(f"{'CORRECTED MEM':<30s} {corr_mean:>12.6f} {err_corr:>9.2f}% {1.0:>10.3f}")

    return {
        "case": "bb_bug_fix", "ny": ny, "u_wall": u_wall,
        "F_analytical": F_analytical,
        "old_me_mean": old_mean, "corrected_me_mean": corr_mean,
        "old_me_error_pct": err_old, "corrected_me_error_pct": err_corr,
        "ratio_old_to_corrected": ratio,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Force computation method comparison tests")
    parser.add_argument("--case", required=True,
                        choices=["couette", "poiseuille", "cylinder", "sphere", "suboff", "bb_bug", "all"],
                        help="Test case to run")
    parser.add_argument("--device", default="sdaa:16", help="SDAA device (e.g., sdaa:16)")
    parser.add_argument("--ny", type=int, default=32, help="Grid size (2D cases)")
    parser.add_argument("--nx", type=int, default=200, help="Grid size x (3D cases)")
    parser.add_argument("--n-steps", type=int, default=None, help="Override n_steps")
    args = parser.parse_args()

    results = {}

    if args.case in ("couette", "all"):
        ns = args.n_steps or 2000
        results["couette"] = run_couette(device=args.device, ny=args.ny, n_steps=ns)
    if args.case in ("poiseuille", "all"):
        ns = args.n_steps or 3000
        results["poiseuille"] = run_poiseuille(device=args.device, ny=args.ny, n_steps=ns)
    if args.case in ("cylinder", "all"):
        ns = args.n_steps or 5000
        results["cylinder"] = run_cylinder(device=args.device, nx=args.nx, n_steps=ns)
    if args.case in ("sphere", "all"):
        ns = args.n_steps or 3000
        results["sphere"] = run_sphere(device=args.device, nx=64, n_steps=ns)
    if args.case in ("suboff", "all"):
        ns = args.n_steps or 3000
        results["suboff"] = run_suboff(device=args.device, nx=96, n_steps=ns)
    if args.case in ("bb_bug", "all"):
        ns = args.n_steps or 1000
        results["bb_bug"] = run_bb_bug_test(device=args.device, ny=args.ny, n_steps=ns)

    dev_id = args.device.split(":")[1] if ":" in args.device else "0"
    outfile = f"force_methods_results_sdaa{dev_id}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
