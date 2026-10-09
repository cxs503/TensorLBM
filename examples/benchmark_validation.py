#!/usr/bin/env python
"""Standard benchmark validation for TensorLBM common modules on CUDA.

Tests two canonical problems with exact analytical solutions:

1. Poiseuille channel flow (D3Q19 + D3Q27)
   - Body-force driven flow between parallel plates
   - Exact parabolic profile: u(y) = G/(2ν) * y' * (H - y')
   - Validates: equilibrium, macroscopic, collision, streaming, bounce-back

2. Sphere flow in Stokes regime (D3Q19 + D3Q27)
   - Creeping flow past a sphere (Re << 1)
   - Exact drag: Cd = 24/Re (Stokes law)
   - Validates: sphere_mask, far_field_bc, bounce_back, drag integration

Usage:
  PYTHONPATH=src python examples/benchmark_validation.py [--gpu 0] [--quick]
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    far_field_bc_3d,
    sphere_mask,
)
from tensorlbm.boundaries_d3q27 import (
    bounce_back_cells_27,
    far_field_bc_27,
)
from tensorlbm.cumulant import collide_cumulant_d3q27

# ---------------------------------------------------------------------------
# Lattice imports
# ---------------------------------------------------------------------------
from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import (
    C as C27,
)
from tensorlbm.d3q27 import (
    collide_bgk27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
)
from tensorlbm.solver3d import (
    collide_bgk3d,
    collide_mrt3d,
    correct_mass3d,
    stream3d_roll,
)

# D3Q27 roll streaming (defined in runner, but we need it standalone)
_D3Q27_SHIFTS = [
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (1, 0, -1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, -1, 1),
    (0, 1, -1),
    (0, -1, -1),
    (1, 1, 1),
    (-1, 1, 1),
    (1, -1, 1),
    (-1, -1, 1),
    (1, 1, -1),
    (-1, 1, -1),
    (1, -1, -1),
    (-1, -1, -1),
]


def stream27_roll(f: torch.Tensor) -> torch.Tensor:
    """Memory-efficient D3Q27 streaming using torch.roll."""
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _D3Q27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------
def _equilibrium(lattice, rho, ux, uy, uz, device=None):
    if lattice == "D3Q19":
        return equilibrium3d(rho, ux, uy, uz, device=device)
    return equilibrium27(rho, ux, uy, uz, device=device)


def _macroscopic(lattice, f):
    if lattice == "D3Q19":
        return macroscopic3d(f)
    return macroscopic27(f)


def _stream(lattice, f):
    if lattice == "D3Q19":
        return stream3d_roll(f)
    return stream27_roll(f)


def _bounce_back(lattice, f, mask):
    if lattice == "D3Q19":
        return bounce_back_cells_3d(f, mask)
    return bounce_back_cells_27(f, mask)


def _far_field_bc(lattice, f, u_in, obstacle_mask=None):
    if lattice == "D3Q19":
        return far_field_bc_3d(f, u_in, obstacle_mask=obstacle_mask)
    return far_field_bc_27(f, u_in, obstacle_mask=obstacle_mask)


def _collide(lattice, collision, f, tau):
    if lattice == "D3Q19":
        if collision == "BGK":
            return collide_bgk3d(f, tau)
        if collision == "MRT":
            return collide_mrt3d(f, tau)
    else:
        if collision == "BGK":
            return collide_bgk27(f, tau)
        if collision == "MRT":
            return collide_mrt27(f, tau)
        if collision == "CUMULANT":
            return collide_cumulant_d3q27(f, tau)
    raise ValueError(f"Unknown {collision} for {lattice}")


def _guo_force(lattice, f, force_x, tau, rho, ux, uy, uz):
    """Apply body force via Exact Difference Method (EDM).

    Δf = f_eq(ρ, u + Δu) - f_eq(ρ, u),  where Δu = F/ρ.

    Works with any collision operator (BGK, MRT, Cumulant).
    Equivalent to Guo scheme at steady state.
    """
    du = force_x / rho  # velocity increment
    feq_u = _equilibrium(lattice, rho, ux, uy, uz, device=f.device)
    feq_u_plus = _equilibrium(lattice, rho, ux + du, uy, uz, device=f.device)
    return f + (feq_u_plus - feq_u)


def _correct_mass(lattice, f, target_mass):
    if lattice == "D3Q19":
        return correct_mass3d(f, target_mass)
    # D3Q27 mass correction: same principle
    current = f.sum()
    if current.abs() < 1e-30:
        return f
    return f * (target_mass / current)


# ---------------------------------------------------------------------------
# Benchmark 1: Poiseuille channel flow
# ---------------------------------------------------------------------------
def run_poiseuille(
    lattice="D3Q19",
    collision="BGK",
    ny=32,
    nx=64,
    nz=4,
    force=1e-5,
    tau=1.0,
    n_steps=10000,
    device="cuda:0",
):
    """Poiseuille flow between parallel plates with exact analytical solution.

    Channel walls at y=0 and y=ny-1 (bounce-back no-slip).
    Body force G drives flow in x-direction.
    Periodic in x and z (natural with roll streaming).

    Exact: u(y) = G/(2ν) * y' * (H - y'), where y' = y - 0.5, H = ny - 2
    """
    dev = torch.device(device)
    nu = (tau - 0.5) / 3.0
    H = float(ny - 2)

    # Solid mask: walls at y=0 and y=ny-1
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    # Initialize: uniform density, zero velocity
    rho0 = torch.ones(nz, ny, nx, device=dev)
    f = _equilibrium(
        lattice,
        rho0,
        torch.zeros_like(rho0),
        torch.zeros_like(rho0),
        torch.zeros_like(rho0),
        device=dev,
    )
    initial_mass = float(f.sum().item())

    u_max_exact = force * H**2 / (8.0 * nu)
    cf_exact = 8.0 * nu / (u_max_exact * H) if u_max_exact > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"Poiseuille Flow: {lattice} {collision}")
    print(f"Grid: {nx}x{ny}x{nz}, tau={tau}, nu={nu:.6f}, force={force}")
    print(f"Steps={n_steps}, device={device}")
    print(f"u_max_exact={u_max_exact:.8f}, Cf_exact={cf_exact:.6f}")
    print(f"{'=' * 60}")

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # Collision (standard, no force)
        f = _collide(lattice, collision, f, tau)

        # Guo body force (x-direction)
        rho, ux, uy, uz = _macroscopic(lattice, f)
        f = _guo_force(lattice, f, force, tau, rho, ux, uy, uz)

        # Streaming (periodic via roll)
        f = _stream(lattice, f)

        # Bounce-back on walls
        f = _bounce_back(lattice, f, solid)

        # Mass correction every 200 steps
        if step % 200 == 0:
            f = _correct_mass(lattice, f, initial_mass)

        if step % 2000 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux, _, _ = _macroscopic(lattice, f)
            u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
            u_max = float(u_profile.max().item())
            print(
                f"  step {step:5d}/{n_steps}: u_max={u_max:.8f} "
                f"({u_max / u_max_exact * 100:.2f}% exact), {elapsed:.1f}s"
            )

    # Extract final velocity profile
    _, ux, _, _ = _macroscopic(lattice, f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))  # average over z and x

    # Exact solution
    y_interior = torch.arange(1, ny - 1, device=dev, dtype=torch.float32)
    u_exact = force / (2.0 * nu) * (y_interior - 0.5) * (H - (y_interior - 0.5))

    u_num = u_profile.cpu().numpy()
    u_ex = u_exact.cpu().numpy()
    u_max_num = float(u_num.max())

    # Relative L2 error
    l2_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(u_num, u_ex))) / math.sqrt(
        sum(b**2 for b in u_ex)
    )
    max_err = max(abs(a - b) for a, b in zip(u_num, u_ex)) / abs(u_max_exact)

    # Wall shear stress from numerical gradient
    du_dy_wall = u_num[0] / 0.5  # forward diff at wall
    tau_wall = nu * du_dy_wall
    cf_num = 2.0 * tau_wall / u_max_num**2 if u_max_num > 1e-12 else 0.0
    cf_err = abs(cf_num - cf_exact) / abs(cf_exact) * 100 if cf_exact > 0 else 0.0

    print("\n  --- Results ---")
    print(
        f"  u_max:  num={u_max_num:.8f}, exact={u_max_exact:.8f}, "
        f"err={abs(u_max_num - u_max_exact) / u_max_exact * 100:.3f}%"
    )
    print(f"  L2 error: {l2_err * 100:.3f}%")
    print(f"  Max error: {max_err * 100:.3f}%")
    print(f"  Cf:     num={cf_num:.6f}, exact={cf_exact:.6f}, err={cf_err:.2f}%")

    # Print profile comparison
    print(f"\n  {'y':>5s}  {'u_num':>12s}  {'u_exact':>12s}  {'err':>12s}")
    for yv, un, ue in zip(y_interior.cpu().numpy(), u_num, u_ex):
        print(f"  {yv:5.1f}  {un:12.8f}  {ue:12.8f}  {abs(un - ue):12.2e}")

    passed = max_err < 0.01  # <1% max error
    print(f"\n  {'PASS' if passed else 'FAIL'}: max_err={max_err * 100:.3f}% (target < 1%)")
    return {
        "benchmark": "poiseuille",
        "lattice": lattice,
        "collision": collision,
        "u_max_num": u_max_num,
        "u_max_exact": u_max_exact,
        "u_max_err_pct": abs(u_max_num - u_max_exact) / u_max_exact * 100,
        "l2_err_pct": l2_err * 100,
        "max_err_pct": max_err * 100,
        "cf_num": cf_num,
        "cf_exact": cf_exact,
        "cf_err_pct": cf_err,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Sphere flow (Stokes regime)
# ---------------------------------------------------------------------------
def run_sphere_stokes(
    lattice="D3Q19",
    collision="BGK",
    nx=64,
    ny=40,
    nz=40,
    radius=6.0,
    u_in=0.01,
    re=0.5,
    n_steps=5000,
    device="cuda:0",
):
    """Creeping flow past a sphere — Stokes drag Cd = 24/Re.

    Low Reynolds number (Re << 1) so inertial effects are negligible.
    Sphere centered at (nx/4, ny/2, nz/2) with far-field BC.
    """
    dev = torch.device(device)
    nu = u_in * 2.0 * radius / re
    tau = 3.0 * nu + 0.5

    # Sphere geometry
    cx = nx * 0.25
    cy = ny / 2.0
    cz = nz / 2.0
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=dev)

    # Surface mask: solid cells with at least one fluid neighbour (6-connectivity)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, dims=0)
        | torch.roll(fluid, -1, dims=0)
        | torch.roll(fluid, 1, dims=1)
        | torch.roll(fluid, -1, dims=1)
        | torch.roll(fluid, 1, dims=2)
        | torch.roll(fluid, -1, dims=2)
    )

    # Cross-section area of sphere for drag coefficient
    A_sphere = math.pi * radius**2
    # Dynamic pressure
    dyn_pressure = 0.5 * 1.0 * u_in**2 * A_sphere

    # Stokes drag: Cd = 24/Re
    cd_stokes = 24.0 / re

    print(f"\n{'=' * 60}")
    print(f"Sphere Flow (Stokes): {lattice} {collision}")
    print(f"Grid: {nx}x{ny}x{nz}, radius={radius}, Re={re}")
    print(f"u_in={u_in}, nu={nu:.6f}, tau={tau:.4f}")
    print(f"Steps={n_steps}, device={device}")
    print(f"Cd_stokes={cd_stokes:.4f}")
    print(f"{'=' * 60}")

    # Initialize: uniform flow
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[solid] = 0.0
    f = _equilibrium(lattice, rho0, ux0, uy0, uz0, device=dev)
    initial_mass = float(f.sum().item())

    t0 = time.time()
    cd_history = []

    for step in range(1, n_steps + 1):
        # Collision
        f = _collide(lattice, collision, f, tau)

        # Streaming
        f = _stream(lattice, f)

        # Far-field BC (free stream inlet + lateral, zero-grad outlet)
        f = _far_field_bc(lattice, f, u_in, obstacle_mask=solid)

        # Bounce-back on sphere — compute momentum exchange (MEM drag)
        f_before = f.clone()
        f = _bounce_back(lattice, f, solid)
        delta_f = f - f_before  # non-zero only at solid cells

        # MEM: drag_x = Σ_q c_qx * Δf_q  at surface cells only
        if lattice == "D3Q19":
            Cmat = C19.to(dev).to(f.dtype)
        else:
            Cmat = C27.to(dev).to(f.dtype)
        # Mask to surface cells only (exclude interior solid)
        delta_f_surf = delta_f * surface.unsqueeze(0).to(f.dtype)
        drag_force = float((Cmat[:, 0:1].view(-1, 1, 1, 1) * delta_f_surf).sum().item())

        # Mass correction every 200 steps
        if step % 200 == 0:
            f = _correct_mass(lattice, f, initial_mass)

        # Compute Cd every 500 steps
        if step % 500 == 0 or step == n_steps:
            cd_total = -drag_force / dyn_pressure if dyn_pressure > 0 else 0.0
            cd_history.append(cd_total)

            elapsed = time.time() - t0
            print(
                f"  step {step:5d}/{n_steps}: Cd={cd_total:.4f} "
                f"(Stokes={cd_stokes:.4f}, err={abs(cd_total - cd_stokes) / cd_stokes * 100:.1f}%), "
                f"{elapsed:.1f}s"
            )

    # Use time-averaged Cd from last 50% of steps
    warmup = len(cd_history) // 2
    cd_avg = sum(cd_history[warmup:]) / max(len(cd_history[warmup:]), 1)
    cd_err = abs(cd_avg - cd_stokes) / cd_stokes * 100

    print("\n  --- Results ---")
    print(f"  Cd_num (avg):   {cd_avg:.4f}")
    print(f"  Cd_stokes:      {cd_stokes:.4f}")
    print(f"  Cd error:       {cd_err:.1f}%")

    # Stokes regime is hard to get exact with LBM (discretization errors)
    # Accept < 15% error for low Re
    passed = cd_err < 15.0
    print(f"\n  {'PASS' if passed else 'FAIL'}: Cd_err={cd_err:.1f}% (target < 15%)")
    return {
        "benchmark": "sphere_stokes",
        "lattice": lattice,
        "collision": collision,
        "re": re,
        "cd_num": cd_avg,
        "cd_stokes": cd_stokes,
        "cd_err_pct": cd_err,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TensorLBM benchmark validation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer steps")
    parser.add_argument("--poiseuille-only", action="store_true")
    parser.add_argument("--sphere-only", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    dev = torch.device(device)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(dev)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(dev).total_memory / 1e9:.1f} GB")

    # Step counts
    if args.quick:
        poiseuille_steps = 3000
        sphere_steps = 1500
    else:
        poiseuille_steps = 10000
        sphere_steps = 5000

    results = []

    # --- Benchmark 1: Poiseuille flow ---
    if not args.sphere_only:
        configs = [
            ("D3Q19", "BGK", 32, 64, 4),
            ("D3Q19", "MRT", 32, 64, 4),
            ("D3Q27", "BGK", 32, 64, 4),
            ("D3Q27", "MRT", 32, 64, 4),
            ("D3Q27", "CUMULANT", 32, 64, 4),
        ]
        for lattice, collision, ny, nx, nz in configs:
            r = run_poiseuille(
                lattice=lattice,
                collision=collision,
                ny=ny,
                nx=nx,
                nz=nz,
                force=1e-5,
                tau=1.0,
                n_steps=poiseuille_steps,
                device=device,
            )
            results.append(r)

    # --- Benchmark 2: Sphere Stokes flow ---
    if not args.poiseuille_only:
        configs = [
            ("D3Q19", "BGK", 64, 40, 40, 6.0, 0.01, 0.5),
            ("D3Q27", "BGK", 64, 40, 40, 6.0, 0.01, 0.5),
            ("D3Q27", "CUMULANT", 64, 40, 40, 6.0, 0.01, 0.5),
        ]
        for lattice, collision, nx, ny, nz, radius, u_in, re in configs:
            r = run_sphere_stokes(
                lattice=lattice,
                collision=collision,
                nx=nx,
                ny=ny,
                nz=nz,
                radius=radius,
                u_in=u_in,
                re=re,
                n_steps=sphere_steps,
                device=device,
            )
            results.append(r)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Benchmark':<20s} {'Lattice':<8s} {'Collision':<10s} "
        f"{'Metric':<12s} {'Num':>10s} {'Exact':>10s} {'Err%':>8s} {'Result':>6s}"
    )
    print("-" * 90)
    for r in results:
        if r["benchmark"] == "poiseuille":
            metric = "u_max"
            num = r["u_max_num"]
            exact = r["u_max_exact"]
            err = r["u_max_err_pct"]
        else:
            metric = "Cd"
            num = r["cd_num"]
            exact = r["cd_stokes"]
            err = r["cd_err_pct"]
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['benchmark']:<20s} {r['lattice']:<8s} {r['collision']:<10s} "
            f"{metric:<12s} {num:10.6f} {exact:10.6f} {err:8.2f} {status:>6s}"
        )

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n{n_pass}/{len(results)} benchmarks passed")


if __name__ == "__main__":
    main()
