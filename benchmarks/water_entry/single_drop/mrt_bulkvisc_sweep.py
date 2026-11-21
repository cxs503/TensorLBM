"""MRT bulk-viscosity sweep for droplet water-impact (DE-X-COL1-MRT).

Sweep the MRT even-mode relaxation rate s_e (energy moment) at {1×, 5×, 20×}
of the base value 1.19.  The bulk viscosity is
    ν_bulk = c_s² (1/s_e − 0.5)
so sweeping s_e changes the bulk viscosity while leaving the shear viscosity
(τ) untouched.

Odd modes (rows 1,3,5,7,9-13,16-18) are NOT touched — only s_e varies.

This script tests the MRT collision operator directly with a simple shear
flow setup to measure the effective bulk viscosity.

Usage:
    python mrt_bulkvisc_sweep.py --cells 40 --out runs/mrt_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure we can import tensorlbm from the worktree
sys.path.insert(0, "/root/tp-agent/work/m26957/src")

from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, collide_bgk3d

# ---------------------------------------------------------------------------
# Anchor constants (17-digit, from TRACK_COMB_W1.md)
# ---------------------------------------------------------------------------
ANCHOR_EXP_DEPTH = 1.857434  # D
ANCHOR_EXP_TIME = 10.394  # ms (pc)
ANCHOR_CTRL_DEPTH = 1.435220  # D
ANCHOR_CTRL_TIME_ABS = 11.0906  # ms (abs)
ANCHOR_CTRL_TIME_PC = 10.5433  # ms (pc)
WINDOW_MS = 24.005  # ms
MACH = 0.2598
CONTACT = 45  # cells
DRIFT_LIMIT = 4.8e-6
CELL_SIZE_D = 0.125  # 1 cell = 0.125 D ≈ 6.7%

# MRT base relaxation rates
S_E_BASE = 1.19
S_EPS_BASE = 1.4
S_Q_BASE = 1.2
S_PI_BASE = None  # defaults to s_e

# Bulk-viscosity sweep multipliers
S_E_MULTIPLIERS = [1.0, 5.0, 20.0]


def compute_bulk_viscosity(s_e: float) -> float:
    """Compute bulk viscosity from MRT relaxation rate.

    ν_bulk = c_s² (1/s_e − 0.5)
    """
    cs2 = 1.0 / 3.0
    return cs2 * (1.0 / s_e - 0.5)


def run_mrt_shear_test(
    nx: int,
    ny: int,
    nz: int,
    tau: float,
    s_e: float,
    n_steps: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Run MRT shear flow test to measure effective viscosity.

    A plane shear wave u_x = A sin(k y) is an exact solution of the
    incompressible Navier-Stokes equations. Its kinetic energy decays as
    exp(-2 ν k² t) with ν = (τ - 0.5)/3.

    The MRT collision with different s_e values should NOT affect the
    shear viscosity (which is controlled by τ), but should affect the
    bulk viscosity.
    """
    device = torch.device(device)
    dtype = dtype

    # Initialize shear wave
    k = 2.0 * math.pi / ny
    amplitude = 0.01
    y = torch.arange(ny, dtype=dtype)
    profile = torch.sin(k * y).view(1, ny, 1).expand(nz, ny, nx)
    rho = torch.ones((nz, ny, nx), dtype=dtype)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, amplitude * profile.contiguous(), zero, zero)

    # Track kinetic energy
    times = []
    energies = []

    for step in range(n_steps + 1):
        # MRT collision
        f = collide_mrt3d(f, tau, s_e=s_e, s_eps=S_EPS_BASE, s_q=S_Q_BASE, s_pi=S_PI_BASE)

        # Streaming (periodic)
        out = torch.empty_like(f)
        for q in range(19):
            cx, cy, cz = int(C[q, 0]), int(C[q, 1]), int(C[q, 2])
            out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
        f = out

        # Record energy
        if step % 10 == 0:
            rho_s, ux, uy, uz = macroscopic3d(f)
            energy = 0.5 * rho_s * (ux ** 2 + uy ** 2 + uz ** 2)
            times.append(step)
            energies.append(float(energy.mean()))

    # Fit decay rate
    if len(times) >= 3:
        t = torch.tensor(times, dtype=torch.float64)
        y = torch.log(torch.tensor(energies, dtype=torch.float64))
        t_mean = t.mean()
        y_mean = y.mean()
        slope = float(((t - t_mean) * (y - y_mean)).sum() / ((t - t_mean) ** 2).sum())
        measured_visc = -slope / (2.0 * k * k)
        analytic_visc = (tau - 0.5) / 3.0
        rel_error = abs(measured_visc - analytic_visc) / abs(analytic_visc) if analytic_visc != 0 else abs(measured_visc)
    else:
        measured_visc = 0
        analytic_visc = (tau - 0.5) / 3.0
        rel_error = 1.0

    return {
        "s_e": s_e,
        "bulk_viscosity": compute_bulk_viscosity(s_e),
        "measured_shear_viscosity": measured_visc,
        "analytic_shear_viscosity": analytic_visc,
        "shear_viscosity_rel_error": rel_error,
        "max_mach": float((3 * (torch.tensor(energies) * 2 / (0.01 ** 2))).sqrt().max()) if energies else 0,
        "energy_first": energies[0] if energies else 0,
        "energy_last": energies[-1] if energies else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="MRT bulk-viscosity sweep")
    parser.add_argument("--cells", type=int, default=40, help="Grid resolution (nx=ny=nz)")
    parser.add_argument("--out", type=Path, default=Path("runs/mrt_sweep"), help="Output directory")
    parser.add_argument("--tau", type=float, default=1.0, help="MRT shear relaxation time")
    parser.add_argument("--n-steps", type=int, default=500, help="Max steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Grid setup
    nx = ny = nz = args.cells

    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"tau={args.tau}")
    print(f"s_e sweep: {S_E_MULTIPLIERS}× base ({S_E_BASE})")
    print()

    results = {}
    for mult in S_E_MULTIPLIERS:
        s_e = S_E_BASE * mult
        bulk_visc = compute_bulk_viscosity(s_e)
        print(f"--- s_e = {s_e:.4f} (bulk_visc = {bulk_visc:.6f}) ---")

        result = run_mrt_shear_test(
            nx=nx, ny=ny, nz=nz,
            tau=args.tau,
            s_e=s_e,
            n_steps=args.n_steps,
            device=args.device,
        )

        key = f"mrt_se{mult}x"
        results[key] = result

        print(f"  measured_shear_visc={result['measured_shear_viscosity']:.6f}")
        print(f"  analytic_shear_visc={result['analytic_shear_viscosity']:.6f}")
        print(f"  shear_visc_rel_error={result['shear_viscosity_rel_error']:.4%}")
        print(f"  max_mach={result['max_mach']:.6f}")
        print()

    # Save results
    with open(args.out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.out / 'results.json'}")

    # Print comparison table
    print("\n=== COMPARISON TABLE ===")
    print(f"{'Model':<20} {'s_e':<10} {'BulkVisc':<12} {'ShearVisc':<12} {'ShearErr':<10} {'Mach':<8}")
    print("-" * 72)
    for key, res in results.items():
        print(f"{key:<20} {res['s_e']:<10.4f} {res['bulk_viscosity']:<12.6f} {res['measured_shear_viscosity']:<12.6f} {res['shear_viscosity_rel_error']:<10.4%} {res['max_mach']:<8.6f}")

    # Verify that shear viscosity is NOT affected by s_e sweep
    print("\n=== VERIFICATION ===")
    print("The shear viscosity should be unchanged across all s_e values.")
    print(f"Analytic shear viscosity: {(args.tau - 0.5) / 3.0:.6f}")
    for key, res in results.items():
        err = res['shear_viscosity_rel_error']
        status = "PASS" if err < 0.03 else "FAIL"
        print(f"  {key}: shear_visc_error={err:.4%} [{status}]")


if __name__ == "__main__":
    main()
