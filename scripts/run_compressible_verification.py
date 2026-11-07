"""Verification script for compressible shock wave simulations.

Runs normal shock parameter scan with optimized solver.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tensorlbm.compressible.solver import (
    GodunovSolver1D,
    normal_shock_relations,
    oblique_shock_relations,
)


def run_normal_shock_scan(output_dir: Path):
    """Run normal shock parameter scan: Mach 2/3/5/8/10 × 3 grids."""
    mach_numbers = [2.0, 3.0, 5.0, 8.0, 10.0]
    grid_sizes = [128, 256, 512]
    gamma = 1.4

    results = []

    for M1 in mach_numbers:
        rh = normal_shock_relations(M1, gamma)
        print(f"\nMach {M1:.0f} - Exact: rho2/rho1={rh.rho_ratio:.6f}, p2/p1={rh.p_ratio:.6f}, T2/T1={rh.T_ratio:.6f}")

        for nx in grid_sizes:
            solver = GodunovSolver1D(nx=nx, gamma=gamma, cfl=0.4)
            solver.initialize_normal_shock(M1=M1, gamma=gamma, shock_pos=0.5)

            # Run fewer steps for speed (shock is stationary, so few steps needed)
            n_steps = min(300, nx * 2)
            solver.run(n_steps, report_interval=n_steps)

            # Extract post-shock values (right half)
            nx_mid = nx // 2
            rho_post = solver.rho[nx_mid:].mean()
            u_post = solver.u[nx_mid:].mean()
            p_post = solver.p[nx_mid:].mean()
            T_post = p_post / (rho_post * (gamma - 1.0))

            # Exact values
            rho1, p1 = 1.0, 1.0
            T1 = p1 / (rho1 * (gamma - 1.0))
            cs1 = math.sqrt(gamma * p1 / rho1)
            u1 = M1 * cs1

            rho2_exact = rho1 * rh.rho_ratio
            p2_exact = p1 * rh.p_ratio
            T2_exact = T1 * rh.T_ratio
            u2_exact = rh.M2 * math.sqrt(gamma * p2_exact / rho2_exact)

            err_rho = abs(rho_post - rho2_exact) / rho2_exact * 100
            err_p = abs(p_post - p2_exact) / p2_exact * 100
            err_T = abs(T_post - T2_exact) / T2_exact * 100
            err_u = abs(u_post - u2_exact) / abs(u2_exact) * 100 if abs(u2_exact) > 1e-10 else 0

            result = {
                "M1": M1, "nx": nx,
                "rho_exact": rho2_exact, "rho_num": rho_post, "rho_err_pct": err_rho,
                "p_exact": p2_exact, "p_num": p_post, "p_err_pct": err_p,
                "T_exact": T2_exact, "T_num": T_post, "T_err_pct": err_T,
                "u_exact": u2_exact, "u_num": u_post, "u_err_pct": err_u,
            }
            results.append(result)
            print(f"  nx={nx:4d}: rho_err={err_rho:.4f}%, p_err={err_p:.4f}%, T_err={err_T:.4f}%")

    # Save results
    with open(output_dir / "normal_shock_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate plots
    generate_plots(results, output_dir, mach_numbers, grid_sizes)

    return results


def generate_plots(results, output_dir, mach_numbers, grid_sizes):
    """Generate verification plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Density error vs Mach
    ax = axes[0, 0]
    for nx in grid_sizes:
        machs = [r["M1"] for r in results if r["nx"] == nx]
        errs = [r["rho_err_pct"] for r in results if r["nx"] == nx]
        ax.plot(machs, errs, "o-", label=f"nx={nx}")
    ax.set_xlabel("Mach Number"); ax.set_ylabel("Density Error (%)")
    ax.set_title("Post-Shock Density Error vs Mach"); ax.legend(); ax.grid(True, alpha=0.3)

    # Pressure error vs Mach
    ax = axes[0, 1]
    for nx in grid_sizes:
        machs = [r["M1"] for r in results if r["nx"] == nx]
        errs = [r["p_err_pct"] for r in results if r["nx"] == nx]
        ax.plot(machs, errs, "s-", label=f"nx={nx}")
    ax.set_xlabel("Mach Number"); ax.set_ylabel("Pressure Error (%)")
    ax.set_title("Post-Shock Pressure Error vs Mach"); ax.legend(); ax.grid(True, alpha=0.3)

    # Temperature error vs Mach
    ax = axes[1, 0]
    for nx in grid_sizes:
        machs = [r["M1"] for r in results if r["nx"] == nx]
        errs = [r["T_err_pct"] for r in results if r["nx"] == nx]
        ax.plot(machs, errs, "^-", label=f"nx={nx}")
    ax.set_xlabel("Mach Number"); ax.set_ylabel("Temperature Error (%)")
    ax.set_title("Post-Shock Temperature Error vs Mach"); ax.legend(); ax.grid(True, alpha=0.3)

    # Convergence
    ax = axes[1, 1]
    for M1 in [2.0, 5.0, 10.0]:
        nxs = [r["nx"] for r in results if r["M1"] == M1]
        errs = [r["rho_err_pct"] for r in results if r["M1"] == M1]
        ax.loglog(nxs, errs, "o-", label=f"M={M1:.0f}")
    ax.set_xlabel("Grid Points"); ax.set_ylabel("Density Error (%)")
    ax.set_title("Convergence Study"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "normal_shock_verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Generate profile plots for M=2, 5, 10
    for M1 in [2.0, 5.0, 10.0]:
        generate_profile_plot(M1, 256, output_dir)


def generate_profile_plot(M1, nx, output_dir):
    """Generate shock profile plot."""
    gamma = 1.4
    solver = GodunovSolver1D(nx=nx, gamma=gamma, cfl=0.4)
    solver.initialize_normal_shock(M1=M1, gamma=gamma, shock_pos=0.5)
    solver.run(min(300, nx * 2), report_interval=min(300, nx * 2))

    rh = normal_shock_relations(M1, gamma)
    rho1, p1 = 1.0, 1.0
    T1 = p1 / (rho1 * (gamma - 1.0))
    cs1 = math.sqrt(gamma * p1 / rho1)
    u1 = M1 * cs1
    rho2 = rho1 * rh.rho_ratio
    p2 = p1 * rh.p_ratio
    T2 = T1 * rh.T_ratio
    u2 = rh.M2 * math.sqrt(gamma * p2 / rho2)

    x = np.arange(nx) / nx

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(x, solver.rho, "b-", lw=1.5, label="Numerical")
    ax.axhline(rho1, color="r", ls="--", alpha=0.7, label=f"Exact ρ₁={rho1:.2f}")
    ax.axhline(rho2, color="g", ls="--", alpha=0.7, label=f"Exact ρ₂={rho2:.2f}")
    ax.set_xlabel("x/L"); ax.set_ylabel("Density")
    ax.set_title(f"Density (M={M1}, nx={nx})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x, solver.p, "b-", lw=1.5, label="Numerical")
    ax.axhline(p1, color="r", ls="--", alpha=0.7, label=f"Exact p₁={p1:.2f}")
    ax.axhline(p2, color="g", ls="--", alpha=0.7, label=f"Exact p₂={p2:.2f}")
    ax.set_xlabel("x/L"); ax.set_ylabel("Pressure")
    ax.set_title(f"Pressure (M={M1}, nx={nx})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(x, solver.u, "b-", lw=1.5, label="Numerical")
    ax.axhline(u1, color="r", ls="--", alpha=0.7, label=f"Exact u₁={u1:.3f}")
    ax.axhline(u2, color="g", ls="--", alpha=0.7, label=f"Exact u₂={u2:.3f}")
    ax.set_xlabel("x/L"); ax.set_ylabel("Velocity")
    ax.set_title(f"Velocity (M={M1}, nx={nx})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"shock_profile_M{M1:.0f}_nx{nx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    output_dir = Path(__file__).parent.parent / "runs" / "m5249" / "compressible_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Compressible Shock Wave Verification Campaign")
    print("=" * 60)

    results = run_normal_shock_scan(output_dir)

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    max_err_rho = max(r["rho_err_pct"] for r in results)
    max_err_p = max(r["p_err_pct"] for r in results)
    max_err_T = max(r["T_err_pct"] for r in results)

    print(f"Max density error:  {max_err_rho:.4f}%")
    print(f"Max pressure error: {max_err_p:.4f}%")
    print(f"Max temperature error: {max_err_T:.4f}%")
    print(f"All errors < 3%: {max_err_rho < 3 and max_err_p < 3 and max_err_T < 3}")

    summary = {
        "campaign": "IN21-L1",
        "mission_id": 5249,
        "total_cases": len(results),
        "max_rho_err_pct": max_err_rho,
        "max_p_err_pct": max_err_p,
        "max_T_err_pct": max_err_T,
        "all_under_3pct": bool(max_err_rho < 3 and max_err_p < 3 and max_err_T < 3),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
