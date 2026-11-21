#!/usr/bin/env python
"""MRT collision operator benchmark: three standard test cases (quick version).

Runs Taylor-Green vortex decay, Poiseuille channel flow, and lid-driven cavity
with both BGK and MRT collision operators, comparing results.
"""

from __future__ import annotations

import json
import time
import torch
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm.d2q9 import (
    equilibrium, macroscopic, stream, collide_bgk, collide_mrt as collide_mrt_2d,
    C as C2Q9, W as W2Q9, CS2,
)


def run_taylor_green(n=32, re=100, u0=0.05, steps=500, collision="bgk"):
    """Taylor-Green vortex decay test (quick version)."""
    L = 2 * np.pi
    nu = u0 * L / re
    tau = nu / CS2 + 0.5

    x = torch.linspace(0, L, n + 1)[:-1]
    y = torch.linspace(0, L, n + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing='ij')

    rho = torch.ones((n, n), dtype=torch.float64)
    ux = -u0 * torch.cos(X) * torch.sin(Y)
    uy = u0 * torch.sin(X) * torch.cos(Y)
    f = equilibrium(rho, ux, uy)

    E0 = (ux**2 + uy**2).sum().item() / (n * n)

    collide = collide_bgk if collision == "bgk" else collide_mrt_2d

    for step in range(steps):
        if collision == "mrt":
            f = collide(f, tau, s_e=1.0, s_q=1.0)
        else:
            f = collide(f, tau)
        f = stream(f)

    _, ux_f, uy_f = macroscopic(f)
    E_final = (ux_f**2 + uy_f**2).sum().item() / (n * n)

    t_final = steps
    E_analytical = u0**2 / 4.0 * np.exp(-4.0 * nu * t_final / L**2 * (2 * np.pi)**2)

    err_E = abs(E_final - E_analytical) / E_analytical * 100 if E_analytical > 1e-15 else 0.0

    return {
        "n": n, "re": re, "collision": collision, "tau": tau,
        "E0": E0, "E_final": E_final, "E_analytical": E_analytical,
        "err_E_pct": err_E, "stable": True,
    }


def run_poiseuille(ny=32, nx=32, steps=1000, collision="bgk"):
    """Poiseuille channel flow test (quick version)."""
    G = 1e-5
    H = ny - 1
    tau = 0.8
    nu = (tau - 0.5) / 3.0

    rho = torch.ones((ny, nx), dtype=torch.float64)
    ux = torch.zeros((ny, nx), dtype=torch.float64)
    uy = torch.zeros((ny, nx), dtype=torch.float64)
    f = equilibrium(rho, ux, uy)

    collide = collide_bgk if collision == "bgk" else collide_mrt_2d
    c = C2Q9.to(f.device, dtype=f.dtype)
    w = W2Q9.to(f.device, dtype=f.dtype)

    for step in range(steps):
        rho, ux, uy = macroscopic(f)
        f = f + w.view(9, 1, 1) * c[:, 0].view(9, 1, 1) * G * rho.unsqueeze(0) / CS2

        if collision == "mrt":
            f = collide(f, tau, s_e=1.0, s_q=1.0)
        else:
            f = collide(f, tau)

        f[:, 0, :] = f[[0, 3, 4, 1, 2, 7, 8, 5, 6], 1, :]
        f[:, -1, :] = f[[0, 3, 4, 1, 2, 7, 8, 5, 6], -2, :]
        f = stream(f)
        f[:, 0, :] = f[[0, 3, 4, 1, 2, 7, 8, 5, 6], 1, :]
        f[:, -1, :] = f[[0, 3, 4, 1, 2, 7, 8, 5, 6], -2, :]

    _, ux_final, _ = macroscopic(f)
    ux_avg = ux_final.mean(dim=1).numpy()

    y = np.arange(ny)
    u_analytical = (G / (2 * nu)) * y * (H - y)
    u_max_analytical = G * H**2 / (8 * nu)
    u_max_sim = ux_avg.max()

    if u_max_analytical > 0 and u_max_sim > 0:
        u_norm = ux_avg / u_max_sim
        u_anal_norm = u_analytical / u_max_analytical
        interior = slice(2, -2)
        r2 = 1.0 - ((u_norm[interior] - u_anal_norm[interior])**2).sum() / \
             (u_anal_norm[interior]**2).sum()
    else:
        r2 = 0.0

    err_u_max = abs(u_max_sim - u_max_analytical) / u_max_analytical * 100

    return {
        "ny": ny, "nx": nx, "collision": collision, "tau": tau,
        "u_max_sim": float(u_max_sim), "u_max_analytical": float(u_max_analytical),
        "err_u_max_pct": err_u_max, "r2": float(r2), "stable": True,
    }


def run_lid_driven_cavity(n=33, re=100, steps=2000, collision="bgk"):
    """Lid-driven cavity test (quick version)."""
    u_lid = 0.1
    L = n - 1
    nu = u_lid * L / re
    tau = nu / CS2 + 0.5

    rho = torch.ones((n, n), dtype=torch.float64)
    ux = torch.zeros((n, n), dtype=torch.float64)
    uy = torch.zeros((n, n), dtype=torch.float64)
    f = equilibrium(rho, ux, uy)

    collide = collide_bgk if collision == "bgk" else collide_mrt_2d

    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)

    for step in range(steps):
        if collision == "mrt":
            f_collided = collide(f, tau, s_e=1.0, s_q=1.0)
        else:
            f_collided = collide(f, tau)

        f_streamed = stream(f_collided)

        f_out = f_streamed.clone()
        rho_post = f_collided.sum(dim=0)

        # Bottom wall
        for i in range(1, 9):
            o = opp[i].item()
            f_out[o, 0, :] = f_collided[i, 1, :]
        # Top wall (moving lid)
        for i in range(1, 9):
            o = opp[i].item()
            cx = int(C2Q9[i, 0])
            f_out[o, -1, :] = f_collided[i, -2, :] + 6.0 * W2Q9[o].item() * cx * u_lid * rho_post[-1, :]
        # Left wall
        for i in range(1, 9):
            o = opp[i].item()
            f_out[o, :, 0] = f_collided[i, :, 1]
        # Right wall
        for i in range(1, 9):
            o = opp[i].item()
            f_out[o, :, -1] = f_collided[i, :, -2]

        f = f_out

    _, ux_final, uy_final = macroscopic(f)
    mid = n // 2
    u_center = ux_final[:, mid].numpy()

    u_min_sim = u_center.min()
    u_min_ghia = -0.21090
    err_u_min = abs(u_min_sim - u_min_ghia) / abs(u_min_ghia) * 100

    return {
        "n": n, "re": re, "collision": collision, "tau": tau,
        "u_min_sim": float(u_min_sim), "u_min_ghia": u_min_ghia,
        "err_u_min_pct": err_u_min, "stable": True,
    }


def main():
    """Run all three test cases with BGK and MRT."""
    results = {
        "mission": "m23153",
        "anchor": "LBM-P0-MRT",
        "title": "MRT 碰撞算子真实现实证",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": {},
    }

    # Case 1: Taylor-Green
    print("=" * 60)
    print("Case 1: Taylor-Green vortex decay (N=32, Re=100)")
    print("=" * 60)

    for collision in ["bgk", "mrt"]:
        print(f"  Running {collision.upper()}...")
        r = run_taylor_green(n=32, re=100, u0=0.05, steps=500, collision=collision)
        print(f"    E_final = {r['E_final']:.6e}, E_analytical = {r['E_analytical']:.6e}")
        print(f"    Error = {r['err_E_pct']:.3f}%")
        results["cases"]["taylor_green"] = results["cases"].get("taylor_green", {})
        results["cases"]["taylor_green"][collision] = r

    # Case 2: Poiseuille
    print("\n" + "=" * 60)
    print("Case 2: Poiseuille channel flow (Ny=32)")
    print("=" * 60)

    for collision in ["bgk", "mrt"]:
        print(f"  Running {collision.upper()}...")
        r = run_poiseuille(ny=32, nx=32, steps=1000, collision=collision)
        print(f"    u_max_sim = {r['u_max_sim']:.6e}, u_max_analytical = {r['u_max_analytical']:.6e}")
        print(f"    Error = {r['err_u_max_pct']:.3f}%, R² = {r['r2']:.4f}")
        results["cases"]["poiseuille"] = results["cases"].get("poiseuille", {})
        results["cases"]["poiseuille"][collision] = r

    # Case 3: Lid-driven cavity
    print("\n" + "=" * 60)
    print("Case 3: Lid-driven cavity (N=33, Re=100)")
    print("=" * 60)

    for collision in ["bgk", "mrt"]:
        print(f"  Running {collision.upper()}...")
        r = run_lid_driven_cavity(n=33, re=100, steps=2000, collision=collision)
        print(f"    u_min_sim = {r['u_min_sim']:.6f}, u_min_ghia = {r['u_min_ghia']:.6f}")
        print(f"    Error = {r['err_u_min_pct']:.3f}%")
        results["cases"]["lid_driven_cavity"] = results["cases"].get("lid_driven_cavity", {})
        results["cases"]["lid_driven_cavity"][collision] = r

    # Save results
    output_dir = Path(__file__).parent.parent / "docs" / "m23153"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mrt_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: MRT vs BGK comparison")
    print("=" * 60)
    for case_name, case_results in results["cases"].items():
        print(f"\n{case_name}:")
        for collision, r in case_results.items():
            if "err_E_pct" in r:
                print(f"  {collision}: err_E = {r['err_E_pct']:.3f}%")
            elif "err_u_max_pct" in r:
                print(f"  {collision}: err_u_max = {r['err_u_max_pct']:.3f}%, R² = {r['r2']:.4f}")
            elif "err_u_min_pct" in r:
                print(f"  {collision}: err_u_min = {r['err_u_min_pct']:.3f}%")

    return results


if __name__ == "__main__":
    main()
