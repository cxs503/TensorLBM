#!/usr/bin/env python
"""Entry point for AA-pattern fused stream-collide cross-validation.

Usage:
    python src/tensorlbm/aa_fused_xval_entry.py [--n N] [--re RE] [--steps STEPS] [--out DIR]

Produces:
    - JSON results file with numerical equivalence, accuracy vs Ghia, and performance
    - PNG figures comparing velocity profiles
    - Summary report

Anchor: XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.aa_fused_stream_collide import (
    aa_fused_step,
    separate_step,
    _macroscopic,
    _equilibrium,
    C,
    W,
)
from tensorlbm.verification.lid_driven_cavity.reference_data import get_ghia_reference


def run_cavity(n: int, re: float, u_lid: float, max_steps: int,
               method: str = "aa", dtype: torch.dtype = torch.float64,
               device: torch.device = torch.device("cpu")) -> dict:
    """Run lid-driven cavity with specified method."""
    L = n - 1
    nu = u_lid * L / re
    tau = nu / (1.0 / 3.0) + 0.5

    rho = torch.ones((n, n), dtype=dtype, device=device)
    ux = torch.zeros((n, n), dtype=dtype, device=device)
    uy = torch.zeros((n, n), dtype=dtype, device=device)

    c = C.to(device=device, dtype=dtype)
    w = W.to(device=device, dtype=dtype)

    uc = c[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(-1, 1, 1) * uy.unsqueeze(0)
    u2 = ux**2 + uy**2
    f = w.view(-1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
    )

    solid = torch.zeros((n, n), dtype=torch.bool, device=device)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True

    step_fn = aa_fused_step if method == "aa" else separate_step

    start = time.perf_counter()
    for _ in range(max_steps):
        f = step_fn(f, tau, solid, u_lid=u_lid, lid_row=-1)
    elapsed = time.perf_counter() - start

    # Extract profiles (fluid nodes only: indices 1..n-2)
    rho_f, ux_f, uy_f = _macroscopic(f)
    mid = n // 2
    y_norm = np.linspace(0, 1, n)[1:-1]
    x_norm = np.linspace(0, 1, n)[1:-1]
    u_center = (ux_f[1:-1, mid] / u_lid).cpu().numpy()
    v_center = (uy_f[mid, 1:-1] / u_lid).cpu().numpy()

    mlups = (n * n * max_steps) / (elapsed * 1e6)

    return {
        "y": y_norm,
        "u": u_center,
        "x": x_norm,
        "v": v_center,
        "elapsed": elapsed,
        "mlups": mlups,
        "mass": float(f.sum().item()),
        "tau": float(tau),
    }


def compute_errors(y_sim, u_sim, x_sim, v_sim, ref) -> dict:
    """Compute error metrics vs Ghia reference (interior points only)."""
    # Ghia reference includes boundary points; we compare only interior
    mask_u = (ref.y > 0) & (ref.y < 1)
    mask_v = (ref.x > 0) & (ref.x < 1)

    # Interpolate simulation at Ghia points
    sim_y_full = np.linspace(0, 1, len(y_sim) + 2)
    sim_x_full = np.linspace(0, 1, len(x_sim) + 2)
    u_interp = np.interp(ref.y[mask_u], sim_y_full[1:-1], u_sim)
    v_interp = np.interp(ref.x[mask_v], sim_x_full[1:-1], v_sim)

    l2_u = float(np.sqrt(np.mean((u_interp - ref.u[mask_u]) ** 2)))
    l2_v = float(np.sqrt(np.mean((v_interp - ref.v[mask_v]) ** 2)))
    max_u = float(np.max(np.abs(u_interp - ref.u[mask_u])))
    max_v = float(np.max(np.abs(v_interp - ref.v[mask_v])))

    return {"l2_u": l2_u, "l2_v": l2_v, "max_u": max_u, "max_v": max_v}


def generate_profiles_figure(y_aa, u_aa, x_aa, v_aa, y_sep, u_sep, x_sep, v_sep,
                              ref, n, re, out_path: str):
    """Generate comparison figure of velocity profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(ref.u, ref.y, "ko", markersize=4, label="Ghia et al. (1982)", zorder=3)
    ax.plot(u_aa, y_aa, "b-", linewidth=1.5, label="AA-pattern fused")
    ax.plot(u_sep, y_sep, "r--", linewidth=1.5, label="Separate stream+collide")
    ax.set_xlabel("u / U_lid")
    ax.set_ylabel("y / L")
    ax.set_title(f"Vertical centerline velocity (Re={re}, n={n})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ref.v, ref.x, "ko", markersize=4, label="Ghia et al. (1982)", zorder=3)
    ax.plot(v_aa, x_aa, "b-", linewidth=1.5, label="AA-pattern fused")
    ax.plot(v_sep, x_sep, "r--", linewidth=1.5, label="Separate stream+collide")
    ax.set_xlabel("v / U_lid")
    ax.set_ylabel("x / L")
    ax.set_title(f"Horizontal centerline velocity (Re={re}, n={n})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="AA-pattern fused stream-collide XVAL")
    parser.add_argument("--n", type=int, default=65, help="Grid resolution")
    parser.add_argument("--re", type=float, default=100, help="Reynolds number")
    parser.add_argument("--steps", type=int, default=8000, help="Time steps")
    parser.add_argument("--u-lid", type=float, default=0.1, help="Lid velocity")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).parent.parent.parent / "artifacts" / "xval_aa_pattern"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"XVAL: AA-pattern fused stream-collide")
    print(f"  n={args.n}, Re={args.re}, steps={args.steps}, u_lid={args.u_lid}")

    print("\n--- Running AA-pattern fused ---")
    result_aa = run_cavity(args.n, args.re, args.u_lid, args.steps, method="aa")
    print(f"  Time: {result_aa['elapsed']:.3f}s, MLUPS: {result_aa['mlups']:.1f}")

    print("\n--- Running separate stream+collide ---")
    result_sep = run_cavity(args.n, args.re, args.u_lid, args.steps, method="separate")
    print(f"  Time: {result_sep['elapsed']:.3f}s, MLUPS: {result_sep['mlups']:.1f}")

    print("\n--- Mass Comparison ---")
    mass_aa = result_aa["mass"]
    mass_sep = result_sep["mass"]
    mass_diff = abs(mass_aa - mass_sep) / mass_aa
    print(f"  Mass AA: {mass_aa:.2f}, Mass Sep: {mass_sep:.2f}, rel diff: {mass_diff:.2e}")

    print("\n--- Ghia Reference Comparison (Re=100) ---")
    ref = get_ghia_reference(100)

    errors_aa = compute_errors(result_aa["y"], result_aa["u"],
                               result_aa["x"], result_aa["v"], ref)
    errors_sep = compute_errors(result_sep["y"], result_sep["u"],
                                result_sep["x"], result_sep["v"], ref)

    print(f"  AA-pattern:  L2_u={errors_aa['l2_u']:.6f}, L2_v={errors_aa['l2_v']:.6f}")
    print(f"  Separate:    L2_u={errors_sep['l2_u']:.6f}, L2_v={errors_sep['l2_v']:.6f}")

    print("\n--- Performance ---")
    speedup = result_sep["elapsed"] / result_aa["elapsed"]
    print(f"  AA-pattern: {result_aa['mlups']:.1f} MLUPS")
    print(f"  Separate:   {result_sep['mlups']:.1f} MLUPS")

    aa_pass = bool(errors_aa['l2_u'] < 0.15 and errors_aa['l2_v'] < 0.15)
    sep_pass = bool(errors_sep['l2_u'] < 0.15 and errors_sep['l2_v'] < 0.15)
    print(f"\n--- Verdict ---")
    print(f"  AA-pattern vs Ghia < 15%: {'PASS' if aa_pass else 'FAIL'}")
    print(f"  Separate vs Ghia < 15%: {'PASS' if sep_pass else 'FAIL'}")

    # Generate figure
    fig_path = str(out_dir / "profiles_comparison.png")
    generate_profiles_figure(
        result_aa["y"], result_aa["u"], result_aa["x"], result_aa["v"],
        result_sep["y"], result_sep["u"], result_sep["x"], result_sep["v"],
        ref, args.n, args.re, fig_path
    )
    print(f"\nFigure saved to: {fig_path}")

    # Build result JSON (all values are Python native types)
    result = {
        "experiment": "aa_pattern_fused_stream_collide_xval",
        "anchor": "XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN",
        "case": {"n": args.n, "re": args.re, "u_lid": args.u_lid, "max_steps": args.steps},
        "numerical_comparison": {
            "mass_aa": mass_aa,
            "mass_sep": mass_sep,
            "mass_rel_diff": float(mass_diff),
        },
        "aa_pattern": {
            "elapsed_s": float(result_aa["elapsed"]),
            "mlups": float(result_aa["mlups"]),
            "tau": float(result_aa["tau"]),
            "errors_vs_ghia": errors_aa,
            "pass_vs_ghia_15pct": aa_pass,
        },
        "separate": {
            "elapsed_s": float(result_sep["elapsed"]),
            "mlups": float(result_sep["mlups"]),
            "tau": float(result_sep["tau"]),
            "errors_vs_ghia": errors_sep,
            "pass_vs_ghia_15pct": sep_pass,
        },
        "performance": {
            "speedup_sep_over_aa": float(speedup),
        },
        "verdict": {
            "aa_pattern_accuracy": "PASS" if aa_pass else "FAIL",
            "separate_accuracy": "PASS" if sep_pass else "FAIL",
        },
        "artifacts": {
            "profiles_figure": fig_path,
        },
    }

    out_path = out_dir / "xval_aa_pattern_fused_stream_collide.json"
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"Results saved to: {out_path}")

    return result


if __name__ == "__main__":
    main()
