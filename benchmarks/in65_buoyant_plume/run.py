#!/usr/bin/env python
"""IN65-N03 benchmark runner: scalar transport + Boussinesq verification.

Produces:
  - runs/in65_verification/in65_verification.json: full verification report
  - artifacts/in65_figures/fig1_1d_advection_diffusion.png
  - artifacts/in65_figures/fig2_plume_self_similarity.png
  - artifacts/in65_figures/fig3_lfl_hazard_zone.png

Usage:
    python benchmarks/in65_buoyant_plume/run.py
    python benchmarks/in65_buoyant_plume/run.py --smoke  # quick test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.scalar_verification_in65 import (
    run_1d_advection_diffusion,
    run_mass_conservation_test,
    run_buoyant_plume_self_similarity,
    run_buoyant_plume_lbm_check,
    run_negative_control_no_buoyancy,
    run_lfl_wind_speed_sweep,
    run_lfl_grid_convergence,
)


def run_all_gates() -> dict:
    """Run all verification gates."""
    t0 = time.time()
    report = {
        "mission": "IN65-N03",
        "title": "TensorLBM Scalar Transport + Boussinesq Buoyancy Verification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # G1: 1D advection-diffusion
    print("[G1] 1-D advection-diffusion Gaussian anchor...")
    g1 = run_1d_advection_diffusion(nx=256, n_steps=1000)
    report["G1"] = {k: v for k, v in g1.items() if k not in ("x", "C_numeric", "C_exact")}
    report["G1"]["_x"] = g1["x"]
    report["G1"]["_C_numeric"] = g1["C_numeric"]
    report["G1"]["_C_exact"] = g1["C_exact"]
    print(f"  L2 error = {g1['l2_relative_error']:.4f} → {'PASS' if g1['pass_l2'] else 'FAIL'}")
    print(f"  Peak error = {g1['peak_error_pct']:.2f}% → {'PASS' if g1['pass_peak'] else 'FAIL'}")

    # G2: Mass conservation
    print("\n[G2] Mass conservation...")
    g2 = run_mass_conservation_test(nx=64, ny=64, n_steps=2000)
    report["G2"] = g2
    print(f"  Max drift = {g2['max_relative_drift']:.2e} → {'PASS' if g2['pass'] else 'FAIL'}")

    # G3: Buoyant plume self-similarity (analytical)
    print("\n[G3] Buoyant plume self-similarity Q(z) ∝ z^(5/3)...")
    g3 = run_buoyant_plume_self_similarity()
    report["G3"] = g3
    print(f"  Exponent = {g3['exponent']:.3f} → {'PASS' if g3['pass'] else 'FAIL'}")

    # G3b: LBM sanity check (small grid)
    print("\n[G3b] LBM buoyancy sanity check...")
    g3b = run_buoyant_plume_lbm_check(nx=16, ny=32, n_steps=200)
    report["G3b"] = g3b
    print(f"  uy_max = {g3b['uy_max']:.6f} → {'PASS' if g3b['buoyancy_drives_upward'] else 'FAIL'}")

    # G4: Negative control
    print("\n[G4] Negative control g'=0 → Q(z) linear...")
    g4 = run_negative_control_no_buoyancy()
    report["G4"] = g4
    print(f"  Exponent = {g4['exponent']:.3f} → {'PASS' if g4['pass'] else 'FAIL'}")

    # G5: LFL wind speed sweep
    print("\n[G5] LFL contour vs wind speed...")
    g5 = run_lfl_wind_speed_sweep(mass_rate=10.0)
    report["G5"] = g5
    print(f"  Lengths = {[round(l, 1) for l in g5['lengths']]}")
    print(f"  Strictly decreasing → {'PASS' if g5['pass'] else 'FAIL'}")

    # G6: Grid convergence
    print("\n[G6] LFL grid convergence...")
    g6 = run_lfl_grid_convergence()
    report["G6"] = g6
    print(f"  Max change = {g6['max_relative_change']:.4f} → {'PASS' if g6['pass'] else 'FAIL'}")

    elapsed = time.time() - t0
    report["elapsed_seconds"] = elapsed

    gates = {"G1": g1["pass_l2"] and g1["pass_peak"],
             "G2": g2["pass"],
             "G3": g3["pass"],
             "G3b": g3b["buoyancy_drives_upward"],
             "G4": g4["pass"],
             "G5": g5["pass"],
             "G6": g6["pass"]}
    report["gates"] = gates
    report["all_pass"] = all(gates.values())
    report["n_pass"] = sum(gates.values())
    report["n_total"] = len(gates)

    print(f"\n{'='*60}")
    print(f"VERIFICATION SUMMARY: {report['n_pass']}/{report['n_total']} gates passed ({elapsed:.1f}s)")
    for gate, passed in gates.items():
        print(f"  {gate}: {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"{'='*60}")

    return report


def make_fig1_advection_diffusion(output_dir: Path, report: dict) -> Path:
    """Figure 1: 1-D advection-diffusion Gaussian vs analytical."""
    g1 = report["G1"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: profile comparison
    ax = axes[0]
    x = np.array(g1["_x"])
    C_num = np.array(g1["_C_numeric"])
    C_exact = np.array(g1["_C_exact"])

    ax.plot(x, C_exact, "k-", lw=2, label="Analytical")
    ax.plot(x, C_num, "r--", lw=1.5, label="LBM D2Q5")
    ax.set_xlabel("x (lattice units)", fontsize=11)
    ax.set_ylabel("C (concentration)", fontsize=11)
    ax.set_title("1-D Advection-Diffusion: Gaussian Pulse", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 256)

    # Right: error
    ax = axes[1]
    error = np.abs(C_num - C_exact) / np.maximum(np.abs(C_exact), 1e-15) * 100
    ax.semilogy(x, error, "b-", lw=1.5)
    ax.axhline(2.0, color="r", ls="--", lw=1, label="2% gate")
    ax.set_xlabel("x (lattice units)", fontsize=11)
    ax.set_ylabel("Relative error (%)", fontsize=11)
    ax.set_title(f"L2 error = {g1['l2_relative_error']*100:.2f}%, Peak = {g1['peak_error_pct']:.2f}%",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 256)
    ax.set_ylim(1e-4, 1e3)

    fig.tight_layout()
    path = output_dir / "fig1_1d_advection_diffusion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_fig2_plume_self_similarity(output_dir: Path, report: dict) -> Path:
    """Figure 2: Plume self-similarity Q(z) ∝ z^(5/3) and negative control."""
    g3 = report["G3"]
    g4 = report["G4"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: buoyant plume
    ax = axes[0]
    z = np.array(g3["heights"])
    Q = np.array(g3["fluxes"])
    z_fit = np.linspace(z.min(), z.max(), 100)

    coeff = (6.0 * 0.083 / 5.0) ** (1.0 / 3.0) * (1.0 ** (1.0 / 3.0))
    Q_theory = coeff * z_fit ** (5.0 / 3.0)

    ax.loglog(z, Q, "bo-", ms=8, lw=2, label=f"Analytical MTT (n={g3['exponent']:.3f})")
    ax.loglog(z_fit, Q_theory, "k--", lw=1.5, label=r"Theory $z^{5/3}$")
    ax.set_xlabel("z (height from source)", fontsize=11)
    ax.set_ylabel("Q(z) (volume flux)", fontsize=11)
    ax.set_title("Buoyant Plume: Q(z) ∝ z^(5/3)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")

    # Right: negative control
    ax = axes[1]
    z = np.array(g4["heights"])
    Q = np.array(g4["fluxes"])
    z_fit = np.linspace(z.min(), z.max(), 100)
    Q_theory = 0.32 * z_fit  # linear

    ax.loglog(z, Q, "rs-", ms=8, lw=2, label=f"Jet (n={g4['exponent']:.3f})")
    ax.loglog(z_fit, Q_theory, "k--", lw=1.5, label=r"Theory $z^{1}$")
    ax.set_xlabel("z (height from source)", fontsize=11)
    ax.set_ylabel("Q(z) (volume flux)", fontsize=11)
    ax.set_title("Negative Control g'=0: Q(z) ∝ z (momentum jet)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = output_dir / "fig2_plume_self_similarity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_fig3_lfl_hazard_zone(output_dir: Path, report: dict) -> Path:
    """Figure 3: LFL hazard zone - wind speed sweep."""
    g5 = report["G5"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: contour length vs wind speed
    ax = axes[0]
    U = np.array([r["wind_speed"] for r in g5["results"]])
    L = np.array([r["contour_length"] for r in g5["results"]])

    ax.semilogy(U, L, "go-", ms=10, lw=2, label="LFL contour length")
    ax.set_xlabel("Wind speed U (m/s)", fontsize=11)
    ax.set_ylabel("LFL contour length (m)", fontsize=11)
    ax.set_title(f"LFL Hazard Zone: {g5['fuel']} (Q={g5['mass_rate']} kg/s)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(U)

    # Right: contour area vs wind speed
    ax = axes[1]
    A = np.array([r["contour_area"] for r in g5["results"]])
    W = np.array([r["contour_width"] for r in g5["results"]])

    ax.semilogy(U, A, "b^-", ms=10, lw=2, label="Contour area")
    ax.semilogy(U, W, "m>--", ms=8, lw=1.5, label="Contour width")
    ax.set_xlabel("Wind speed U (m/s)", fontsize=11)
    ax.set_ylabel("Dimension (m² or m)", fontsize=11)
    ax.set_title("Hazard Zone Dimensions vs Wind Speed", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(U)

    fig.tight_layout()
    path = output_dir / "fig3_lfl_hazard_zone.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="IN65-N03 benchmark runner")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    t0 = time.time()

    # Output directories
    artifact_dir = REPO_ROOT / "runs" / "in65_verification"
    figure_dir = REPO_ROOT / "artifacts" / "in65_figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IN65-N03: TensorLBM Scalar Transport + Boussinesq Verification")
    print("=" * 70)

    # Run all gates
    report = run_all_gates()

    # Generate figures
    print("\nGenerating figures...")
    fig1 = make_fig1_advection_diffusion(figure_dir, report)
    print(f"  {fig1}")
    fig2 = make_fig2_plume_self_similarity(figure_dir, report)
    print(f"  {fig2}")
    fig3 = make_fig3_lfl_hazard_zone(figure_dir, report)
    print(f"  {fig3}")

    # Save report
    report_path = artifact_dir / "in65_verification.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"All gates passed: {report['all_pass']}")

    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
