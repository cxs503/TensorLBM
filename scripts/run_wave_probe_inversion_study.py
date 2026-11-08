#!/usr/bin/env python3
"""Run the full wave probe inversion identifiability study.

Produces:
1. Identifiability scan over (Δx/λ, s/λ, L/λ)
2. Error maps and minimum protocol envelope
3. LBM hold-out test
4. Paper-quality report with figures

Usage:
    PYTHONPATH=src python scripts/run_wave_probe_inversion_study.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tensorlbm.wave_probe_inversion.synthetic_tank import (
    SyntheticWaveConfig,
    SyntheticWaveTank,
)
from tensorlbm.wave_probe_inversion.inversion import (
    MultiProbeLeastSquares,
    ThreePointGodaInversion,
    fit_harmonic,
)
from tensorlbm.wave_probe_inversion.identifiability import IdentifiabilityScan


def run_identifiability_scan():
    """Run the full identifiability scan."""
    print("=" * 60)
    print("IDENTIFIABILITY SCAN")
    print("=" * 60)

    scan = IdentifiabilityScan(method="multi_probe_ls")

    # Scan parameters
    dx_values = [1/40, 1/30, 1/20, 1/15, 1/10, 1/8]
    s_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
    L_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    print(f"Scanning {len(dx_values)} × {len(s_values)} × {len(L_values)} = "
          f"{len(dx_values)*len(s_values)*len(L_values)} configurations...")

    t0 = time.time()
    result = scan.run_scan(
        dx_values=dx_values,
        s_values=s_values,
        L_values=L_values,
        R_star=0.1,
        alpha_star=0.001,
        n_probes=5,
    )
    elapsed = time.time() - t0

    print(f"Scan completed in {elapsed:.1f}s")
    print(f"Total points: {len(result.points)}")

    if result.min_protocol:
        print(f"\nMinimum protocol achieving targets:")
        print(f"  Δx/λ ≤ {result.min_protocol['dx_over_lambda']:.4f}")
        print(f"  s/λ ∈ [{result.min_protocol['s_over_lambda']:.2f}]")
        print(f"  L/λ ≥ {result.min_protocol['L_over_lambda']:.1f}")
        print(f"  R_error = {result.min_protocol['R_error']:.4f}")
        print(f"  kr_error = {result.min_protocol['kr_error']:.4e}")
        print(f"  alpha_error = {result.min_protocol['alpha_error']:.4f}")
        print(f"  c_error = {result.min_protocol['c_error']:.4e}")
    else:
        print("\nNo protocol achieved all targets!")

    return result


def generate_error_maps(scan_result, output_dir):
    """Generate error map figures."""
    print("\nGenerating error maps...")

    os.makedirs(output_dir, exist_ok=True)

    # Extract data into arrays
    points = scan_result.points

    # Group by (s/λ, L/λ) for each Δx/λ
    dx_values = sorted(set(p.dx_over_lambda for p in points))
    s_values = sorted(set(p.s_over_lambda for p in points))
    L_values = sorted(set(p.L_over_lambda for p in points))

    # For each error metric, create a heatmap
    metrics = [
        ("R_error", "Reflection Error |R̂ - R*|"),
        ("kr_error", "Wavenumber Error |k̂_r - k*| / k*"),
        ("alpha_error", "Attenuation Error |α̂ - α*| / α*"),
        ("c_error", "Phase Speed Error |ĉ - c*| / c*"),
    ]

    for metric_name, metric_label in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{metric_label}\n(R*=0.1, α*=0.001, 5 probes)", fontsize=14)

        for idx, dx in enumerate(dx_values[:6]):  # Show first 6 dx values
            ax = axes[idx // 3, idx % 3]

            # Build matrix
            data = np.zeros((len(s_values), len(L_values)))
            for p in points:
                if p.dx_over_lambda == dx:
                    i = s_values.index(p.s_over_lambda)
                    j = L_values.index(p.L_over_lambda)
                    data[i, j] = getattr(p, metric_name)

            # Plot heatmap
            im = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                norm=mcolors.LogNorm(vmin=max(1e-6, data[data > 0].min()),
                                     vmax=max(data.max(), 1e-2))
            )
            ax.set_title(f"Δx/λ = 1/{int(1/dx)}")
            ax.set_xlabel("L/λ")
            ax.set_ylabel("s/λ")
            ax.set_xticks(range(len(L_values)))
            ax.set_xticklabels([f"{L:.1f}" for L in L_values], rotation=45)
            ax.set_yticks(range(len(s_values)))
            ax.set_yticklabels([f"{s:.2f}" for s in s_values])

            # Add target contour
            if metric_name == "R_error":
                target = 0.03
            elif metric_name == "kr_error":
                target = 0.01
            elif metric_name == "alpha_error":
                target = 0.10
            else:
                target = 0.03

            # Mark passing cells
            for i in range(len(s_values)):
                for j in range(len(L_values)):
                    if data[i, j] < target:
                        ax.plot(j, i, "g+", markersize=8, markeredgewidth=2)

        plt.colorbar(im, ax=axes, label=metric_label, shrink=0.8)
        plt.tight_layout()

        filename = f"error_map_{metric_name}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}")

    # Summary plot: passing fraction vs L/λ for different s/λ
    fig, ax = plt.subplots(figsize=(10, 6))

    for s in [0.2, 0.3, 0.4, 0.5, 0.6]:
        passing_frac = []
        for L in L_values:
            n_pass = 0
            n_total = 0
            for p in points:
                if p.s_over_lambda == s and p.L_over_lambda == L:
                    n_total += 1
                    if (p.R_error < 0.03 and p.kr_error < 0.01 and
                        p.alpha_error < 0.10 and p.c_error < 0.03):
                        n_pass += 1
            passing_frac.append(n_pass / n_total if n_total > 0 else 0)

        ax.plot(L_values, passing_frac, "o-", label=f"s/λ = {s:.2f}")

    ax.set_xlabel("L/λ (propagation length)")
    ax.set_ylabel("Fraction of Δx/λ values passing all targets")
    ax.set_title("Protocol Envelope: Passing Fraction vs Propagation Length\n"
                 "(Targets: |ΔR|<0.03, |Δk/k|<1%, |Δα/α|<10%, |Δc/c|<3%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "protocol_envelope.png"), dpi=150)
    plt.close()
    print("  Saved protocol_envelope.png")


def run_analytical_self_check():
    """Run analytical self-check (criterion a)."""
    print("\n" + "=" * 60)
    print("ANALYTICAL SELF-CHECK (criterion a)")
    print("=" * 60)

    cfg = SyntheticWaveConfig(
        wavelength=100.0,
        amplitude=1.0,
        depth=50.0,
        gravity=9.81,
        R_star=0.1,
        alpha_star=0.0,
        n_harmonics=1,
        n_periods=20.0,
        samples_per_period=40,
        warm_periods=6.0,
    )
    tank = SyntheticWaveTank(cfg)

    x_probes = [50.0, 70.0, 90.0, 110.0, 130.0]
    t, etas = tank.synthesize_probes(x_probes, 0.0)
    c_probes = np.array([fit_harmonic(t, eta, cfg.omega) for eta in etas])

    inv = MultiProbeLeastSquares(cfg.omega, cfg.k_r, 0.0)
    result = inv.invert(np.array(x_probes), c_probes)

    R_error = abs(result.R_hat - cfg.R_star)
    kr_error = abs(result.k_r_hat - cfg.k_r) / cfg.k_r

    print(f"True R* = {cfg.R_star}, R̂ = {result.R_hat:.6f}, |ΔR| = {R_error:.2e}")
    print(f"True k* = {cfg.k_r:.6f}, k̂ = {result.k_r_hat:.6f}, |Δk/k| = {kr_error:.2e}")
    print(f"R_error < 1e-3: {'PASS' if R_error < 1e-3 else 'FAIL'}")
    print(f"kr_error < 1e-4: {'PASS' if kr_error < 1e-4 else 'FAIL'}")

    return {
        "R_true": cfg.R_star,
        "R_hat": result.R_hat,
        "R_error": R_error,
        "k_true": cfg.k_r,
        "k_hat": result.k_r_hat,
        "kr_error": kr_error,
        "R_pass": bool(R_error < 1e-3),
        "kr_pass": bool(kr_error < 1e-4),
    }


def run_identifiability_law_check():
    """Run identifiability law check (criterion b)."""
    print("\n" + "=" * 60)
    print("IDENTIFIABILITY LAW CHECK (criterion b)")
    print("=" * 60)

    scan = IdentifiabilityScan(method="multi_probe_ls")

    # Test with the recommended protocol
    point = scan.run_single(
        dx_over_lambda=1/20,
        s_over_lambda=0.5,
        L_over_lambda=2.0,
        R_star=0.1,
        alpha_star=0.001,
        n_probes=5,
    )

    print(f"Protocol: Δx/λ=1/20, s/λ=0.5, L/λ=2.0")
    print(f"R_error = {point.R_error:.4f} (target < 0.03): {'PASS' if point.R_error < 0.03 else 'FAIL'}")
    print(f"kr_error = {point.kr_error:.4e} (target < 0.01): {'PASS' if point.kr_error < 0.01 else 'FAIL'}")
    print(f"alpha_error = {point.alpha_error:.4f} (target < 0.10): {'PASS' if point.alpha_error < 0.10 else 'FAIL'}")
    print(f"c_error = {point.c_error:.4e} (target < 0.03): {'PASS' if point.c_error < 0.03 else 'FAIL'}")

    return {
        "protocol": {"dx_over_lambda": 1/20, "s_over_lambda": 0.5, "L_over_lambda": 2.0},
        "R_error": point.R_error,
        "kr_error": point.kr_error,
        "alpha_error": point.alpha_error,
        "c_error": point.c_error,
        "all_pass": bool((point.R_error < 0.03 and point.kr_error < 0.01 and
                     point.alpha_error < 0.10 and point.c_error < 0.03)),
    }


def generate_waveform_figure(output_dir):
    """Generate waveform comparison figure."""
    print("\nGenerating waveform figure...")

    os.makedirs(output_dir, exist_ok=True)

    cfg = SyntheticWaveConfig(
        wavelength=100.0,
        amplitude=1.0,
        depth=50.0,
        gravity=9.81,
        R_star=0.1,
        alpha_star=0.001,
        n_harmonics=2,
        n_periods=10.0,
        samples_per_period=40,
        warm_periods=6.0,
    )
    tank = SyntheticWaveTank(cfg)

    x_probes = [50.0, 100.0, 150.0]
    t, etas = tank.synthesize_probes(x_probes, 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Synthetic Probe Time Series\n(R*=0.1, α*=0.001, Stokes 2nd order)",
                 fontsize=14)

    for i, (x, eta) in enumerate(zip(x_probes, etas)):
        ax = axes[i]
        ax.plot(t / cfg.period, eta, "b-", linewidth=0.8)
        ax.set_ylabel("η (m)")
        ax.set_title(f"Probe at x = {x:.0f} (x/λ = {x/cfg.wavelength:.2f})")
        ax.grid(True, alpha=0.3)

        # Mark measurement window
        warm_steps = int(cfg.warm_periods * cfg.samples_per_period)
        ax.axvline(t[warm_steps] / cfg.period, color="r", linestyle="--",
                   alpha=0.5, label="Measurement start")

    axes[-1].set_xlabel("t / T")
    axes[0].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "synthetic_waveforms.png"), dpi=150)
    plt.close()
    print("  Saved synthetic_waveforms.png")


def main():
    """Run the full study."""
    output_dir = os.path.join(ROOT, "artifacts", "wave_probe_inversion")
    os.makedirs(output_dir, exist_ok=True)

    print("WAVE PROBE INVERSION IDENTIFIABILITY STUDY")
    print("=" * 60)

    # 1. Analytical self-check
    self_check = run_analytical_self_check()

    # 2. Identifiability law check
    law_check = run_identifiability_law_check()

    # 3. Full identifiability scan
    scan_result = run_identifiability_scan()

    # 4. Generate figures
    generate_waveform_figure(output_dir)
    generate_error_maps(scan_result, output_dir)

    # 5. Save results
    results = {
        "analytical_self_check": self_check,
        "identifiability_law": law_check,
        "scan_summary": {
            "n_points": len(scan_result.points),
            "min_protocol": scan_result.min_protocol,
            "targets": {
                "R_error": scan_result.target_R_error,
                "kr_error": scan_result.target_kr_error,
                "alpha_error": scan_result.target_alpha_error,
                "c_error": scan_result.target_c_error,
            },
        },
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Analytical self-check: R_error={self_check['R_error']:.2e}, "
          f"kr_error={self_check['kr_error']:.2e}")
    print(f"  → R < 1e-3: {'PASS' if self_check['R_pass'] else 'FAIL'}")
    print(f"  → k < 1e-4: {'PASS' if self_check['kr_pass'] else 'FAIL'}")
    print(f"Identifiability law: all targets met: {'PASS' if law_check['all_pass'] else 'FAIL'}")
    if scan_result.min_protocol:
        mp = scan_result.min_protocol
        print(f"Minimum protocol: Δx/λ≤{mp['dx_over_lambda']:.4f}, "
              f"s/λ={mp['s_over_lambda']:.2f}, L/λ≥{mp['L_over_lambda']:.1f}")

    print("\nFigures saved to:", output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
