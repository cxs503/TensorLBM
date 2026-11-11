#!/usr/bin/env python3
"""Benchmark: 3D square duct Poiseuille flow (CASE-POISEUILLE-3D).

Reference: Shah, R. K. & London, A. L. (1978).
"Laminar Flow Forced Convection in Ducts", Academic Press.
- Square duct (aspect ratio = 1): f·Re = 56.908 (Table 49, p. 103)

Usage:
    python bench_square_duct_poiseuille.py [--resolution N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "TensorLBM" / "src"))

from tensorlbm.verification.square_duct_poiseuille import (
    SquareDuctConfig, run_square_duct, shah_london_fRe, square_duct_velocity_profile
)
from tensorlbm.verification.square_duct_poiseuille.solver import compute_fRe

REF_FRE = 56.908
REF_SOURCE = "Shah & London (1978), Laminar Flow Forced Convection in Ducts, Table 49, p. 103"


def run_benchmark(resolution=32, tau=1.0, steps=20000, force=1e-5,
                  output_dir="runs/square_duct_poiseuille"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=== 3D Square Duct Poiseuille Benchmark ===")
    print(f"Resolution: {resolution}x{resolution} (cross-section)")
    print(f"tau={tau}, force={force}, steps={steps}")
    print(f"Reference f·Re = {REF_FRE} ({REF_SOURCE})")
    print()

    cfg = SquareDuctConfig(
        nx=8, ny=resolution, nz=resolution, tau=tau, force=force,
        steps=steps, warmup=steps // 4, sample_interval=100,
        label=f"square_duct_{resolution}",
    )

    print("Running simulation...")
    t0 = time.time()
    result = run_square_duct(cfg)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")
    print(f"Mean velocity: {result['run_info']['u_mean']:.6e}")
    print(f"Max velocity: {result['run_info']['u_max']:.6e}")
    print(f"Computed f·Re: {result['fRe']:.4f}")
    print(f"Reference f·Re: {REF_FRE:.4f}")

    fRe_computed = result["fRe"]
    error_pct = abs(fRe_computed - REF_FRE) / REF_FRE * 100
    passed = error_pct < 5.0
    print(f"Error: {error_pct:.2f}%")
    print(f"PASS (<5%): {'YES' if passed else 'NO'}")

    # Dimensionless numbers
    Re = result["run_info"]["u_mean"] * cfg.hydraulic_diameter / cfg.nu
    Ma = result["run_info"]["u_mean"] / np.sqrt(1/3)

    results = {
        "case": "CASE-POISEUILLE-3D",
        "reference": {"fRe": REF_FRE, "source": REF_SOURCE},
        "simulation": {
            "resolution": resolution, "tau": tau, "force": force, "steps": steps,
            "nu": cfg.nu, "D_h": cfg.hydraulic_diameter,
            "u_mean": result["run_info"]["u_mean"],
            "u_max": result["run_info"]["u_max"],
            "u_max_u_mean_ratio": result["run_info"]["u_max"] / result["run_info"]["u_mean"],
            "fRe_computed": fRe_computed,
            "error_pct": error_pct, "passed": passed,
            "elapsed_s": elapsed, "steps_per_sec": result["run_info"]["steps_per_sec"],
        },
        "dimensionless_numbers": {"Re": Re, "Ma": Ma},
    }

    json_path = output_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    generate_figures(result, cfg, output_path)

    repro_cmd = f"python bench_square_duct_poiseuille.py --resolution {resolution} --tau {tau} --steps {steps} --force {force}"
    with open(output_path / "repro_cmd.txt", "w") as f:
        f.write(repro_cmd + "\n")

    return results


def generate_figures(result, cfg, output_path):
    y_coords = result["y_coords"]
    uy_profile = result["uy_profile"]
    z_coords = result["z_coords"]
    uz_profile = result["uz_profile"]
    u_field = result["u_field"]

    a = float(cfg.ny)
    dpdx = -cfg.force
    mu = cfg.nu

    y_ana = np.linspace(0, cfg.ny - 1, 100)
    y_c = y_ana - (cfg.ny - 1) / 2.0
    u_ana_y = square_duct_velocity_profile(y_c, np.zeros_like(y_c), a, dpdx, mu, n_terms=50)

    z_ana = np.linspace(0, cfg.nz - 1, 100)
    z_c = z_ana - (cfg.nz - 1) / 2.0
    u_ana_z = square_duct_velocity_profile(np.zeros_like(z_c), z_c, a, dpdx, mu, n_terms=50)

    u_max_num = result["run_info"]["u_max"]
    u_max_ana = float(np.max(u_ana_y))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(uy_profile / u_max_num, y_coords / cfg.ny, 'b-o', markersize=3, label='LBM')
    ax.plot(u_ana_y / u_max_ana, y_ana / cfg.ny, 'r--', linewidth=2, label='Analytical')
    ax.set_xlabel('u / u_max')
    ax.set_ylabel('y / H')
    ax.set_title('Velocity Profile (y-direction)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(z_coords / cfg.nz, uz_profile / u_max_num, 'b-o', markersize=3, label='LBM')
    ax.plot(z_ana, u_ana_z / u_max_ana, 'r--', linewidth=2, label='Analytical')
    ax.set_xlabel('z / H')
    ax.set_ylabel('u / u_max')
    ax.set_title('Velocity Profile (z-direction)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    Y, Z = np.meshgrid(np.arange(cfg.ny), np.arange(cfg.nz), indexing='ij')
    cf = ax.contourf(Z / cfg.nz, Y / cfg.ny, u_field / u_max_num, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='u / u_max')
    ax.set_xlabel('z / H')
    ax.set_ylabel('y / H')
    ax.set_title('Velocity Contour (x=0)')
    ax.set_aspect('equal')

    plt.tight_layout()
    fig_path = output_path / "velocity_profiles.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    np.save(output_path / "u_field.npy", u_field)
    np.save(output_path / "uy_profile.npy", uy_profile)
    np.save(output_path / "uz_profile.npy", uz_profile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--force", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="runs/square_duct_poiseuille")
    args = parser.parse_args()

    results = run_benchmark(
        resolution=args.resolution, tau=args.tau,
        steps=args.steps, force=args.force, output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Case: CASE-POISEUILLE-3D")
    print(f"Reference: {REF_SOURCE}")
    print(f"Reference f·Re: {REF_FRE:.4f}")
    print(f"Computed f·Re: {results['simulation']['fRe_computed']:.4f}")
    print(f"Error: {results['simulation']['error_pct']:.2f}%")
    print(f"PASS (<5%): {'YES' if results['simulation']['passed'] else 'NO'}")
    print(f"Reynolds number: {results['dimensionless_numbers']['Re']:.2f}")
    print(f"Mach number: {results['dimensionless_numbers']['Ma']:.4e}")
    print(f"u_max/u_mean: {results['simulation']['u_max_u_mean_ratio']:.4f} (ref: ~2.096)")
    print("=" * 60)

    return 0 if results["simulation"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
