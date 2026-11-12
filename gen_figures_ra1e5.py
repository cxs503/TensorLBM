#!/usr/bin/env python3
"""Generate figures for Ra=1e5 natural convection cavity benchmark.

Produces:
1. Temperature field contour plot
2. Velocity field (quiver + magnitude)
3. Nu convergence history
4. Comparison bar chart (ref vs got for Nu, u_max, v_max)
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "thermal_cavity"))

import torch
import driver

# de Vahl Davis 1983 reference
REF_DVD_1E5 = {
    "nu": 4.519,
    "u_max": 34.72,
    "v_max": 68.60,
}


def load_result():
    result_path = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "thermal_cavity" / "result.json"
    with open(result_path) as f:
        return json.load(f)


def get_ra1e5_cases(data):
    cases = data.get("cases", [])
    return [c for c in cases if abs(c.get("ra", 0) - 1e5) < 1e3]


def run_and_get_fields(nx, ra=1e5, pr=0.71, tau=0.6, max_steps=5000):
    """Run a short simulation and return final fields for visualization."""
    device = torch.device("cpu")
    torch.set_num_threads(32)
    f, g, wall, p = driver.make_fields(nx, ra, pr, tau, device)
    tau_T, g_beta = p["tau_T"], p["g_beta"]

    for step in range(1, max_steps + 1):
        f, g = driver.advance(f, g, wall, tau, tau_T, g_beta)

    rho, ux, uy = driver.macroscopic(f)
    T = g.sum(dim=0)
    return T.numpy(), ux.numpy(), uy.numpy(), p


def plot_temperature_field(T, nx, out_path):
    """Temperature contour plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(T, cmap="RdYlBu_r", origin="lower", aspect="equal",
                   vmin=0, vmax=1, extent=[0, 1, 0, 1])
    ax.set_xlabel("x/H")
    ax.set_ylabel("y/H")
    ax.set_title(f"Temperature Field (Ra=1e5, N={nx})")
    cbar = plt.colorbar(im, ax=ax, label="T")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_velocity_field(ux, uy, nx, out_path):
    """Velocity magnitude contour + quiver."""
    speed = np.sqrt(ux**2 + uy**2)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(speed, cmap="viridis", origin="lower", aspect="equal", extent=[0, 1, 0, 1])
    # Quiver (subsampled)
    skip = max(1, nx // 20)
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nx))
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              ux[::skip, ::skip], uy[::skip, ::skip],
              color="white", alpha=0.6, scale=50, width=0.003)
    ax.set_xlabel("x/H")
    ax.set_ylabel("y/H")
    ax.set_title(f"Velocity Magnitude (Ra=1e5, N={nx})")
    cbar = plt.colorbar(im, ax=ax, label="|u|")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_nu_history(cases, out_path):
    """Nu convergence history for all Ra=1e5 cases."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for case in sorted(cases, key=lambda c: c.get("nx", 0)):
        tag = case.get("tag", "unknown")
        nx = case.get("nx", 0)
        # We don't have full history in result.json, just final values
        # Plot final Nu as horizontal line
        nu = case.get("nu")
        if nu is not None:
            ax.axhline(nu, label=f"N={nx}: Nu={nu:.3f}", alpha=0.7)
    ax.axhline(REF_DVD_1E5["nu"], color="red", linestyle="--", linewidth=2,
               label=f"de Vahl Davis: Nu={REF_DVD_1E5['nu']}")
    ax.set_xlabel("Grid Resolution")
    ax.set_ylabel("Nusselt Number")
    ax.set_title("Ra=1e5: Nusselt Number vs Grid Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_comparison(cases, out_path):
    """Bar chart: ref vs got for Nu, u_max, v_max."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    quantities = [
        ("nu", "Nu", REF_DVD_1E5["nu"]),
        ("u_max_nd", "u_max", REF_DVD_1E5["u_max"]),
        ("v_max_nd", "v_max", REF_DVD_1E5["v_max"]),
    ]

    for ax, (key, label, ref_val) in zip(axes, quantities):
        x = np.arange(len(cases))
        width = 0.35
        vals = [c.get(key, 0) for c in cases]
        labels = [f"N={c.get('nx', '?')}" for c in cases]

        ax.bar(x - width/2, [ref_val] * len(cases), width, label="Reference", alpha=0.7, color="red")
        ax.bar(x + width/2, vals, width, label="TensorLBM", alpha=0.7, color="blue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(label)
        ax.set_title(f"{label} (Ra=1e5)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def main():
    out_dir = Path(__file__).resolve().parents[1] / "figures" / "ra1e5"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_result()
    cases = get_ra1e5_cases(data)

    if not cases:
        print("No Ra=1e5 cases found. Running short simulation for figures...")
        # Run a short simulation for visualization
        T, ux, uy, p = run_and_get_fields(128, max_steps=5000)
        plot_temperature_field(T, 128, str(out_dir / "temperature_ra1e5_n128.png"))
        plot_velocity_field(ux, uy, 128, str(out_dir / "velocity_ra1e5_n128.png"))
        print("Generated field plots from short run.")
        return

    print(f"Found {len(cases)} Ra=1e5 cases: {[c.get('tag') for c in cases]}")

    # Plot comparison
    plot_comparison(cases, str(out_dir / "comparison_ra1e5.png"))
    plot_nu_history(cases, str(out_dir / "nu_history_ra1e5.png"))

    # Try to get fields from the finest available case
    best_case = max(cases, key=lambda c: c.get("nx", 0))
    nx = best_case.get("nx", 128)
    print(f"Generating field plots for N={nx}...")
    T, ux, uy, p = run_and_get_fields(nx, max_steps=5000)
    plot_temperature_field(T, nx, str(out_dir / f"temperature_ra1e5_n{nx}.png"))
    plot_velocity_field(ux, uy, nx, str(out_dir / f"velocity_ra1e5_n{nx}.png"))

    print("Done.")


if __name__ == "__main__":
    main()
