#!/usr/bin/env python
"""Generate figures for the Taylor–Aris dispersion report."""

import json
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "p96w2_n05"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "p96w2_n05"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _encode_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def fig1_diffusion_validation():
    """C1: Pure diffusion σ² = 2Dt validation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, D_target in enumerate([0.02, 0.05, 0.10]):
        data = json.loads((ARTIFACTS_DIR / f"c1_D{D_target:.4f}.json").read_text())
        ax.plot([D_target], [data["D_eff"]], "o", color=colors[i], markersize=12,
                label=f"D={D_target:.2f} → D_eff={data['D_eff']:.4f}")

    # Reference line
    ax.plot([0.015, 0.11], [0.015, 0.11], "k--", alpha=0.5, label="Perfect agreement")
    ax.set_xlabel("Input diffusivity D")
    ax.set_ylabel("Measured effective diffusivity D_eff")
    ax.set_title("C1: Pure Diffusion Validation (σ² = 2Dt)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    png = _encode_png(fig)
    plt.close(fig)
    return {"name": "c1_diffusion_validation", "caption": "Pure diffusion: measured D_eff vs input D for three diffusivity values. All within 1% error.", "png_base64": png}


def fig2_numerical_diffusion():
    """C2: Numerical diffusion calibration."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = json.loads((ARTIFACTS_DIR / "c2_calibration.json").read_text())["results"]
    tau_vals = sorted(set(r["tau_s"] for r in data))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tau_vals)))

    for tau, color in zip(tau_vals, colors):
        pts = [r for r in data if r["tau_s"] == tau]
        ux = [r["ux"] for r in pts]
        dnorm = [r["D_num/(u*dx)"] for r in pts]
        ax.plot(ux, dnorm, "o-", color=color, label=f"τ_s={tau}")

    ax.set_xlabel("Velocity u (l.u.)")
    ax.set_ylabel("D_num / (u·Δx)")
    ax.set_title("C2: Numerical Diffusion Calibration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    png = _encode_png(fig)
    plt.close(fig)
    return {"name": "c2_numerical_diffusion", "caption": "Normalized numerical diffusion D_num/(u·Δx) vs velocity for different τ_s values.", "png_base64": png}


def fig3_taylor_aris():
    """C3: Taylor–Aris dispersion validation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = json.loads((ARTIFACTS_DIR / "c3_Pe10.json").read_text())
    bars = ax.bar(["Measured", "Analytical"],
                  [data["D_eff_measured"], data["D_eff_analytical"]],
                  color=["#1f77b4", "#ff7f0e"], width=0.5)
    ax.set_ylabel("Effective diffusivity D_eff")
    ax.set_title(f"C3: Taylor–Aris Dispersion (Pe={data['Pe_target']})\n"
                 f"D_eff_meas={data['D_eff_measured']:.4f}, D_eff_analytical={data['D_eff_analytical']:.4f}\n"
                 f"Error: {data['relative_error']:.2%}")
    ax.grid(True, alpha=0.3, axis="y")

    png = _encode_png(fig)
    plt.close(fig)
    return {"name": "c3_taylor_aris", "caption": f"Taylor–Aris dispersion at Pe={data['Pe_target']}: measured vs analytical D_eff. Error: {data['relative_error']:.2%}.", "png_base64": png}


def fig4_early_shear():
    """C4: Early-time shear enhancement."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = json.loads((ARTIFACTS_DIR / "c4_early_shear.json").read_text())
    ax.bar(["Molecular D", "Early D_eff"],
           [data["D"], data["D_early"]],
           color=["#2ca02c", "#d62728"], width=0.5)
    ax.set_ylabel("Diffusivity")
    ax.set_title(f"C4: Early-Time Shear Enhancement\n"
                 f"Enhancement factor: {data['enhancement_factor']:.2f}×")
    ax.grid(True, alpha=0.3, axis="y")

    png = _encode_png(fig)
    plt.close(fig)
    return {"name": "c4_early_shear", "caption": f"Early-time effective diffusivity exceeds molecular D by {data['enhancement_factor']:.2f}× due to shear.", "png_base64": png}


def main():
    figures = [
        fig1_diffusion_validation(),
        fig2_numerical_diffusion(),
        fig3_taylor_aris(),
        fig4_early_shear(),
    ]

    # Save figures as PNG files
    for fig_data in figures:
        png_path = FIGURES_DIR / f"{fig_data['name']}.png"
        png_bytes = base64.b64decode(fig_data["png_base64"])
        png_path.write_bytes(png_bytes)
        print(f"Saved {png_path}")

    # Save figure metadata
    meta_path = FIGURES_DIR / "figures.json"
    meta_path.write_text(json.dumps(figures, indent=2))
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
