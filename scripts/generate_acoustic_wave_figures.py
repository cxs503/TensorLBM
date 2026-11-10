#!/usr/bin/env python
"""Generate figures for acoustic wave propagation verification.

Produces:
1. 1D waveform comparison (numerical vs analytical)
2. 2D density field snapshots
3. Dispersion relation (Re(ω) vs k)
4. Dissipation curve (Im(ω) vs k²)
"""

from __future__ import annotations

import base64
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.acoustic_wave import (
    CS_LATTICE,
    CS2_LATTICE,
    GaussianPulseConfig,
    SinusoidalWaveConfig,
    run_1d_sinusoidal_wave,
    run_2d_gaussian_pulse,
    run_dissipation_scan,
)


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_1d_waveform_figure() -> tuple[plt.Figure, str]:
    """Generate 1D waveform comparison figure."""
    res = run_1d_sinusoidal_wave(
        nx=512, tau=1.0, n_steps=200,
        wave=SinusoidalWaveConfig(amplitude=1e-4, wavelength=128.0),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(512)
    ax.plot(x, res["rho_final"], "b-", linewidth=1.5, label="LBM numerical", alpha=0.8)
    ax.plot(x, res["rho_analytical"], "r--", linewidth=1.0, label="Analytical (with dissipation)")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("ρ (lattice units)")
    ax.set_title(
        f"1D Sinusoidal Acoustic Wave: τ=1.0, λ=128, t=200\n"
        f"L2 error = {res['l2_error']:.4%}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Zoom inset
    axins = ax.inset_axes([0.65, 0.1, 0.3, 0.35])
    x_zoom = slice(200, 300)
    axins.plot(x[x_zoom], res["rho_final"][x_zoom], "b-", linewidth=1.5)
    axins.plot(x[x_zoom], res["rho_analytical"][x_zoom], "r--", linewidth=1.0)
    axins.set_xlim(200, 300)
    axins.set_ylim(0.99994, 1.00006)
    axins.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="gray")

    return fig, fig_to_base64(fig)


def generate_2d_pulse_figure() -> tuple[plt.Figure, str]:
    """Generate 2D Gaussian pulse density field figure."""
    res = run_2d_gaussian_pulse(
        nx=128, ny=128, tau=1.0, n_steps=100,
        pulse=GaussianPulseConfig(amplitude=1e-4, center=(64, 64), sigma=10.0),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    rho0 = res["rho_history"][0]
    rho_mid = res["rho_history"][len(res["rho_history"]) // 2]
    rho_final = res["rho_history"][-1]

    for ax, rho, title in zip(
        axes, [rho0, rho_mid, rho_final],
        ["t = 0 (initial)", f"t = {res['time_history'][len(res['time_history'])//2]:.0f}", f"t = {res['time_history'][-1]:.0f}"]
    ):
        pert = (rho - 1.0) * 1e4  # Scale perturbation for visibility
        im = ax.imshow(pert, cmap="RdBu_r", origin="lower", vmin=-1.2, vmax=1.2)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="ρ' × 10⁴")

    fig.suptitle("2D Gaussian Acoustic Pulse Propagation (τ=1.0)", fontsize=12)
    plt.tight_layout()

    return fig, fig_to_base64(fig)


def generate_dispersion_figure() -> tuple[plt.Figure, str]:
    """Generate dispersion and dissipation figures."""
    res = run_dissipation_scan(tau=1.0, k_min=0.02, k_max=0.5, n_k=20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Dispersion: Re(ω) vs k
    k = np.array(res["k_values"])
    omega_re = np.array(res["omega_re_acoustic"])
    omega_im = np.array(res["omega_im_acoustic"])

    ax1.plot(k, omega_re, "bo-", markersize=4, label="LBM (von Neumann)")
    ax1.plot(k, CS_LATTICE * k, "r--", linewidth=1.5, label=f"c_s·k (c_s = 1/√3 = {CS_LATTICE:.4f})")
    ax1.set_xlabel("Wave number k")
    ax1.set_ylabel("Re(ω)")
    ax1.set_title("Dispersion Relation: Re(ω) vs k")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Dissipation: -Im(ω) vs k²
    k2 = k ** 2
    nu_theory = CS2_LATTICE * 0.5  # τ = 1.0

    ax2.plot(k2, -omega_im, "bo-", markersize=4, label="LBM (von Neumann)")
    ax2.plot(k2, nu_theory * k2, "r--", linewidth=1.5, label=f"ν_num·k² (ν_num = {nu_theory:.4f})")
    ax2.set_xlabel("k²")
    ax2.set_ylabel("-Im(ω)")
    ax2.set_title(f"Dissipation: -Im(ω) vs k² (R² = {res['r_squared']:.6f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, fig_to_base64(fig)


def main():
    """Generate all figures and save to artifacts/."""
    os.makedirs("artifacts", exist_ok=True)

    print("Generating 1D waveform figure...")
    fig1, b64_1d = generate_1d_waveform_figure()
    fig1.savefig("artifacts/acoustic_wave_1d_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    print("Generating 2D pulse figure...")
    fig2, b64_2d = generate_2d_pulse_figure()
    fig2.savefig("artifacts/acoustic_wave_2d_pulse.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("Generating dispersion/dissipation figure...")
    fig3, b64_disp = generate_dispersion_figure()
    fig3.savefig("artifacts/acoustic_wave_dispersion_dissipation.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print("Figures saved to artifacts/")
    return {
        "1d_waveform": b64_1d,
        "2d_pulse": b64_2d,
        "dispersion": b64_disp,
    }


if __name__ == "__main__":
    main()
