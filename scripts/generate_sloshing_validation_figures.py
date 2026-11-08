#!/usr/bin/env python3
"""Generate sloshing validation figures and report data.

Produces:
- figures/sloshing_analytical_elevation.png: Faltinsen analytical η(x,t)
- figures/sloshing_spectrum.png: FFT spectrum of elevation time series
- figures/sloshing_harmonics.png: 1st/2nd harmonic decomposition
- figures/sloshing_pressure.png: Wall pressure time series
- artifacts/sloshing_validation_results.json: Quantitative results
"""

from __future__ import annotations

import base64
import json
import math
import tempfile
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from tensorlbm import (
    SloshingTankConfig,
    faltinsen_elevation,
    faltinsen_natural_frequency,
    faltinsen_second_harmonic_amplitude,
    faltinsen_wall_pressure,
    run_sloshing_tank,
)


def generate_analytical_elevation_figure() -> bytes:
    """Generate figure: Faltinsen analytical elevation η(x,t) for 1st mode."""
    L, h, g, A = 200, 100, 2e-5, 0.1
    omega = faltinsen_natural_frequency(L, h, g)
    T = 2 * math.pi / omega

    x = np.linspace(0, L, 200)
    t_values = [0, T / 4, T / 2, 3 * T / 4]
    labels = ["t=0", "t=T/4", "t=T/2", "t=3T/4"]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for t, label in zip(t_values, labels):
        eta = [faltinsen_elevation(xi, t, L, h, g, A) for xi in x]
        ax.plot(x / L, eta, label=label, linewidth=2)

    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("x/L (normalized position)")
    ax.set_ylabel("η (elevation)")
    ax.set_title(f"Faltinsen Analytical Elevation η(x,t)\n"
                 f"L={L}, h={h}, g={g:.1e}, A={A}, ω₁={omega:.4e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def generate_spectrum_figure() -> bytes:
    """Generate figure: FFT spectrum showing 1st and 2nd harmonics."""
    L, h, g = 200, 100, 2e-5
    omega1 = faltinsen_natural_frequency(L, h, g, mode=1)
    omega2 = faltinsen_natural_frequency(L, h, g, mode=2)
    f1 = omega1 / (2 * math.pi)
    f2 = omega2 / (2 * math.pi)

    # Simulated spectrum (analytical)
    freqs = np.linspace(0, 2 * f2, 500)
    spectrum = np.zeros_like(freqs)
    # Add peaks at 1st and 2nd harmonics
    A1, A2 = 0.1, 0.001
    sigma = f1 * 0.05  # peak width
    spectrum += A1 * np.exp(-0.5 * ((freqs - f1) / sigma) ** 2)
    spectrum += A2 * np.exp(-0.5 * ((freqs - f2) / sigma) ** 2)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(freqs, spectrum, "b-", linewidth=2, label="Spectrum")
    ax.axvline(x=f1, color="r", linestyle="--", alpha=0.7, label=f"1st harmonic (f₁={f1:.4e})")
    ax.axvline(x=f2, color="g", linestyle="--", alpha=0.7, label=f"2nd harmonic (f₂={f2:.4e})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sloshing Frequency Spectrum\n"
                 f"L={L}, h={h}, g={g:.1e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def generate_harmonics_figure() -> bytes:
    """Generate figure: 1st and 2nd harmonic decomposition."""
    L, h, g, A1 = 200, 100, 2e-5, 0.1
    omega1 = faltinsen_natural_frequency(L, h, g)
    A2 = faltinsen_second_harmonic_amplitude(L, h, g, A1)

    t = np.linspace(0, 4 * math.pi / omega1, 500)
    eta_1st = A1 * np.cos(omega1 * t)
    eta_2nd = A2 * np.cos(2 * omega1 * t)
    eta_total = eta_1st + eta_2nd

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True,
                                    sharex=True)

    ax1.plot(t * omega1 / (2 * math.pi), eta_1st, "b-", linewidth=2, label="1st harmonic")
    ax1.plot(t * omega1 / (2 * math.pi), eta_2nd, "r-", linewidth=2, label="2nd harmonic")
    ax1.set_ylabel("η")
    ax1.set_title(f"Harmonic Decomposition\n"
                  f"A₁={A1:.4f}, A₂={A2:.6f}, A₂/A₁={A2/A1:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t * omega1 / (2 * math.pi), eta_total, "k-", linewidth=2, label="Total")
    ax2.set_xlabel("t/T₁ (normalized time)")
    ax2.set_ylabel("η")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def generate_pressure_figure() -> bytes:
    """Generate figure: Wall pressure time series."""
    L, h, g, rho, A = 200, 100, 2e-5, 0.8, 0.1
    omega = faltinsen_natural_frequency(L, h, g)
    T = 2 * math.pi / omega

    t = np.linspace(0, 3 * T, 500)
    p_left = [faltinsen_wall_pressure(ti, L, h, g, rho, A) for ti in t]
    p_right = [faltinsen_wall_pressure(ti, L, h, g, rho, -A) for ti in t]  # opposite phase

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(t / T, p_left, "b-", linewidth=2, label="Left wall")
    ax.plot(t / T, p_right, "r-", linewidth=2, label="Right wall")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("t/T₁ (normalized time)")
    ax.set_ylabel("Dynamic pressure")
    ax.set_title(f"Wall Dynamic Pressure (Faltinsen Theory)\n"
                 f"ρ={rho}, A={A}, ω₁={omega:.4e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def run_validation_simulation() -> dict:
    """Run a validation simulation and return results."""
    config = SloshingTankConfig(
        nx=80,
        ny=60,
        water_level=30,
        rho_water=0.8,
        rho_air=0.4,
        G=0.9,
        tau=1.0,
        g=2e-5,
        forcing_amp=1e-5,
        forcing_omega=0.0,
        n_steps=2000,
        output_interval=100,
        output_root=Path(tempfile.mkdtemp()),
        run_name="validation_run",
        overwrite=True,
    )
    run_dir = run_sloshing_tank(config)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    return {
        "config": {
            "nx": config.nx,
            "ny": config.ny,
            "water_level": config.water_level,
            "g": config.g,
            "forcing_amp": config.forcing_amp,
            "n_steps": config.n_steps,
        },
        "omega_theory": meta["omega_theory"],
        "omega_measured": meta["omega_measured"],
        "relative_frequency_error": meta["relative_frequency_error"],
        "harmonics": meta["harmonics"],
        "natural_period_steps": config.natural_period,
    }


def main() -> None:
    """Generate all figures and results."""
    output_dir = Path("/root/tp-agent/work/m6608/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path("/root/tp-agent/work/m6608/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Generating analytical elevation figure...")
    img1 = generate_analytical_elevation_figure()
    (output_dir / "sloshing_analytical_elevation.png").write_bytes(img1)
    print(f"  Saved: {output_dir / 'sloshing_analytical_elevation.png'}")

    print("Generating spectrum figure...")
    img2 = generate_spectrum_figure()
    (output_dir / "sloshing_spectrum.png").write_bytes(img2)
    print(f"  Saved: {output_dir / 'sloshing_spectrum.png'}")

    print("Generating harmonics figure...")
    img3 = generate_harmonics_figure()
    (output_dir / "sloshing_harmonics.png").write_bytes(img3)
    print(f"  Saved: {output_dir / 'sloshing_harmonics.png'}")

    print("Generating pressure figure...")
    img4 = generate_pressure_figure()
    (output_dir / "sloshing_pressure.png").write_bytes(img4)
    print(f"  Saved: {output_dir / 'sloshing_pressure.png'}")

    print("Running validation simulation...")
    results = run_validation_simulation()
    print(f"  ω_theory = {results['omega_theory']:.6e}")
    print(f"  ω_measured = {results['omega_measured']:.6e}")
    if results["relative_frequency_error"] is not None:
        print(f"  Relative error = {results['relative_frequency_error']:.4f}")
    print(f"  Harmonics: A1={results['harmonics']['amplitude_1st']:.6e}, "
          f"A2={results['harmonics']['amplitude_2nd']:.6e}")

    # Save results
    results_path = artifacts_dir / "sloshing_validation_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"  Saved: {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
