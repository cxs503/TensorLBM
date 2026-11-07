#!/usr/bin/env python3
"""FWH acoustic validation benchmark: Curle dipole, Doppler, airfoil noise.

Generates figures and result.json for the IN41-N07 validation campaign.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.acoustics_validation import (
    curle_analytical_amplitude,
    CurleDipoleConfig,
    DopplerConfig,
    AirfoilNoiseConfig,
    validate_curle_dipole,
    validate_doppler_moving_source,
    validate_airfoil_noise,
    build_curle_surface,
    curle_analytical_pressure,
    build_airfoil_surface,
)
from tensorlbm.acoustics import compute_fwh_far_field, oaspl, AcousticObserver

OUTPUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_curle_dipole():
    """Generate Curle dipole comparison figures."""
    cfg = CurleDipoleConfig()
    r_obs = cfg.r_obs
    frequencies = [250.0, 500.0, 1000.0]
    angles = [0.0, 45.0, 60.0]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Curle Dipole: FWH vs Analytical Solution", fontsize=14, fontweight="bold")

    for fi, freq_hz in enumerate(frequencies):
        case_cfg = CurleDipoleConfig(freq_hz=freq_hz, r_obs=r_obs)
        surface = build_curle_surface(case_cfg)

        observers = [
            AcousticObserver(x=r_obs * math.cos(math.radians(th)),
                           y=r_obs * math.sin(math.radians(th)),
                           label=f"θ={th:.0f}°")
            for th in angles
        ]

        p_prime_sim, time_list = compute_fwh_far_field(surface, observers)

        for ai, (theta_deg, obs) in enumerate(zip(angles, observers)):
            ax = axes[fi, ai]
            theta = math.radians(theta_deg)

            # Analytical pressure
            p_ana = np.array([curle_analytical_pressure(r_obs, theta, t, case_cfg) for t in time_list])

            # Simulated pressure
            p_sim = p_prime_sim[ai].numpy()

            # Skip transient
            arrival_time = r_obs / cfg.c0
            skip = int(math.ceil(arrival_time / cfg.dt)) + 10

            t_plot = np.array(time_list[skip:]) * 1000  # ms
            p_sim_plot = p_sim[skip:]
            p_ana_plot = p_ana[skip:]

            ax.plot(t_plot, p_ana_plot, "b-", linewidth=1.5, label="Analytical (Curle)", alpha=0.8)
            ax.plot(t_plot, p_sim_plot, "r--", linewidth=1.0, label="FWH Integration", alpha=0.8)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("p\' (Pa)")
            ax.set_title(f"f={freq_hz:.0f} Hz, {obs.label}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "curle_dipole_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # Directivity plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    freq_hz = 1000.0
    case_cfg = CurleDipoleConfig(freq_hz=freq_hz, r_obs=r_obs)
    surface = build_curle_surface(case_cfg)

    theta_range = np.linspace(0, 2 * math.pi, 100)
    amp_ana = [curle_analytical_amplitude(r_obs, th, case_cfg) for th in theta_range]

    # Polar plot
    ax2 = plt.subplot(111, projection="polar")
    ax2.plot(theta_range, amp_ana, "b-", linewidth=2, label="Analytical")
    ax2.set_title(f"Curle Dipole Directivity (f={freq_hz:.0f} Hz)", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True)

    fig2_path = FIG_DIR / "curle_dipole_directivity.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fig2_path}")


def plot_doppler():
    """Generate Doppler shift comparison figure."""
    cfg = DopplerConfig()
    angles = np.linspace(0, 180, 100)

    f_ana = [cfg.f_src / (1.0 - cfg.M * math.cos(math.radians(th))) for th in angles]

    # Simulated points
    result = validate_doppler_moving_source()
    sim_angles = [obs["theta_deg"] for obs in result["observers"]]
    sim_freqs = result["f_sim"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(angles, f_ana, "b-", linewidth=2, label="Analytical: f_src/(1-M·cosθ)")
    ax.plot(sim_angles, sim_freqs, "ro", markersize=10, label="FWH Retarded-Time (simulated)")
    ax.axhline(y=cfg.f_src, color="gray", linestyle="--", alpha=0.5, label=f"Source frequency ({cfg.f_src:.0f} Hz)")
    ax.set_xlabel("Observer Angle θ (degrees)")
    ax.set_ylabel("Observed Frequency (Hz)")
    ax.set_title(f"Doppler Frequency Shift (M={cfg.M:.2f}, f_src={cfg.f_src:.0f} Hz)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = FIG_DIR / "doppler_frequency_shift.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def plot_airfoil_noise():
    """Generate airfoil noise OASPL comparison figure."""
    cfg = AirfoilNoiseConfig()
    surface = build_airfoil_surface(cfg)

    observers = [AcousticObserver(x=cfg.chord, y=cfg.r_obs, label="θ=90°")]
    p_prime, time_list = compute_fwh_far_field(surface, observers)

    # Time series plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    t_ms = np.array(time_list) * 1000
    p_sim = p_prime[0].numpy()

    ax1.plot(t_ms, p_sim, "b-", linewidth=0.5, alpha=0.7)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("p\' (Pa)")
    ax1.set_title("NACA0012 Trailing-Edge Noise: FWH Far-Field Pressure Time Series", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # SPL spectrum
    from tensorlbm.acoustics import compute_spl_spectrum
    spl, freqs = compute_spl_spectrum(p_prime, dt=cfg.dt)

    ax2.semilogx(freqs, spl[0].numpy(), "b-", linewidth=1.0)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("SPL (dB re 20 µPa)")
    ax2.set_title("SPL Spectrum", fontsize=12)
    ax2.grid(True, alpha=0.3, which="both")

    # Add OASPL annotation
    oaspl_val = oaspl(p_prime, dt=cfg.dt)[0]
    ax2.axhline(y=oaspl_val, color="r", linestyle="--", alpha=0.7, label=f"OASPL = {oaspl_val:.1f} dB")
    ax2.legend()

    plt.tight_layout()
    fig_path = FIG_DIR / "airfoil_noise_timeseries_spectrum.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def generate_results_json():
    """Generate result.json with all validation data."""
    curle = validate_curle_dipole()
    doppler = validate_doppler_moving_source()
    airfoil = validate_airfoil_noise()

    # Convert torch tensors to lists for JSON serialization
    curle_cases = []
    for case in curle["cases"]:
        case_data = {
            "freq_hz": case["freq_hz"],
            "observers": [],
        }
        for obs in case["observers"]:
            case_data["observers"].append({
                "label": obs["label"],
                "amplitude_sim": obs["amplitude_sim"],
                "amplitude_ana": obs["amplitude_ana"],
                "error_pct": obs["error_pct"],
                "pass": obs["pass"],
                "ka": obs["ka"],
                "r_over_lambda": obs["r_over_lambda"],
            })
        curle_cases.append(case_data)

    result = {
        "campaign": "IN41-N07",
        "title": "TensorLBM FWH Acoustic Far-Field Validation",
        "date": "2026-11-07",
        "curle_dipole": {
            "description": "Oscillating cylinder acoustic radiation vs Curle analytical solution",
            "config": {
                "cylinder_radius_m": 0.01,
                "velocity_amplitude_m_s": 0.1,
                "fluid_density_kg_m3": 1.225,
                "sound_speed_m_s": 343.0,
                "observer_distance_m": 1.5,
                "n_source_points": 128,
            },
            "cases": curle_cases,
            "pass_3pct": curle["pass_3pct"],
            "n_observers": curle["n_observers"],
            "n_frequencies": curle["n_frequencies"],
        },
        "doppler": {
            "description": "Moving monopole Doppler frequency shift validation",
            "config": {
                "source_frequency_hz": doppler["f_src"],
                "source_velocity_m_s": doppler["M"] * 343.0,
                "mach_number": doppler["M"],
                "sound_speed_m_s": 343.0,
            },
            "observers": doppler["observers"],
            "f_sim": doppler["f_sim"],
            "f_ana": doppler["f_ana"],
            "error_pct": doppler["error_pct"],
            "pass_1pct": doppler["pass_1pct"],
        },
        "airfoil_noise": {
            "description": "NACA0012 trailing-edge noise OASPL vs BPM reference",
            "config": {
                "chord_m": airfoil["chord"],
                "free_stream_velocity_m_s": 55.0,
                "reynolds_number": airfoil["Re"],
                "mach_number": airfoil["M"],
                "observer_distance_m": airfoil["r_obs"],
            },
            "oaspl_sim_db": airfoil["oaspl_sim"],
            "oaspl_ref_db": airfoil["oaspl_ref"],
            "error_db": airfoil["error_db"],
            "pass_5db": airfoil["pass_5db"],
        },
        "summary": {
            "curle_pass_3pct": curle["pass_3pct"],
            "doppler_pass_1pct": doppler["pass_1pct"],
            "airfoil_pass_5db": airfoil["pass_5db"],
            "all_pass": curle["pass_3pct"] and doppler["pass_1pct"] and airfoil["pass_5db"],
        },
        "figures": [
            "figures/curle_dipole_comparison.png",
            "figures/curle_dipole_directivity.png",
            "figures/doppler_frequency_shift.png",
            "figures/airfoil_noise_timeseries_spectrum.png",
        ],
    }

    result_path = OUTPUT_DIR / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {result_path}")
    return result


if __name__ == "__main__":
    print("=== FWH Acoustic Validation Benchmark ===")
    print()
    print("Generating Curle dipole figures...")
    plot_curle_dipole()
    print()
    print("Generating Doppler figures...")
    plot_doppler()
    print()
    print("Generating airfoil noise figures...")
    plot_airfoil_noise()
    print()
    print("Generating result.json...")
    result = generate_results_json()
    print()
    print("=== Summary ===")
    print(f"  Curle dipole: {'PASS' if result['summary']['curle_pass_3pct'] else 'FAIL'}")
    print(f"  Doppler: {'PASS' if result['summary']['doppler_pass_1pct'] else 'FAIL'}")
    print(f"  Airfoil noise: {'PASS' if result['summary']['airfoil_pass_5db'] else 'FAIL'}")
    print(f"  ALL: {'PASS' if result['summary']['all_pass'] else 'FAIL'}")
