#!/usr/bin/env python3
"""Generate figures for wave tank benchmark report."""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tensorlbm.wave_tank_2d import (
    StokesWaveParams,
    stokes_second_order_surface,
    compute_relaxation_weight,
    two_point_reflection_coefficient,
    harmonic_analysis,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_stokes_surface():
    """Figure 1: Stokes 2nd order surface elevation profiles."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, kh in enumerate([0.7, 1.5, 3.0]):
        ax = axes[idx]
        wp = StokesWaveParams(amplitude=1.0, wavenumber=kh/50.0, gravity=1e-4, depth=50.0)
        x = np.linspace(0, wp.wavelength * 2, 500)
        eta1 = wp.amplitude * np.cos(wp.wavenumber * x)
        eta2 = stokes_second_order_surface(x, 0.0, wp)

        ax.plot(x, eta1, '--', label='1st order (Airy)', alpha=0.7)
        ax.plot(x, eta2, '-', label='2nd order (Stokes)', linewidth=2)
        ax.set_xlabel('x (lattice units)')
        ax.set_ylabel('η (lattice units)')
        ax.set_title(f'kh = {kh}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Stokes Second-Order Surface Elevation η(x, t=0)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_stokes_surface.png'), dpi=150)
    plt.close()
    print("Saved fig_stokes_surface.png")


def fig_dispersion_relation():
    """Figure 2: Dispersion relation verification."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: ω² vs gk·tanh(kh)
    kh_vals = np.linspace(0.1, 5.0, 100)
    h = 50.0
    g = 1e-4
    k_vals = kh_vals / h
    omega_exact = np.sqrt(g * k_vals * np.tanh(kh_vals))

    ax1.plot(kh_vals, omega_exact**2 / (g * k_vals), 'b-', linewidth=2, label='tanh(kh)')
    ax1.plot(kh_vals, np.ones_like(kh_vals), 'k--', alpha=0.5, label='deep water (→1)')
    ax1.plot(kh_vals, kh_vals, 'r--', alpha=0.5, label='shallow water (→kh)')
    ax1.set_xlabel('kh')
    ax1.set_ylabel('ω² / (gk)')
    ax1.set_title('Dispersion Relation: ω² = gk·tanh(kh)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)

    # Right: Phase speed
    c_exact = omega_exact / k_vals
    c_deep = np.sqrt(g / k_vals)
    c_shallow = np.sqrt(g * h) * np.ones_like(k_vals)

    ax2.plot(kh_vals, c_exact, 'b-', linewidth=2, label='Exact c = ω/k')
    ax2.plot(kh_vals, c_deep, 'k--', alpha=0.5, label='Deep water c = √(g/k)')
    ax2.plot(kh_vals, c_shallow, 'r--', alpha=0.5, label='Shallow water c = √(gh)')
    ax2.set_xlabel('kh')
    ax2.set_ylabel('Phase speed c')
    ax2.set_title('Phase Speed vs. Water Depth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_dispersion.png'), dpi=150)
    plt.close()
    print("Saved fig_dispersion.png")


def fig_relaxation_zone():
    """Figure 3: Relaxation zone weight profiles."""
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.linspace(0, 100, 200)
    lam_gen = compute_relaxation_weight(x, 0.0, 40.0, "generation")
    lam_abs = compute_relaxation_weight(x, 60.0, 100.0, "absorption")

    if isinstance(lam_gen, np.ndarray):
        lam_gen = np.array(lam_gen)
    else:
        lam_gen = np.array([compute_relaxation_weight(xi, 0.0, 40.0, "generation") for xi in x])
    if isinstance(lam_abs, np.ndarray):
        lam_abs = np.array(lam_abs)
    else:
        lam_abs = np.array([compute_relaxation_weight(xi, 60.0, 100.0, "absorption") for xi in x])

    ax.plot(x, lam_gen, 'b-', linewidth=2, label='Generation zone (0-40)')
    ax.plot(x, lam_abs, 'r-', linewidth=2, label='Absorption zone (60-100)')
    ax.axvspan(0, 40, alpha=0.1, color='blue')
    ax.axvspan(60, 100, alpha=0.1, color='red')
    ax.axvspan(40, 60, alpha=0.05, color='gray')
    ax.text(20, 0.5, 'Generation\nZone', ha='center', fontsize=9, color='blue')
    ax.text(50, 0.5, 'Tank\nInterior', ha='center', fontsize=9, color='gray')
    ax.text(80, 0.5, 'Absorption\nZone', ha='center', fontsize=9, color='red')
    ax.set_xlabel('x (lattice units)')
    ax.set_ylabel('Relaxation weight λ(x)')
    ax.set_title('Relaxation Zone Blending Weight Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_relaxation_zone.png'), dpi=150)
    plt.close()
    print("Saved fig_relaxation_zone.png")


def fig_reflection_coefficient():
    """Figure 4: Reflection coefficient verification with synthetic signals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    k = 0.05
    omega = 0.1
    x1, x2 = 50.0, 80.0
    n = 500
    dt = 1.0
    t = np.arange(n) * dt

    # Left: Pure incident wave (Kr ≈ 0)
    a_I = 2.0
    eta1_inc = a_I * np.cos(k * x1 - omega * t)
    eta2_inc = a_I * np.cos(k * x2 - omega * t)

    axes[0].plot(t, eta1_inc, label=f'η₁(x={x1})')
    axes[0].plot(t, eta2_inc, label=f'η₂(x={x2})', alpha=0.7)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('η')
    axes[0].set_title('Pure Incident Wave (Kr ≈ 0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    result_inc = two_point_reflection_coefficient(
        eta1_inc.tolist(), eta2_inc.tolist(), dt, x1, x2, k, omega
    )
    axes[0].text(0.5, 0.95, f'Kr = {result_inc["Kr"]:.4f}',
                 transform=axes[0].transAxes, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: Standing wave (Kr ≈ 1)
    eta1_std = 2 * a_I * np.cos(k * x1) * np.cos(omega * t)
    eta2_std = 2 * a_I * np.cos(k * x2) * np.cos(omega * t)

    axes[1].plot(t, eta1_std, label=f'η₁(x={x1})')
    axes[1].plot(t, eta2_std, label=f'η₂(x={x2})', alpha=0.7)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('η')
    axes[1].set_title('Standing Wave (Kr ≈ 1)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    result_std = two_point_reflection_coefficient(
        eta1_std.tolist(), eta2_std.tolist(), dt, x1, x2, k, omega
    )
    axes[1].text(0.5, 0.95, f'Kr = {result_std["Kr"]:.4f}',
                 transform=axes[1].transAxes, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_reflection_coefficient.png'), dpi=150)
    plt.close()
    print("Saved fig_reflection_coefficient.png")


def fig_harmonic_analysis():
    """Figure 5: Harmonic analysis of synthetic Stokes wave."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Synthetic signal: 1st + 2nd harmonic
    n = 400
    dt = 1.0
    omega = 2 * math.pi / 80.0
    t = np.arange(n) * dt
    a1_true = 3.0
    a2_true = 0.5
    signal = a1_true * np.sin(omega * t) + a2_true * np.sin(2 * omega * t)

    # Time series
    axes[0].plot(t, signal, 'b-', linewidth=0.8)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('η')
    axes[0].set_title('Synthetic Stokes Wave Signal')
    axes[0].grid(True, alpha=0.3)

    # FFT
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(n, d=dt)
    amplitudes = 2.0 * np.abs(fft_vals) / n

    axes[1].stem(freqs[:30], amplitudes[:30], linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[1].axvline(omega / (2*np.pi), color='r', linestyle='--', alpha=0.7, label=f'f₁={omega/(2*np.pi):.4f}')
    axes[1].axvline(2*omega / (2*np.pi), color='g', linestyle='--', alpha=0.7, label=f'f₂={2*omega/(2*np.pi):.4f}')
    axes[1].set_xlabel('Frequency')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Harmonic Analysis (DFT)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    result = harmonic_analysis(signal.tolist(), dt, omega, n_harmonics=3)
    text = f'A₁ = {result["A1"]:.3f} (true: {a1_true})\nA₂ = {result["A2"]:.3f} (true: {a2_true})'
    axes[1].text(0.7, 0.9, text, transform=axes[1].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_harmonic_analysis.png'), dpi=150)
    plt.close()
    print("Saved fig_harmonic_analysis.png")


def fig_second_order_comparison():
    """Figure 6: First vs second order wave profiles for different ka."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, ka in enumerate([0.02, 0.08]):
        ax = axes[idx]
        h = 50.0
        k = 1.5 / h  # kh = 1.5
        a = ka / k
        wp = StokesWaveParams(amplitude=a, wavenumber=k, gravity=1e-4, depth=h)

        x = np.linspace(0, wp.wavelength * 2, 500)
        eta1 = wp.amplitude * np.cos(wp.wavenumber * x)
        eta2 = np.array([stokes_second_order_surface(xi, 0.0, wp) for xi in x])

        ax.plot(x, eta1, '--', label='1st order (Airy)', alpha=0.7)
        ax.plot(x, eta2, '-', label='2nd order (Stokes)', linewidth=2)
        ax.set_xlabel('x (lattice units)')
        ax.set_ylabel('η (lattice units)')
        ax.set_title(f'ka = {ka}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate second-order amplitude
        a2 = wp.second_order_amplitude
        ax.text(0.02, 0.95, f'a₂ = {a2:.4f}\na₂/a₁ = {a2/a*100:.1f}%',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Effect of Wave Steepness ka on Second-Order Correction', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_second_order.png'), dpi=150)
    plt.close()
    print("Saved fig_second_order.png")


if __name__ == "__main__":
    fig_stokes_surface()
    fig_dispersion_relation()
    fig_relaxation_zone()
    fig_reflection_coefficient()
    fig_harmonic_analysis()
    fig_second_order_comparison()
    print("\nAll figures generated in artifacts/")
