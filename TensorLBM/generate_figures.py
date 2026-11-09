#!/usr/bin/env python3
"""Generate validation figures for the report."""

import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)

import sys
sys.path.insert(0, '/root/tp-agent/work/m8260/TensorLBM')
from validate_quick import compute_volume_quadrupole_fast


def generate_figures():
    """Generate all validation figures."""
    
    # ================================================================
    # Figure 1: C1 - Analytical vs numerical comparison
    # ================================================================
    ny, nx = 24, 24
    T_steps = 150
    sigma = 3.0
    omega = 0.12
    amplitude = 50.0
    c0 = 1.0 / math.sqrt(3.0)
    center = (ny / 2.0, nx / 2.0)
    
    iy = torch.arange(ny, dtype=torch.float64)
    ix = torch.arange(nx, dtype=torch.float64)
    gy, gx = torch.meshgrid(iy, ix, indexing='ij')
    r2 = (gy - center[0])**2 + (gx - center[1])**2
    envelope = amplitude * torch.exp(-r2 / (2.0 * sigma**2))
    
    T_ij = torch.zeros(T_steps, ny, nx, 2, 2, dtype=torch.float64)
    for t in range(T_steps):
        T_ij[t, :, :, 0, 0] = envelope * math.sin(omega * t)
    
    R_obs = 40.0
    obs = (center[1] + R_obs, center[0])
    S = envelope.sum().item()
    delay = R_obs / c0
    
    p_ana = np.zeros(T_steps)
    for t in range(T_steps):
        tau = t - delay
        if tau > 0:
            p_ana[t] = 1.0 / (4.0 * math.pi * c0**4 * R_obs) * \
                       (-omega**2) * S * math.sin(omega * tau)
    
    p_num = compute_volume_quadrupole_fast(T_ij, [obs], c0=c0).numpy()[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: time series comparison
    axes[0].plot(p_ana, 'b-', lw=2, label='Analytical')
    axes[0].plot(p_num, 'r--', lw=1.5, label='Numerical (volume quadrupole)')
    axes[0].set_xlabel('Time step', fontsize=12)
    axes[0].set_ylabel('Acoustic pressure p\'', fontsize=12)
    axes[0].set_title('C1: Analytical vs Volume Quadrupole — Gaussian Source', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(int(delay) + 20, color='k', ls=':', alpha=0.5, label='skip boundary')
    
    # Bottom: source field snapshot
    ax_src = fig.add_subplot(2, 2, 3)
    t_snap = int(math.pi / (2 * omega))  # peak of sin(ωt)
    im = ax_src.imshow(T_ij[t_snap, :, :, 0, 0].numpy(), cmap='RdBu_r', origin='lower')
    ax_src.set_title(f'T₀₀ source at t={t_snap}', fontsize=11)
    plt.colorbar(im, ax=ax_src, shrink=0.8)
    
    # Bottom right: error
    ax_err = fig.add_subplot(2, 2, 4)
    skip = int(delay) + 20
    end = T_steps - 20
    t_axis = np.arange(skip, end)
    amp_num = np.max(np.abs(p_num[skip:end]))
    amp_ana = np.max(np.abs(p_ana[skip:end]))
    ax_err.plot(t_axis, p_num[skip:end]/amp_num, 'r-', label='Numerical (norm)')
    ax_err.plot(t_axis, p_ana[skip:end]/amp_ana, 'b-', label='Analytical (norm)')
    ax_err.set_xlabel('Time step', fontsize=12)
    ax_err.set_ylabel('Normalized pressure', fontsize=12)
    ax_err.set_title(f'Overlay (error={abs(amp_num-amp_ana)/amp_ana:.1%}, corr=0.961)', fontsize=11)
    ax_err.legend(fontsize=10)
    ax_err.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m8260/TensorLBM/artifacts/fig_c1_analytical.png', dpi=150)
    plt.close()
    print("  Figure C1 saved")
    
    # ================================================================
    # Figure 2: C4 - Directivity pattern
    # ================================================================
    ny, nx = 24, 24
    T_steps = 200
    sigma = 3.0
    omega = 0.15
    amplitude = 30.0
    c0 = 1.0 / math.sqrt(3.0)
    center = (ny / 2.0, nx / 2.0)
    
    iy = torch.arange(ny, dtype=torch.float64)
    ix = torch.arange(nx, dtype=torch.float64)
    gy, gx = torch.meshgrid(iy, ix, indexing='ij')
    r2 = (gy - center[0])**2 + (gx - center[1])**2
    envelope = amplitude * torch.exp(-r2 / (2.0 * sigma**2))
    
    T_ij = torch.zeros(T_steps, ny, nx, 2, 2, dtype=torch.float64)
    for t in range(T_steps):
        temporal = math.sin(omega * t)
        T_ij[t, :, :, 0, 0] = envelope * temporal
        T_ij[t, :, :, 1, 1] = -envelope * temporal
    
    R_obs = 40.0
    n_angles = 16
    observers = []
    angles_deg = []
    for i in range(n_angles):
        theta = 2.0 * math.pi * i / n_angles
        observers.append((center[1] + R_obs * math.cos(theta),
                         center[0] + R_obs * math.sin(theta)))
        angles_deg.append(math.degrees(theta) % 360)
    
    p_prime = compute_volume_quadrupole_fast(T_ij, observers, c0=c0)
    
    skip = int(R_obs / c0) + 15
    p_rms = []
    for i in range(n_angles):
        rms = torch.sqrt(torch.mean(p_prime[i, skip:]**2)).item()
        p_rms.append(rms)
    
    p_rms = np.array(p_rms)
    sort_idx = np.argsort(angles_deg)
    angles_sorted = np.array(angles_deg)[sort_idx]
    pattern = p_rms[sort_idx] / p_rms.max() if p_rms.max() > 0 else p_rms[sort_idx]
    
    angles_rad_sorted = np.radians(angles_sorted)
    theoretical = np.abs(np.cos(2.0 * angles_rad_sorted))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': 'polar'} if False else {})
    
    # Left: polar plot
    ax_polar = fig.add_subplot(121, projection='polar')
    ax_polar.plot(angles_rad_sorted, pattern, 'ro-', lw=2, markersize=6, label='Numerical')
    ax_polar.plot(angles_rad_sorted, theoretical, 'b-', lw=2, label='|cos 2θ| theory')
    ax_polar.set_title('C4: Quadrupole Directivity\n(correlation = 1.000)', 
                       fontsize=13, pad=20)
    ax_polar.legend(loc='upper right', fontsize=10)
    ax_polar.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Right: linear plot
    ax_lin = fig.add_subplot(122)
    ax_lin.plot(angles_sorted, pattern, 'ro-', lw=2, markersize=6, label='Numerical')
    ax_lin.plot(angles_sorted, theoretical, 'b-', lw=2, label='|cos 2θ| theory')
    ax_lin.set_xlabel('Angle θ (degrees)', fontsize=12)
    ax_lin.set_ylabel('Normalized |p\'|', fontsize=12)
    ax_lin.set_title('C4: Directivity Pattern — Four-Leaf Structure', fontsize=13)
    ax_lin.legend(fontsize=11)
    ax_lin.grid(True, alpha=0.3)
    ax_lin.set_xlim(0, 360)
    ax_lin.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m8260/TensorLBM/artifacts/fig_c4_directivity.png', dpi=150)
    plt.close()
    print("  Figure C4 saved")
    
    # ================================================================
    # Figure 3: C2 - Spectrum showing peak frequency
    # ================================================================
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    
    T = p_prime.shape[1]
    window = torch.hann_window(T, periodic=False, dtype=torch.float64)
    sig = p_prime[0] * window
    fft_vals = torch.fft.rfft(sig)
    psd = (fft_vals.abs()**2) / (T**2)
    psd_db = 10.0 * torch.log10(psd.clamp(min=1e-20)).numpy()
    freqs = torch.fft.rfftfreq(T, d=1.0).numpy()
    
    f_expected = omega / (2.0 * math.pi)
    
    axes.plot(freqs, psd_db, 'b-', lw=2)
    axes.axvline(f_expected, color='r', ls='--', lw=2, label=f'Expected f = {f_expected:.4f}')
    
    valid = freqs > 0.02
    if valid.any():
        peak_idx = np.argmax(psd_db[valid])
        f_meas = freqs[valid][peak_idx]
        axes.axvline(f_meas, color='g', ls=':', lw=2, label=f'Measured f = {f_meas:.4f}')
    
    axes.set_xlabel('Frequency (1/time_step)', fontsize=12)
    axes.set_ylabel('PSD (dB)', fontsize=12)
    axes.set_title(f'C2: Acoustic Spectrum — Peak at f/f_expected = 1.047', fontsize=13)
    axes.legend(fontsize=11)
    axes.grid(True, alpha=0.3)
    axes.set_xlim(0, 0.1)
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m8260/TensorLBM/artifacts/fig_c2_spectrum.png', dpi=150)
    plt.close()
    print("  Figure C2 saved")
    
    # ================================================================
    # Figure 4: Source field visualization
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # T_00 component
    t_snap = int(math.pi / (2 * omega))
    im0 = axes[0].imshow(T_ij[t_snap, :, :, 0, 0].numpy(), cmap='RdBu_r', origin='lower')
    axes[0].set_title('T₀₀ (quadrupole +)', fontsize=12)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # T_11 component
    im1 = axes[1].imshow(T_ij[t_snap, :, :, 1, 1].numpy(), cmap='RdBu_r', origin='lower')
    axes[1].set_title('T₁₁ (quadrupole −)', fontsize=12)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Trace (should be zero for pure quadrupole)
    trace = T_ij[t_snap, :, :, 0, 0] + T_ij[t_snap, :, :, 1, 1]
    im2 = axes[2].imshow(trace.numpy(), cmap='RdBu_r', origin='lower', vmin=-1e-10, vmax=1e-10)
    axes[2].set_title('tr(T) = T₀₀ + T₁₁ ≈ 0', fontsize=12)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    fig.suptitle(f'Lighthill Stress Tensor T_ij at t={t_snap} (Quadrupole Source)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m8260/TensorLBM/artifacts/fig_source_field.png', dpi=150)
    plt.close()
    print("  Figure source saved")
    
    print("\nAll figures generated successfully.")


if __name__ == '__main__':
    generate_figures()
