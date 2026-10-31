#!/usr/bin/env python3
"""Generate example visualization plots based on literature values.

These plots demonstrate the expected output format for the Schäfer-Turek benchmark.
The data is synthetic but representative of typical LBM results at Re=100.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def generate_example_plots(output_dir: Path):
    """Generate example plots showing expected results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Synthetic time series based on typical LBM results for Re=100 cylinder
    np.random.seed(42)
    n_points = 2000
    
    # Time steps
    steps = np.arange(n_points) * 10
    
    # Drag coefficient: mean ~3.2 with small oscillations
    cd_mean = 3.25
    cd_oscillation = 0.15 * np.sin(2 * np.pi * steps / 800)  # Period ~800 steps
    cd_noise = 0.05 * np.random.randn(n_points)
    cd = cd_mean + cd_oscillation + cd_noise
    
    # Lift coefficient: oscillating around 0
    cl_oscillation = 0.08 * np.sin(2 * np.pi * steps / 800 + np.pi/4)
    cl_noise = 0.02 * np.random.randn(n_points)
    cl = cl_oscillation + cl_noise
    
    # Reference values
    cd_ref = 3.2260
    st_ref = 0.165
    
    # Plot 1: Force time history
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(steps, cd, 'b-', linewidth=0.6, alpha=0.8, label='$C_D(t)$')
    ax1.axhline(y=np.mean(cd), color='r', linestyle='--', linewidth=2, 
                label=f'$\\overline{{C_D}}$ = {np.mean(cd):.4f}')
    ax1.axhline(y=cd_ref, color='g', linestyle=':', linewidth=2, 
                label=f'Reference: {cd_ref}')
    ax1.axhspan(cd_ref * 0.92, cd_ref * 1.08, alpha=0.15, color='green', 
                label='±8% tolerance')
    ax1.set_ylabel('Drag Coefficient $C_D$', fontsize=13)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Schäfer-Turek 2D-2 Benchmark (Re=100): Force Time History\n(Example visualization based on literature)', 
                  fontsize=14, fontweight='bold')
    
    ax2.plot(steps, cl, 'r-', linewidth=0.6, alpha=0.8, label='$C_L(t)$')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Time Step', fontsize=13)
    ax2.set_ylabel('Lift Coefficient $C_L$', fontsize=13)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p1 = output_dir / "example_force_history.png"
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {p1}")
    
    # Plot 2: FFT spectrum
    cl_centered = cl - np.mean(cl)
    fft_vals = np.fft.rfft(cl_centered)
    fft_mag = np.abs(fft_vals)
    fft_freq = np.fft.rfftfreq(n_points, d=10)
    
    # Convert to Strouhal number (assuming D=20 lu, U=0.05)
    D_lu = 20.0
    u_mean = 0.05
    st_axis = fft_freq * D_lu / u_mean
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(st_axis, fft_mag / fft_mag.max(), 'b-', linewidth=1.5, label='FFT magnitude')
    
    # Find peak
    peak_idx = np.argmax(fft_mag[1:]) + 1
    st_peak = st_axis[peak_idx]
    ax.axvline(x=st_peak, color='r', linestyle='--', linewidth=2.5,
               label=f'Peak: St = {st_peak:.4f}')
    ax.axvspan(0.16, 0.17, alpha=0.25, color='green', label='Target [0.16, 0.17]')
    ax.axvline(x=st_ref, color='g', linestyle=':', linewidth=2, label=f'Reference: {st_ref}')
    
    ax.set_xlabel('Strouhal Number $St = fD/U$', fontsize=13)
    ax.set_ylabel('Normalized FFT Magnitude', fontsize=13)
    ax.set_title('Lift Coefficient FFT Spectrum (Vortex Shedding Frequency)\n(Example visualization)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.4)
    
    plt.tight_layout()
    p2 = output_dir / "example_fft_strouhal.png"
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {p2}")
    
    # Plot 3: Phase portrait
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(cl, cd, 'b-', linewidth=0.4, alpha=0.6)
    ax.set_xlabel('Lift Coefficient $C_L$', fontsize=13)
    ax.set_ylabel('Drag Coefficient $C_D$', fontsize=13)
    ax.set_title('Phase Portrait: $C_L$ vs $C_D$ (Periodic Regime)\n(Example visualization)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    p3 = output_dir / "example_phase_portrait.png"
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {p3}")
    
    # Plot 4: Synthetic vorticity field
    nx, ny = 200, 80
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create synthetic vortex shedding pattern
    x_c, y_c = 60, 40
    R = 10
    
    # Background flow
    vorticity = np.zeros((ny, nx))
    
    # Add von Kármán vortex street pattern
    lambda_x = 80  # Wavelength
    for i in range(3):
        x_vortex = x_c + 30 + i * lambda_x
        # Alternating vortices
        vorticity += 0.015 * np.exp(-((X - x_vortex)**2 + (Y - (y_c + 15))**2) / 50)
        vorticity -= 0.015 * np.exp(-((X - x_vortex)**2 + (Y - (y_c - 15))**2) / 50)
    
    # Add wake
    wake_mask = X > x_c + R
    vorticity *= np.exp(-(X - x_c) / 100) * wake_mask
    
    # Mask cylinder
    cylinder_mask = (X - x_c)**2 + (Y - y_c)**2 <= R**2
    vorticity[cylinder_mask] = np.nan
    
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(vorticity, origin='lower', cmap='RdBu_r', 
                   vmin=-0.015, vmax=0.015, aspect='equal')
    
    # Draw cylinder
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(x_c + R*np.cos(theta), y_c + R*np.sin(theta), 'k-', linewidth=2)
    
    ax.set_xlabel('x (lattice units)', fontsize=13)
    ax.set_ylabel('y (lattice units)', fontsize=13)
    ax.set_title('Vorticity Field $\\omega_z$ — Schäfer-Turek 2D-2 (Re=100)\n(Example visualization)', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$\\omega_z$ (1/time_step)', fontsize=12)
    
    plt.tight_layout()
    p4 = output_dir / "example_vorticity_field.png"
    fig.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {p4}")
    
    print(f"\nAll example plots saved to {output_dir}")
    return [p1, p2, p3, p4]


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "artifacts_examples"
    generate_example_plots(output_dir)
