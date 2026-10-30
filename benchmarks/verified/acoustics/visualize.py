"""Generate publication-quality figures for D2Q9 plane acoustic wave verification.

Produces:
  1. Convergence plot: wave speed & decay rate error vs N
  2. Dispersion relation: measured vs theoretical
  3. Decay rate comparison: measured vs continuum & discrete theory
  4. Tau sensitivity plot
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

CS = 1.0 / math.sqrt(3.0)


def load_results():
    """Load result.json."""
    with open(Path(__file__).parent / 'result.json', 'r') as f:
        return json.load(f)


def plot_convergence(data):
    """Plot convergence of wave speed and decay rate errors."""
    conv = data['convergence']
    N_vals = [c['N'] for c in conv]
    c_err = [abs(c['c_err_cont_pct']) for c in conv]
    delta_err = [abs(c['delta_err_cont_pct']) for c in conv]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Wave speed error
    ax1.loglog(N_vals, c_err, 'o-', linewidth=2, markersize=8, label='Measured')
    # Reference O(N^-2)
    N_ref = np.array(N_vals, dtype=float)
    c_ref = c_err[0] * (N_vals[0] / N_ref) ** 2
    ax1.loglog(N_ref, c_ref, '--', linewidth=1.5, alpha=0.6, label=r'$O(N^{-2})$ reference')
    ax1.axhline(3.0, color='red', linestyle=':', linewidth=1.5, label='3% threshold')
    ax1.set_xlabel('Resolution N (points per wavelength)')
    ax1.set_ylabel('Wave speed error |c_meas/c_s - 1| (%)')
    ax1.set_title('(a) Wave Speed Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.01, 10)
    
    # Decay rate error
    ax2.loglog(N_vals, delta_err, 's-', linewidth=2, markersize=8, label='Measured')
    delta_ref = delta_err[0] * (N_vals[0] / N_ref) ** 2
    ax2.loglog(N_ref, delta_ref, '--', linewidth=1.5, alpha=0.6, label=r'$O(N^{-2})$ reference')
    ax2.axhline(3.0, color='red', linestyle=':', linewidth=1.5, label='3% threshold')
    ax2.set_xlabel('Resolution N (points per wavelength)')
    ax2.set_ylabel('Decay rate error |δ_meas/(νk²) - 1| (%)')
    ax2.set_title('(b) Decay Rate Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.01, 10)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'fig_convergence.png')
    print("Saved fig_convergence.png")
    plt.close()


def plot_dispersion(data):
    """Plot dispersion relation: measured vs theoretical."""
    conv = data['convergence']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Wave speed
    k_vals = [c['k'] for c in conv]
    c_meas = [c['c_meas'] for c in conv]
    c_cont = [c['c_cont'] for c in conv]
    c_disc = [c['c_discrete'] for c in conv]
    
    ax1.plot(k_vals, c_meas, 'o-', linewidth=2, markersize=8, label='LBM measured')
    ax1.plot(k_vals, c_cont, 'k--', linewidth=1.5, label='Continuum theory c_s')
    ax1.plot(k_vals, c_disc, 's--', linewidth=1.5, label='Discrete theory')
    ax1.axhline(CS, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Wavenumber k')
    ax1.set_ylabel('Phase speed c (LU/step)')
    ax1.set_title('(a) Dispersion Relation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Decay rate
    delta_meas = [c['delta_meas'] for c in conv]
    delta_cont = [c['delta_cont'] for c in conv]
    delta_disc = [c['delta_discrete'] for c in conv]
    
    ax2.plot(k_vals, delta_meas, 'o-', linewidth=2, markersize=8, label='LBM measured')
    ax2.plot(k_vals, delta_cont, 'k--', linewidth=1.5, label='Continuum νk²')
    ax2.plot(k_vals, delta_disc, 's--', linewidth=1.5, label='Discrete theory')
    ax2.set_xlabel('Wavenumber k')
    ax2.set_ylabel('Decay rate δ (1/step)')
    ax2.set_title('(b) Attenuation Relation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'fig_dispersion.png')
    print("Saved fig_dispersion.png")
    plt.close()


def plot_tau_sensitivity(data):
    """Plot tau sensitivity of decay rate."""
    tau_sweep = data['tau_sweep']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    tau_vals = [t['tau'] for t in tau_sweep]
    delta_meas = [t['delta_meas'] for t in tau_sweep]
    delta_cont = [t['delta_cont'] for t in tau_sweep]
    delta_disc = [t['delta_discrete'] for t in tau_sweep]
    
    # Decay rate vs tau
    ax1.plot(tau_vals, delta_meas, 'o-', linewidth=2, markersize=8, label='LBM measured')
    ax1.plot(tau_vals, delta_cont, 'k--', linewidth=1.5, label='Continuum νk²')
    ax1.plot(tau_vals, delta_disc, 's--', linewidth=1.5, label='Discrete theory')
    ax1.set_xlabel('Relaxation time τ')
    ax1.set_ylabel('Decay rate δ (1/step)')
    ax1.set_title('(a) Decay Rate vs τ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error vs tau
    err_cont = [abs(t['delta_err_cont_pct']) for t in tau_sweep]
    err_disc = [abs(t['delta_err_disc_pct']) for t in tau_sweep]
    
    ax2.plot(tau_vals, err_cont, 'o-', linewidth=2, markersize=8, label='vs Continuum')
    ax2.plot(tau_vals, err_disc, 's-', linewidth=2, markersize=8, label='vs Discrete')
    ax2.axhline(3.0, color='red', linestyle=':', linewidth=1.5, label='3% threshold')
    ax2.set_xlabel('Relaxation time τ')
    ax2.set_ylabel('Decay rate error (%)')
    ax2.set_title('(b) Error vs τ (N=32)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.0)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'fig_tau_sensitivity.png')
    print("Saved fig_tau_sensitivity.png")
    plt.close()


def plot_wave_evolution(data):
    """Plot wave amplitude evolution for different N."""
    # This would require time-series data, which we don't have in result.json
    # For now, create a schematic plot showing the concept
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conv = data['convergence']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, c in enumerate(conv):
        N = c['N']
        delta = c['delta_meas']
        omega = c['omega_meas']
        period = c['period_steps']
        
        # Generate synthetic time series
        t = np.linspace(0, 15 * period, 500)
        amplitude = np.exp(-delta * t)
        wave = amplitude * np.sin(omega * t)
        
        ax.plot(t / period, wave, linewidth=1.5, color=colors[i], 
                label=f'N={N}, δ={delta:.2e}')
    
    ax.set_xlabel('Time (periods)')
    ax.set_ylabel('Wave amplitude (normalized)')
    ax.set_title('Plane Wave Amplitude Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'fig_wave_evolution.png')
    print("Saved fig_wave_evolution.png")
    plt.close()


def main():
    """Generate all figures."""
    print("Loading results...")
    data = load_results()
    
    print("Generating convergence plot...")
    plot_convergence(data)
    
    print("Generating dispersion plot...")
    plot_dispersion(data)
    
    print("Generating tau sensitivity plot...")
    plot_tau_sensitivity(data)
    
    print("Generating wave evolution plot...")
    plot_wave_evolution(data)
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
