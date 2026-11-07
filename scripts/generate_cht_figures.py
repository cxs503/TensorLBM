"""Generate CHT validation figures for the report."""

import math
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.cht_validation import (
    analytical_composite_wall,
    analytical_erf,
    analytical_graetz_nu,
    analytical_churchill_chu_nu,
    run_erf_penetration,
)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_composite_wall_figure():
    """Figure 1: 1D composite wall temperature profile vs analytical."""
    x = torch.arange(50, dtype=torch.float32)
    T_ana = analytical_composite_wall(
        x, h_water=15.0, t_s=8.0, t_i=27.0,
        k_f=1.0, k_s=10.0, k_i=0.1,
        T_hot=1.0, T_cold=0.0,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x.numpy(), T_ana.numpy(), 'b-', linewidth=2, label='Analytical (series resistance)')

    # Mark layer boundaries
    t_i, t_s, h_w = 27.0, 8.0, 15.0
    ax.axvline(x=t_i, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=t_i + t_s, color='gray', linestyle='--', alpha=0.5)
    ax.text(t_i/2, 0.9, 'Insulation', ha='center', fontsize=10, alpha=0.7)
    ax.text(t_i + t_s/2, 0.9, 'Steel', ha='center', fontsize=10, alpha=0.7)
    ax.text(t_i + t_s + h_w/2, 0.9, 'Fluid', ha='center', fontsize=10, alpha=0.7)

    ax.set_xlabel('Position (lattice units)', fontsize=12)
    ax.set_ylabel('Temperature (normalized)', fontsize=12)
    ax.set_title('1D Composite Wall: Analytical Series Resistance Solution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.05, 1.05)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_erf_figure():
    """Figure 2: erf heat penetration - simulation vs analytical."""
    result = run_erf_penetration(N=80, n_steps=3000, alpha_lu=0.1, device='cpu')
    T_sim = result['T_profile']
    x_lu = result['x_lu']
    T_ana = analytical_erf(x_lu, float(result['n_steps']), result['alpha_lu'],
                           result['T_surface'], result['T_initial'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_lu.numpy(), T_sim.numpy(), 'ro-', markersize=4, label='LBM Simulation', alpha=0.7)
    ax.plot(x_lu.numpy(), T_ana.numpy(), 'b-', linewidth=2, label='Analytical (erf solution)')

    ax.set_xlabel('Position x (lattice units)', fontsize=12)
    ax.set_ylabel('Temperature (normalized)', fontsize=12)
    ax.set_title(f'Semi-Infinite Solid Transient Conduction (t={result["n_steps"]}, α={result["alpha_lu"]})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_graetz_figure():
    """Figure 3: Graetz problem Nu number vs Shah-London."""
    x_star_vals = np.logspace(-3, -0.5, 100)
    Nu_vals = [analytical_graetz_nu(100, 7, float(xs)) for xs in x_star_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(x_star_vals, Nu_vals, 'b-', linewidth=2, label='Shah-London correlation')
    ax.axhline(y=4.364, color='gray', linestyle='--', alpha=0.5, label='Fully developed Nu = 4.364')

    ax.set_xlabel('Dimensionless axial coordinate x*', fontsize=12)
    ax.set_ylabel('Nusselt number Nu', fontsize=12)
    ax.set_title('Graetz Problem: Thermal Entrance Region Nu Correlation', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_churchill_chu_figure():
    """Figure 4: Churchill-Chu Nu vs Ra."""
    Ra_vals = np.logspace(3, 12, 100)
    Nu_vals = [analytical_churchill_chu_nu(float(Ra), 0.71) for Ra in Ra_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(Ra_vals, Nu_vals, 'b-', linewidth=2, label='Churchill-Chu (Pr=0.71)')

    ax.set_xlabel('Rayleigh number Ra', fontsize=12)
    ax.set_ylabel('Nusselt number Nu', fontsize=12)
    ax.set_title('Natural Convection on Vertical Plate: Churchill-Chu Correlation', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


if __name__ == '__main__':
    print("Generating figures...")
    fig1 = generate_composite_wall_figure()
    print(f"Figure 1 (composite wall): {len(fig1)} bytes base64")
    
    fig2 = generate_erf_figure()
    print(f"Figure 2 (erf penetration): {len(fig2)} bytes base64")
    
    fig3 = generate_graetz_figure()
    print(f"Figure 3 (Graetz Nu): {len(fig3)} bytes base64")
    
    fig4 = generate_churchill_chu_figure()
    print(f"Figure 4 (Churchill-Chu): {len(fig4)} bytes base64")
    
    # Save figures
    import base64
    for i, b64 in enumerate([fig1, fig2, fig3, fig4], 1):
        with open(f'/root/tp-agent/work/m5486/figures/cht_fig{i}.png', 'wb') as f:
            f.write(base64.b64decode(b64))
    print("Figures saved to figures/")
