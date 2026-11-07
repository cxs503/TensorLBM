"""Generate figures for the yield-stress LBM campaign report."""

import sys
sys.path.insert(0, 'src')

import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.yield_stress_analytical import (
    bingham_couette_profile,
    bingham_poiseuille_profile,
    newtonian_poiseuille_profile,
)
from tensorlbm.yield_stress_channel import (
    run_bingham_couette,
    run_bingham_poiseuille,
)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_couette_figure():
    """Generate Couette flow comparison figure."""
    H = 8
    u_wall = 0.05
    tau_y = 0.001
    mu_0 = 0.02
    m = 100.0

    result = run_bingham_couette(
        H, u_wall, tau_y, mu_0, m,
        max_iter=5000, tol=1e-4, dtype=torch.float64,
    )

    analytic_u, y_phys = bingham_couette_profile(H, u_wall, tau_y, mu_0)
    u_sim = result["u_sim"].numpy()
    y = np.arange(1, H + 1, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(u_sim, y, 'ro-', label='LBM', markersize=4)
    ax.plot(analytic_u.numpy(), y_phys.numpy(), 'b--', label='Analytic', linewidth=2)
    ax.set_xlabel('Velocity (lattice units)')
    ax.set_ylabel('y (lattice units)')
    ax.set_title(f'Bingham Couette Flow (H={H}, tau_y={tau_y}, mu_0={mu_0})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'L2 error: {result["l2_error"]:.4f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig_to_base64(fig)


def generate_poiseuille_figure():
    """Generate Poiseuille flow comparison figure."""
    H = 8
    ax_force = 1e-5
    tau_y = 2e-5
    mu_0 = 0.02
    m = 100.0

    result = run_bingham_poiseuille(
        H, ax_force, tau_y, mu_0, m,
        max_iter=5000, tol=1e-4, dtype=torch.float64,
    )

    analytic_u, y_phys = bingham_poiseuille_profile(H, ax_force, tau_y, mu_0)
    u_sim = result["u_sim"].numpy()
    y = np.arange(1, H + 1, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(u_sim, y, 'ro-', label='LBM', markersize=4)
    ax.plot(analytic_u.numpy(), y_phys.numpy(), 'b--', label='Analytic (Bingham)', linewidth=2)

    # Newtonian for comparison
    newton_u, _ = newtonian_poiseuille_profile(H, ax_force, mu_0)
    ax.plot(newton_u.numpy(), y_phys.numpy(), 'g:', label='Newtonian', linewidth=1.5)

    ax.set_xlabel('Velocity (lattice units)')
    ax.set_ylabel('y (lattice units)')
    ax.set_title(f'Bingham Poiseuille Flow (H={H}, a={ax_force}, tau_y={tau_y})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'L2 error: {result["l2_error"]:.4f}\n'
            f'u_max_sim: {result["u_max_sim"]:.2e}\n'
            f'u_max_analytic: {result["u_max_analytic"]:.2e}',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig_to_base64(fig)


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)

    print("Generating Couette figure...")
    couette_b64 = generate_couette_figure()
    with open('figures/couette.png', 'wb') as f:
        f.write(base64.b64decode(couette_b64))
    print(f"Saved figures/couette.png ({len(couette_b64)} bytes)")

    print("Generating Poiseuille figure...")
    poiseuille_b64 = generate_poiseuille_figure()
    with open('figures/poiseuille.png', 'wb') as f:
        f.write(base64.b64decode(poiseuille_b64))
    print(f"Saved figures/poiseuille.png ({len(poiseuille_b64)} bytes)")
