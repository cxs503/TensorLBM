"""Generate validation figures for linear-interpolated BB alpha scheme.

Produces:
1. Analytical parabolic velocity profile
2. Comparison of interpolated BB vs standard BB boundary treatment
3. q-value distribution for curved boundary
4. Convergence rate illustration
"""

from __future__ import annotations

import os
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorlbm.verification.linear_interpolated_bb_alpha.solver import (
    analytical_circular_poiseuille,
)
from tensorlbm.boundary.inamuro_wetnode import (
    create_curved_boundary_masks,
    compute_boundary_distance_from_levelset,
)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_analytical_profile():
    """Figure 1: Analytical parabolic velocity profile."""
    R = 25.0
    F = 1e-5
    nu = 1.0 / 6.0

    r = np.linspace(0, R, 100)
    u = analytical_circular_poiseuille(r, R, F, nu)
    u_max = F * R**2 / (4.0 * nu)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, u, 'b-', linewidth=2, label='Analytical: $u(r) = u_{max}(1 - (r/R)^2)$')
    ax.plot([0, R], [u_max, u_max], 'r--', linewidth=1, label=f'$u_{{max}} = {u_max:.6f}$')
    ax.plot([R, R], [0, u_max], 'k:', linewidth=1, label=f'$R = {R}$')
    ax.set_xlabel('Radial position $r$', fontsize=12)
    ax.set_ylabel('Velocity $u(r)$', fontsize=12)
    ax.set_title('Analytical Circular Poiseuille Velocity Profile', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, R * 1.1)
    ax.set_ylim(0, u_max * 1.1)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_q_distribution():
    """Figure 2: Distribution of q values for curved boundary."""
    n = 65
    center = (n - 1) / 2.0
    radius = 25.0
    fluid, solid, levelset = create_curved_boundary_masks(
        n, n, center, center, radius, interior_solid=True
    )

    # Collect all q values
    all_q = []
    for i in range(1, 9):
        q, mask = compute_boundary_distance_from_levelset(fluid, levelset, i)
        if mask.any():
            all_q.extend(q[mask].numpy().tolist())

    all_q = np.array(all_q)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(all_q, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='q = 0.5 (scheme transition)')
    axes[0].set_xlabel('Boundary distance $q$', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'Distribution of q values\n(n={n}, R={radius}, {len(all_q)} boundary nodes)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Boundary visualization
    ax2 = axes[1]
    boundary_vis = np.zeros((n, n))
    boundary_vis[fluid] = 1.0
    boundary_vis[solid] = 0.0

    # Overlay q values on boundary
    q_overlay = np.full((n, n), np.nan)
    for i in range(1, 9):
        q, mask = compute_boundary_distance_from_levelset(fluid, levelset, i)
        q_overlay[mask] = np.maximum(q_overlay[mask], q[mask])

    im = ax2.imshow(boundary_vis, cmap='gray', origin='lower')
    ax2.imshow(np.ma.masked_invalid(q_overlay), cmap='viridis', alpha=0.7, origin='lower',
               vmin=0, vmax=1)
    ax2.set_title('Curved boundary with q values', fontsize=12)
    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('y', fontsize=10)
    plt.colorbar(im, ax=ax2, label='Fluid/Solid')

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_convergence_illustration():
    """Figure 3: Illustration of expected convergence behavior."""
    # Theoretical convergence rates
    h = np.array([1/16, 1/32, 1/48, 1/64, 1/81, 1/100])

    # First-order (standard BB)
    error_bb = 0.2 * h / h[0]

    # Second-order (interpolated BB)
    error_interp = 0.15 * (h / h[0])**2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(1/h, error_bb, 'ro-', linewidth=2, markersize=8, label='Standard BB (1st order)')
    ax.loglog(1/h, error_interp, 'bs-', linewidth=2, markersize=8, label='Interpolated BB (2nd order)')

    # Reference lines
    h_ref = np.array([1/16, 1/100])
    ax.loglog(1/h_ref, 0.2 * h_ref / h_ref[0], 'r:', linewidth=1, alpha=0.5)
    ax.loglog(1/h_ref, 0.15 * (h_ref / h_ref[0])**2, 'b:', linewidth=1, alpha=0.5)

    ax.set_xlabel('Grid resolution $N$', fontsize=12)
    ax.set_ylabel('L2 velocity error', fontsize=12)
    ax.set_title('Expected Convergence: Linear-Interpolated BB Alpha Scheme', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(10, 120)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_scheme_diagram():
    """Figure 4: Schematic of the alpha scheme interpolation."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Draw lattice nodes
    x_f = 0.0  # fluid node
    x_s = 1.0  # solid node
    x_b = 0.6  # boundary position (q = 0.6)

    # Draw nodes
    ax.plot(x_f, 0, 'go', markersize=15, label='Fluid node $x_f$', zorder=5)
    ax.plot(x_s, 0, 'rs', markersize=15, label='Solid node $x_s$', zorder=5)
    ax.axvline(x=x_b, color='blue', linewidth=2, linestyle='-', label=f'Boundary $x_b$ (q={x_b-x_f:.1f})', zorder=3)

    # Draw distance annotations
    ax.annotate('', xy=(x_b, -0.3), xytext=(x_f, -0.3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text((x_f + x_b) / 2, -0.45, '$q \\cdot \\Delta x$', ha='center', fontsize=11)

    ax.annotate('', xy=(x_s, -0.3), xytext=(x_b, -0.3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text((x_b + x_s) / 2, -0.45, '$(1-q) \\cdot \\Delta x$', ha='center', fontsize=11)

    # Draw lattice direction arrow
    ax.annotate('', xy=(x_s + 0.1, 0.3), xytext=(x_f - 0.1, 0.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(0.5, 0.4, 'Lattice direction $c_i$', ha='center', fontsize=11, color='gray')

    # Add formula
    formula = (
        r"For $q \geq 0.5$: $f_i = \frac{1}{2q} f_{\bar{i}}(x_f) + \frac{2q-1}{2q} f_{\bar{i}}(x_f - c_i)$" + "\n"
        r"For $q < 0.5$: $f_i = 2q f_{\bar{i}}(x_f) + (1-2q) f_i(x_f)$"
    )
    ax.text(0.5, 0.7, formula, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(-0.3, 1.4)
    ax.set_ylim(-0.6, 1.0)
    ax.set_xlabel('Position along lattice direction', fontsize=12)
    ax.set_title('Linear-Interpolated Bounce-Back (Alpha Scheme) Schematic', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2, axis='x')

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def main():
    """Generate all figures and save to disk."""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    figures = {
        'analytical_profile': generate_analytical_profile(),
        'q_distribution': generate_q_distribution(),
        'convergence_illustration': generate_convergence_illustration(),
        'scheme_diagram': generate_scheme_diagram(),
    }

    # Save base64 strings for report embedding
    import json
    output_file = os.path.join(output_dir, 'linear_interpolated_bb_alpha_figures.json')
    with open(output_file, 'w') as f:
        json.dump({k: v for k, v in figures.items()}, f)

    # Also save as PNG files
    for name, b64 in figures.items():
        png_path = os.path.join(output_dir, f'{name}.png')
        with open(png_path, 'wb') as f:
            f.write(base64.b64decode(b64))
        print(f"Saved: {png_path}")

    print(f"\nAll figures saved to {output_dir}")
    return figures


if __name__ == '__main__':
    main()
