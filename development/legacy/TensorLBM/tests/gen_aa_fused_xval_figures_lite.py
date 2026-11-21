"""Generate key figures for AA-pattern XVAL report (lightweight version)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from tensorlbm.aa_fused_stream_collide import (
    aa_fused_step,
    separate_step,
    _macroscopic,
)
from tensorlbm.verification.lid_driven_cavity.reference_data import (
    get_ghia_reference,
)


def run_cavity_for_plot(n, re=100, u_lid=0.1, max_steps=3000):
    """Run cavity simulation for plotting."""
    L = n - 1
    nu = u_lid * L / re
    tau = nu / (1.0/3.0) + 0.5

    device = torch.device("cpu")
    dtype = torch.float64

    rho = torch.ones((n, n), dtype=dtype, device=device)
    ux = torch.zeros((n, n), dtype=dtype, device=device)
    uy = torch.zeros((n, n), dtype=dtype, device=device)

    C = torch.tensor([
        [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [-1, -1], [1, -1],
    ], dtype=torch.int64, device=device)
    W = torch.tensor([
        4./9., 1./9., 1./9., 1./9., 1./9.,
        1./36., 1./36., 1./36., 1./36.,
    ], dtype=dtype, device=device)

    uc = C[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) + C[:, 1].view(-1, 1, 1) * uy.unsqueeze(0)
    u2 = ux**2 + uy**2
    f = W.view(-1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
    )

    solid = torch.zeros((n, n), dtype=torch.bool, device=device)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True

    f_aa = f.clone()
    for _ in range(max_steps):
        f_aa = aa_fused_step(f_aa, tau, solid, u_lid=u_lid, lid_row=-1)

    f_sep = f.clone()
    for _ in range(max_steps):
        f_sep = separate_step(f_sep, tau, solid, u_lid=u_lid, lid_row=-1)

    rho_aa, ux_aa, uy_aa = _macroscopic(f_aa)
    rho_sep, ux_sep, uy_sep = _macroscopic(f_sep)

    mid = n // 2
    u_aa = (ux_aa[:, mid] / u_lid).numpy()
    v_aa = (uy_aa[mid, :] / u_lid).numpy()
    u_sep = (ux_sep[:, mid] / u_lid).numpy()
    v_sep = (uy_sep[mid, :] / u_lid).numpy()

    y_norm = np.linspace(0, 1, n)
    x_norm = np.linspace(0, 1, n)

    return y_norm, u_aa, x_norm, v_aa, y_norm, u_sep, x_norm, v_sep


def main():
    out_dir = Path(__file__).parent.parent.parent / "artifacts" / "xval_aa_pattern" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Velocity profiles comparison (n=49, moderate grid)
    print("Generating velocity profile comparison (n=49)...")
    y, u_aa, x, v_aa, y_s, u_sep, x_s, v_sep = run_cavity_for_plot(49, max_steps=5000)
    ref = get_ghia_reference(100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # u profile
    axes[0].plot(ref.u, ref.y, 'ko-', markersize=4, label='Ghia et al. (1982)', zorder=3)
    axes[0].plot(u_aa, y, 'b-', linewidth=1.5, label='AA-pattern fused')
    axes[0].plot(u_sep, y_s, 'r--', linewidth=1.5, label='Separate stream+collide')
    axes[0].set_xlabel('u / U_lid')
    axes[0].set_ylabel('y / L')
    axes[0].set_title('Vertical centerline u-velocity (Re=100, n=49)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # v profile
    axes[1].plot(ref.v, ref.x, 'ko-', markersize=4, label='Ghia et al. (1982)', zorder=3)
    axes[1].plot(v_aa, x, 'b-', linewidth=1.5, label='AA-pattern fused')
    axes[1].plot(v_sep, x_s, 'r--', linewidth=1.5, label='Separate stream+collide')
    axes[1].set_xlabel('v / U_lid')
    axes[1].set_ylabel('x / L')
    axes[1].set_title('Horizontal centerline v-velocity (Re=100, n=49)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "velocity_profiles_n49.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

    # Figure 2: Convergence study (3 grid sizes)
    print("Generating convergence study (n=33, 49, 65)...")
    grid_sizes = [33, 49, 65]
    errors_aa_list = []
    errors_sep_list = []

    for n in grid_sizes:
        y, u_aa, x, v_aa, y_s, u_sep, x_s, v_sep = run_cavity_for_plot(n, max_steps=5000)
        ref = get_ghia_reference(100)

        u_interp_aa = np.interp(ref.y, y, u_aa)
        v_interp_aa = np.interp(ref.x, x, v_aa)
        l2_aa = float(np.sqrt(np.mean((u_interp_aa - ref.u)**2) + np.mean((v_interp_aa - ref.v)**2)))

        u_interp_sep = np.interp(ref.y, y_s, u_sep)
        v_interp_sep = np.interp(ref.x, x_s, v_sep)
        l2_sep = float(np.sqrt(np.mean((u_interp_sep - ref.u)**2) + np.mean((v_interp_sep - ref.v)**2)))

        errors_aa_list.append(l2_aa)
        errors_sep_list.append(l2_sep)
        print(f"  n={n}: AA L2={l2_aa:.4f}, Sep L2={l2_sep:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid_sizes, errors_aa_list, 'bo-', linewidth=2, markersize=8, label='AA-pattern fused')
    ax.plot(grid_sizes, errors_sep_list, 'rs--', linewidth=2, markersize=8, label='Separate stream+collide')
    ax.axhline(y=0.15, color='gray', linestyle=':', linewidth=1.5, label='15% threshold')
    ax.set_xlabel('Grid size (n × n)')
    ax.set_ylabel('Combined L2 error (u, v)')
    ax.set_title('Convergence: L2 error vs grid size (Re=100)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    fig_path = out_dir / "convergence_study.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
