#!/usr/bin/env python3
"""Generate cross-validation report figures for TensorLBM.

Produces:
1. Poiseuille velocity profile comparison (sim vs analytical)
2. Lid-driven cavity centerline velocity comparison (sim vs Ghia)
3. Convergence plot (error vs grid resolution)
4. Bitwise reproducibility verification diagram
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def generate_poiseuille_figure(output_dir: Path):
    """Generate Poiseuille velocity profile comparison figure."""
    from tensorlbm.ice_hydro.lbm_channel import (
        CX, CY, WT, ChannelConfig, D2Q9Channel,
    )

    G = 1.0e-5
    TAU = 0.8
    H = 16
    STEPS = 5000

    ny = H + 4
    cfg = ChannelConfig(
        nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
        cover_amplitude=0.0, cover_depth=float(ny - 3),
        bc="no_slip", forcing="guo",
    )
    ch = D2Q9Channel(cfg)
    rows = np.where(ch.fluid[0, :])[0]
    n_fluid = rows.size
    nu = (TAU - 0.5) / 3.0
    y = rows - 0.5
    u_ana = G / (2.0 * nu) * y * (n_fluid - y)
    u0 = np.zeros(ny)
    u0[rows] = u_ana
    ch.init_equilibrium(u0)

    acc = np.zeros(n_fluid)
    n_win = max(STEPS // 4, 1)
    n_acc = 0
    for step in range(1, STEPS + 1):
        ch.f = ch._stream_from(ch._collide(ch.f))
        if step > STEPS - n_win:
            rho = ch.f.sum(axis=0)
            ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
            acc += ux[0, rows]
            n_acc += 1
    u = acc / n_acc

    eps_inf = float(np.max(np.abs(u - u_ana)) / u_ana.max())
    eps_L2 = float(np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: velocity profile
    ax = axes[0]
    y_norm = y / n_fluid
    ax.plot(u_ana, y_norm, 'k-', linewidth=2, label='Analytical')
    ax.plot(u, y_norm, 'ro', markersize=4, label=f'LBM (H={H})')
    ax.set_xlabel('u / u_max', fontsize=12)
    ax.set_ylabel('y / H', fontsize=12)
    ax.set_title(f'Poiseuille Flow Profile (Re={G*H**2/nu:.0f})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: error profile
    ax = axes[1]
    err = (u - u_ana) / u_ana.max()
    ax.plot(y_norm, err * 100, 'b-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('y / H', fontsize=12)
    ax.set_ylabel('Relative error (%)', fontsize=12)
    ax.set_title(f'Error: ε∞={eps_inf*100:.3f}%, ε₂={eps_L2*100:.3f}%', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'xval_poiseuille_profile.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'xval_poiseuille_profile.png'}")

    return {"eps_inf": eps_inf, "eps_L2": eps_L2}


def generate_cavity_figure(output_dir: Path):
    """Generate lid-driven cavity comparison figure (coarse grid for speed)."""
    from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
    from tensorlbm.verification.lid_driven_cavity.reference_data import get_ghia_reference
    from tensorlbm.verification.lid_driven_cavity.verification import compute_errors

    cfg = CavityConfig(n=33, re=100, max_steps=5000, u_lid=0.1)
    solver = CavitySolver(cfg)
    for _ in range(3000):
        solver.step()

    y, u, x, v = solver.get_profiles()
    ref = get_ghia_reference(100)
    errors = compute_errors(y, u, x, v, ref)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: u along vertical centerline
    ax = axes[0]
    ax.plot(ref.u, ref.y, 'k-', linewidth=2, label='Ghia (1982)')
    ax.plot(u, y, 'ro', markersize=3, label=f'LBM (n=33)')
    ax.set_xlabel('u / U_lid', fontsize=12)
    ax.set_ylabel('y / L', fontsize=12)
    ax.set_title('Vertical Centerline u-velocity', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: v along horizontal centerline
    ax = axes[1]
    ax.plot(ref.x, ref.v, 'k-', linewidth=2, label='Ghia (1982)')
    ax.plot(x, v, 'ro', markersize=3, label=f'LBM (n=33)')
    ax.set_xlabel('x / L', fontsize=12)
    ax.set_ylabel('v / U_lid', fontsize=12)
    ax.set_title(f'Horizontal Centerline v-velocity (L₂_u={errors["l2_u"]:.3f})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'xval_cavity_profile.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'xval_cavity_profile.png'}")

    return errors


def generate_convergence_figure(output_dir: Path):
    """Generate convergence plot for Poiseuille flow."""
    from tensorlbm.ice_hydro.lbm_channel import (
        CX, CY, WT, ChannelConfig, D2Q9Channel,
    )

    G = 1.0e-5
    TAU = 0.8

    def run_poiseuille(H, steps):
        ny = H + 4
        cfg = ChannelConfig(
            nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
            cover_amplitude=0.0, cover_depth=float(ny - 3),
            bc="no_slip", forcing="guo",
        )
        ch = D2Q9Channel(cfg)
        rows = np.where(ch.fluid[0, :])[0]
        n_fluid = rows.size
        nu = (TAU - 0.5) / 3.0
        y = rows - 0.5
        u_ana = G / (2.0 * nu) * y * (n_fluid - y)
        u0 = np.zeros(ny)
        u0[rows] = u_ana
        ch.init_equilibrium(u0)

        acc = np.zeros(n_fluid)
        n_win = max(steps // 4, 1)
        n_acc = 0
        for step in range(1, steps + 1):
            ch.f = ch._stream_from(ch._collide(ch.f))
            if step > steps - n_win:
                rho = ch.f.sum(axis=0)
                ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
                acc += ux[0, rows]
                n_acc += 1
        u = acc / n_acc
        return float(np.max(np.abs(u - u_ana)) / u_ana.max())

    resolutions = [8, 16, 32]
    steps_list = [2000, 5000, 10000]
    errors = []
    for H, steps in zip(resolutions, steps_list):
        eps = run_poiseuille(H, steps)
        errors.append(eps)
        print(f"  H={H}: eps_inf={eps:.6f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(resolutions, errors, 'ro-', linewidth=2, markersize=8, label='LBM (BGK + Guo)')

    # Reference second-order line
    ref_x = np.array(resolutions)
    ref_y = errors[0] * (resolutions[0] / ref_x) ** 2
    ax.loglog(ref_x, ref_y, 'k--', linewidth=1, alpha=0.5, label='O(Δx²)')

    ax.set_xlabel('Grid resolution H', fontsize=12)
    ax.set_ylabel('ε∞ (max relative error)', fontsize=12)
    ax.set_title('Poiseuille Flow: Grid Convergence', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Add convergence ratio annotation
    if len(errors) >= 2:
        ratio = errors[0] / errors[1]
        ax.annotate(f'ratio={ratio:.1f}', xy=(resolutions[1], errors[1]),
                    xytext=(20, 10), textcoords='offset points', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / 'xval_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'xval_convergence.png'}")

    return dict(zip(resolutions, errors))


def generate_bitwise_repro_figure(output_dir: Path):
    """Generate bitwise reproducibility verification figure."""
    from tensorlbm.verification.xval.bitwise_refdata import compute_state_hash
    from tensorlbm.ice_hydro.lbm_channel import (
        CX, CY, WT, ChannelConfig, D2Q9Channel,
    )

    G = 1.0e-5
    TAU = 0.8
    H = 16
    STEPS = 5000

    # Run twice
    def run_poiseuille():
        ny = H + 4
        cfg = ChannelConfig(
            nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
            cover_amplitude=0.0, cover_depth=float(ny - 3),
            bc="no_slip", forcing="guo",
        )
        ch = D2Q9Channel(cfg)
        rows = np.where(ch.fluid[0, :])[0]
        n_fluid = rows.size
        nu = (TAU - 0.5) / 3.0
        y = rows - 0.5
        u_ana = G / (2.0 * nu) * y * (n_fluid - y)
        u0 = np.zeros(ny)
        u0[rows] = u_ana
        ch.init_equilibrium(u0)

        for step in range(1, STEPS + 1):
            ch.f = ch._stream_from(ch._collide(ch.f))
        return ch.f

    f1 = run_poiseuille()
    f2 = run_poiseuille()
    h1 = compute_state_hash(f1)
    h2 = compute_state_hash(f2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Left: f1 snapshot
    ax = axes[0]
    ax.imshow(f1[0, :, :], cmap='RdBu_r')
    ax.set_title('Run 1: f[0] snapshot', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Middle: f2 snapshot
    ax = axes[1]
    ax.imshow(f2[0, :, :], cmap='RdBu_r')
    ax.set_title('Run 2: f[0] snapshot', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Right: difference (should be exactly zero)
    ax = axes[2]
    diff = f1 - f2
    max_diff = np.max(np.abs(diff))
    ax.imshow(diff[0, :, :], cmap='RdBu_r', vmin=-1e-16, vmax=1e-16)
    ax.set_title(f'Diff (max={max_diff:.1e})', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    reproducible = h1 == h2
    fig.suptitle(
        f'Bitwise Reproducibility: {"PASS ✓" if reproducible else "FAIL ✗"}\n'
        f'SHA-256: {h1[:32]}...',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()
    fig.savefig(output_dir / 'xval_bitwise_repro.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'xval_bitwise_repro.png'}")

    return {"reproducible": reproducible, "hash": h1}


def main():
    output_dir = Path("figures_xval")
    output_dir.mkdir(exist_ok=True)

    print("Generating Poiseuille profile figure...")
    p_metrics = generate_poiseuille_figure(output_dir)

    print("Generating cavity profile figure...")
    c_metrics = generate_cavity_figure(output_dir)

    print("Generating convergence figure...")
    conv_metrics = generate_convergence_figure(output_dir)

    print("Generating bitwise reproducibility figure...")
    repro_metrics = generate_bitwise_repro_figure(output_dir)

    print("\nAll figures generated.")
    print(f"Poiseuille: eps_inf={p_metrics['eps_inf']*100:.3f}%, eps_L2={p_metrics['eps_L2']*100:.3f}%")
    print(f"Cavity: l2_u={c_metrics['l2_u']:.4f}, l2_v={c_metrics['l2_v']:.4f}")
    print(f"Convergence: {conv_metrics}")
    print(f"Bitwise reproducible: {repro_metrics['reproducible']}")


if __name__ == "__main__":
    main()
