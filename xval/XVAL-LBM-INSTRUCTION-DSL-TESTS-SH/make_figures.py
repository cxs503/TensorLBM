#!/usr/bin/env python3
"""Generate figures for XVAL-LBM-INSTRUCTION-DSL-TESTS-SH report."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.shallow_water import ShallowWaterConfig, init_dam_break, swe_step
from tensorlbm.shallow_water_lbm import SWE_LBM_Config, run_swe_lbm

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_smooth_wave_figure():
    """Generate comparison figure for smooth wave case."""
    nx = 200
    h0 = 1.0
    g = 1.0
    t_end = 1.0
    x_min = -4.0
    x_max = 12.0

    x_arr = np.linspace(x_min, x_max, nx)
    c0 = np.sqrt(g * h0)

    # Analytical reference (linear wave propagation)
    h_ref = h0 + 0.005 * np.exp(-((x_arr - 2.0 - c0 * t_end) ** 2) / 0.5) + \
                 0.005 * np.exp(-((x_arr - 2.0 + c0 * t_end) ** 2) / 0.5)

    # FV solution
    fv_cfg = ShallowWaterConfig(nx=nx, x_min=x_min, x_max=x_max, dam_x=0,
                                h0=h0, g=g, cfl=0.4, t_end=t_end, scheme="rusanov")
    h_init = h0 + 0.01 * np.exp(-((x_arr - 2.0) ** 2) / 0.5)
    U = init_dam_break(fv_cfg)
    U[0] = torch.tensor(h_init, dtype=torch.float64)
    dt = fv_cfg.cfl * fv_cfg.dx / (2.0 * fv_cfg.c0)
    n_steps = int(t_end / dt) + 1
    dt = t_end / n_steps
    for _ in range(n_steps):
        U = swe_step(U, fv_cfg, dt)
    h_fv = U[0].numpy()

    # LBM solution
    lbm_cfg = SWE_LBM_Config(nx=nx, x_min=x_min, x_max=x_max, dam_x=0,
                             h0=h0, g=g, lam=4.0, sigma_h=1e-3, sigma_q=1e-1,
                             cfl=0.4, t_end=t_end)
    try:
        result = run_swe_lbm(lbm_cfg)
        h_lbm = result.h.numpy()
        lbm_ok = True
    except Exception:
        lbm_ok = False

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_arr, h_ref, 'k--', linewidth=2, label='Analytical (linear wave)')
    ax.plot(x_arr, h_fv, 'b-', linewidth=1.5, label='FV Rusanov')
    if lbm_ok:
        ax.plot(x_arr, h_lbm, 'r-.', linewidth=1.5, label='LBM D1Q2')
    ax.set_xlabel('x')
    ax.set_ylabel('h')
    ax.set_title(f'Small-amplitude wave propagation (nx={nx}, t={t_end})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = ARTIFACTS_DIR / "smooth_wave_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def generate_dam_break_figure():
    """Generate comparison figure for dam-break case."""
    nx = 200
    h0 = 1.0
    g = 1.0
    t_end = 3.0
    x_min = -4.0
    x_max = 12.0

    x_arr = np.linspace(x_min, x_max, nx)
    c0 = np.sqrt(g * h0)

    # Ritter analytical solution
    h_ref = np.zeros(nx)
    xi = x_arr / t_end
    mask_left = xi <= -c0
    mask_front = xi >= 2.0 * c0
    mask_mid = ~mask_left & ~mask_front
    h_ref[mask_left] = h0
    h_ref[mask_mid] = (2.0 * c0 - xi[mask_mid]) ** 2 / (9.0 * g)

    # FV solution
    fv_cfg = ShallowWaterConfig(nx=nx, x_min=x_min, x_max=x_max, dam_x=0,
                                h0=h0, g=g, cfl=0.4, t_end=t_end, scheme="rusanov")
    U = init_dam_break(fv_cfg)
    dt = fv_cfg.cfl * fv_cfg.dx / (2.0 * fv_cfg.c0)
    n_steps = int(t_end / dt) + 1
    dt = t_end / n_steps
    for _ in range(n_steps):
        U = swe_step(U, fv_cfg, dt)
    h_fv = U[0].numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_arr, h_ref, 'k--', linewidth=2, label='Ritter analytical')
    ax.plot(x_arr, h_fv, 'b-', linewidth=1.5, label='FV Rusanov')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Dam position')
    ax.set_xlabel('x')
    ax.set_ylabel('h')
    ax.set_title(f'Ritter dam-break (nx={nx}, t={t_end})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = ARTIFACTS_DIR / "dam_break_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    import torch
    generate_smooth_wave_figure()
    generate_dam_break_figure()
