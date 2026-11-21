"""Generate figures for XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS report."""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from run_xval import oracle_poiseuille, tlbm_poiseuille

def generate_figures():
    cases = [
        {'ny':16, 'nx':32, 'tau':0.8, 'g_accel':1e-5, 'n_steps':4000, 'label':'C1_ny16_tau0.8'},
        {'ny':32, 'nx':64, 'tau':0.8, 'g_accel':1e-5, 'n_steps':10000, 'label':'C2_ny32_tau0.8'},
    ]
    fig_paths = []

    for c in cases:
        label = c.pop('label')
        ro = oracle_poiseuille(**c)
        rt = tlbm_poiseuille(**c)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: velocity profiles
        ax = axes[0]
        y = ro['y']
        ax.plot(ro['ua'], y, 'k-', linewidth=2, label='Analytic')
        ax.plot(ro['u'], y, 'ro', markersize=4, label='Oracle (Maarten)')
        ax.plot(rt['u'], y, 'b^', markersize=4, label='TensorLBM')
        ax.set_xlabel('u (lattice units)')
        ax.set_ylabel('y (lattice units)')
        ax.set_title(f'Poiseuille Profile – {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: error profile
        ax = axes[1]
        ua = ro['ua']
        scale = np.max(np.abs(ua)) if np.max(np.abs(ua)) > 0 else 1.0
        err_o = (ro['u'] - ua) / scale
        err_t = (rt['u'] - ua) / scale
        ax.plot(y, err_o * 100, 'r-o', markersize=4, label='Oracle error')
        ax.plot(y, err_t * 100, 'b-^', markersize=4, label='TensorLBM error')
        ax.set_xlabel('y (lattice units)')
        ax.set_ylabel('Relative error (%)')
        ax.set_title(f'Error vs Analytic – {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(HERE, f'figure_{label}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        fig_paths.append(path)
        print(f'Saved {path}')

    # Summary figure: convergence with resolution
    fig, ax = plt.subplots(figsize=(8, 5))
    nys = [16, 32]
    eps_infs = []
    for ny in nys:
        steps = 4000 if ny == 16 else 10000
        r = oracle_poiseuille(ny=ny, nx=2*ny, tau=0.8, g_accel=1e-5, n_steps=steps)
        eps_infs.append(r['eps_inf'] * 100)
    ax.semilogy(nys, eps_infs, 'ro-', markersize=8, label='Oracle eps_inf')
    ax.set_xlabel('Channel resolution (ny)')
    ax.set_ylabel('eps_inf (%)')
    ax.set_title('Poiseuille Flow: Convergence with Resolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(HERE, 'figure_convergence.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    fig_paths.append(path)
    print(f'Saved {path}')

    return fig_paths

if __name__ == '__main__':
    paths = generate_figures()
    print(f'Generated {len(paths)} figures')
