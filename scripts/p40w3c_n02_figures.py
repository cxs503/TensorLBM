"""Generate figures for P40W3C-N02 cavitation validation report."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.cavitation_airfoil import (
    CavitationFlowConfig,
    MixtureCavitationSolver,
    reference_data,
)

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def generate_lcav_comparison():
    """L_cav/c vs sigma: simulation vs experiment."""
    ref = reference_data()
    
    # Load results
    art = ROOT / "artifacts" / "p40w3c_n02" / "summary.json"
    data = json.loads(art.read_text())
    sheet = data["sheet_results"]
    
    sigmas = [r["sigma"] for r in sheet]
    l_cav_sim = [r["l_cav"] for r in sheet]
    l_cav_exp = ref["sheet"]["l_cav_exp"]
    l_cav_merkle = ref["sheet"]["l_cav_merkle"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, l_cav_exp, 'o-', label='Brennen (1994) experiment', linewidth=2, markersize=8)
    ax.plot(sigmas, l_cav_merkle, 's--', label='Merkle (1998) numerical', linewidth=2, markersize=8)
    ax.plot(sigmas, l_cav_sim, '^:', label='This work (cpc=20)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Cavitation number σ', fontsize=12)
    ax.set_ylabel('Cavity length L_cav / c', fontsize=12)
    ax.set_title('Sheet Cavitation: NACA0015, α=5°, Re=10⁵', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.1)
    ax.set_ylim(0, 0.8)
    
    plt.tight_layout()
    plt.savefig(FIG / "p40w3c_n02_lcav_vs_sigma.png", dpi=150)
    plt.close()
    print(f"Saved {FIG / 'p40w3c_n02_lcav_vs_sigma.png'}")


def generate_delta_comparison():
    """delta_cav/c vs sigma: simulation vs experiment."""
    ref = reference_data()
    
    art = ROOT / "artifacts" / "p40w3c_n02" / "summary.json"
    data = json.loads(art.read_text())
    sheet = data["sheet_results"]
    
    sigmas = [r["sigma"] for r in sheet]
    delta_sim = [r["delta_cav"] for r in sheet]
    delta_exp = ref["sheet"]["delta_cav_exp"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, delta_exp, 'o-', label='Brennen (1994) experiment', linewidth=2, markersize=8)
    ax.plot(sigmas, delta_sim, '^:', label='This work (cpc=20)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Cavitation number σ', fontsize=12)
    ax.set_ylabel('Maximum cavity thickness δ_cav / c', fontsize=12)
    ax.set_title('Sheet Cavitation Thickness: NACA0015, α=5°, Re=10⁵', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.1)
    ax.set_ylim(0, 0.08)
    
    plt.tight_layout()
    plt.savefig(FIG / "p40w3c_n02_delta_vs_sigma.png", dpi=150)
    plt.close()
    print(f"Saved {FIG / 'p40w3c_n02_delta_vs_sigma.png'}")


def generate_pressure_field():
    """Pressure field snapshot showing cavitation region."""
    cfg = CavitationFlowConfig(
        airfoil='NACA0015',
        alpha_deg=5.0,
        sigma=0.6,
        cells_per_chord=20,
        domain_x_chords=4.0,
        domain_y_chords=2.0,
        le_x_chords=1.0,
        le_y_chords=1.0,
        n_steps=280,
        cfl=0.2,
        c_dest=0.0,
        c_prod=0.0,
    )
    
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    
    for _ in range(280):
        solver.step()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Get pressure field
    p = solver.p.cpu().numpy()
    h = cfg.h
    nx, ny = cfg.nx, cfg.ny
    
    # Create coordinate arrays (cell centers)
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Clip pressure for visualization
    p_clip = np.clip(p, -2.0, 1.0)
    
    # pcolormesh needs C to be (nx, ny) for indexing='ij'
    im = ax.pcolormesh(X, Y, p_clip, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='Pressure p / (ρU²)')
    
    # Mark vapour pressure contour
    cs = ax.contour(X, Y, p_clip, levels=[cfg.p_v], colors='black', linewidths=2)
    ax.clabel(cs, inline=True, fontsize=10, fmt='p_v=%.2f')
    
    # Mark airfoil (solid mask)
    solid = solver.solid.cpu().numpy().astype(float)
    ax.contour(X, Y, solid, levels=[0.5], colors='black', linewidths=2)
    
    ax.set_xlabel('x / c', fontsize=12)
    ax.set_ylabel('y / c', fontsize=12)
    ax.set_title('Pressure Field: NACA0015, σ=0.6, α=5° (black line: p = p_v)', fontsize=14)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(FIG / "p40w3c_n02_pressure_field.png", dpi=150)
    plt.close()
    print(f"Saved {FIG / 'p40w3c_n02_pressure_field.png'}")


def main():
    print("Generating figures for P40W3C-N02 report...")
    generate_lcav_comparison()
    generate_delta_comparison()
    generate_pressure_field()
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
