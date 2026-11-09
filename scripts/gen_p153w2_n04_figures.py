"""Generate verification figures for P153W2-N04 campaign report."""

import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm.compressible.solver import normal_shock_relations, GodunovSolver1D
from tensorlbm.compressible.airfoil import (
    naca0012_surface_points, thin_airfoil_theory, prandtl_glauert_correction,
    AGARD_NACA0012_REFERENCE
)
from tensorlbm.compressible.ackeret import n_wave_signature, ackeret_pressure_coefficient
from tensorlbm.compressible.isentropic import (
    isentropic_p_p0, isentropic_rho_rho0, isentropic_T_T0
)
from tensorlbm.compressible.naca0012_lbm import NACA0012LBMConfig, NACA0012LBMSolver

OUTPUT_DIR = Path(__file__).parent.parent / "artifacts" / "p153w2_n04"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig1_normal_shock_profile():
    """Figure 1: Normal shock density and pressure profiles vs RH solution."""
    M1 = 2.0
    solver = GodunovSolver1D(nx=512, gamma=1.4, cfl=0.4)
    solver.initialize_normal_shock(M1=M1, gamma=1.4, shock_pos=0.5)
    solver.run(500, report_interval=500)
    
    rh = normal_shock_relations(M1, gamma=1.4)
    rho1, p1 = 1.0, 1.0
    rho2 = rho1 * rh.rho_ratio
    p2 = p1 * rh.p_ratio
    
    x = np.linspace(0, 1, solver.nx)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Density
    ax1.plot(x, solver.rho, 'b-', label='Godunov', linewidth=1.5)
    ax1.axhline(rho1, color='r', linestyle='--', alpha=0.7, label=f'RH upstream (ρ={rho1:.2f})')
    ax1.axhline(rho2, color='g', linestyle='--', alpha=0.7, label=f'RH downstream (ρ={rho2:.2f})')
    ax1.set_xlabel('x/L')
    ax1.set_ylabel('ρ')
    ax1.set_title(f'Normal Shock Density Profile (M₁={M1})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Pressure
    ax2.plot(x, solver.p, 'b-', label='Godunov', linewidth=1.5)
    ax2.axhline(p1, color='r', linestyle='--', alpha=0.7, label=f'RH upstream (p={p1:.2f})')
    ax2.axhline(p2, color='g', linestyle='--', alpha=0.7, label=f'RH downstream (p={p2:.2f})')
    ax2.set_xlabel('x/L')
    ax2.set_ylabel('p')
    ax2.set_title(f'Normal Shock Pressure Profile (M₁={M1})')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "fig1_normal_shock.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def fig2_naca0012_cp():
    """Figure 2: NACA0012 Cp distribution at Ma=0.15."""
    cfg = NACA0012LBMConfig(
        nx=160, ny=80,
        Ma_inf=0.15, alpha_deg=2.0,
        chord_pixels=48,
        n_steps=1500, output_interval=500,
        c_h=0.05, omega_f=0.8,
    )
    solver = NACA0012LBMSolver(cfg)
    solver.run(1500)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if solver.cp_surface is not None and solver.x_surface is not None:
        # Separate upper and lower surfaces
        upper_mask = solver.y_surface > cfg.chord_offset_y * cfg.ny
        lower_mask = ~upper_mask
        
        if upper_mask.any():
            ax.plot(solver.x_surface[upper_mask], solver.cp_surface[upper_mask],
                   'b-o', markersize=3, label='Upper surface (LBM)', linewidth=1)
        if lower_mask.any():
            ax.plot(solver.x_surface[lower_mask], solver.cp_surface[lower_mask],
                   'r-o', markersize=3, label='Lower surface (LBM)', linewidth=1)
    
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.set_title('NACA0012 Pressure Coefficient (Ma=0.15, α=2°)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    path = OUTPUT_DIR / "fig2_naca0012_cp.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def fig3_n_wave():
    """Figure 3: N-wave pressure signature vs Ackeret theory."""
    Ma = 2.0
    theta_deg = 10.0
    
    sig = n_wave_signature(Ma=Ma, wedge_angle_deg=theta_deg)
    
    # Ackeret prediction
    theta_rad = math.radians(theta_deg)
    beta = math.sqrt(Ma**2 - 1.0)
    dp_ackeret = 1.4 * Ma**2 * theta_rad / beta
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(sig.x_over_L, sig.dp_over_p_inf, 'b-', linewidth=2, label='N-wave (LBM)')
    ax.axhline(dp_ackeret, color='r', linestyle='--', alpha=0.7,
               label=f'Ackeret theory (Δp/p∞={dp_ackeret:.3f})')
    ax.axhline(-dp_ackeret, color='r', linestyle='--', alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    
    ax.set_xlabel('x/L')
    ax.set_ylabel('Δp/p∞')
    ax.set_title(f'Sonic Boom N-Wave Signature (Ma={Ma}, θ={theta_deg}°)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "fig3_n_wave.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def fig4_isentropic():
    """Figure 4: Isentropic flow relations."""
    Ma_range = np.linspace(0.1, 2.0, 100)
    
    p_p0 = [isentropic_p_p0(M, 1.4) for M in Ma_range]
    rho_rho0 = [isentropic_rho_rho0(M, 1.4) for M in Ma_range]
    T_T0 = [isentropic_T_T0(M, 1.4) for M in Ma_range]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(Ma_range, p_p0, 'b-', linewidth=2, label='p/p₀')
    ax.plot(Ma_range, rho_rho0, 'r-', linewidth=2, label='ρ/ρ₀')
    ax.plot(Ma_range, T_T0, 'g-', linewidth=2, label='T/T₀')
    
    ax.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Ma=1 (sonic)')
    
    ax.set_xlabel('Mach number')
    ax.set_ylabel('Ratio')
    ax.set_title('Isentropic Flow Relations (γ=1.4)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "fig4_isentropic.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def fig5_shock_strength():
    """Figure 5: Shock strength vs upstream Mach number."""
    M1_range = np.linspace(1.01, 5.0, 100)
    
    rho_ratios = []
    p_ratios = []
    T_ratios = []
    
    for M1 in M1_range:
        rh = normal_shock_relations(M1, gamma=1.4)
        rho_ratios.append(rh.rho_ratio)
        p_ratios.append(rh.p_ratio)
        T_ratios.append(rh.T_ratio)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(M1_range, rho_ratios, 'b-', linewidth=2, label='ρ₂/ρ₁')
    ax.plot(M1_range, p_ratios, 'r-', linewidth=2, label='p₂/p₁')
    ax.plot(M1_range, T_ratios, 'g-', linewidth=2, label='T₂/T₁')
    
    ax.set_xlabel('Upstream Mach number M₁')
    ax.set_ylabel('Ratio')
    ax.set_title('Normal Shock Strength vs Mach Number (γ=1.4)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "fig5_shock_strength.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Generating verification figures...")
    fig1_normal_shock_profile()
    fig2_naca0012_cp()
    fig3_n_wave()
    fig4_isentropic()
    fig5_shock_strength()
    print("All figures generated.")
