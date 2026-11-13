"""Generate verification figures for circular Poiseuille flow.

Produces:
1. Velocity profile comparison (BB vs Inamuro vs analytical)
2. Error convergence with grid refinement
3. Velocity field contour plot
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import json
import os

from tensorlbm.verification.circular_poiseuille.solver import (
    CircularPoiseuilleConfig,
    CircularPoiseuilleSolver,
)
from tensorlbm.verification.circular_poiseuille.analytical import (
    poiseuille_velocity_profile,
)


def run_and_collect(n, radius, body_force, tau, max_steps, use_inamuro):
    """Run simulation and return results."""
    cfg = CircularPoiseuilleConfig(
        n=n, radius=radius, body_force=body_force, tau=tau,
        max_steps=max_steps, use_inamuro=use_inamuro
    )
    solver = CircularPoiseuilleSolver(cfg)
    solver.run(resid_tol=1e-6, monitor_interval=1000)
    
    r, u = solver.get_radial_profile()
    rho, ux, uy = solver.get_full_field()
    u_max = solver.compute_max_velocity()
    
    return {
        'r': r, 'u': u, 'ux': ux, 'uy': uy, 'rho': rho,
        'u_max': u_max, 'config': cfg,
        'fluid_mask': solver.fluid_mask.cpu().numpy(),
    }


def generate_profile_comparison(output_dir='figures'):
    """Generate velocity profile comparison figure."""
    os.makedirs(output_dir, exist_ok=True)
    
    n = 49
    radius = 18.0
    body_force = 1e-5
    tau = 1.0
    max_steps = 15000
    
    print(f"Running BB simulation (n={n})...")
    bb = run_and_collect(n, radius, body_force, tau, max_steps, use_inamuro=False)
    
    print(f"Running Inamuro simulation (n={n})...")
    inamuro = run_and_collect(n, radius, body_force, tau, max_steps, use_inamuro=True)
    
    # Analytical solution
    nu = (tau - 0.5) / 3.0
    u_max_analytical = body_force * radius**2 / (4.0 * nu)
    
    r_ana = np.linspace(0, radius, 100)
    u_ana = poiseuille_velocity_profile(r_ana, radius, u_max_analytical)
    
    # Normalize velocities
    u_bb_norm = bb['u'] / u_max_analytical
    u_in_norm = inamuro['u'] / u_max_analytical
    u_ana_norm = u_ana / u_max_analytical
    r_norm = bb['r'] / radius
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot analytical
    ax.plot(r_norm, u_ana_norm, 'k-', linewidth=2, label='Analytical (parabolic)')
    
    # Plot BB
    ax.plot(r_norm, u_bb_norm, 'r--', linewidth=1.5, label='Standard Bounce-Back')
    
    # Plot Inamuro
    ax.plot(r_norm, u_in_norm, 'b-', linewidth=1.5, label='Inamuro Wet-Node')
    
    ax.set_xlabel('r / R (normalized radial position)', fontsize=12)
    ax.set_ylabel('u / u_max (normalized velocity)', fontsize=12)
    ax.set_title('Circular Poiseuille Flow: Velocity Profile Comparison\n'
                 f'n={n}, R={radius}, Re≈{u_max_analytical * radius / nu:.0f}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    png_path = os.path.join(output_dir, 'circular_poiseuille_profile_comparison.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute errors
    mask = bb['r'] <= radius
    r_v = bb['r'][mask]
    
    u_bb_v = bb['u'][mask]
    u_in_v = inamuro['u'][mask]
    u_ana_v = poiseuille_velocity_profile(r_v, radius, u_max_analytical)
    
    l2_bb = np.sqrt(np.mean((u_bb_v - u_ana_v)**2)) / u_max_analytical
    l2_in = np.sqrt(np.mean((u_in_v - u_ana_v)**2)) / u_max_analytical
    
    max_err_bb = np.max(np.abs(u_bb_v - u_ana_v)) / u_max_analytical
    max_err_in = np.max(np.abs(u_in_v - u_ana_v)) / u_max_analytical
    
    print(f"\nProfile Comparison Results:")
    print(f"  Analytical u_max: {u_max_analytical:.6e}")
    print(f"  BB u_max:         {bb['u_max']:.6e} (error: {abs(bb['u_max']-u_max_analytical)/u_max_analytical*100:.2f}%)")
    print(f"  Inamuro u_max:    {inamuro['u_max']:.6e} (error: {abs(inamuro['u_max']-u_max_analytical)/u_max_analytical*100:.2f}%)")
    print(f"  BB L2 error:      {l2_bb:.4f}")
    print(f"  Inamuro L2 error: {l2_in:.4f}")
    print(f"  BB max error:     {max_err_bb:.4f}")
    print(f"  Inamuro max error:{max_err_in:.4f}")
    
    return {
        'l2_bb': l2_bb, 'l2_in': l2_in,
        'max_err_bb': max_err_bb, 'max_err_in': max_err_in,
        'u_max_analytical': u_max_analytical,
        'u_max_bb': bb['u_max'], 'u_max_in': inamuro['u_max'],
        'png_path': png_path,
    }


def generate_field_contour(output_dir='figures'):
    """Generate velocity field contour plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    n = 49
    radius = 18.0
    body_force = 1e-5
    tau = 1.0
    max_steps = 15000
    
    print(f"Running Inamuro simulation for field plot...")
    inamuro = run_and_collect(n, radius, body_force, tau, max_steps, use_inamuro=True)
    
    ux = inamuro['ux']
    fluid = inamuro['fluid_mask']
    
    # Mask solid region
    ux_plot = ux.copy()
    ux_plot[~fluid] = np.nan
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    im = ax.imshow(ux_plot, cmap='viridis', origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax, label='u_x (lattice units)')
    
    ax.set_xlabel('x (lattice units)', fontsize=12)
    ax.set_ylabel('y (lattice units)', fontsize=12)
    ax.set_title(f'Circular Poiseuille Flow: Velocity Field\n'
                 f'Inamuro Wet-Node, n={n}, R={radius}', fontsize=13)
    
    plt.tight_layout()
    png_path = os.path.join(output_dir, 'circular_poiseuille_velocity_field.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'png_path': png_path}


def generate_convergence_study(output_dir='figures'):
    """Generate convergence study figure."""
    os.makedirs(output_dir, exist_ok=True)
    
    resolutions = [33, 41, 49]
    errors_bb = []
    errors_in = []
    
    for n in resolutions:
        radius = n * 0.38
        body_force = 1e-5
        tau = 1.0
        max_steps = 10000
        
        nu = (tau - 0.5) / 3.0
        u_max_analytical = body_force * radius**2 / (4.0 * nu)
        
        print(f"Running convergence study n={n}...")
        
        # BB
        cfg_bb = CircularPoiseuilleConfig(
            n=n, radius=radius, body_force=body_force, tau=tau,
            max_steps=max_steps, use_inamuro=False
        )
        solver_bb = CircularPoiseuilleSolver(cfg_bb)
        solver_bb.run(resid_tol=1e-6, monitor_interval=1000)
        r_bb, u_bb = solver_bb.get_radial_profile()
        
        mask = r_bb <= radius
        u_ana = poiseuille_velocity_profile(r_bb[mask], radius, u_max_analytical)
        l2_bb = np.sqrt(np.mean((u_bb[mask] - u_ana)**2)) / u_max_analytical
        errors_bb.append(l2_bb)
        
        # Inamuro
        cfg_in = CircularPoiseuilleConfig(
            n=n, radius=radius, body_force=body_force, tau=tau,
            max_steps=max_steps, use_inamuro=True
        )
        solver_in = CircularPoiseuilleSolver(cfg_in)
        solver_in.run(resid_tol=1e-6, monitor_interval=1000)
        r_in, u_in = solver_in.get_radial_profile()
        
        u_ana = poiseuille_velocity_profile(r_in[mask], radius, u_max_analytical)
        l2_in = np.sqrt(np.mean((u_in[mask] - u_ana)**2)) / u_max_analytical
        errors_in.append(l2_in)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    ax.plot(resolutions, errors_bb, 'r--o', linewidth=1.5, label='Standard Bounce-Back')
    ax.plot(resolutions, errors_in, 'b-o', linewidth=1.5, label='Inamuro Wet-Node')
    
    # Reference slope lines
    n_ref = np.array(resolutions)
    ax.plot(n_ref, errors_in[0] * (n_ref[0] / n_ref)**2, 'k:', 
            linewidth=1, alpha=0.5, label='O(Δx²) reference')
    
    ax.set_xlabel('Grid resolution n', fontsize=12)
    ax.set_ylabel('L2 velocity error', fontsize=12)
    ax.set_title('Convergence Study: L2 Error vs Grid Resolution', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    png_path = os.path.join(output_dir, 'circular_poiseuille_convergence.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nConvergence Results:")
    for n, e_bb, e_in in zip(resolutions, errors_bb, errors_in):
        print(f"  n={n}: BB={e_bb:.4f}, Inamuro={e_in:.4f}")
    
    return {
        'resolutions': resolutions,
        'errors_bb': errors_bb,
        'errors_in': errors_in,
        'png_path': png_path,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Circular Poiseuille Flow Verification Figures")
    print("=" * 60)
    
    results = {}
    
    print("\n--- Profile Comparison ---")
    results['profile'] = generate_profile_comparison()
    
    print("\n--- Velocity Field ---")
    results['field'] = generate_field_contour()
    
    print("\n--- Convergence Study ---")
    results['convergence'] = generate_convergence_study()
    
    # Save results
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_json = {k: {kk: convert(vv) for kk, vv in v.items() if kk != 'png_path'} 
                    for k, v in results.items()}
    results_json['png_paths'] = {k: v.get('png_path', '') for k, v in results.items()}
    
    with open(os.path.join(output_dir, 'circular_poiseuille_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)
