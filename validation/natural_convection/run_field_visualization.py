#!/usr/bin/env python3
"""
Temperature field visualization for natural convection square cavity.
Runs the simulation and saves temperature contour plots.
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from tensorlbm.d2q9 import equilibrium, macroscopic, stream
from tensorlbm.thermal import (
    temperature_equilibrium, temperature_collision, temperature_stream,
    apply_temperature_boundaries, buoyancy_force, collide_bgk_force,
    pre_streaming_bounce_back, cavity_wall_mask, nusselt_number, thermal_params,
)


def run_and_save_field(nx, ra, pr, tau, steps, report_every, label, output_dir):
    """Run simulation and save temperature field + velocity streamlines."""
    device = torch.device('cpu')
    ny = nx
    p = thermal_params(nx, ra, pr, tau)
    nu_lat, alpha_lat, tau_T, H, g_beta = p["nu"], p["alpha"], p["tau_T"], p["H"], p["g_beta"]
    dT = 1.0
    t_hot, t_cold = 1.0, 0.0

    # Initial conditions
    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)
    T0 = torch.linspace(t_cold, t_hot, nx, device=device).unsqueeze(0).expand(ny, nx).contiguous()
    g = temperature_equilibrium(T0, u0, u0)

    wall = cavity_wall_mask(ny, nx, device)

    t0 = time.time()
    nu_history = []
    for step in range(1, steps + 1):
        rho, ux, uy, T = macroscopic(f), None, None, g.sum(dim=0)
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        F = buoyancy_force(rho, T, g_beta, t_ref=t_cold)
        
        g = temperature_collision(g, tau_T, ux, uy)
        g = temperature_stream(g)
        g = apply_temperature_boundaries(g, t_hot, t_cold)
        
        f_pre = f
        f = collide_bgk_force(f, tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)

        if step % report_every == 0 or step == steps:
            rho, ux, uy, T = macroscopic(f), None, None, None
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            nu_cur = nusselt_number(T, H, dT, t_hot, t_cold, mode="grad2")
            nu_history.append({"step": step, "nu": nu_cur["nu"]})
            if step % (report_every * 5) == 0 or step == steps:
                elapsed = time.time() - t0
                print(f"  [{label}] step={step}, Nu={nu_cur['nu']:.4f}, elapsed={elapsed:.0f}s")

    # Final field
    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    T_np = T.detach().cpu().numpy()
    ux_np = ux.detach().cpu().numpy()
    uy_np = uy.detach().cpu().numpy()
    u_mag = np.sqrt(ux_np**2 + uy_np**2)
    
    nu_final = nusselt_number(T.detach().cpu(), H, dT, t_hot, t_cold, mode="grad2")
    
    elapsed = time.time() - t0
    print(f"  [{label}] Final: Nu={nu_final['nu']:.4f}, elapsed={elapsed:.0f}s")
    
    # Plot temperature contour
    fig, ax = plt.subplots(figsize=(7, 6))
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    
    cf = ax.contourf(X, Y, T_np, levels=30, cmap='coolwarm')
    cbar = plt.colorbar(cf, ax=ax, label='Temperature (T)')
    
    # Streamlines (subsampled)
    skip = max(1, nx // 15)
    ax.streamplot(X[::skip, ::skip], Y[::skip, ::skip], 
                  ux_np[::skip, ::skip], uy_np[::skip, ::skip],
                  color='gray', linewidth=0.5, density=1.2)
    
    ax.set_xlabel('x / L', fontsize=12)
    ax.set_ylabel('y / L', fontsize=12)
    ax.set_title(f'Temperature Field & Streamlines\nRa = {ra:.0e}, Pr = 0.71, Nu_avg = {nu_final["nu"]:.3f}', 
                 fontsize=13)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    save_path = output_dir / f"temperature_field_ra{int(ra)}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    return {
        "nu_final": nu_final["nu"],
        "nu_left": nu_final["nu_left"],
        "nu_right": nu_final["nu_right"],
        "nu_history": nu_history,
        "T_field": T_np,
        "u_mag": u_mag,
        "elapsed_s": elapsed,
    }


def main():
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Temperature Field Visualization")
    print("Natural Convection Square Cavity")
    print("="*60)
    
    # Use moderate grid for CPU feasibility
    nx = 41
    
    results = {}
    
    # Case 1: Ra = 1e4
    print("\n--- Case 1: Ra = 1e4 ---")
    results[1e4] = run_and_save_field(
        nx=nx, ra=1e4, pr=0.71, tau=0.6, 
        steps=8000, report_every=1000, 
        label="Ra1e4", output_dir=output_dir
    )
    
    # Case 2: Ra = 1e5
    print("\n--- Case 2: Ra = 1e5 ---")
    results[1e5] = run_and_save_field(
        nx=nx, ra=1e5, pr=0.71, tau=0.6,
        steps=12000, report_every=1000,
        label="Ra1e5", output_dir=output_dir
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for ra, res in results.items():
        nu_ref = 2.194 if ra == 1e4 else 4.446
        err = abs(res['nu_final'] - nu_ref) / nu_ref * 100
        status = "PASS" if err <= 5.0 else "FAIL"
        print(f"Ra={ra:.0e}: Nu={res['nu_final']:.4f} (ref={nu_ref:.3f}, err={err:.2f}%) [{status}]")
    
    # Save combined data
    import json
    data = {
        "cases": {
            f"Ra{int(ra)}": {
                "nu_final": res["nu_final"],
                "nu_left": res["nu_left"],
                "nu_right": res["nu_right"],
                "elapsed_s": res["elapsed_s"],
            }
            for ra, res in results.items()
        }
    }
    with open(output_dir / "field_data.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
