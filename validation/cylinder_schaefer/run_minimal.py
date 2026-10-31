#!/usr/bin/env python3
"""Minimal Schäfer-Turek benchmark proof-of-concept.

This is a reduced-resolution test to demonstrate the methodology.
For full validation, use run_benchmark.py with higher resolution.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
)


def run_minimal_benchmark():
    """Run a minimal benchmark to demonstrate the methodology."""
    print("=== Schäfer-Turek 2D-2 Minimal Proof-of-Concept ===")
    print("Note: This is a reduced-resolution test for demonstration.")
    print("      Full validation requires higher resolution and longer runs.\n")
    
    dev = torch.device("cpu")
    
    # Reduced resolution for quick test
    D_lu = 12.0  # Cylinder diameter (smaller than recommended)
    R_lu = D_lu / 2.0
    
    # Geometry
    H_lu = 4.1 * D_lu
    ny = int(round(H_lu)) + 1
    x_c = int(round(2.0 * D_lu))
    y_c = (ny - 1) // 2
    nx = x_c + int(round(10.0 * D_lu))
    
    print(f"Domain: nx={nx}, ny={ny}")
    print(f"Cylinder: center=({x_c}, {y_c}), R={R_lu}")
    print(f"Channel height: H={ny-1} lu (H/D={(ny-1)/D_lu:.2f})")
    
    # Flow parameters
    Re = 100.0
    u_mean = 0.06
    u_max = 1.5 * u_mean
    
    nu = u_mean * D_lu / Re
    tau = 3.0 * nu + 0.5
    
    print(f"u_mean={u_mean}, nu={nu:.4f}, tau={tau:.4f}")
    
    # Create geometry
    mask = cylinder_mask(nx, ny, x_c, y_c, R_lu, device=dev)
    wall_mask = make_channel_wall_mask(ny, nx, mask, device=dev)
    combined_mask = wall_mask | mask
    
    print(f"Obstacle cells: {mask.sum().item()}")
    
    # Initialize with parabolic profile
    y = torch.arange(ny, device=dev, dtype=torch.float32)
    H = float(ny - 1)
    profile = u_max * 4.0 * y * (H - y) / (H * H)
    
    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = profile.unsqueeze(1).expand(ny, nx).clone()
    uy0 = torch.zeros(ny, nx, device=dev)
    
    f = equilibrium(rho0, ux0, uy0, device=dev)
    f[:, mask] = 0.0
    
    # Time stepping
    n_steps = 5000
    sample_interval = 10
    transient_steps = int(0.4 * n_steps)
    
    print(f"\nRunning {n_steps} steps...")
    t_start = time.time()
    
    steps_list = []
    cd_list = []
    cl_list = []
    
    A_ref = D_lu
    q = 0.5 * u_mean**2
    
    for step in range(1, n_steps + 1):
        # Collision
        f = collide_bgk(f, tau=tau)
        
        # Streaming
        f = stream(f)
        
        # Forces
        fx, fy = compute_obstacle_forces(f, mask)
        
        # Inlet BC: equilibrium with parabolic profile
        rho_col0 = f.sum(dim=0)[:, 0]
        feq_in = equilibrium(
            rho_col0.unsqueeze(1),
            profile.unsqueeze(1),
            torch.zeros(ny, 1, device=dev),
            device=dev,
        )
        f[:, :, 0] = feq_in[:, :, 0]
        
        # Outlet BC: zero gradient
        f[:, :, -1] = f[:, :, -2]
        
        # Bounce-back
        f = bounce_back_cells(f, combined_mask)
        
        # Record
        if step % sample_interval == 0:
            fx_val = float(fx.item())
            fy_val = float(fy.item())
            
            # Check for NaN
            if np.isnan(fx_val) or np.isnan(fy_val):
                print(f"WARNING: NaN detected at step {step}")
                break
            
            cd = fx_val / (q * A_ref)
            cl = fy_val / (q * A_ref)
            steps_list.append(step)
            cd_list.append(cd)
            cl_list.append(cl)
        
        if step % 1000 == 0:
            elapsed = time.time() - t_start
            if len(cd_list) > 0:
                cd_recent = np.mean(cd_list[-50:])
                print(f"  Step {step}/{n_steps}, Cd_recent={cd_recent:.3f}, t={elapsed:.1f}s")
    
    elapsed = time.time() - t_start
    print(f"\nCompleted {len(steps_list)} samples in {elapsed:.1f}s")
    
    if len(steps_list) < 100:
        print("ERROR: Insufficient data for analysis")
        return None
    
    # Analysis
    steps_arr = np.array(steps_list)
    cd_arr = np.array(cd_list)
    cl_arr = np.array(cl_list)
    
    periodic_mask = steps_arr > transient_steps
    cd_periodic = cd_arr[periodic_mask]
    cl_periodic = cl_arr[periodic_mask]
    
    cd_mean = float(np.mean(cd_periodic))
    cd_std = float(np.std(cd_periodic))
    cl_rms = float(np.sqrt(np.mean(cl_periodic**2)))
    
    # Strouhal number
    dt = sample_interval
    cl_centered = cl_periodic - np.mean(cl_periodic)
    fft_vals = np.fft.rfft(cl_centered)
    fft_mag = np.abs(fft_vals)
    fft_freq = np.fft.rfftfreq(len(cl_centered), d=dt)
    
    if len(fft_mag) > 1:
        peak_idx = np.argmax(fft_mag[1:]) + 1
        f_shedding = fft_freq[peak_idx]
        St = f_shedding * D_lu / u_mean
    else:
        St = 0.0
    
    # Validation
    cd_target = 3.2260
    cd_error = abs(cd_mean - cd_target) / cd_target * 100
    cd_pass = cd_error <= 8.0
    st_pass = 0.16 <= St <= 0.17
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Cd_mean = {cd_mean:.4f} (target: {cd_target})")
    print(f"  Error: {cd_error:.2f}%")
    print(f"  PASS: {cd_pass}")
    print(f"St = {St:.4f} (target: 0.16-0.17)")
    print(f"  PASS: {st_pass}")
    print(f"Cl_rms = {cl_rms:.4f}")
    print(f"{'='*60}")
    
    # Save results
    output_dir = Path(__file__).parent / "artifacts_minimal"
    output_dir.mkdir(exist_ok=True)
    
    # Save time series
    ts_data = np.column_stack([steps_arr, cd_arr, cl_arr])
    np.savetxt(
        output_dir / "force_history.csv",
        ts_data,
        delimiter=",",
        header="step,Cd,Cl",
        comments="",
    )
    
    # Generate plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    ax1.plot(steps_arr, cd_arr, 'b-', linewidth=0.8, label='Cd(t)')
    ax1.axhline(y=cd_mean, color='r', linestyle='--', linewidth=1.5, 
                label=f'Cd_mean = {cd_mean:.3f}')
    ax1.axhline(y=cd_target, color='g', linestyle=':', linewidth=1.5, 
                label=f'Reference: {cd_target}')
    ax1.set_ylabel('Drag Coefficient $C_D$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Schäfer-Turek 2D-2 (Re=100) - Minimal Proof-of-Concept')
    
    ax2.plot(steps_arr, cl_arr, 'r-', linewidth=0.8, label='Cl(t)')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Lift Coefficient $C_L$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p1 = output_dir / "force_history.png"
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {p1}")
    
    # FFT plot
    fig, ax = plt.subplots(figsize=(8, 4))
    st_axis = fft_freq * D_lu / u_mean
    ax.plot(st_axis, fft_mag / fft_mag.max(), 'b-', linewidth=1)
    ax.axvline(x=St, color='r', linestyle='--', linewidth=2, label=f'Peak: St={St:.3f}')
    ax.axvspan(0.16, 0.17, alpha=0.2, color='green', label='Target [0.16, 0.17]')
    ax.set_xlabel('Strouhal Number')
    ax.set_ylabel('Normalized FFT Magnitude')
    ax.set_title('Lift Coefficient FFT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.4)
    
    plt.tight_layout()
    p2 = output_dir / "fft_strouhal.png"
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p2}")
    
    # Save summary
    summary = {
        "benchmark": "Schäfer-Turek 2D-2 (Minimal PoC)",
        "Re": Re,
        "resolution_lu": D_lu,
        "domain": {"nx": nx, "ny": ny},
        "cd_mean": cd_mean,
        "cd_target": cd_target,
        "cd_error_pct": cd_error,
        "cd_pass": cd_pass,
        "st": St,
        "st_pass": st_pass,
        "note": "Reduced resolution proof-of-concept. Full validation requires D≥20 lu and ≥20000 steps.",
    }
    
    import json
    with open(output_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return summary


if __name__ == "__main__":
    run_minimal_benchmark()
