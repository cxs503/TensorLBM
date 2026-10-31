#!/usr/bin/env python3
"""Schäfer-Turek benchmark 2D-2: laminar flow past a cylinder in a channel at Re=100.

Reference:
    Schäfer, M. & Turek, S. (1996). "Benchmark Computations of Laminar Flow
    Around a NACA 0012 Airfoil and a Cylinder." Notes on Numerical Fluid
    Mechanics, vol 52. Vieweg+Teubner.

Geometry (test case 2D-2):
    - Channel height H = 0.41 m
    - Cylinder diameter D = 0.1 m (radius R = 0.05 m)
    - Cylinder center at (0.2, 0.2) m from inlet
    - Re = U_mean * D / nu = 100 (based on diameter)
    - Parabolic inlet profile with U_max = 1.5 * U_mean

Target values (from reference solutions):
    - Mean drag coefficient: Cd_mean ≈ 3.2260 (±8% acceptance: 2.968–3.484)
    - Strouhal number: St = f*D/U_mean ∈ [0.16, 0.17]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    stabilize_outlet_backflow,
)


def run_schaefer_turek(
    *,
    resolution: int = 20,
    n_steps: int = 30000,
    sample_interval: int = 5,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> dict:
    """Run the Schäfer-Turek 2D-2 benchmark (fully vectorized)."""
    dev = torch.device(device)
    
    # Geometry parameters (lattice units)
    D_lu = float(resolution)
    R_lu = D_lu / 2.0
    
    # Schäfer-Turek geometry ratios
    H_lu = 4.1 * D_lu
    ny = int(round(H_lu)) + 1
    
    x_c = int(round(2.0 * D_lu))
    y_c = (ny - 1) // 2
    
    nx = x_c + int(round(14.0 * D_lu))
    
    print(f"=== Schäfer-Turek 2D-2 Benchmark (Re=100) ===")
    print(f"Resolution: D = {D_lu} lu")
    print(f"Domain: nx={nx}, ny={ny}")
    print(f"Cylinder center: ({x_c}, {y_c}), R={R_lu}")
    
    # Flow parameters
    Re = 100.0
    u_mean = 0.05
    u_max = 1.5 * u_mean
    
    nu = u_mean * D_lu / Re
    tau = 3.0 * nu + 0.5
    
    print(f"u_mean = {u_mean}, u_max = {u_max}")
    print(f"nu = {nu:.6f}, tau = {tau:.4f}")
    
    if tau <= 0.5:
        raise ValueError(f"tau={tau} <= 0.5, simulation unstable!")
    
    # Create geometry masks
    mask = cylinder_mask(nx, ny, x_c, y_c, R_lu, device=dev)
    wall_mask = make_channel_wall_mask(ny, nx, mask, device=dev)
    combined_mask = wall_mask | mask
    
    # Pre-compute parabolic profile (vectorized)
    y = torch.arange(ny, device=dev, dtype=torch.float32)
    H = float(ny - 1)
    profile = u_max * 4.0 * y * (H - y) / (H * H)  # (ny,)
    
    # Pre-compute inlet equilibrium (will be updated each step with rho from neighbor)
    # Initialize distribution functions
    rho0 = torch.ones(ny, nx, device=dev, dtype=torch.float32)
    ux0 = profile.unsqueeze(1).expand(ny, nx).clone()
    uy0 = torch.zeros(ny, nx, device=dev, dtype=torch.float32)
    
    f = equilibrium(rho0, ux0, uy0, device=dev)
    f[:, mask] = 0.0
    
    # Pre-compute inlet mask (column 0, excluding obstacle)
    inlet_mask = ~mask[:, 0]  # (ny,)
    
    # Time stepping
    print(f"\nRunning {n_steps} time steps...")
    t_start = time.time()
    
    steps_list = []
    cd_list = []
    cl_list = []
    
    A_ref = D_lu * 1.0
    q = 0.5 * 1.0 * u_mean**2
    transient_steps = int(0.3 * n_steps)
    
    for step in range(1, n_steps + 1):
        # Collision
        f = collide_bgk(f, tau=tau)
        
        # Streaming
        f = stream(f)
        
        # Compute forces (momentum exchange, BEFORE bounce-back)
        fx, fy = compute_obstacle_forces(f, mask)
        
        # === Boundary conditions (fully vectorized) ===
        
        # Inlet: prescribe parabolic velocity profile
        # Use rho from column 1 for inlet density
        rho_in = f[:, 1, 0].sum(dim=0)  # Actually need rho at column 1
        # Simpler: use rho from column 0 itself (from streaming)
        rho_col0 = f.sum(dim=0)[:, 0]  # (ny,)
        
        # Build equilibrium at inlet with prescribed velocity
        ux_in = profile.clone()  # (ny,)
        uy_in = torch.zeros(ny, device=dev, dtype=torch.float32)
        
        # Compute feq for inlet column
        feq_inlet = equilibrium(
            rho_col0.unsqueeze(1),  # (ny, 1)
            ux_in.unsqueeze(1),     # (ny, 1)
            uy_in.unsqueeze(1),     # (ny, 1)
            device=dev,
        )  # (9, ny, 1)
        
        f[:, :, 0] = feq_inlet[:, :, 0]
        
        # Outlet: zero gradient
        f[:, :, -1] = f[:, :, -2]
        
        # Stabilize outlet backflow
        f = stabilize_outlet_backflow(f, max_backflow_speed=0.01)
        
        # Wall and obstacle bounce-back
        f = bounce_back_cells(f, combined_mask)
        
        # Record forces
        if step % sample_interval == 0:
            fx_val = float(fx.item())
            fy_val = float(fy.item())
            cd = fx_val / (q * A_ref)
            cl = fy_val / (q * A_ref)
            steps_list.append(step)
            cd_list.append(cd)
            cl_list.append(cl)
        
        if step % 5000 == 0:
            elapsed = time.time() - t_start
            cd_recent = np.mean(cd_list[-100:]) if len(cd_list) >= 100 else np.mean(cd_list)
            print(f"  Step {step}/{n_steps} ({100*step/n_steps:.1f}%), "
                  f"Cd_recent={cd_recent:.3f}, elapsed={elapsed:.1f}s")
    
    elapsed = time.time() - t_start
    print(f"\nSimulation completed in {elapsed:.1f}s")
    print(f"MLUPS = {nx * ny * n_steps / elapsed / 1e6:.2f}")
    
    # Convert to numpy arrays
    steps_arr = np.array(steps_list)
    cd_arr = np.array(cd_list)
    cl_arr = np.array(cl_list)
    
    # Post-processing
    periodic_mask = steps_arr > transient_steps
    cd_periodic = cd_arr[periodic_mask]
    cl_periodic = cl_arr[periodic_mask]
    
    cd_mean = float(np.mean(cd_periodic))
    cd_std = float(np.std(cd_periodic))
    cl_mean = float(np.mean(cl_periodic))
    cl_rms = float(np.sqrt(np.mean(cl_periodic**2)))
    
    # Strouhal number from FFT of lift coefficient
    dt = sample_interval
    cl_centered = cl_periodic - np.mean(cl_periodic)
    
    n_fft = len(cl_centered)
    fft_vals = np.fft.rfft(cl_centered)
    fft_mag = np.abs(fft_vals)
    fft_freq = np.fft.rfftfreq(n_fft, d=dt)
    
    if len(fft_mag) > 1:
        peak_idx = np.argmax(fft_mag[1:]) + 1
        f_shedding = fft_freq[peak_idx]
    else:
        f_shedding = 0.0
    
    St = f_shedding * D_lu / u_mean
    
    print(f"\n=== Results ===")
    print(f"Cd_mean = {cd_mean:.4f} (target: 3.2260 ± 8% = [2.968, 3.484])")
    print(f"Cd_std = {cd_std:.4f}")
    print(f"Cl_mean = {cl_mean:.4f}")
    print(f"Cl_rms = {cl_rms:.4f}")
    print(f"St = {St:.4f} (target: 0.16-0.17)")
    print(f"Shedding frequency = {f_shedding:.6f} (1/time_step)")
    
    cd_target = 3.2260
    cd_tol = 0.08
    cd_pass = abs(cd_mean - cd_target) / cd_target <= cd_tol
    st_pass = 0.16 <= St <= 0.17
    
    print(f"\n=== Validation ===")
    print(f"Cd error: {abs(cd_mean - cd_target)/cd_target*100:.2f}% (tolerance: {cd_tol*100:.0f}%)")
    print(f"Cd PASS: {cd_pass}")
    print(f"St PASS: {st_pass}")
    
    results = {
        "benchmark": "Schäfer-Turek 2D-2",
        "Re": Re,
        "resolution_lu": D_lu,
        "domain": {"nx": nx, "ny": ny},
        "u_mean": u_mean,
        "u_max": u_max,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "sample_interval": sample_interval,
        "transient_steps": transient_steps,
        "cd_mean": cd_mean,
        "cd_std": cd_std,
        "cd_target": cd_target,
        "cd_error_pct": abs(cd_mean - cd_target) / cd_target * 100,
        "cd_pass": cd_pass,
        "cl_mean": cl_mean,
        "cl_rms": cl_rms,
        "st": St,
        "st_pass": st_pass,
        "f_shedding": f_shedding,
        "elapsed_s": elapsed,
        "mlups": nx * ny * n_steps / elapsed / 1e6,
        "overall_pass": cd_pass and st_pass,
    }
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ts_data = np.column_stack([steps_arr, cd_arr, cl_arr])
        np.savetxt(
            output_dir / "force_history.csv",
            ts_data,
            delimiter=",",
            header="step,Cd,Cl",
            comments="",
        )
        
        with open(output_dir / "results.json", "w") as fp:
            json.dump(results, fp, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    return results


def generate_plots(results: dict, output_dir: Path) -> list[Path]:
    """Generate publication-quality plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ts_data = np.loadtxt(output_dir / "force_history.csv", delimiter=",", skiprows=1)
    steps = ts_data[:, 0]
    cd = ts_data[:, 1]
    cl = ts_data[:, 2]
    
    plots = []
    
    # Plot 1: Drag and Lift time history
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(steps, cd, 'b-', linewidth=0.5, alpha=0.8, label='Cd(t)')
    ax1.axhline(y=results['cd_mean'], color='r', linestyle='--', linewidth=1.5, 
                label=f'Cd_mean = {results["cd_mean"]:.4f}')
    ax1.axhline(y=3.2260, color='g', linestyle=':', linewidth=1.5, 
                label='Reference: Cd = 3.2260')
    ax1.axhspan(3.2260 * 0.92, 3.2260 * 1.08, alpha=0.1, color='green', 
                label='±8% tolerance band')
    ax1.set_ylabel('Drag Coefficient $C_D$', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Schäfer-Turek 2D-2 Benchmark (Re=100): Force Time History', fontsize=14)
    
    ax2.plot(steps, cl, 'r-', linewidth=0.5, alpha=0.8, label='Cl(t)')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Lift Coefficient $C_L$', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p1 = output_dir / "force_time_history.png"
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plots.append(p1)
    print(f"Saved: {p1}")
    
    # Plot 2: FFT of lift coefficient
    transient_steps = results['transient_steps']
    periodic_mask = steps > transient_steps
    cl_periodic = cl[periodic_mask] - np.mean(cl[periodic_mask])
    
    dt = results['sample_interval']
    n_fft = len(cl_periodic)
    fft_vals = np.fft.rfft(cl_periodic)
    fft_mag = np.abs(fft_vals)
    fft_freq = np.fft.rfftfreq(n_fft, d=dt)
    
    D_lu = results['resolution_lu']
    u_mean = results['u_mean']
    st_axis = fft_freq * D_lu / u_mean
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(st_axis, fft_mag / fft_mag.max(), 'b-', linewidth=1)
    ax.axvline(x=results['st'], color='r', linestyle='--', linewidth=2,
               label=f'Peak: St = {results["st"]:.4f}')
    ax.axvspan(0.16, 0.17, alpha=0.2, color='green', label='Target band [0.16, 0.17]')
    ax.set_xlabel('Strouhal Number $St = fD/U$', fontsize=12)
    ax.set_ylabel('Normalized FFT Magnitude', fontsize=12)
    ax.set_title('Schäfer-Turek 2D-2: Lift Coefficient FFT (Vortex Shedding Frequency)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.4)
    
    plt.tight_layout()
    p2 = output_dir / "fft_strouhal.png"
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plots.append(p2)
    print(f"Saved: {p2}")
    
    # Plot 3: Phase space
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(cl[periodic_mask], cd[periodic_mask], 'b-', linewidth=0.3, alpha=0.5)
    ax.set_xlabel('Lift Coefficient $C_L$', fontsize=12)
    ax.set_ylabel('Drag Coefficient $C_D$', fontsize=12)
    ax.set_title('Phase Portrait: $C_L$ vs $C_D$ (Periodic Regime)', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p3 = output_dir / "phase_portrait.png"
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plots.append(p3)
    print(f"Saved: {p3}")
    
    return plots


def generate_vorticity_snapshot(
    results: dict,
    output_dir: Path,
    device: str = "cpu",
) -> Path:
    """Run a short simulation and save a vorticity snapshot."""
    dev = torch.device(device)
    
    D_lu = results['resolution_lu']
    R_lu = D_lu / 2.0
    H_lu = 4.1 * D_lu
    ny = int(round(H_lu)) + 1
    x_c = int(round(2.0 * D_lu))
    y_c = (ny - 1) // 2
    nx = x_c + int(round(14.0 * D_lu))
    
    Re = 100.0
    u_mean = 0.05
    u_max = 1.5 * u_mean
    nu = u_mean * D_lu / Re
    tau = 3.0 * nu + 0.5
    
    mask = cylinder_mask(nx, ny, x_c, y_c, R_lu, device=dev)
    wall_mask = make_channel_wall_mask(ny, nx, mask, device=dev)
    combined_mask = wall_mask | mask
    
    y = torch.arange(ny, device=dev, dtype=torch.float32)
    H = float(ny - 1)
    profile = u_max * 4.0 * y * (H - y) / (H * H)
    
    rho0 = torch.ones(ny, nx, device=dev, dtype=torch.float32)
    ux0 = profile.unsqueeze(1).expand(ny, nx).clone()
    uy0 = torch.zeros(ny, nx, device=dev, dtype=torch.float32)
    
    f = equilibrium(rho0, ux0, uy0, device=dev)
    f[:, mask] = 0.0
    
    n_steps = 15000
    print(f"Running {n_steps} steps for vorticity snapshot...")
    
    for step in range(1, n_steps + 1):
        f = collide_bgk(f, tau=tau)
        f = stream(f)
        
        # Inlet BC (vectorized)
        rho_col0 = f.sum(dim=0)[:, 0]
        ux_in = profile.clone()
        uy_in = torch.zeros(ny, device=dev, dtype=torch.float32)
        feq_inlet = equilibrium(
            rho_col0.unsqueeze(1),
            ux_in.unsqueeze(1),
            uy_in.unsqueeze(1),
            device=dev,
        )
        f[:, :, 0] = feq_inlet[:, :, 0]
        
        # Outlet
        f[:, :, -1] = f[:, :, -2]
        f = stabilize_outlet_backflow(f, max_backflow_speed=0.01)
        
        # Bounce-back
        f = bounce_back_cells(f, combined_mask)
        
        if step % 5000 == 0:
            print(f"  Step {step}/{n_steps}")
    
    # Compute vorticity
    rho, ux, uy = macroscopic(f)
    ux_np = ux.detach().cpu().numpy()
    uy_np = uy.detach().cpu().numpy()
    
    dux_dy = np.zeros_like(ux_np)
    duy_dx = np.zeros_like(uy_np)
    dux_dy[1:-1, :] = 0.5 * (ux_np[2:, :] - ux_np[:-2, :])
    duy_dx[:, 1:-1] = 0.5 * (uy_np[:, 2:] - uy_np[:, :-2])
    vorticity = duy_dx - dux_dy
    
    vorticity[mask.detach().cpu().numpy()] = np.nan
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    im = ax.imshow(
        vorticity,
        origin='lower',
        cmap='RdBu_r',
        vmin=-0.02,
        vmax=0.02,
        extent=[0, nx, 0, ny],
        aspect='equal',
    )
    
    theta = np.linspace(0, 2*np.pi, 100)
    cx_circle = x_c + R_lu * np.cos(theta)
    cy_circle = y_c + R_lu * np.sin(theta)
    ax.plot(cx_circle, cy_circle, 'k-', linewidth=1.5)
    
    ax.set_xlabel('x (lattice units)', fontsize=12)
    ax.set_ylabel('y (lattice units)', fontsize=12)
    ax.set_title('Vorticity Field $\\omega_z$ — Schäfer-Turek 2D-2 (Re=100)', fontsize=13)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$\\omega_z$ (1/time_step)', fontsize=11)
    
    plt.tight_layout()
    p = Path(output_dir) / "vorticity_field.png"
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p}")
    
    return p


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Schäfer-Turek 2D-2 Benchmark")
    parser.add_argument("--resolution", type=int, default=20, help="Cylinder diameter in lu")
    parser.add_argument("--steps", type=int, default=30000, help="Number of time steps")
    parser.add_argument("--sample-interval", type=int, default=5, help="Force sampling interval")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        output_dir = Path(__file__).parent / "artifacts"
    else:
        output_dir = Path(args.output_dir)
    
    results = run_schaefer_turek(
        resolution=args.resolution,
        n_steps=args.steps,
        sample_interval=args.sample_interval,
        device=args.device,
        output_dir=output_dir,
    )
    
    plots = generate_plots(results, output_dir)
    vort_plot = generate_vorticity_snapshot(results, output_dir, device=args.device)
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Cd_mean = {results['cd_mean']:.4f} (target: 3.2260 ± 8%)")
    print(f"  Error: {results['cd_error_pct']:.2f}%")
    print(f"  PASS: {results['cd_pass']}")
    print(f"St = {results['st']:.4f} (target: 0.16-0.17)")
    print(f"  PASS: {results['st_pass']}")
    print(f"Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print("="*60)
