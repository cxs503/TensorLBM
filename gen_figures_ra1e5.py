#!/usr/bin/env python3
"""Generate figures for Ra=1e5 natural convection cavity benchmark.

Produces:
1. Temperature field contour
2. Velocity magnitude field
3. Nusselt number convergence history
4. Comparison bar chart (Nu, u_max, v_max vs reference)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "benchmarks/verified/thermal_cavity")

import driver  # noqa: E402


def load_result(path: str) -> dict:
    """Load result JSON."""
    with open(path) as f:
        return json.load(f)


def plot_temperature_field(result: dict, out_path: str):
    """Plot temperature field contour for the finest grid."""
    # Find finest grid
    grids = result["grids"]
    nx_max = max(int(k) for k in grids.keys())
    res = grids[str(nx_max)]
    
    # Re-run to get fields (or load from saved)
    # For now, we'll create a placeholder
    print(f"[plot] Temperature field for N={nx_max} (placeholder - need field data)")
    
    # Create a simple contour plot using the history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, f"Temperature field\nN={nx_max}\n(Requires field data save)", 
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title("Temperature Field (placeholder)")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_convergence(result: dict, out_path: str):
    """Plot Nusselt number convergence history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ref_nu = result.get("ref_values", {}).get("nu", 4.519)
    
    for idx, (nx_str, res) in enumerate(result["grids"].items()):
        nx = int(nx_str)
        hist = res.get("history", [])
        if not hist:
            continue
        
        steps = [h["step"] for h in hist]
        nus = [h["nu"] for h in hist]
        u_maxs = [h["u_max_nd"] for h in hist]
        v_maxs = [h["v_max_nd"] for h in hist]
        
        # Nu convergence
        axes[0].plot(steps, nus, label=f"N={nx}", linewidth=2)
        axes[0].axhline(ref_nu, color='red', linestyle='--', alpha=0.5, label=f"Ref={ref_nu}")
        
        # u_max convergence
        axes[1].plot(steps, u_maxs, label=f"N={nx}", linewidth=2)
        
        # v_max convergence
        axes[2].plot(steps, v_maxs, label=f"N={nx}", linewidth=2)
    
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Nu")
    axes[0].set_title("Nusselt Number Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("u_max (dimensionless)")
    axes[1].set_title("Maximum Horizontal Velocity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("v_max (dimensionless)")
    axes[2].set_title("Maximum Vertical Velocity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f"Ra={result['ra']:.0e}, Pr={result['pr']}", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_comparison(result: dict, out_path: str):
    """Plot comparison with reference values."""
    ref = result.get("ref_values", {})
    if not ref:
        print("[plot] No reference values, skipping comparison plot")
        return
    
    grids = result["grids"]
    nx_list = sorted([int(k) for k in grids.keys()])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    quantities = [
        ("nu", "Nusselt Number", ref.get("nu", 4.519)),
        ("u_max_nd", "u_max (dimensionless)", ref.get("u_max", 34.72)),
        ("v_max_nd", "v_max (dimensionless)", ref.get("v_max", 68.60)),
    ]
    
    for ax, (key, label, ref_val) in zip(axes, quantities):
        sim_vals = [grids[str(nx)]["obs"][key] for nx in nx_list]
        errors = [100.0 * abs(v - ref_val) / ref_val for v in sim_vals]
        
        x = np.arange(len(nx_list))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [ref_val]*len(nx_list), width, label='Reference', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, sim_vals, width, label='TensorLBM', color='blue', alpha=0.7)
        
        ax.set_xlabel("Grid Size N")
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n(Error: {', '.join([f'N{n}={e:.1f}%' for n, e in zip(nx_list, errors)])})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in nx_list])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"Comparison with de Vahl Davis 1983 (Ra={result['ra']:.0e})", 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_field_snapshots(result: dict, out_dir: str):
    """Generate actual field snapshots by re-running the simulation briefly."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Use the finest grid
    grids = result["grids"]
    nx_max = max(int(k) for k in grids.keys())
    res = grids[str(nx_max)]
    ra = result["ra"]
    
    print(f"[plot] Generating field snapshots for N={nx_max}, Ra={ra:.0e}")
    
    # Re-run to get fields
    device = torch.device("cpu")
    torch.set_num_threads(8)
    
    f, g, wall, p = driver.make_fields(nx_max, ra, 0.71, 0.6, device)
    
    # Run for a few steps to get a developed flow
    n_steps = min(50000, res.get("steps_run", 50000))
    print(f"[plot] Running {n_steps} steps to develop flow...")
    
    for step in range(1, n_steps + 1):
        f, g = driver.advance(f, g, wall, 0.6, p["tau_T"], p["g_beta"])
        if step % 10000 == 0:
            print(f"  step={step}", flush=True)
    
    # Extract fields
    rho, ux, uy = driver.macroscopic(f)
    T = g.sum(dim=0)
    
    # Convert to numpy
    T_np = T.detach().cpu().numpy()
    ux_np = ux.detach().cpu().numpy()
    uy_np = uy.detach().cpu().numpy()
    u_mag = np.sqrt(ux_np**2 + uy_np**2)
    
    # Plot temperature field
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(T_np, cmap='hot', origin='lower', aspect='equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Temperature Field (N={nx_max}, Ra={ra:.0e})")
    cbar = plt.colorbar(im, ax=ax, label="T")
    fig.savefig(f"{out_dir}/temperature_field.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out_dir}/temperature_field.png")
    
    # Plot velocity magnitude
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(u_mag, cmap='viridis', origin='lower', aspect='equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Velocity Magnitude (N={nx_max}, Ra={ra:.0e})")
    cbar = plt.colorbar(im, ax=ax, label="|u|")
    fig.savefig(f"{out_dir}/velocity_magnitude.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out_dir}/velocity_magnitude.png")
    
    # Plot streamlines
    fig, ax = plt.subplots(figsize=(8, 7))
    x = np.arange(nx_max)
    y = np.arange(nx_max)
    X, Y = np.meshgrid(x, y)
    
    # Subsample for clarity
    step = max(1, nx_max // 30)
    ax.streamplot(X[::step, ::step], Y[::step, ::step], 
                  ux_np[::step, ::step], uy_np[::step, ::step],
                  color=u_mag[::step, ::step], cmap='viridis', 
                  density=1.5, arrowsize=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Streamlines (N={nx_max}, Ra={ra:.0e})")
    ax.set_aspect('equal')
    fig.savefig(f"{out_dir}/streamlines.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved {out_dir}/streamlines.png")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="Path to result JSON")
    ap.add_argument("--out-dir", default="artifacts/figures_ra1e5")
    args = ap.parse_args()
    
    result = load_result(args.result)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[main] Generating figures for {args.result}")
    
    # Generate plots
    plot_convergence(result, str(out_dir / "convergence.png"))
    plot_comparison(result, str(out_dir / "comparison.png"))
    plot_field_snapshots(result, str(out_dir))
    
    print(f"\n[main] All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
