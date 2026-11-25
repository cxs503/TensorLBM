"""Cumulant collision probe for single-drop water entry.

This module provides diagnostic utilities for the D3Q19 cumulant collision
operator in the Körner free-surface solver.  It measures:

1. Mass conservation drift across collision steps
2. Ghost-mode decay rates (3rd/4th order)
3. Stability at high lattice Mach numbers (Ma > 0.2)
4. Comparison with BGK/MRT baselines

Usage
-----
Run from the repository root::

    python -m benchmarks.water_entry.single_drop.cumulant_probe \\
        --tau 0.6 --C_s 0.0 --collision cumulant \\
        --nz 40 --ny 40 --nx 40 --drop_radius 10 \\
        --impact_speed 0.15 --output results/cumulant_probe.npz

The probe runs a single-drop water-entry simulation and writes diagnostic
data to the specified output file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the source directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
    init_f_distribution,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cumulant collision probe for single-drop water entry"
    )
    parser.add_argument("--tau", type=float, default=0.6, help="Shear relaxation time")
    parser.add_argument("--C_s", type=float, default=0.0, help="Subgrid-scale coefficient")
    parser.add_argument(
        "--collision", type=str, default="cumulant", choices=["bgk", "mrt", "cumulant"]
    )
    parser.add_argument("--nz", type=int, default=40, help="Grid size (z)")
    parser.add_argument("--ny", type=int, default=40, help="Grid size (y)")
    parser.add_argument("--nx", type=int, default=40, help="Grid size (x)")
    parser.add_argument(
        "--drop_radius", type=float, default=10.0, help="Drop radius in lattice units"
    )
    parser.add_argument(
        "--impact_speed", type=float, default=0.15, help="Impact speed in lattice units"
    )
    parser.add_argument("--num_steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument(
        "--output", type=str, default="results/cumulant_probe.npz", help="Output file path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def init_drop_water_entry(nz, ny, nx, drop_radius, impact_speed, device="cpu"):
    """Initialize single-drop water entry configuration.

    Args:
        nz, ny, nx: Grid dimensions.
        drop_radius: Drop radius in lattice units.
        impact_speed: Impact speed in lattice units (positive = downward).
        device: Torch device.

    Returns:
        flags, mass, f, rho_liquid, fill, body_force
    """
    # Initialize flags: GAS everywhere, then create drop as LIQUID
    flags = torch.full((nz, ny, nx), GAS, dtype=torch.int32, device=device)
    mass = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)

    # Drop center
    cx, cy, cz = nx // 2, ny // 2, nz // 4
    r2 = drop_radius * drop_radius

    # Create drop as LIQUID
    y, x = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
    )
    z = torch.arange(nz, device=device)

    # Drop sphere
    drop_mask = ((x - cx) ** 2 + (y - cy) ** 2 + (z[:, None, None] - cz) ** 2) <= r2
    flags[drop_mask] = LIQUID
    mass[drop_mask] = 1.0  # Liquid fill = 1
    fill[drop_mask] = 1.0

    # Initialize free surface: INTERFACE cells around the drop
    # Simple approach: mark cells adjacent to LIQUID as INTERFACE
    interface_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                shifted = torch.roll(drop_mask.long(), shifts=(dz, dy, dx), dims=(0, 1, 2)).bool()
                interface_mask |= shifted & ~drop_mask
    flags[interface_mask] = INTERFACE
    mass[interface_mask] = 0.5  # Interface fill = 0.5
    fill[interface_mask] = 0.5

    # Initialize distributions
    rho_liquid = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    rho_gas = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)

    # Body force: gravity in -y direction (impact speed is downward)
    body_force = torch.zeros((nz, ny, nx, 3), dtype=torch.float32, device=device)
    body_force[:, :, :, 1] = -0.005  # Gravity

    # Initial velocity: drop moving downward at impact_speed
    ux = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    uy = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    uz = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    uy[flags == LIQUID] = -impact_speed  # Negative y = downward

    f = init_f_distribution(flags, rho_liquid, ux, uy, uz, device=device)

    return flags, mass, f, rho_liquid, fill, body_force


def run_probe(args):
    """Run the cumulant collision probe."""
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    print(f"Initializing single-drop water entry: {args.nz}x{args.ny}x{args.nx}")
    print(f"Drop radius: {args.drop_radius}, Impact speed: {args.impact_speed}")
    print(f"Collision: {args.collision}, tau: {args.tau}, C_s: {args.C_s}")

    flags, mass, f, rho_liquid, fill, body_force = init_drop_water_entry(
        args.nz, args.ny, args.nx, args.drop_radius, args.impact_speed, device
    )

    # Diagnostic storage
    mass_history = []
    ghost_mode_history = []
    divergence_detected = False

    print(f"Running {args.num_steps} steps...")
    for step in range(args.num_steps):
        # Run one free-surface step
        f, flags, mass, rho_liquid, fill, ledger = free_surface_step(
            f,
            flags,
            mass,
            rho_liquid,
            fill,
            body_force,
            tau=args.tau,
            C_s=args.C_s,
            collision=args.collision,
            device=device,
        )

        # Check for divergence
        if torch.isnan(f).any() or torch.isinf(f).any():
            print(f"Divergence detected at step {step}")
            divergence_detected = True
            break

        # Track mass conservation
        mass_total = mass.sum().item()
        mass_history.append(mass_total)

        # Track ghost modes (simplified: check non-equilibrium stress norm)
        from tensorlbm.d3q19 import equilibrium3d, macroscopic3d

        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        ghost_norm = torch.sqrt((f_neq**2).sum(0)).mean().item()
        ghost_mode_history.append(ghost_norm)

        if step % 50 == 0:
            print(f"Step {step}: mass={mass_total:.6f}, ghost_norm={ghost_norm:.6e}")

    # Compute diagnostics
    mass_start = mass_history[0] if mass_history else 1.0
    mass_end = mass_history[-1] if mass_history else 1.0
    mass_drift = abs(mass_end - mass_start) / mass_start

    result = {
        "args": vars(args),
        "mass_history": mass_history,
        "ghost_mode_history": ghost_mode_history,
        "mass_drift_relative": mass_drift,
        "divergence_detected": divergence_detected,
        "num_steps": len(mass_history),
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy for saving
    np.savez(
        str(output_path),
        mass_history=np.array(mass_history),
        ghost_mode_history=np.array(ghost_mode_history),
        mass_drift_relative=np.float64(mass_drift),
        divergence_detected=np.bool_(divergence_detected),
        args=json.dumps(vars(args)),
    )

    print(f"Results saved to {output_path}")
    print(f"Mass drift: {mass_drift:.2e} (relative)")
    print(f"Divergence: {divergence_detected}")

    return result


def main():
    """Main entry point."""
    args = parse_args()
    result = run_probe(args)

    # Print summary
    print("\n=== Cumulant Collision Probe Summary ===")
    print(f"Collision operator: {args.collision}")
    print(f"Mass drift (relative): {result['mass_drift_relative']:.2e}")
    print(f"Divergence detected: {result['divergence_detected']}")
    print(
        f"Final ghost mode norm: {result['ghost_mode_history'][-1]:.2e}"
        if result["ghost_mode_history"]
        else "N/A"
    )

    # Pass/fail based on predeclared thresholds
    if result["mass_drift_relative"] > 1e-5:
        print("FAIL: Mass drift exceeds threshold (1e-5)")
    elif result["divergence_detected"]:
        print("FAIL: Divergence detected")
    else:
        print("PASS: All thresholds met")


if __name__ == "__main__":
    main()
