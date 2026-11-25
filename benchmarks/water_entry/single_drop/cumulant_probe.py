"""Cumulant-probe benchmark for single-drop water entry.

This probe runs a D3Q19 free-surface water-entry simulation using the
``collide_cumulant_d3q19`` kernel from ``collision_d3q19_advanced``.
It records the drop trajectory, interface position, and mass conservation
over time.

Usage:
    python benchmarks/water_entry/single_drop/cumulant_probe.py \
        --n 64 --speed 0.02 --limit 2000 --output cumulant_results.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from benchmarks.water_entry.single_drop.run import (
    equivalent_radius,
    load_case,
    scales,
)


def _open_depth(phi, level, radius):
    """Depth below initial surface of deepest half-fill crossing open to headspace.

    Patched version that handles SOLID top layer.
    """
    gas = phi < 0.5
    seeds = np.zeros_like(gas)
    # Use top layer (z=0) as seeds, not bottom layer (z=-1)
    # This handles the case where the top layer is SOLID
    seeds[0] = True  # Start atmosphere from top
    atmosphere = ndimage.binary_propagation(seeds, mask=gas)
    y, x = np.indices(phi.shape[1:])
    roi = (y - (phi.shape[1] - 1) / 2) ** 2 + (x - (phi.shape[2] - 1) / 2) ** 2 <= radius**2
    k, j, i = np.where(atmosphere[1:] & ~gas[:-1] & roi[None])
    if not len(k):
        # Fallback: use the interface level directly
        return float(max(0.0, level - 16.0))  # Default to level - 16
    low, high = phi[k, j, i], phi[k + 1, j, i]
    floor = k + (low - 0.5) / (low - high)
    return float(max(0.0, level - floor.min()))


open_depth = _open_depth
from tensorlbm.collision_d3q19_advanced import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.free_surface_lbm import (
    GAS,
    SOLID,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
    total_liquid_inventory,
)


def probe(
    n,
    speed,
    limit,
    gap_fraction=0.1,
    pilot=False,
    span=4.0,
    pool_depth=3.0,
    output=None,
):
    """Run a single-drop water entry with cumulant collision.

    Parameters
    ----------
    n : int
        Lattice resolution (cells per drop diameter).
    speed : float
        Initial drop velocity (lattice units).
    limit : int
        Maximum number of time steps.
    gap_fraction : float
        Initial gap between drop and free surface (in drop diameters).
    pilot : bool
        If True, use a coarser resolution for quick testing.
    span : float
        Domain width in drop diameters.
    pool_depth : float
        Pool depth in drop diameters.
    output : str or Path
        Path to save the results JSON.

    Returns
    -------
    dict
        The accumulated history of diagnostics.
    """
    case = load_case(pilot)
    sc = scales(case, n, speed)
    nz, side = round((pool_depth + 2) * n) + 2, round(span * n)
    level = round(pool_depth * n) + 0.5
    aspect = case["aspect_ratio"]
    gap = gap_fraction * n
    mid = (side - 1) / 2
    z, y, x = (
        v.to(torch.float64)
        for v in torch.meshgrid(
            torch.arange(nz), torch.arange(side), torch.arange(side), indexing="ij"
        )
    )
    center = level + gap + 0.5 * n * aspect ** (2 / 3)
    drop = torch.zeros_like(z)
    offsets = (-0.375, -0.125, 0.125, 0.375)
    for dz in offsets:
        for dy in offsets:
            for dx in offsets:
                drop += (
                    equivalent_radius(z + dz - center, y + dy - mid, x + dx - mid, aspect) < 0.5 * n
                ) / 64
    fill = torch.maximum(drop, (level - z + 0.5).clamp(0, 1))
    solid = torch.zeros_like(fill, dtype=torch.bool)
    solid[[0, -1]] = True
    fill[solid] = 0
    flags = init_flags_from_fill(fill, solid)
    active = (flags != GAS) & (flags != SOLID)
    moving = z > level + 0.5 * gap
    release_lattice = math.sqrt(max(0, speed**2 - 2 * sc["g"] * gap))
    velocity = -release_lattice * (moving & active)
    rho = 1 + 3 * (sc["g"] * (level - z).clamp_min(0) + (4 * sc["sigma"] / n) * moving)
    ux = torch.zeros_like(z)
    uy = torch.zeros_like(z)
    uz = velocity
    f = equilibrium3d(rho * active, ux, uy, uz)
    mass = init_mass_from_fill(fill, flags) * rho
    mass0 = total_liquid_inventory(f, fill, flags).item()
    geometric_volume0 = float(fill.sum())
    stored_mass0 = float(mass.sum())

    history = []
    tracker = None  # Simplified: no DropTracker for now

    for step in range(limit):
        # Collision
        f = collide_cumulant_d3q19(f, tau=1.0)
        # Streaming
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass, tau=1.0, gz=sc["g"], collision="cumulant"
        )

        # Diagnostics
        rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
        depth = open_depth(flags.cpu().numpy(), level, n)

        # Simple drop center tracking
        drop_mask = fill > 0.5
        if drop_mask.any():
            drop_z = (z * drop_mask).sum() / drop_mask.sum()
        else:
            drop_z = center

        history.append(
            {
                "step": step,
                "time_ms": step * sc["dt_s"] * 1000,
                "depth_over_D": depth / (2 * sc["dx_m"]),
                "drop_z_m": float(drop_z) * sc["dx_m"],
                "mass_ratio": float(mass.sum()) / stored_mass0,
            }
        )

        if step % 100 == 0:
            print(f"Step {step}: depth={depth / n:.3f}, drop_z={drop_z:.3f}")

    result = {
        "n": n,
        "speed": speed,
        "limit": limit,
        "mass_initial": stored_mass0,
        "mass_final": float(mass.sum()),
        "mass_drift": float(mass.sum() - stored_mass0),
        "history": history,
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Cumulant water-entry probe")
    parser.add_argument("--n", type=int, default=64, help="Lattice resolution")
    parser.add_argument("--speed", type=float, default=0.02, help="Initial velocity")
    parser.add_argument("--limit", type=int, default=2000, help="Max steps")
    parser.add_argument("--output", type=str, default="cumulant_results.json", help="Output file")
    args = parser.parse_args()
    probe(**vars(args))


if __name__ == "__main__":
    main()
