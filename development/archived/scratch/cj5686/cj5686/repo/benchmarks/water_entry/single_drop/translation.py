"""Check transport of a detached water sphere with a known translational solution."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from benchmarks.water_entry.single_drop.run import load_case, scales
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
)
from tensorlbm.free_surface_topology_transaction import TopologyTransactionError


def translate(n, steps=40):
    # At uniform pressure and velocity, with no gravity or surface tension,
    # the analytical free surface translates at U without changing shape.
    speed = 1 / 16
    side = 3 * n + 12
    z, y, x = torch.meshgrid(*(torch.arange(side, dtype=torch.float64),) * 3, indexing="ij")
    mid = (side - 1) / 2
    fill = torch.zeros_like(z)
    for dz, dy, dx in itertools.product((-0.375, -0.125, 0.125, 0.375), repeat=3):
        fill += (
            (z + dz - mid) ** 2 + (y + dy - mid) ** 2 + (x + dx - mid) ** 2 < (n / 2) ** 2
        ) / 64
    solid = torch.zeros_like(fill, dtype=torch.bool)
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    density = (flags != GAS).to(fill.dtype)
    zero = torch.zeros_like(fill)
    f = equilibrium3d(density, zero, zero, -speed * density)
    # The same liquid viscosity in lattice units isolates resolution of a
    # fragment inside the 8-cell/D impact calculation.
    tau = 0.5 + 3 * scales(load_case(), 8, 0.06)["mu_l"]
    rows = []
    reason = "window_complete"
    for i in range(steps + 1):
        rho, _, _, uz = macroscopic3d(f)
        weight = torch.where(flags == LIQUID, rho, torch.where(flags == INTERFACE, mass, 0.0))
        displacement = mid - float((z * fill).sum() / fill.sum())
        rows.append(
            dict(
                step=i,
                geometric_displacement_cells=displacement,
                analytic_displacement_cells=i * speed,
                water_mass=float(weight.sum()),
                mass_weighted_downward_velocity=float(-(weight * uz).sum() / weight.sum()),
            )
        )
        if i < steps:
            try:
                f, fill, flags, mass, _ = free_surface_step(
                    f,
                    fill,
                    flags,
                    solid,
                    mass=mass,
                    tau=tau,
                    rho_gas=1.0,
                    gz=0.0,
                    surface_tension=0.0,
                    capacity_redistribution=True,
                    simple_mass_exchange=True,
                )
            except TopologyTransactionError as error:
                reason = str(error)
                break
    return dict(
        diameter_cells=n,
        speed_cells_per_step=speed,
        relaxation_time=tau,
        stop_reason=reason,
        history=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = [translate(n) for n in (2, 4, 8, 12)]
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
