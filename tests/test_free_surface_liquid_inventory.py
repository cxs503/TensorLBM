"""Closed-domain liquid-inventory accounting for the D3Q19 free-surface solver.

The inventory intentionally combines compressible bulk liquid density with the
bounded interface fill mass; it is not the legacy ``mass.sum()`` accumulator.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
    init_fill_rectangular,
    init_flags_from_fill,
    init_mass_from_fill,
    total_liquid_inventory,
)


def _inventory(f: torch.Tensor, fill: torch.Tensor, flags: torch.Tensor) -> float:
    return float(total_liquid_inventory(f, fill, flags, rho_liquid=1.0))


def test_inventory_counts_bulk_density_but_interface_fill_mass() -> None:
    """Density fluctuations change bulk inventory without changing fill volume."""
    flags = torch.tensor([[[LIQUID, LIQUID, INTERFACE, GAS]]], dtype=torch.int8)
    fill = torch.tensor([[[1.0, 1.0, 0.25, 0.0]]])
    rho = torch.tensor([[[0.90, 1.20, 0.25, 9.0]]])
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, zero, zero, zero)

    # The liquid density fluctuations sum to 2.1; the interface contribution
    # is rho_liquid * fill = 0.25.  Gas density is deliberately irrelevant.
    assert _inventory(f, fill, flags) == pytest.approx(2.35, abs=1.0e-6)
    assert float(fill.sum()) != pytest.approx(_inventory(f, fill, flags))


def _closed_dam_break_state() -> tuple[torch.Tensor, ...]:
    fill, solid = init_fill_rectangular(4, 12, 14, 5, 5, device="cpu")
    # Close the periodic z faces too; all liquid transport remains internal.
    solid[0, :, :] = solid[-1, :, :] = True
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, zero, zero, zero), fill, flags, solid, mass


def _closed_gravity_wave_state() -> tuple[torch.Tensor, ...]:
    nz, ny, nx = 4, 18, 32
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 0, :] = solid[:, -1, :] = True
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    y = torch.arange(ny, dtype=torch.float32).view(1, ny, 1)
    fill = torch.clamp(8.0 + 0.25 * torch.cos(2.0 * math.pi * x / nx) - y, 0.0, 1.0)
    fill = fill.expand(nz, -1, -1).clone()
    fill[solid] = 0.0
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, zero, zero, zero), fill, flags, solid, mass


def _advance(f, fill, flags, solid, mass, *, steps: int, gy: float) -> tuple[torch.Tensor, ...]:
    for _ in range(steps):
        f, fill, flags, mass, _ = free_surface_step(
            f,
            fill,
            flags,
            solid,
            mass=mass,
            tau=0.8,
            gy=gy,
            rho_liquid=1.0,
            rho_gas=0.001,
        )
    return f, fill, flags, mass


@pytest.mark.xfail(
    strict=True,
    reason="P0 blocker: ABB/population and interface-mass ledgers have no conservative common flux",
)
def test_closed_static_plane_conserves_composite_liquid_inventory() -> None:
    nz, ny, nx = 8, 12, 12
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 0, :] = solid[:, -1, :] = True
    fill = torch.zeros((nz, ny, nx))
    fill[1:-1, 5, 1:-1] = 0.5
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    f = equilibrium3d(rho, zero, zero, zero)
    initial = _inventory(f, fill, flags)
    f, fill, flags, _ = _advance(f, fill, flags, solid, mass, steps=4, gy=0.0)
    assert _inventory(f, fill, flags) == pytest.approx(initial, abs=1.0e-5 * initial)


@pytest.mark.xfail(
    strict=True,
    reason="P0 blocker: liquid/interface link flux updates only the interface mass field",
)
def test_closed_dam_break_conserves_composite_liquid_inventory() -> None:
    f, fill, flags, solid, mass = _closed_dam_break_state()
    initial = _inventory(f, fill, flags)
    f, fill, flags, _ = _advance(f, fill, flags, solid, mass, steps=4, gy=-1.0e-4)
    assert _inventory(f, fill, flags) == pytest.approx(initial, abs=2.0e-4 * initial)


def test_closed_gravity_wave_conserves_composite_liquid_inventory() -> None:
    f, fill, flags, solid, mass = _closed_gravity_wave_state()
    initial = _inventory(f, fill, flags)
    f, fill, flags, _ = _advance(f, fill, flags, solid, mass, steps=1, gy=-1.0e-4)
    assert _inventory(f, fill, flags) == pytest.approx(initial, abs=3.0e-5 * initial)
