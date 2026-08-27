"""Frozen mixed-topology ledger for the Körner free-surface mass stencil.

This intentionally holds flags fixed so exchange and ABB reconstruction can be
measured without redistribution/conversion/halo propagation.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
    total_liquid_inventory,
)


def _mixed_liquid_interface_gas_state() -> tuple[torch.Tensor, ...]:
    """Periodic x strips with every LIQUID/GAS link covered by INTERFACE.

    A periodic ``G | G | I | L | L | L`` strip is invalid: its x-wrap creates
    a direct L/G D3Q19 link which has neither ABB reconstruction nor an
    interface mass exchange.  Keep an interface cell at x=0 so this fixture
    tests the L/I ledger rather than an unsupported direct L/G boundary.
    """
    nz, ny, nx = 3, 4, 6
    flags = torch.full((nz, ny, nx), GAS, dtype=torch.int8)
    flags[:, :, 0] = INTERFACE
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3:] = LIQUID
    solid = torch.zeros_like(flags, dtype=torch.bool)
    fill = torch.zeros((nz, ny, nx))
    fill[flags == INTERFACE] = 0.5
    fill[flags == LIQUID] = 1.0
    mass = fill.clone()
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    # A nonuniform velocity makes the L/I link transfer nonzero, so this is
    # not merely a symmetric equilibrium cancellation.
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    ux = 0.04 * torch.sin(2.0 * torch.pi * x / nx).expand_as(fill)
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, ux, zero, zero), fill, flags, solid, mass


def test_frozen_mixed_topology_ledger_attributes_exchange_to_liquid_interface_rule() -> None:
    """Freeze flags and expose the exact mass change of the L/I rule.

    This is intentionally diagnostic rather than a conservation assertion:
    the standard stencil updates only the interface endpoint of a liquid /
    interface link.  A correction needs an independently specified matching
    liquid-cell mass rule, not a global total-mass patch.
    """
    f, fill, flags, solid, mass = _mixed_liquid_interface_gas_state()
    ledger: dict[str, float] = {}
    _, out_fill, out_flags, out_mass, _ = free_surface_step(
        f,
        fill,
        flags,
        solid,
        mass=mass,
        tau=1.0,
        rho_gas=0.001,
        mass_ledger=ledger,
        freeze_topology=True,
    )

    assert torch.equal(out_flags, flags)
    assert torch.allclose(out_fill, out_mass)
    assert ledger["interface_start"] == 12.0
    assert ledger["liquid_start"] == 36.0
    assert ledger["gas_start"] == 0.0
    assert ledger["exchange_gas_delta"] == 0.0
    assert abs(ledger["exchange_liquid_delta"] + 0.4156921) < 2.0e-6
    assert abs((ledger["exchange"] - ledger["start"]) - ledger["exchange_liquid_delta"]) < 2.0e-6


def test_frozen_mixed_topology_conserves_composite_liquid_inventory() -> None:
    """A closed, frozen L/I/G topology must conserve the physical inventory."""
    f, fill, flags, solid, mass = _mixed_liquid_interface_gas_state()
    before = float(total_liquid_inventory(f, fill, flags))

    out_f, out_fill, out_flags, _, _ = free_surface_step(
        f,
        fill,
        flags,
        solid,
        mass=mass,
        tau=1.0,
        rho_gas=0.001,
        freeze_topology=True,
    )

    after = float(total_liquid_inventory(out_f, out_fill, out_flags))
    assert abs(after - before) < 2.0e-6


def test_direct_liquid_gas_link_is_rejected_in_frozen_topology() -> None:
    """Körner free-surface states must separate liquid and gas by interface."""
    f, fill, flags, solid, mass = _mixed_liquid_interface_gas_state()
    flags[:, :, 0] = GAS  # periodic x-wrap creates direct LIQUID/GAS links
    fill[:, :, 0] = 0.0
    mass[:, :, 0] = 0.0

    with pytest.raises(ValueError, match="LIQUID.*GAS.*INTERFACE"):
        free_surface_step(
            f,
            fill,
            flags,
            solid,
            mass=mass,
            tau=1.0,
            rho_gas=0.001,
            freeze_topology=True,
        )
