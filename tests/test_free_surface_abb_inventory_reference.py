"""TDD specification for isolated ABB population-to-inventory accounting."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.free_surface_abb_inventory import (
    ABBInventoryOwnershipError,
    ABBInventoryState,
    abb_reconstruction_density_change,
    apply_closed_abb_inventory_transaction,
)
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, _stream19_roll, free_surface_step


def _state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One frozen GAS/INTERFACE/LIQUID strip with nonzero ABB reconstruction."""
    nz, ny, nx = 3, 3, 5
    flags = torch.full((nz, ny, nx), GAS, dtype=torch.int8)
    # I | G | I | L | L is valid under periodic x streaming.
    flags[:, :, 0] = INTERFACE
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3:] = LIQUID
    fill = torch.zeros((nz, ny, nx))
    fill[flags == INTERFACE] = 0.50
    fill[flags == LIQUID] = 1.0
    solid = torch.zeros_like(flags, dtype=torch.bool)
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    ux = 0.025 * torch.sin(2.0 * torch.pi * x / nx).expand_as(fill)
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, ux, zero, zero), fill, flags, solid, ux


def test_reference_maps_the_current_solver_abb_reconstruction_link_by_link() -> None:
    f, fill, flags, solid, ux = _state()
    ledger: dict[str, float] = {}
    out, _, out_flags, _, _ = free_surface_step(
        f, fill, flags, solid, mass=fill.clone(), tau=1.0, rho_gas=0.001,
        mass_ledger=ledger, freeze_topology=True,
    )
    rho, solver_ux, solver_uy, solver_uz = macroscopic3d(f)
    # With tau=1, the solver's BGK collision replaces f with its local feq.
    f_post = torch.where(
        (flags != GAS).unsqueeze(0),
        equilibrium3d(rho.clamp(min=1e-6, max=3.0), solver_ux, solver_uy, solver_uz),
        torch.zeros_like(f),
    )
    # Solver zeroes pre-existing GAS cells after pull streaming, before ABB.
    streamed = torch.where((flags != GAS).unsqueeze(0), _stream19_roll(f_post), torch.zeros_like(f_post))
    change = abb_reconstruction_density_change(
        f_post, streamed, flags, rho_gas=0.001, ux=solver_ux,
        uy=solver_uy, uz=solver_uz,
    )

    assert torch.equal(out_flags, flags)
    assert torch.allclose(out - streamed, change.per_link_delta, atol=1.0e-7, rtol=0.0)
    assert float(change.per_link_delta.abs().sum()) > 1.0e-6
    assert change.population_delta == pytest.approx(ledger["abb_population_delta"], abs=2.0e-6)


def test_abb_inventory_change_requires_an_explicit_liquid_bulk_owner_per_link() -> None:
    f, _, flags, _, ux = _state()
    rho, solver_ux, solver_uy, solver_uz = macroscopic3d(f)
    f_post = torch.where((flags != GAS).unsqueeze(0), f, torch.zeros_like(f))
    change = abb_reconstruction_density_change(
        f_post, _stream19_roll(f_post), flags, rho_gas=0.001, ux=solver_ux,
        uy=solver_uy, uz=solver_uz,
    )
    inventory = ABBInventoryState.from_flags_and_values(
        flags, bulk_liquid=torch.where(flags == LIQUID, torch.full_like(ux, 10.0), torch.zeros_like(ux)),
        interface_inventory=torch.where(flags == INTERFACE, torch.full_like(ux, 10.0), torch.zeros_like(ux)),
    )

    with pytest.raises(ABBInventoryOwnershipError, match="explicit LIQUID bulk owner"):
        apply_closed_abb_inventory_transaction(inventory, change)


def test_closed_frozen_abb_transaction_conserves_when_every_link_has_an_owner() -> None:
    f, _, flags, _, ux = _state()
    rho, solver_ux, solver_uy, solver_uz = macroscopic3d(f)
    f_post = torch.where((flags != GAS).unsqueeze(0), f, torch.zeros_like(f))
    change = abb_reconstruction_density_change(
        f_post, _stream19_roll(f_post), flags, rho_gas=0.001, ux=solver_ux,
        uy=solver_uy, uz=solver_uz,
    )
    inventory = ABBInventoryState.from_flags_and_values(
        flags, bulk_liquid=torch.where(flags == LIQUID, torch.full_like(ux, 10.0), torch.zeros_like(ux)),
        interface_inventory=torch.where(flags == INTERFACE, torch.full_like(ux, 10.0), torch.zeros_like(ux)),
    )
    # This is an accounting reference, not a solver correction: choosing this
    # nearby liquid strip as owner is explicit policy supplied by the caller.
    owner = torch.full_like(change.per_link_delta, -1, dtype=torch.long)
    # Flat flattened index for (z=0, y=0, x=3), a LIQUID cell.
    owner[change.link_mask] = 3
    after, transaction = apply_closed_abb_inventory_transaction(inventory, change, bulk_owner=owner)

    assert transaction.interface_delta == pytest.approx(change.population_delta, abs=2.0e-6)
    assert transaction.bulk_delta == pytest.approx(-change.population_delta, abs=2.0e-6)
    assert transaction.total_delta == pytest.approx(0.0, abs=2.0e-6)
    assert after.total == pytest.approx(inventory.total, abs=5.0e-5)


def test_multistep_closed_runtime_attributes_unexplained_one_sided_exchange_drift() -> None:
    f, fill, flags, solid, _ = _state()
    mass = fill.clone()
    runtime: dict[str, object] = {}
    for _ in range(3):
        f, fill, flags, mass, _ = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=1.0, rho_gas=0.001,
            freeze_topology=True, runtime_ledger=runtime,
        )

    steps = runtime["steps"]
    assert isinstance(steps, list) and len(steps) == 3
    assert all(step["direct_liquid_gas_links"] == 0 for step in steps)
    assert all(step["mass_unit"] == "lattice liquid mass (sum of independent mass field)" for step in steps)
    assert all(not step["closed_domain_conserved"] for step in steps)
    assert all(abs(step["unexplained_residual"]) > 1.0e-6 for step in steps)
    assert all(step["unexplained_residual"] == pytest.approx(step["liquid_interface_interface_credit"], abs=5.0e-6) for step in steps)
    assert all("one-sided liquid/interface" in step["diagnostic"] for step in steps)
