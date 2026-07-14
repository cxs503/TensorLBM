"""Runtime regression for the L/I paired bulk-debit mass budget."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, free_surface_step


def _closed_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Periodic, topology-frozen G/I/L strip with no direct L/G D3Q19 link."""
    nz, ny, nx = 3, 3, 5
    flags = torch.full((nz, ny, nx), GAS, dtype=torch.int8)
    # I | G | I | L | L, including the periodic x seam.
    flags[:, :, 0] = INTERFACE
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3:] = LIQUID
    fill = torch.zeros((nz, ny, nx))
    fill[flags == INTERFACE] = 0.5
    fill[flags == LIQUID] = 1.0
    solid = torch.zeros_like(flags, dtype=torch.bool)
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    ux = 0.025 * torch.sin(2.0 * torch.pi * x / nx).expand_as(fill)
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, ux, zero, zero), fill, flags, solid


def _run(*, paired: bool) -> list[dict[str, object]]:
    f, fill, flags, solid = _closed_state()
    mass = fill.clone()
    runtime: dict[str, object] = {}
    for _ in range(3):
        f, fill, flags, mass, _ = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=1.0, rho_gas=0.001,
            freeze_topology=True, runtime_ledger=runtime,
            paired_liquid_interface_debit=paired,
        )
    steps = runtime["steps"]
    assert isinstance(steps, list)
    return steps  # type: ignore[return-value]


def test_three_step_closed_runtime_pairs_each_liquid_interface_credit_without_global_correction() -> None:
    legacy = _run(paired=False)
    paired = _run(paired=True)
    assert len(legacy) == len(paired) == 3

    for old, new in zip(legacy, paired):
        assert old["direct_liquid_gas_links"] == new["direct_liquid_gas_links"] == 0
        assert old["liquid_interface_paired"] is False
        assert new["liquid_interface_paired"] is True
        assert abs(float(new["liquid_interface_interface_credit"])) > 1.0e-6
        assert float(new["liquid_interface_bulk_debit"]) == pytest.approx(
            -float(new["liquid_interface_interface_credit"]), abs=5.0e-6
        )
        assert float(new["liquid_interface_paired_residual"]) == pytest.approx(0.0, abs=5.0e-6)
        # The legacy mode's tracked-mass drift is precisely its unpaired L/I
        # credit; paired mode removes that contribution link by link.
        assert float(old["unexplained_residual"]) == pytest.approx(
            float(old["liquid_interface_interface_credit"]), abs=5.0e-6
        )
        assert abs(float(new["mass_drift"])) < abs(float(old["mass_drift"])) + 5.0e-6
        assert abs(float(new["unexplained_residual"])) < 5.0e-6
        assert new["closed_domain_conserved"] is True
        assert "not a physical/PV closure claim" in str(new["diagnostic"])
