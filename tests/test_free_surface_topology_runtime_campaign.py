"""Actual closed-domain topology-changing free-surface runtime campaign."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, free_surface_step


def _direct_lg_links(flags: torch.Tensor) -> int:
    """Independent full-18 D3Q19 direct-L/G audit, including periodic seams."""
    liquid = flags == LIQUID
    sources = torch.stack([
        flags.roll((int(C[q, 2]), int(C[q, 1]), int(C[q, 0])), dims=(0, 1, 2)) == GAS
        for q in range(1, 19)
    ])
    return int((liquid.unsqueeze(0) & sources).sum())


def _topology_changing_closed_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Periodic compact interface seed whose centre converts on the real step."""
    shape = (5, 6, 7)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    fill = torch.zeros(shape)
    centre = (2, 3, 3)
    flags[centre] = INTERFACE
    fill[centre] = 1.0
    # A full initial D3Q19 interface envelope preserves the no-direct-L/G rule
    # when the centre becomes LIQUID on the first actual solver timestep.
    for q in range(1, 19):
        dz, dy, dx = int(C[q, 2]), int(C[q, 1]), int(C[q, 0])
        source = tuple(
            (index - delta) % extent
            for index, delta, extent in zip(centre, (dz, dy, dx), shape)
        )
        flags[source] = INTERFACE
        fill[source] = 0.5
    zero = torch.zeros(shape)
    populations = equilibrium3d(torch.ones(shape), zero, zero, zero)
    return populations, fill, flags, torch.zeros(shape, dtype=torch.bool)


def test_ten_actual_topology_changing_steps_have_local_paired_liquid_interface_budget() -> None:
    f, fill, flags, solid = _topology_changing_closed_state()
    mass = fill.clone()
    initial_mass = float(mass.sum())
    runtime: dict[str, object] = {}
    liquid_counts = [int((flags == LIQUID).sum())]

    for _ in range(10):
        assert _direct_lg_links(flags) == 0
        f, fill, flags, mass, _ = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=1.0, rho_gas=1.0e-3,
            runtime_ledger=runtime, paired_liquid_interface_debit=True,
        )
        assert _direct_lg_links(flags) == 0
        assert bool(torch.isfinite(f).all())
        assert bool(torch.isfinite(fill).all())
        assert bool(torch.isfinite(mass).all())
        liquid_counts.append(int((flags == LIQUID).sum()))

    steps = runtime["steps"]
    assert isinstance(steps, list)
    assert len(steps) == 10
    assert max(liquid_counts) > min(liquid_counts)  # actual conversion changed topology
    assert abs(float(mass.sum()) - initial_mass) <= 3.0e-6

    for step in steps:
        assert step["directLG"] == step["direct_liquid_gas_links"] == 0
        assert step["paired"] is True
        assert abs(float(step["paired_residual"])) <= 1.0e-6
        assert float(step["drift"]) == pytest.approx(float(step["mass_drift"]), abs=0.0)
        assert float(step["unexplained"]) == pytest.approx(float(step["unexplained_residual"]), abs=0.0)
        assert float(step["conversion"]) == pytest.approx(
            float(step["mass_after_conversion"]) - float(step["mass_after_clamp"]), abs=1.0e-8
        )
        assert float(step["redistribution"]) == pytest.approx(
            float(step["mass_after_redistribution"]) - float(step["mass_after_exchange"]), abs=1.0e-8
        )

    assert any(abs(float(step["conversion"])) > 1.0e-4 for step in steps)
    assert any(abs(float(step["redistribution"])) > 1.0e-4 for step in steps)
