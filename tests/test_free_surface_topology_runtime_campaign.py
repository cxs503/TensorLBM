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


def test_runtime_ledger_separates_gross_operator_activity_from_drift_attribution() -> None:
    """Cancellation is gross activity, not evidence of a drift root cause.

    This deliberately exercises the real conversion path, rather than feeding
    synthetic ledger numbers into a reporting helper. ABB is retained as a
    population-only observation.  Steps 3 and 5 have real, opposite conversion
    and redistribution deltas; their roundoff drift must be withheld rather
    than falsely assigned to the larger gross operator.
    """
    f, fill, flags, solid = _topology_changing_closed_state()
    runtime: dict[str, object] = {}
    mass = fill.clone()
    for _ in range(10):
        f, fill, flags, mass, _ = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=1.0, rho_gas=1.0e-3,
            runtime_ledger=runtime, paired_liquid_interface_debit=True,
        )

    steps = runtime["steps"]
    curve = runtime["operator_curve"]
    assert isinstance(steps, list) and isinstance(curve, list)
    assert len(curve) == len(steps) == 10
    for step, point in zip(steps, curve):
        attribution = step["operator_attribution"]
        assert point["step"] == step["step"]
        assert attribution["gross_activity_event_id"].startswith(f"step:{step['step']}:operator:")
        assert attribution["gross_activity_operator"] in {
            "conversion", "redistribution", "clamp", "isolation", "boundary",
            "abb", "interface_paired_debit",
        }
        events = attribution["events"]
        assert {event["operator"] for event in events} == {
            "conversion", "redistribution", "clamp", "isolation", "boundary",
            "abb", "interface_paired_debit",
        }
        assert all(event["event_id"].startswith(f"step:{step['step']}:operator:") for event in events)
        abb = next(event for event in events if event["operator"] == "abb")
        assert abb["tracked_mass"] is False
        reconciliation = step["residual_reconciliation"]
        assert reconciliation["sum_tracked_deltas"] == pytest.approx(
            reconciliation["expected_drift"], abs=0.0
        )
        assert reconciliation["observed_drift"] == pytest.approx(step["mass_drift"], abs=0.0)

    for index in (2, 4):  # steps 3 and 5
        step = steps[index]
        attribution = step["operator_attribution"]
        assert abs(float(step["conversion"])) > 1.0e-4
        assert abs(float(step["redistribution"])) > 1.0e-4
        assert float(step["conversion"]) * float(step["redistribution"]) < 0.0
        assert attribution["gross_activity_operator"] in {"conversion", "redistribution"}
        assert attribution["dominant_operator"] == "withheld/unexplained"
        assert attribution["dominant_event_id"] is None
        assert attribution["reason"] in {
            "multiple_tracked_operators_no_unique_residual_cause",
            "tracked_deltas_do_not_reconcile_observed_drift",
        }
        assert abs(float(step["residual_reconciliation"]["residual"])) <= 1.0e-6
