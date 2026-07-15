"""TDD contract for the diagnostic free-surface ownership ledger R1.

This ledger tracks existing solver state ownership.  It is explicitly not a
physical/PV closure or a solver mass correction.
"""
from __future__ import annotations

import pytest
import torch
from dataclasses import replace

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, free_surface_step
from tensorlbm.free_surface_ownership_ledger import (
    OwnershipLedgerError,
    OwnershipLedgerState,
    build_ownership_ledger,
)


def _liquid_interface_fact() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flags = torch.full((1, 1, 5), GAS, dtype=torch.int8)
    flags[0, 0, 2] = INTERFACE
    flags[0, 0, 3] = LIQUID
    links = torch.zeros((19, 1, 1, 5))
    # q=-x pulls target x=2 from its LIQUID source x=3.
    links[2, 0, 0, 2] = 0.75
    mask = links != 0
    return flags, links, mask


def test_link_records_pair_exactly_with_explicit_liquid_and_interface_owners() -> None:
    flags, links, mask = _liquid_interface_fact()

    state = build_ownership_ledger(
        flags=flags, mass_delta_liquid=links, liquid_interface_mask=mask,
        paired_liquid_interface_debit=True,
    )

    assert isinstance(state, OwnershipLedgerState)
    assert state.status == "OBSERVED_NOT_PHYSICAL_CLOSURE"
    # Even a fully paired L/I observation cannot assign ABB population changes
    # to a liquid inventory owner; this is not physical/PV closure.
    assert state.unresolved_categories == ("abb_population_inventory_owner_withheld",)
    assert len(state.liquid_interface_transfers) == 1
    record = state.liquid_interface_transfers[0]
    assert record.source.owner_phase == "LIQUID"
    assert record.source.cell == (0, 0, 3)
    assert record.target.owner_phase == "INTERFACE"
    assert record.target.cell == (0, 0, 2)
    assert record.credit == pytest.approx(0.75)
    assert record.debit == pytest.approx(-0.75)
    assert record.net == pytest.approx(0.0)
    assert record.ownership == "PAIRED"


def test_unpaired_liquid_interface_credit_is_explicitly_withheld() -> None:
    flags, links, mask = _liquid_interface_fact()

    state = build_ownership_ledger(
        flags=flags, mass_delta_liquid=links, liquid_interface_mask=mask,
        paired_liquid_interface_debit=False,
    )

    record = state.liquid_interface_transfers[0]
    assert record.ownership == "UNPAIRED/WITHHELD"
    assert record.debit is None
    assert record.net is None
    assert "unpaired_liquid_interface_debit" in state.unresolved_categories


def test_redistribution_conversion_and_abb_remain_observed_not_physical_closure() -> None:
    flags, links, mask = _liquid_interface_fact()
    evidence = {
        "redistribution_links": ({"donor": (0, 0, 3), "receiver": (0, 0, 2), "mass_delta": 0.2},),
        "conversion_cells": ({
            "cell": (0, 0, 2), "flag_before": INTERFACE, "flag_after": LIQUID,
            "mass_before": 1.2, "mass_after": 1.0, "mass_delta": -0.2,
        },),
    }

    state = build_ownership_ledger(
        flags=flags, mass_delta_liquid=links, liquid_interface_mask=mask,
        paired_liquid_interface_debit=True, conversion_evidence=evidence,
        abb_population_delta=1.5,
    )

    assert state.redistributions[0].donor.owner_phase == "LIQUID"
    assert state.redistributions[0].receiver.owner_phase == "INTERFACE"
    assert state.conversions[0].before_owner_phase == "INTERFACE"
    assert state.conversions[0].after_owner_phase == "LIQUID"
    assert state.abb_records[0].population_only is True
    assert state.abb_records[0].inventory_owner_status == "WITHHELD"
    assert "abb_population_inventory_owner_withheld" in state.unresolved_categories
    assert state.status == "OBSERVED_NOT_PHYSICAL_CLOSURE"


@pytest.mark.parametrize("kind", ["missing", "nonliquid"])
def test_paired_liquid_interface_owner_validation_fails_closed(kind: str) -> None:
    flags, links, mask = _liquid_interface_fact()
    if kind == "missing":
        with pytest.raises(OwnershipLedgerError, match="requires both mass_delta_liquid"):
            build_ownership_ledger(
                flags=flags, mass_delta_liquid=links, liquid_interface_mask=None,
                paired_liquid_interface_debit=True,
            )
        return
    if kind == "nonliquid":
        flags[0, 0, 3] = GAS

    with pytest.raises(OwnershipLedgerError, match="LIQUID source owner"):
        build_ownership_ledger(
            flags=flags, mass_delta_liquid=links, liquid_interface_mask=mask,
            paired_liquid_interface_debit=True,
        )


def _runtime_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (3, 3, 5)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    flags[:, :, 0] = INTERFACE
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3:] = LIQUID
    fill = torch.zeros(shape)
    fill[flags == INTERFACE] = 0.5
    fill[flags == LIQUID] = 1.0
    solid = torch.zeros_like(flags, dtype=torch.bool)
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, zero, zero, zero), fill, flags, solid


def test_ownership_only_request_captures_plan_evidence_without_runtime_ledger(monkeypatch) -> None:
    import tensorlbm.free_surface_lbm as solver

    original = solver.build_topology_transaction
    evidence = {
        "redistribution_links": (),
        "conversion_cells": ({
            "cell": (0, 0, 0), "flag_before": INTERFACE, "flag_after": LIQUID,
            "mass_before": 0.5, "mass_after": 1.0, "mass_delta": 0.5,
        },),
    }

    def tracked(*args, **kwargs):
        assert kwargs["capture_evidence"] is True
        plan = original(*args, **kwargs)
        return replace(plan, conversion_evidence=evidence)

    monkeypatch.setattr(solver, "build_topology_transaction", tracked)
    f, fill, flags, solid = _runtime_state()
    ownership: dict[str, object] = {}
    solver.free_surface_step(
        f, fill, flags, solid, mass=fill.clone(), rho_gas=0.001,
        ownership_ledger=ownership, paired_liquid_interface_debit=True,
    )
    assert isinstance(ownership, dict)
    state = ownership["steps"][-1]
    assert isinstance(state, OwnershipLedgerState)
    assert len(state.conversions) == 1


def test_ownership_build_failure_publishes_no_partial_ledgers(monkeypatch) -> None:
    import tensorlbm.free_surface_lbm as solver

    monkeypatch.setattr(solver, "_append_ownership_ledger", lambda *args, **kwargs: (_ for _ in ()).throw(OwnershipLedgerError("injected")))
    f, fill, flags, solid = _runtime_state()
    runtime = {"sentinel": "runtime", "steps": [{"old": True}], "operator_curve": [{"old": True}]}
    ownership = {"sentinel": "ownership", "steps": [{"old": True}], "latest": {"old": True}}
    expected_runtime = {"sentinel": "runtime", "steps": [{"old": True}], "operator_curve": [{"old": True}]}
    expected_ownership = {"sentinel": "ownership", "steps": [{"old": True}], "latest": {"old": True}}
    with pytest.raises(OwnershipLedgerError, match="injected"):
        solver.free_surface_step(
            f, fill, flags, solid, mass=fill.clone(), rho_gas=0.001,
            runtime_ledger=runtime, ownership_ledger=ownership,
            paired_liquid_interface_debit=True,
        )
    assert runtime == expected_runtime
    assert ownership == expected_ownership


@pytest.mark.parametrize("freeze_topology", [True, False])
def test_actual_step_appends_observed_ownership_state_without_mutating_runtime_schema(
    freeze_topology: bool,
) -> None:
    f, fill, flags, solid = _runtime_state()
    runtime: dict[str, object] = {"sentinel": "preserved"}
    ownership: dict[str, object] = {}

    free_surface_step(
        f, fill, flags, solid, mass=fill.clone(), rho_gas=0.001,
        freeze_topology=freeze_topology, runtime_ledger=runtime,
        ownership_ledger=ownership, paired_liquid_interface_debit=True,
    )

    assert runtime["sentinel"] == "preserved"
    assert isinstance(runtime["steps"], list)
    assert isinstance(ownership["steps"], list)
    state = ownership["steps"][-1]
    assert isinstance(state, OwnershipLedgerState)
    assert state.status == "OBSERVED_NOT_PHYSICAL_CLOSURE"
    assert all(record.ownership == "PAIRED" for record in state.liquid_interface_transfers)
    assert all(record.inventory_owner_status == "WITHHELD" for record in state.abb_records)
