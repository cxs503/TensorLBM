"""Contracts for the cold cell-level conversion density representation audit."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from tensorlbm.free_surface_closure_experiment import run_free_surface_closure_experiment
from tensorlbm.free_surface_conversion_density_audit import (
    DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE,
    WITHHELD_MISSING_CONVERSION_EVIDENCE,
    build_conversion_density_audit,
)
from tensorlbm.free_surface_lbm import free_surface_step
from tensorlbm.free_surface_topology_transaction import (
    TopologyTransactionError,
    build_topology_transaction,
)


I_TO_L = {
    "conversion_cells": (
        {
            "cell": (1, 2, 3), "flag_before": 2, "flag_after": 1,
            "fill_before": 0.75, "fill_after": 1.0,
            "mass_before": 0.75, "mass_after": 1.0,
            "population_before": 0.8, "population_after": 0.9,
        },
    ),
}


def _nested(value):
    if isinstance(value, tuple) and value and all(isinstance(item, tuple) and len(item) == 2 for item in value):
        return {key: _nested(item) for key, item in value}
    return value


def test_synthetic_i_to_l_exact_representation_math_oracle() -> None:
    audit = build_conversion_density_audit(I_TO_L, rho_liquid=1.0)

    assert audit.status == DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE
    assert audit.conversion_inventory_delta == pytest.approx(0.15)
    assert audit.sum_cell_production_inventory_delta == pytest.approx(0.15)
    assert audit.representation_switch_delta == pytest.approx(0.15)
    assert audit.i_to_l_population_nominal_density_gap == pytest.approx(-0.1)
    assert audit.i_to_l_population_nominal_density_gap_cells == 1
    cell = audit.cells[0]
    assert cell.classification == "I_TO_L"
    assert cell.fill_rho_before == pytest.approx(0.75)
    assert cell.population_density_after == pytest.approx(0.9)
    assert cell.production_inventory_before == pytest.approx(0.75)
    assert cell.production_inventory_after == pytest.approx(0.9)
    assert cell.population_nominal_density_gap == pytest.approx(-0.1)


def test_actual_b_step_three_i_to_g_cells_sum_to_observed_conversion_inventory_delta() -> None:
    report = run_free_surface_closure_experiment()
    step = next(case for case in report.cases if case.case_id == "B_forced_conversion_deterministic").steps[2]
    runtime = _nested(step.runtime_ledger)
    reconciliation = _nested(step.inventory_reconciliation)
    observed = reconciliation["stage_deltas"]["after_topology_conversion"]["total_liquid_inventory"]
    audit = build_conversion_density_audit(
        runtime["conversion_evidence"], rho_liquid=1.0,
        observed_conversion_inventory_delta=observed,
    )

    assert audit.status == DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE
    assert audit.i_to_g_count > 0
    assert audit.observed_conversion_inventory_delta == pytest.approx(observed)
    assert audit.cell_sum_inventory_residual == pytest.approx(
        audit.sum_cell_production_inventory_delta - observed,
    )
    assert not audit.cell_sum_matches_conversion_inventory_delta


def test_actual_c_step_four_i_to_g_cells_report_conversion_sum_without_i_to_l_claim() -> None:
    report = run_free_surface_closure_experiment()
    step = next(case for case in report.cases if case.case_id == "C_dam_break_style_tiny_dynamic_topology").steps[3]
    runtime = _nested(step.runtime_ledger)
    reconciliation = _nested(step.inventory_reconciliation)
    observed = reconciliation["stage_deltas"]["after_topology_conversion"]["total_liquid_inventory"]
    audit = build_conversion_density_audit(
        runtime["conversion_evidence"], rho_liquid=1.0,
        observed_conversion_inventory_delta=observed,
    )

    assert audit.status == DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE
    assert audit.i_to_g_count > 0
    assert audit.observed_conversion_inventory_delta == pytest.approx(observed)
    assert audit.cell_sum_inventory_residual == pytest.approx(
        audit.sum_cell_production_inventory_delta - observed,
    )
    assert not audit.cell_sum_matches_conversion_inventory_delta



def test_actual_zero_independent_mass_delta_i_to_l_is_preserved_in_transaction_evidence() -> None:
    shape = (3, 3, 5)
    flags = torch.full(shape, 0, dtype=torch.int8)
    flags[1, 1, 2] = 2
    fill = torch.zeros(shape)
    fill[1, 1, 2] = 1.0
    mass = fill.clone()
    solid = torch.zeros(shape, dtype=torch.bool)
    zero = torch.zeros(shape)
    f = torch.stack([torch.full(shape, 1.0 / 19.0) for _ in range(19)])
    empty = torch.zeros(shape, dtype=torch.bool)
    to_liq = empty.clone()
    to_liq[1, 1, 2] = True
    plan = build_topology_transaction(
        f, fill, flags, mass,
        to_iface=empty, to_liq=to_liq, to_gas=empty, recv_new=empty,
        redistribution_increment=torch.zeros_like(mass),
        rho_liquid=1.0, rho_gas=0.001, solid_mask=solid,
        gas_flag=0, liquid_flag=1, interface_flag=2, solid_flag=3,
        ux=zero, uy=zero, uz=zero, capture_evidence=True,
    )
    evidence = plan.conversion_evidence
    assert evidence is not None
    cells = evidence["conversion_cells"]
    assert isinstance(cells, tuple)
    assert len(cells) == 1
    cell = cells[0]
    assert isinstance(cell, dict)
    assert cell["flag_before"] == 2
    assert cell["flag_after"] == 1
    assert cell["mass_delta"] == pytest.approx(0.0)
    audit = build_conversion_density_audit(evidence, rho_liquid=1.0)
    assert audit.i_to_l_count == 1
    assert audit.cells[0].population_nominal_density_gap == pytest.approx(0.0, abs=2.0e-7)


def test_missing_evidence_is_explicitly_withheld_not_zero() -> None:
    audit = build_conversion_density_audit(None, rho_liquid=1.0)
    assert audit.status == DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE
    assert audit.withheld_reason == WITHHELD_MISSING_CONVERSION_EVIDENCE
    assert audit.conversion_inventory_delta is None
    assert audit.cells == ()


def test_evidence_and_result_are_immutable_and_do_not_alias_input() -> None:
    evidence = {"conversion_cells": [dict(I_TO_L["conversion_cells"][0])]}
    audit = build_conversion_density_audit(evidence, rho_liquid=1.0)
    evidence["conversion_cells"][0]["population_after"] = 99.0

    assert audit.cells[0].population_density_after == pytest.approx(0.9)
    with pytest.raises(FrozenInstanceError):
        audit.cells[0].classification = "mutated"  # type: ignore[misc]


def test_no_audit_argument_preserves_solver_output_bitwise() -> None:
    from tensorlbm.free_surface_closure_experiment import _conversion_state

    f, fill, flags, solid = _conversion_state()
    normal = free_surface_step(f, fill, flags, solid, mass=fill.clone(), rho_gas=1.0e-3)
    audit_ledger: dict[str, object] = {}
    diagnosed = free_surface_step(
        f, fill, flags, solid, mass=fill.clone(), rho_gas=1.0e-3,
        conversion_density_audit_ledger=audit_ledger,
    )
    assert audit_ledger["status"] == DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE
    for plain, observed in zip(normal, diagnosed):
        assert torch.equal(plain, observed)


def test_audit_ledger_is_atomic_when_transaction_build_fails(monkeypatch) -> None:
    import tensorlbm.free_surface_lbm as module
    from tensorlbm.free_surface_closure_experiment import _conversion_state

    monkeypatch.setattr(
        module, "build_topology_transaction",
        lambda *args, **kwargs: (_ for _ in ()).throw(TopologyTransactionError("injected")),
    )
    f, fill, flags, solid = _conversion_state()
    ledger = {"nested": {"entries": ["unchanged"]}}
    with pytest.raises(TopologyTransactionError, match="injected"):
        free_surface_step(
            f, fill, flags, solid, mass=fill.clone(),
            conversion_density_audit_ledger=ledger,
        )
    assert ledger == {"nested": {"entries": ["unchanged"]}}
