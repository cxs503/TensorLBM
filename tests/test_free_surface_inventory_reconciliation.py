"""R1 contracts for actual-state free-surface inventory reconciliation."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.free_surface_closure_experiment import run_free_surface_closure_experiment
from tensorlbm.free_surface_lbm import free_surface_step
from tensorlbm.free_surface_topology_transaction import TopologyTransactionError


CANONICAL_STAGES = (
    "before_collision",
    "after_collision_and_forcing",
    "after_stream_and_gas_zero",
    "after_abb",
    "after_wall_boundary",
    "after_mass_exchange",
    "after_topology_redistribution",
    "after_topology_clamp",
    "after_topology_conversion",
    "after_topology_halo_isolation_boundary",
)


def _nested(value):
    if isinstance(value, tuple) and value and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    ):
        return {key: _nested(item) for key, item in value}
    return value


def test_forced_step_three_separates_pre_topology_combined_from_conversion() -> None:
    report = run_free_surface_closure_experiment()
    forced = next(case for case in report.cases if case.case_id == "B_forced_conversion_deterministic")
    step = forced.steps[2]
    assert step.tracked_independent_mass_drift == pytest.approx(0.0, abs=2.0e-6)

    reconciliation = _nested(step.inventory_reconciliation)
    assert reconciliation["status"] == "DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE"
    assert reconciliation["abb_inventory_status"] == "POPULATION_ONLY_WITHHELD"
    assert reconciliation["operator_attribution_status"] == "OBSERVED_COMBINED_NOT_ATOMIC"
    stages = _nested(reconciliation["stages"])
    deltas = _nested(reconciliation["stage_deltas"])
    assert tuple(stages) == CANONICAL_STAGES
    assert "after_exchange" not in stages
    assert reconciliation["pre_topology_combined_total_liquid_inventory_delta"] == pytest.approx(
        sum(
            deltas[name]["total_liquid_inventory"]
            for name in (
                "after_collision_and_forcing",
                "after_stream_and_gas_zero",
                "after_abb",
                "after_wall_boundary",
                "after_mass_exchange",
            )
        ),
        abs=2.0e-7,
    )
    assert deltas["after_mass_exchange"]["total_liquid_inventory"] != pytest.approx(0.0, abs=2.0e-7)
    assert deltas["after_topology_conversion"]["total_liquid_inventory"] < 0.0
    assert reconciliation["observed_total_liquid_inventory_delta"] == pytest.approx(
        reconciliation["sum_stage_total_liquid_inventory_delta"], abs=2.0e-7,
    )


def test_frozen_paired_off_on_have_stage_evidence_and_dynamic_case_reconciles() -> None:
    report = run_free_surface_closure_experiment()
    cases = {case.case_id: case for case in report.cases}
    for name in ("A_frozen_topology_paired_off", "A_frozen_topology_paired_on"):
        for step in cases[name].steps:
            reconciliation = _nested(step.inventory_reconciliation)
            assert reconciliation["status"] == "DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE"
            assert reconciliation["stage_deltas"]["after_topology_conversion"]["total_liquid_inventory"] == 0.0
    dynamic = cases["C_dam_break_style_tiny_dynamic_topology"]
    assert len(dynamic.steps) == 10
    for step in dynamic.steps:
        reconciliation = _nested(step.inventory_reconciliation)
        assert reconciliation["observed_total_liquid_inventory_delta"] == pytest.approx(
            reconciliation["sum_stage_total_liquid_inventory_delta"], abs=2.0e-6,
        )


def test_dynamic_step_four_has_positive_pre_topology_and_negative_conversion_without_conversion_only_claim() -> None:
    report = run_free_surface_closure_experiment()
    dynamic = next(case for case in report.cases if case.case_id == "C_dam_break_style_tiny_dynamic_topology")
    reconciliation = _nested(dynamic.steps[3].inventory_reconciliation)
    deltas = _nested(reconciliation["stage_deltas"])

    assert reconciliation["operator_attribution_status"] == "OBSERVED_COMBINED_NOT_ATOMIC"
    assert reconciliation["pre_topology_combined_total_liquid_inventory_delta"] > 0.0
    assert deltas["after_topology_conversion"]["total_liquid_inventory"] < 0.0
    assert reconciliation["observed_total_liquid_inventory_delta"] == pytest.approx(
        reconciliation["sum_stage_total_liquid_inventory_delta"], abs=2.0e-6,
    )


def test_no_diagnostic_has_bitwise_parity_with_cold_diagnostic() -> None:
    from tensorlbm.free_surface_closure_experiment import _conversion_state

    f, fill, flags, solid = _conversion_state()
    normal = free_surface_step(f, fill, flags, solid, mass=fill.clone(), rho_gas=1.0e-3)
    ledger: dict[str, object] = {}
    diagnosed = free_surface_step(
        f, fill, flags, solid, mass=fill.clone(), rho_gas=1.0e-3,
        inventory_reconciliation_ledger=ledger,
    )
    assert ledger
    for plain, observed in zip(normal, diagnosed):
        assert torch.equal(plain, observed)


def test_inventory_ledger_is_atomic_when_transaction_build_fails(monkeypatch) -> None:
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
            f, fill, flags, solid, mass=fill.clone(), inventory_reconciliation_ledger=ledger,
        )
    assert ledger == {"nested": {"entries": ["unchanged"]}}


def test_inventory_report_is_frozen_and_does_not_alias_caller_ledger() -> None:
    report = run_free_surface_closure_experiment()
    step = report.cases[0].steps[0]
    assert step.inventory_reconciliation is not None
    with pytest.raises(TypeError):
        step.inventory_reconciliation[0] = ("mutated", ())  # type: ignore[index]
