"""Contract tests for the R1 dynamic-topology closure diagnostic experiment."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID
from tensorlbm.free_surface_closure_experiment import (
    DIAGNOSTIC_NOT_PHYSICAL_CLOSURE,
    FAILED_DIAGNOSTIC,
    WITHHELD,
    ClosureExperimentError,
    run_free_surface_closure_experiment,
)


def test_real_step_matrix_records_independent_inventory_ledger_ownership_and_topology_evidence() -> None:
    report = run_free_surface_closure_experiment()

    assert report.status == DIAGNOSTIC_NOT_PHYSICAL_CLOSURE
    assert report.physical_closure_claim is False
    assert report.global_mass_correction_applied is False
    assert {case.case_id for case in report.cases} == {
        "A_frozen_topology_paired_off",
        "A_frozen_topology_paired_on",
        "B_forced_conversion_deterministic",
        "C_dam_break_style_tiny_dynamic_topology",
    }
    assert all(case.status in {WITHHELD, FAILED_DIAGNOSTIC} for case in report.cases)

    for case in report.cases:
        assert len(case.steps) == case.requested_steps
        assert len(case.mass_drift_curve) == case.requested_steps + 1
        assert len(case.inventory_drift_curve) == case.requested_steps + 1
        assert case.initial.independent_mass == pytest.approx(case.mass_drift_curve[0], abs=0.0)
        assert case.initial.total_liquid_inventory == pytest.approx(case.inventory_drift_curve[0], abs=0.0)
        assert case.final.independent_mass == pytest.approx(case.mass_drift_curve[-1], abs=0.0)
        assert case.final.total_liquid_inventory == pytest.approx(case.inventory_drift_curve[-1], abs=0.0)
        assert is_dataclass(case)
        with pytest.raises(FrozenInstanceError):
            case.status = "PASS"  # type: ignore[misc]
        for step in case.steps:
            assert step.finite is True
            assert step.direct_liquid_gas_links == 0
            assert step.runtime_ledger is not None
            assert step.ownership_ledger is not None
            assert step.ledger_reconciliation_residual is not None
            assert step.abb_population_only is True
            assert step.topology_event_evidence_available in {True, False}
            assert {event.operator for event in step.topology_events} <= {
                "conversion", "redistribution", "abb", "liquid_interface",
                "clamp", "isolation", "boundary", "other",
            }
            assert "abb_population_inventory_owner_withheld" in step.ownership_unresolved_categories

    forced = next(case for case in report.cases if case.case_id == "B_forced_conversion_deterministic")
    dynamic = next(case for case in report.cases if case.case_id == "C_dam_break_style_tiny_dynamic_topology")
    assert any(
        event.operator == "conversion" and event.event_count > 0
        for step in forced.steps for event in step.topology_events
    )
    assert any(
        event.operator == "redistribution" and event.event_count > 0
        for step in forced.steps for event in step.topology_events
    )
    assert any(step.topology_event_evidence_available for step in forced.steps)
    assert any(event.operator in {"conversion", "redistribution"} for step in dynamic.steps for event in step.topology_events)
    assert all(
        event.operator != "other"
        for case in report.cases for step in case.steps for event in step.topology_events
    )


def test_experiment_is_reproducible_and_never_promotes_withheld_diagnostics_to_pass() -> None:
    first = run_free_surface_closure_experiment()
    second = run_free_surface_closure_experiment()

    assert first == second
    assert "PASS" not in {first.status, *(case.status for case in first.cases)}
    assert all(case.physical_closure_claim is False for case in first.cases)


def test_invalid_topology_is_fail_closed_with_retained_reason() -> None:
    shape = (3, 3, 5)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    flags[:, :, 1] = LIQUID
    flags[:, :, 2] = GAS  # direct D3Q19 L/G link: invalid before a solver update
    fill = torch.where(flags == LIQUID, torch.ones(shape), torch.zeros(shape))
    zero = torch.zeros(shape)
    f = equilibrium3d(torch.ones(shape), zero, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)

    report = run_free_surface_closure_experiment(extra_cases=(("invalid_topology", f, fill, flags, solid, 1, False, True),))
    case = report.cases[-1]

    assert case.status == FAILED_DIAGNOSTIC
    assert case.failure_reason is not None
    assert "direct" in case.failure_reason.lower()
    assert len(case.steps) == 1
    assert case.steps[0].finite is True
    assert case.steps[0].direct_liquid_gas_links > 0


def test_nonfinite_state_is_fail_closed_with_retained_reason() -> None:
    shape = (3, 3, 5)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    flags[:, :, 0] = INTERFACE
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3:] = LIQUID
    fill = torch.where(flags == LIQUID, torch.ones(shape), torch.zeros(shape))
    fill[flags == INTERFACE] = 0.5
    zero = torch.zeros(shape)
    f = equilibrium3d(torch.ones(shape), zero, zero, zero)
    f[0, 0, 0, 0] = float("nan")
    solid = torch.zeros(shape, dtype=torch.bool)

    report = run_free_surface_closure_experiment(extra_cases=(("nonfinite", f, fill, flags, solid, 1, True, True),))
    case = report.cases[-1]

    assert case.status == FAILED_DIAGNOSTIC
    assert case.failure_reason == "non-finite input state"
    assert case.steps[0].finite is False


@pytest.mark.parametrize(
    "extra_case, message",
    [
        (("short",), "8-item"),
        (("bad", None, None, None, None, 1, False, True), "f must be a torch.Tensor"),
        (("bad_steps", None, None, None, None, 0, False, True), "f must be a torch.Tensor"),
    ],
)
def test_experiment_fails_closed_on_malformed_extra_case(extra_case, message: str) -> None:
    with pytest.raises(ClosureExperimentError, match=message):
        run_free_surface_closure_experiment(extra_cases=(extra_case,))


@pytest.mark.parametrize("bad_dtype", [torch.int64, torch.float64, torch.bool])
def test_experiment_rejects_non_float32_field_bytes(bad_dtype: torch.dtype) -> None:
    shape = (3, 3, 5)
    f = equilibrium3d(torch.ones(shape), torch.zeros(shape), torch.zeros(shape), torch.zeros(shape))
    fill = torch.zeros(shape)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    solid = torch.zeros(shape, dtype=torch.bool)
    bad_f = f.to(bad_dtype)
    with pytest.raises(ClosureExperimentError, match="float32"):
        run_free_surface_closure_experiment(extra_cases=(("bad_dtype", bad_f, fill, flags, solid, 1, False, True),))
    with pytest.raises(ClosureExperimentError, match="float32"):
        run_free_surface_closure_experiment(extra_cases=(("bad_dtype", f, fill.to(bad_dtype), flags, solid, 1, False, True),))




@pytest.mark.parametrize("field_name", ["f", "fill", "flags", "solid"])
def test_experiment_rejects_sparse_case_fields(field_name: str) -> None:
    shape = (3, 3, 5)
    f = equilibrium3d(torch.ones(shape), torch.zeros(shape), torch.zeros(shape), torch.zeros(shape))
    fill = torch.zeros(shape)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    solid = torch.zeros(shape, dtype=torch.bool)
    fields = {"f": f, "fill": fill, "flags": flags, "solid": solid}
    fields[field_name] = fields[field_name].to_sparse()
    with pytest.raises(ClosureExperimentError, match="dense strided"):
        run_free_surface_closure_experiment(extra_cases=((
            "sparse", fields["f"], fields["fill"], fields["flags"], fields["solid"], 1, False, True,
        ),))


def test_experiment_rejects_empty_spatial_domain() -> None:
    f = torch.zeros((19, 0, 0, 0), dtype=torch.float32)
    fill = torch.zeros((0, 0, 0), dtype=torch.float32)
    flags = torch.zeros((0, 0, 0), dtype=torch.int8)
    solid = torch.zeros((0, 0, 0), dtype=torch.bool)
    with pytest.raises(ClosureExperimentError, match="spatial dimensions"):
        run_free_surface_closure_experiment(extra_cases=(("empty", f, fill, flags, solid, 1, False, True),))


def test_experiment_rejects_non_tuple_extra_cases() -> None:
    with pytest.raises(ClosureExperimentError, match="extra_cases"):
        run_free_surface_closure_experiment(extra_cases=None)  # type: ignore[arg-type]


def test_experiment_rejects_nonpositive_requested_steps() -> None:
    shape = (3, 3, 5)
    f = equilibrium3d(torch.ones(shape), torch.zeros(shape), torch.zeros(shape), torch.zeros(shape))
    fill = torch.zeros(shape)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    solid = torch.zeros(shape, dtype=torch.bool)
    with pytest.raises(ClosureExperimentError, match="positive"):
        run_free_surface_closure_experiment(extra_cases=(("bad", f, fill, flags, solid, 0, False, True),))
