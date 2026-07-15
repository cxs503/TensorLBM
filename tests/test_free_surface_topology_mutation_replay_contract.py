"""Cold same-order topology mutation replay-contract tests."""
from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, SOLID
from tensorlbm.free_surface_topology_transaction import (
    ReplayEvidence,
    build_topology_transaction,
)
from tensorlbm.free_surface_topology_mutation_replay_contract import (
    AVAILABLE_REPLAYED_EXACT,
    MISSING_INPUT_WITHHELD,
    WITHHELD,
    audit_topology_mutation_replay,
)


def _inputs() -> dict[str, Any]:
    shape = (5, 5, 5)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    flags[2, 2, 2] = INTERFACE
    flags[2, 2, 3] = INTERFACE
    flags[2, 2, 1] = INTERFACE
    fill = torch.zeros(shape)
    fill[flags == INTERFACE] = 0.5
    mass = fill.clone()
    zero = torch.zeros(shape)
    solid = torch.zeros(shape, dtype=torch.bool)
    to_gas = torch.zeros_like(solid)
    to_gas[2, 2, 2] = True
    return {
        "f": equilibrium3d(torch.ones(shape), zero, zero, zero),
        "fill": fill, "flags": flags, "mass": mass,
        "to_iface": torch.zeros_like(solid), "to_liq": torch.zeros_like(solid),
        "to_gas": to_gas, "recv_new": torch.zeros_like(solid),
        "redistribution_increment": zero, "rho_liquid": 1.0, "rho_gas": 1.0,
        "solid_mask": solid, "gas_flag": GAS, "liquid_flag": LIQUID,
        "interface_flag": INTERFACE, "solid_flag": SOLID,
        "ux": zero, "uy": zero, "uz": zero,
    }


def _assert_exact(report: Any) -> None:
    assert report.status == AVAILABLE_REPLAYED_EXACT
    assert report.mutates_solver_state is False
    assert report.physical_claim is False
    assert report.final_candidate_exact is True
    assert all(phase.status == AVAILABLE_REPLAYED_EXACT for phase in report.phases)


def test_complete_capture_replays_exactly_and_exposes_no_mutable_phase_evidence() -> None:
    plan = build_topology_transaction(**_inputs(), capture_replay_stages=True)
    assert plan.replay_evidence is not None
    exposed = plan.replay_stages
    assert exposed is not None
    exposed["clamp"][0].fill_(99.0)
    del exposed["halo_boundary"]
    _assert_exact(audit_topology_mutation_replay(plan.replay_evidence))


@pytest.mark.parametrize("tamper", ("delete", "replace", "fill"))
def test_tampering_a_caller_owned_compatibility_view_cannot_change_audit_evidence(tamper: str) -> None:
    plan = build_topology_transaction(**_inputs(), capture_replay_stages=True)
    exposed = plan.replay_stages
    assert exposed is not None
    if tamper == "delete":
        del exposed["clamp"]
    elif tamper == "replace":
        f, fill, flags, mass = exposed["clamp"]
        exposed["clamp"] = (torch.zeros_like(f), torch.zeros_like(fill), torch.zeros_like(flags), torch.zeros_like(mass))
    else:
        exposed["clamp"][3].fill_(123.0)
    _assert_exact(audit_topology_mutation_replay(plan.replay_evidence))


def test_tampered_serialized_evidence_is_withheld() -> None:
    plan = build_topology_transaction(**_inputs(), capture_replay_stages=True)
    evidence = plan.replay_evidence
    assert evidence is not None
    tampered = ReplayEvidence(
        evidence.invocation_payload[:-1] + bytes([evidence.invocation_payload[-1] ^ 1]),
        evidence.invocation_sha256, evidence.phase_payload, evidence.phase_sha256,
        evidence.candidate_payload, evidence.candidate_sha256, evidence.tensor_records,
    )
    assert audit_topology_mutation_replay(tampered).status == WITHHELD


def test_publicly_forged_hash_matching_evidence_is_withheld() -> None:
    plan = build_topology_transaction(**_inputs(), capture_replay_stages=True)
    evidence = plan.replay_evidence
    assert evidence is not None
    forged = ReplayEvidence(
        evidence.invocation_payload, evidence.invocation_sha256,
        evidence.phase_payload, evidence.phase_sha256,
        evidence.candidate_payload, evidence.candidate_sha256, evidence.tensor_records,
    )
    assert audit_topology_mutation_replay(forged).status == WITHHELD


def test_missing_or_legacy_mapping_input_is_withheld() -> None:
    report = audit_topology_mutation_replay({"f": torch.zeros(1)})
    assert report.status == WITHHELD
    assert all(phase.status == MISSING_INPUT_WITHHELD for phase in report.phases)


def test_audit_never_imports_or_calls_default_solver() -> None:
    import tensorlbm.free_surface_topology_mutation_replay_contract as module

    source = inspect.getsource(module)
    assert "free_surface_lbm" not in source
    assert "free_surface_step" not in source


def test_b_and_c_actual_builder_capture_replays_all_phases_exactly() -> None:
    from tensorlbm.free_surface_closure_experiment import (
        FAILED_DIAGNOSTIC,
        run_free_surface_closure_experiment,
    )

    report = run_free_surface_closure_experiment(
        enable_i_to_g_ownership_closure=True, capture_replay_stages=True,
    )
    for case_id, requested_steps in (
        ("B_forced_conversion_deterministic", 3),
        ("C_dam_break_style_tiny_dynamic_topology", 10),
    ):
        result = next(case for case in report.cases if case.case_id == case_id)
        assert result.physical_closure_claim is False
        assert result.requested_steps == requested_steps
        assert result.status == FAILED_DIAGNOSTIC
        assert len(result.steps) == 3
        assert result.steps[-1].failure_reason is not None
        assert result.steps[-1].replay_evidence is None
        successful = result.steps[:-1]
        assert len(successful) == 2
        for step in successful:
            assert step.replay_evidence is not None
            _assert_exact(audit_topology_mutation_replay(step.replay_evidence))


def test_b_and_c_without_capture_remain_withheld(monkeypatch: pytest.MonkeyPatch) -> None:
    import tensorlbm.free_surface_lbm as solver
    from tensorlbm.free_surface_closure_experiment import _conversion_state, _run_case

    captures: list[object] = []
    original = solver.build_topology_transaction

    def capture(*args: Any, **kwargs: Any):
        assert kwargs["capture_replay_stages"] is False
        plan = original(*args, **kwargs)
        captures.append(plan.replay_evidence)
        return plan

    monkeypatch.setattr(solver, "build_topology_transaction", capture)
    f, fill, flags, solid = _conversion_state()
    _run_case(
        "B_forced_conversion_deterministic", f, fill, flags, solid, 3, False, True,
        enable_i_to_g_ownership_closure=True,
    )
    assert captures and all(item is None for item in captures)
    from tensorlbm.free_surface_closure_experiment import run_free_surface_closure_experiment
    report = run_free_surface_closure_experiment()
    assert all(step.replay_evidence is None for case in report.cases for step in case.steps)
    assert audit_topology_mutation_replay(None).status == WITHHELD
