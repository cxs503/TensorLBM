"""TDD R1 decision gate for Körner I→G f-ownership policy evidence."""
from __future__ import annotations

import pytest

from tensorlbm.free_surface_population_policy_evidence import (
    WITHHELD_MISSING_POLICY_EVIDENCE,
    IToGPopulationPolicy,
    evaluate_i_to_g_policy_evidence,
    evaluate_production_i_to_g_policy_evidence,
)


def _production(policy: IToGPopulationPolicy, payload: object) -> dict[str, object]:
    return {
        "provenance": "production_free_surface_step_runtime_ledger",
        "actual_f_population_transfer": True,
        "policy_evidence": {policy.value: payload},
    }


@pytest.mark.parametrize(
    ("policy", "payload"),
    [
        (
            IToGPopulationPolicy.EXPLICIT_BOUNDARY_RECONSTRUCTION,
            {
                "operator_id": "korner-i-to-g-boundary-r2",
                "source_cells": ((1, 1, 1),),
                "reconstructed_cells": ((1, 1, 2),),
                "qwise_reconstruction": "documented-q-wise-rule",
                "boundary_state": "published-interface-boundary-state",
                "replay_reference": "capture/step-1",
            },
        ),
        (
            IToGPopulationPolicy.CONSERVATIVE_PARTITION_TRANSFER,
            {
                "operator_id": "korner-i-to-g-partition-r2",
                "source_cells": ((1, 1, 1),),
                "destination_cells": ((1, 1, 2),),
                "qwise_transfer_map": "documented-q-wise-map",
                "partition_weights": "published-multi-owner-weights",
                "momentum_treatment": "published-momentum-rule",
                "replay_reference": "capture/step-1",
            },
        ),
        (
            IToGPopulationPolicy.GAS_BOUNDARY_RESERVOIR,
            {
                "operator_id": "korner-i-to-g-reservoir-r2",
                "source_cells": ((1, 1, 1),),
                "reservoir_id": "gas-boundary/outer",
                "qwise_reservoir_debit": "published-q-wise-debit",
                "reservoir_accounting": "published-reservoir-ledger",
                "boundary_state": "published-gas-boundary-state",
                "replay_reference": "capture/step-1",
            },
        ),
    ],
)
def test_each_policy_has_explicit_minimum_production_evidence(
    policy: IToGPopulationPolicy, payload: dict[str, object],
) -> None:
    report = evaluate_i_to_g_policy_evidence(policy, _production(policy, payload))
    assert report.status == WITHHELD_MISSING_POLICY_EVIDENCE
    assert report.feasible is False
    assert report.missing_evidence == ()
    assert report.reason == "R1 never authorizes or implements an f ownership policy"


@pytest.mark.parametrize("policy", tuple(IToGPopulationPolicy))
def test_each_policy_withholds_its_missing_minimum_evidence(policy: IToGPopulationPolicy) -> None:
    report = evaluate_i_to_g_policy_evidence(policy, _production(policy, {"operator_id": "only-id"}))
    assert report.status == WITHHELD_MISSING_POLICY_EVIDENCE
    assert report.feasible is False
    assert "operator_id" not in report.missing_evidence
    assert report.missing_evidence


def test_malformed_or_nonproduction_mappings_cannot_be_policy_proof() -> None:
    malformed = evaluate_i_to_g_policy_evidence(
        IToGPopulationPolicy.CONSERVATIVE_PARTITION_TRANSFER,
        _production(IToGPopulationPolicy.CONSERVATIVE_PARTITION_TRANSFER, {"source_cells": [(1, 1, True)]}),
    )
    shaped = evaluate_i_to_g_policy_evidence(
        IToGPopulationPolicy.CONSERVATIVE_PARTITION_TRANSFER,
        {
            "provenance": "shaped_result_mapping_not_claimed_production",
            "actual_f_population_transfer": True,
            "policy_evidence": {"conservative_partition_transfer": {}},
        },
    )
    assert malformed.status == WITHHELD_MISSING_POLICY_EVIDENCE
    assert "operator_id" in malformed.missing_evidence
    assert shaped.status == WITHHELD_MISSING_POLICY_EVIDENCE
    assert "production_provenance" in shaped.missing_evidence


def test_current_real_production_report_explicitly_marks_all_three_options_missing() -> None:
    import torch

    from tensorlbm.d3q19 import C, equilibrium3d
    from tensorlbm.free_surface_lbm import GAS, INTERFACE, free_surface_step
    from tensorlbm.free_surface_production_evidence import extract_runtime_korner_evidence

    # This is a real production free_surface_step result, not a shaped mapping.
    shape = (5, 6, 7)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    fill = torch.zeros(shape)
    centre = (2, 3, 3)
    flags[centre], fill[centre] = INTERFACE, 1.0
    for q in range(1, 19):
        dz, dy, dx = int(C[q, 2]), int(C[q, 1]), int(C[q, 0])
        source = tuple((index - delta) % extent for index, delta, extent in zip(centre, (dz, dy, dx), shape))
        flags[source], fill[source] = INTERFACE, 0.5
    zero = torch.zeros(shape)
    f = equilibrium3d(torch.ones(shape), zero, zero, zero)
    runtime_ledger: dict[str, object] = {}
    replay_capture: dict[str, object] = {}
    free_surface_step(
        f, fill, flags, torch.zeros(shape, dtype=torch.bool), mass=fill.clone(),
        runtime_ledger=runtime_ledger, replay_capture=replay_capture,
        capture_replay_stages=True,
    )
    runtime = extract_runtime_korner_evidence(
        {"runtime_ledger": runtime_ledger, "replay_capture": replay_capture},
        provenance="production_free_surface_step_runtime_ledger",
    )
    reports = evaluate_production_i_to_g_policy_evidence(runtime)

    assert set(reports) == set(IToGPopulationPolicy)
    for policy, report in reports.items():
        assert report.status == WITHHELD_MISSING_POLICY_EVIDENCE
        assert report.feasible is False
        assert "actual_f_population_transfer" in report.missing_evidence
        assert policy.value in report.reason
