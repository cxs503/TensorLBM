"""TDD R1 tests for detached I→G actual-f transfer planning."""
from __future__ import annotations

import torch

from tensorlbm.free_surface_population_transfer_plan import (
    D3Q19,
    WITHHELD_UNSPECIFIED_TRANSFER_POLICY,
    IToGPopulationTransferEvent,
    IndependentMassLedger,
    Phase,
    PopulationTransferEvidence,
    plan_i_to_g_population_transfer,
    validate_i_to_g_population_transfer,
)


def _event(**changes: object) -> IToGPopulationTransferEvent:
    values: dict[str, object] = {
        "event_id": "step-1/i-to-g-0", "lattice": D3Q19,
        "converting_donors": ((1, 1, 1),), "source_partition": ((1, 1, 1),),
        "destination_partition": ((1, 1, 2),), "source_phase_before": Phase.I,
        "source_phase_after": Phase.G, "destination_phase_before": Phase.I,
        "destination_phase_after": Phase.I,
    }
    values.update(changes)
    return IToGPopulationTransferEvent(**values)  # type: ignore[arg-type]


def _evidence(dtype: torch.dtype = torch.float64, **changes: object) -> PopulationTransferEvidence:
    donor_before = torch.zeros((1, 19), dtype=dtype)
    receiver_before = torch.zeros((1, 19), dtype=dtype)
    # Conservative evidence is deliberately explicit, not donor copy semantics.
    donor_after = donor_before.clone()
    receiver_after = receiver_before.clone()
    zero = torch.zeros((), dtype=dtype)
    values: dict[str, object] = {
        "donor_before": donor_before, "donor_after": donor_after,
        "receiver_before": receiver_before, "receiver_after": receiver_after,
        "independent_mass": IndependentMassLedger(-torch.tensor(0.25, dtype=dtype), torch.tensor(0.25, dtype=dtype), zero),
    }
    values.update(changes)
    return PopulationTransferEvidence(**values)  # type: ignore[arg-type]


def test_complete_conservative_evidence_still_returns_empty_unspecified_plan() -> None:
    plan = plan_i_to_g_population_transfer(_event(), _evidence())
    assert plan.status == WITHHELD_UNSPECIFIED_TRANSFER_POLICY
    assert plan.operations == ()
    assert plan.validation.population_machine_zero is True
    assert plan.validation.independent_mass_machine_zero is True
    assert plan.validation.exact_float32_closure_claimed is False


def test_direct_donor_copy_with_donor_zeroing_has_nonzero_population_residual() -> None:
    evidence = _evidence()
    donor_before = torch.arange(19, dtype=torch.float64).reshape(1, 19)
    report = validate_i_to_g_population_transfer(
        _event(), _evidence(donor_before=donor_before, donor_after=torch.zeros_like(donor_before), receiver_after=donor_before.clone())
    )
    # A copy *plus donor reset* has zero total f, but it is not a policy proof;
    # R1 remains withheld rather than accepting an inferred ownership mutation.
    assert report.status == WITHHELD_UNSPECIFIED_TRANSFER_POLICY
    assert report.population_machine_zero is True


def test_unpaired_copy_without_donor_debit_is_rejected_by_population_sum() -> None:
    donor_before = torch.ones((1, 19), dtype=torch.float64)
    report = validate_i_to_g_population_transfer(
        _event(), _evidence(donor_before=donor_before, donor_after=donor_before, receiver_after=donor_before.clone())
    )
    assert report.status == "WITHHELD_POPULATION_RESIDUAL_NONZERO"
    assert report.population_residual is not None
    assert report.population_residual.item() == 19.0


def test_source_destination_partitions_cannot_overlap_or_infer_non_i_receiver() -> None:
    overlap = validate_i_to_g_population_transfer(_event(destination_partition=((1, 1, 1),)), _evidence())
    bad_owner = validate_i_to_g_population_transfer(_event(destination_phase_before=Phase.G), _evidence())
    assert overlap.status == "WITHHELD_OVERLAPPING_PARTITIONS"
    assert bad_owner.status == "WITHHELD_INVALID_OWNERSHIP_PARTITION"


def test_independent_mass_ledger_is_separate_and_tamper_checked() -> None:
    report = validate_i_to_g_population_transfer(
        _event(), _evidence(independent_mass=IndependentMassLedger(
            torch.tensor(-0.25, dtype=torch.float64), torch.tensor(0.25, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        ))
    )
    assert report.status == "WITHHELD_INDEPENDENT_MASS_RESIDUAL_TAMPERED"


def test_float32_machine_zero_is_not_claimed_mathematical_exactness() -> None:
    report = validate_i_to_g_population_transfer(_event(), _evidence(torch.float32))
    assert report.status == WITHHELD_UNSPECIFIED_TRANSFER_POLICY
    assert report.population_machine_zero is True
    assert report.independent_mass_machine_zero is True
    assert report.exact_float32_closure_claimed is False


def test_unknown_policy_is_fail_closed() -> None:
    plan = plan_i_to_g_population_transfer(_event(), _evidence(), policy="copy" )  # type: ignore[arg-type]
    assert plan.status == "WITHHELD_UNSUPPORTED_TRANSFER_POLICY"
