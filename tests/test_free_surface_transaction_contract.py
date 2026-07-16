"""R1 fail-closed contract tests for detached Körner I→G diagnostics."""
from __future__ import annotations

from tensorlbm.free_surface_transaction_contract import (
    CellConversion,
    CellState,
    D3Q19,
    OwnershipEvidence,
    PopulationTransferEvidence,
    RoundoffResidualEvidence,
    TransactionInput,
    WITHHELD_D3Q19_ONLY,
    WITHHELD_NO_POPULATION_TRANSFER,
    WITHHELD_ROUNDOFF_NOT_EXACT,
    diagnose_korner_i_to_g_transaction,
)


def _complete_input(**changes: object) -> TransactionInput:
    values: dict[str, object] = {
        "event_id": "step-17/i-to-g-0",
        "lattice": D3Q19,
        "conversions": (CellConversion((3, 4, 5), CellState.I, CellState.G),),
        "donor_ownership": (
            OwnershipEvidence((3, 4, 5), CellState.I, "independent_mass", "f"),
        ),
        "receiver_ownership": (
            OwnershipEvidence((3, 4, 6), CellState.I, "independent_mass", "f"),
        ),
        "population_transfer": PopulationTransferEvidence(
            actual_f_population_transfer=True,
            source_cells=((3, 4, 5),),
            destination_cells=((3, 4, 6),),
            replay_reference="captured-f-transfer-17",
        ),
        "roundoff": RoundoffResidualEvidence(residual=0.0, claimed_exact_closure=True),
    }
    values.update(changes)
    return TransactionInput(**values)  # type: ignore[arg-type]


def test_complete_d3q19_evidence_is_diagnostic_only_never_physical_acceptance() -> None:
    report = diagnose_korner_i_to_g_transaction(_complete_input())

    assert report.diagnostic_accepted is True
    assert report.physical_accepted is False
    assert report.status == "DIAGNOSTIC_ACCEPTED_PHYSICAL_WITHHELD"
    assert report.verified_states == (CellState.G, CellState.I, CellState.L, CellState.S)


def test_missing_actual_f_population_transfer_is_named_fail_closed_withholding() -> None:
    report = diagnose_korner_i_to_g_transaction(
        _complete_input(population_transfer=PopulationTransferEvidence(False, (), (), None))
    )

    assert report.status == WITHHELD_NO_POPULATION_TRANSFER
    assert report.diagnostic_accepted is False
    assert report.physical_accepted is False


def test_non_d3q19_lattice_is_withheld() -> None:
    report = diagnose_korner_i_to_g_transaction(_complete_input(lattice="D3Q27"))

    assert report.status == WITHHELD_D3Q19_ONLY
    assert report.diagnostic_accepted is False


def test_nonzero_roundoff_residual_cannot_be_promoted_to_exact_closure() -> None:
    report = diagnose_korner_i_to_g_transaction(
        _complete_input(roundoff=RoundoffResidualEvidence(5.960464477539063e-08, True))
    )

    assert report.status == WITHHELD_ROUNDOFF_NOT_EXACT
    assert report.diagnostic_accepted is False
    assert "roundoff residual" in report.reason


def test_receiver_ownership_must_exactly_bind_actual_population_transfer_destinations() -> None:
    report = diagnose_korner_i_to_g_transaction(
        _complete_input(
            receiver_ownership=(OwnershipEvidence((99, 99, 99), CellState.I, "independent_mass", "f"),)
        )
    )

    assert report.status == "WITHHELD_TRANSFER_RECEIVER_OWNERSHIP_MISMATCH"
    assert report.diagnostic_accepted is False
    assert report.physical_accepted is False


def test_invalid_population_transfer_collection_type_is_named_withholding_not_exception() -> None:
    report = diagnose_korner_i_to_g_transaction(
        _complete_input(
            population_transfer=PopulationTransferEvidence(
                actual_f_population_transfer=True,
                source_cells=((3, 4, 5),),
                destination_cells=[(3, 4, 6)],  # type: ignore[arg-type]
                replay_reference="captured-f-transfer-17",
            )
        )
    )

    assert report.status == WITHHELD_NO_POPULATION_TRANSFER
    assert report.diagnostic_accepted is False
    assert report.physical_accepted is False


def test_missing_event_id_and_receiver_ownership_fail_closed() -> None:
    event = diagnose_korner_i_to_g_transaction(_complete_input(event_id=""))
    receiver = diagnose_korner_i_to_g_transaction(_complete_input(receiver_ownership=()))

    assert event.status == "WITHHELD_MISSING_EVENT_ID"
    assert receiver.status == "WITHHELD_MISSING_RECEIVER_OWNERSHIP"
