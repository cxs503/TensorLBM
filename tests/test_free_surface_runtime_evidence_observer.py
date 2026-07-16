"""Runtime-shaped evidence observer tests: detached, explicit, and fail closed."""
from __future__ import annotations

from tensorlbm.free_surface_runtime_evidence_observer import (
    RuntimeKornerEvidence,
    observe_korner_runtime_evidence,
    transaction_input_from_runtime_evidence,
)
from tensorlbm.free_surface_transaction_contract import WITHHELD_NO_POPULATION_TRANSFER


def _explicit_runtime_evidence() -> dict[str, object]:
    return {
        "event_id": "runtime-step-17/i-to-g-0",
        "lattice": "D3Q19",
        "conversions": ({"cell": (3, 4, 5), "before": "I", "after": "G"},),
        "donor_ownership": (
            {"cell": (3, 4, 5), "state": "I", "independent_mass_owner": "independent_mass", "population_owner": "f"},
        ),
        "receiver_ownership": (
            {"cell": (3, 4, 6), "state": "I", "independent_mass_owner": "independent_mass", "population_owner": "f"},
        ),
        "population_transfer": {
            "actual_f_population_transfer": True,
            "source_cells": ((3, 4, 5),),
            "destination_cells": ((3, 4, 6),),
            "replay_reference": "runtime-captured-f-17",
        },
        "roundoff": {"residual": 0.0, "claimed_exact_closure": True},
    }


def test_runtime_shaped_mass_only_evidence_is_withheld_without_inventing_f_transfer() -> None:
    evidence = _explicit_runtime_evidence()
    evidence.pop("population_transfer")
    evidence.update({"fill": 1.0, "mass": 3.0, "flags": "I_TO_G", "transfer_intent": "copy"})

    transaction = transaction_input_from_runtime_evidence(evidence)
    report = observe_korner_runtime_evidence(evidence)

    assert transaction.population_transfer is None
    assert report.status == WITHHELD_NO_POPULATION_TRANSFER
    assert report.diagnostic_accepted is False
    assert report.physical_accepted is False


def test_malformed_runtime_evidence_never_raises() -> None:
    malformed = RuntimeKornerEvidence(
        event_id="bad-runtime-shape",
        lattice="D3Q19",
        conversions=object(),
        donor_ownership={"not": "a sequence"},
        receiver_ownership=None,
        population_transfer={"actual_f_population_transfer": True, "source_cells": object()},
        roundoff={"residual": "not-a-float", "claimed_exact_closure": "yes"},
    )

    report = observe_korner_runtime_evidence(malformed)

    assert report.diagnostic_accepted is False
    assert report.physical_accepted is False
    assert report.status.startswith("WITHHELD_")


def test_full_explicit_runtime_map_is_diagnostic_accepted_but_never_physical() -> None:
    report = observe_korner_runtime_evidence(_explicit_runtime_evidence())

    assert report.status == "DIAGNOSTIC_ACCEPTED_PHYSICAL_WITHHELD"
    assert report.diagnostic_accepted is True
    assert report.physical_accepted is False


def test_partial_actual_transfer_without_replay_payload_is_population_withheld() -> None:
    evidence = _explicit_runtime_evidence()
    transfer = evidence["population_transfer"]
    assert isinstance(transfer, dict)
    transfer.pop("replay_reference")

    report = observe_korner_runtime_evidence(evidence)

    assert report.status == WITHHELD_NO_POPULATION_TRANSFER
