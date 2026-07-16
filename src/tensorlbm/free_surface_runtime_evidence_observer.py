"""Fail-closed cold observer from runtime-shaped Körner evidence to R1 input.

This adapter is deliberately one way: it normalizes only explicit evidence and
passes a newly built :class:`TransactionInput` to the detached transaction
contract.  It does not import or call a free-surface solver, topology mutation,
or ownership ledger.  In particular, ``fill``, mass, flags, or intent fields
are never used to invent an f-population transfer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from tensorlbm.free_surface_transaction_contract import (
    CellConversion,
    CellState,
    D3Q19,
    OwnershipEvidence,
    PopulationTransferEvidence,
    RoundoffResidualEvidence,
    TransactionContractReport,
    TransactionInput,
    diagnose_korner_i_to_g_transaction,
)


@dataclass(frozen=True)
class RuntimeKornerEvidence:
    """Raw runtime-shaped evidence fields, intentionally typed as ``object``.

    Runtime producers may supply a mapping directly instead.  The raw shape is
    preserved until exact, local validation below; no numeric coercion or state
    inference is performed.
    """

    event_id: object = None
    lattice: object = None
    conversions: object = None
    donor_ownership: object = None
    receiver_ownership: object = None
    population_transfer: object = None
    roundoff: object = None


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _cell(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    if any(not isinstance(part, int) or isinstance(part, bool) for part in value):
        return None
    return tuple(value)  # type: ignore[return-value]


def _state(value: object) -> CellState | None:
    try:
        return CellState(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _conversions(value: object) -> tuple[CellConversion, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    result: list[CellConversion] = []
    for raw in value:
        item = _mapping(raw)
        if item is None:
            return ()
        cell, before, after = _cell(item.get("cell")), _state(item.get("before")), _state(item.get("after"))
        if cell is None or before is None or after is None:
            return ()
        result.append(CellConversion(cell, before, after))
    return tuple(result)


def _ownership(value: object) -> tuple[OwnershipEvidence, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    result: list[OwnershipEvidence] = []
    for raw in value:
        item = _mapping(raw)
        if item is None:
            return ()
        cell, state = _cell(item.get("cell")), _state(item.get("state"))
        mass_owner, population_owner = item.get("independent_mass_owner"), item.get("population_owner")
        if cell is None or state is None or not isinstance(mass_owner, str) or not isinstance(population_owner, str):
            return ()
        result.append(OwnershipEvidence(cell, state, mass_owner, population_owner))  # type: ignore[arg-type]
    return tuple(result)


def _cells(value: object) -> tuple[tuple[int, int, int], ...] | None:
    if not isinstance(value, (tuple, list)):
        return None
    result = tuple(_cell(raw) for raw in value)
    return result if all(cell is not None for cell in result) else None  # type: ignore[return-value]


def _population_transfer(value: object) -> PopulationTransferEvidence | None:
    item = _mapping(value)
    if item is None:
        return None
    sources, destinations = _cells(item.get("source_cells")), _cells(item.get("destination_cells"))
    actual, replay = item.get("actual_f_population_transfer"), item.get("replay_reference")
    # These are all actual-event fields.  No other runtime field is a fallback.
    if sources is None or destinations is None:
        return None
    return PopulationTransferEvidence(actual, sources, destinations, replay)  # type: ignore[arg-type]


def _roundoff(value: object) -> RoundoffResidualEvidence | None:
    item = _mapping(value)
    if item is None:
        return None
    residual, exact = item.get("residual"), item.get("claimed_exact_closure")
    return RoundoffResidualEvidence(residual, exact)  # type: ignore[arg-type]


def transaction_input_from_runtime_evidence(evidence: RuntimeKornerEvidence | Mapping[str, object] | object) -> TransactionInput:
    """Build a contract input without deriving any missing population evidence.

    Invalid substructures become absent/empty contract evidence rather than an
    exception.  The contract is therefore the single fail-closed policy owner.
    """
    if isinstance(evidence, RuntimeKornerEvidence):
        raw: Mapping[str, object] = {
            "event_id": evidence.event_id,
            "lattice": evidence.lattice,
            "conversions": evidence.conversions,
            "donor_ownership": evidence.donor_ownership,
            "receiver_ownership": evidence.receiver_ownership,
            "population_transfer": evidence.population_transfer,
            "roundoff": evidence.roundoff,
        }
    else:
        raw = _mapping(evidence) or {}
    return TransactionInput(
        event_id=raw.get("event_id"),  # type: ignore[arg-type]
        lattice=raw.get("lattice"),  # type: ignore[arg-type]
        conversions=_conversions(raw.get("conversions")),
        donor_ownership=_ownership(raw.get("donor_ownership")),
        receiver_ownership=_ownership(raw.get("receiver_ownership")),
        population_transfer=_population_transfer(raw.get("population_transfer")),
        roundoff=_roundoff(raw.get("roundoff")),
    )


def observe_korner_runtime_evidence(evidence: RuntimeKornerEvidence | Mapping[str, object] | object) -> TransactionContractReport:
    """Observe detached runtime evidence and always return a fail-closed report."""
    try:
        return diagnose_korner_i_to_g_transaction(transaction_input_from_runtime_evidence(evidence))
    except Exception:
        # A hostile/non-standard Mapping must not let the observer escape its
        # diagnostic boundary.  This fallback itself contains no f transfer.
        return diagnose_korner_i_to_g_transaction(
            TransactionInput("", D3Q19, (), (), (), None, None)
        )
