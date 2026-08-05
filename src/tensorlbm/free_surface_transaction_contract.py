"""Fail-closed R1 contract for detached D3Q19 Körner I→G transactions.

This cold diagnostic validates evidence only.  It neither calls nor imports the
free-surface solver, topology transaction, ownership ledger, or population
mutation path.  In particular, it cannot turn an arithmetic residual or a
record ledger into physical acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Literal

D3Q19 = "D3Q19"
WITHHELD_D3Q19_ONLY = "WITHHELD_D3Q19_ONLY"
WITHHELD_NO_POPULATION_TRANSFER = "WITHHELD_NO_POPULATION_TRANSFER"
WITHHELD_ROUNDOFF_NOT_EXACT = "WITHHELD_ROUNDOFF_NOT_EXACT"

Cell = tuple[int, int, int]


class CellState(str, Enum):
    """The complete R1 state vocabulary; no implicit state aliases exist."""

    G = "G"
    I = "I"
    L = "L"
    S = "S"


@dataclass(frozen=True)
class CellConversion:
    cell: Cell
    before: CellState
    after: CellState


@dataclass(frozen=True)
class OwnershipEvidence:
    cell: Cell
    state: CellState
    independent_mass_owner: Literal["independent_mass"]
    population_owner: Literal["f"]


@dataclass(frozen=True)
class PopulationTransferEvidence:
    """Evidence of an *actual* f-population transfer, not an intended transfer."""

    actual_f_population_transfer: bool
    source_cells: tuple[Cell, ...]
    destination_cells: tuple[Cell, ...]
    replay_reference: str | None


@dataclass(frozen=True)
class RoundoffResidualEvidence:
    residual: float
    claimed_exact_closure: bool


@dataclass(frozen=True)
class TransactionInput:
    event_id: str
    lattice: str
    conversions: tuple[CellConversion, ...]
    donor_ownership: tuple[OwnershipEvidence, ...]
    receiver_ownership: tuple[OwnershipEvidence, ...]
    population_transfer: PopulationTransferEvidence | None
    roundoff: RoundoffResidualEvidence | None


@dataclass(frozen=True)
class TransactionContractReport:
    """Diagnostic acceptance is intentionally distinct from physical acceptance."""

    status: str
    reason: str
    diagnostic_accepted: bool
    physical_accepted: bool
    verified_states: tuple[CellState, ...]
    event_id: str | None


def _withheld(status: str, reason: str, event_id: str | None) -> TransactionContractReport:
    return TransactionContractReport(status, reason, False, False, tuple(CellState), event_id)


def _valid_cell(cell: object) -> bool:
    return (
        isinstance(cell, tuple)
        and len(cell) == 3
        and all(isinstance(coordinate, int) and not isinstance(coordinate, bool) for coordinate in cell)
    )


def diagnose_korner_i_to_g_transaction(transaction: object) -> TransactionContractReport:
    """Validate R1 preconditions and withhold on every absent or non-exact fact.

    A successful diagnostic result only says the supplied detached evidence is
    internally complete for this contract.  It is never physical acceptance:
    this module has no solver-state mutation, full phase replay, or independent
    physical-closure evidence.  No tolerance, global correction, or dtype
    substitution is applied.
    """
    if not isinstance(transaction, TransactionInput):
        return _withheld("WITHHELD_INVALID_TRANSACTION_INPUT", "TransactionInput is required", None)
    event_id = transaction.event_id if isinstance(transaction.event_id, str) else None
    if not event_id or not event_id.strip():
        return _withheld("WITHHELD_MISSING_EVENT_ID", "transaction input requires a non-empty event id", event_id)
    if transaction.lattice != D3Q19:
        return _withheld(WITHHELD_D3Q19_ONLY, "R1 accepts D3Q19 transactions only", event_id)
    if not transaction.conversions:
        return _withheld("WITHHELD_MISSING_CONVERSION_CELLS", "transaction input requires conversion cells", event_id)
    if any(
        not isinstance(item, CellConversion) or not _valid_cell(item.cell)
        or item.before is not CellState.I or item.after is not CellState.G
        for item in transaction.conversions
    ):
        return _withheld("WITHHELD_INVALID_I_TO_G_CONVERSION", "every conversion must be a valid I→G cell", event_id)
    converting_cells = {item.cell for item in transaction.conversions}
    if not transaction.donor_ownership:
        return _withheld("WITHHELD_MISSING_DONOR_OWNERSHIP", "transaction input requires donor ownership", event_id)
    donors = {item.cell for item in transaction.donor_ownership if isinstance(item, OwnershipEvidence)}
    if converting_cells != donors or any(
        not _valid_cell(item.cell) or item.state is not CellState.I
        or item.independent_mass_owner != "independent_mass" or item.population_owner != "f"
        for item in transaction.donor_ownership
    ):
        return _withheld("WITHHELD_INVALID_DONOR_OWNERSHIP", "each converting I cell requires mass and f ownership", event_id)
    if not transaction.receiver_ownership:
        return _withheld("WITHHELD_MISSING_RECEIVER_OWNERSHIP", "transaction input requires receiver ownership", event_id)
    receiver_cells = {item.cell for item in transaction.receiver_ownership if isinstance(item, OwnershipEvidence)}
    if any(
        not isinstance(item, OwnershipEvidence) or not _valid_cell(item.cell)
        or item.state is not CellState.I or item.independent_mass_owner != "independent_mass"
        or item.population_owner != "f"
        for item in transaction.receiver_ownership
    ):
        return _withheld("WITHHELD_INVALID_RECEIVER_OWNERSHIP", "each receiver requires I-state mass and f ownership", event_id)
    transfer = transaction.population_transfer
    if (
        not isinstance(transfer, PopulationTransferEvidence)
        or transfer.actual_f_population_transfer is not True
        or not transfer.source_cells
        or not transfer.destination_cells
        or not isinstance(transfer.replay_reference, str)
        or not transfer.replay_reference.strip()
        or not isinstance(transfer.source_cells, tuple)
        or not isinstance(transfer.destination_cells, tuple)
        or any(not _valid_cell(cell) for cell in transfer.source_cells)
        or any(not _valid_cell(cell) for cell in transfer.destination_cells)
    ):
        return _withheld(
            WITHHELD_NO_POPULATION_TRANSFER,
            "actual f population-transfer evidence is required; intent or mass-only evidence is insufficient",
            event_id,
        )
    if set(transfer.destination_cells) != receiver_cells:
        return _withheld(
            "WITHHELD_TRANSFER_RECEIVER_OWNERSHIP_MISMATCH",
            "every actual f-transfer destination requires exactly matching I-state mass and f receiver ownership",
            event_id,
        )
    if not converting_cells.issubset(set(transfer.source_cells)):
        return _withheld(WITHHELD_NO_POPULATION_TRANSFER, "every converting donor requires actual f-transfer source evidence", event_id)
    roundoff = transaction.roundoff
    if not isinstance(roundoff, RoundoffResidualEvidence):
        return _withheld("WITHHELD_MISSING_ROUNDOFF_EVIDENCE", "transaction input requires roundoff evidence", event_id)
    if not isinstance(roundoff.residual, float) or not isfinite(roundoff.residual) or roundoff.residual != 0.0:
        return _withheld(
            WITHHELD_ROUNDOFF_NOT_EXACT,
            "a nonzero or non-finite roundoff residual cannot be promoted to exact closure",
            event_id,
        )
    if roundoff.claimed_exact_closure is not True:
        return _withheld("WITHHELD_EXACT_CLOSURE_NOT_CLAIMED", "exact diagnostic closure must be explicitly evidenced", event_id)
    return TransactionContractReport(
        "DIAGNOSTIC_ACCEPTED_PHYSICAL_WITHHELD",
        "D3Q19 diagnostic evidence is complete; physical acceptance remains withheld by R1 scope",
        True,
        False,
        tuple(CellState),
        event_id,
    )
