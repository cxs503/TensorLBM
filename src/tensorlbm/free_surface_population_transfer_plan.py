"""Pure, fail-closed R1 boundary for Körner INTERFACE→GAS f transfer.

This module is deliberately detached from the solver and topology transaction.
It plans no mutation and supports no executable transfer policy in R1.  Its job
is to make the evidence a future writer must provide explicit and to reject the
common, non-conservative "copy a donor f" shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch

D3Q19 = "D3Q19"
Q = 19
WITHHELD_UNSPECIFIED_TRANSFER_POLICY = "WITHHELD_UNSPECIFIED_TRANSFER_POLICY"

Cell = tuple[int, int, int]


class PopulationTransferPlanError(ValueError):
    """The supplied detached transfer evidence is malformed or incomplete."""


class Phase(str, Enum):
    G = "G"
    I = "I"
    L = "L"
    S = "S"


@dataclass(frozen=True)
class IToGPopulationTransferEvent:
    """Explicit partitions for one event; no neighbours are inferred."""

    event_id: str
    lattice: str
    converting_donors: tuple[Cell, ...]
    source_partition: tuple[Cell, ...]
    destination_partition: tuple[Cell, ...]
    source_phase_before: Phase
    source_phase_after: Phase
    destination_phase_before: Phase
    destination_phase_after: Phase


@dataclass(frozen=True)
class IndependentMassLedger:
    """Separate tracked-mass debit/credit evidence, never derived from ``f``.

    ``debit`` and ``credit`` are signed, independently supplied scalar tensors.
    ``residual`` is an evidence record which must equal their same-dtype sum;
    it is not a population residual and cannot justify an f mutation.
    """

    debit: torch.Tensor
    credit: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class PopulationTransferEvidence:
    """Pre/post populations explicitly keyed by event source/destination order.

    Each population tensor has shape ``(N, 19)``.  Rows correspond exactly to
    the respective event partition, and q is the final dimension.  This avoids
    silent broadcast, neighbour discovery, or ownership inference.
    """

    donor_before: torch.Tensor
    donor_after: torch.Tensor
    receiver_before: torch.Tensor
    receiver_after: torch.Tensor
    independent_mass: IndependentMassLedger


@dataclass(frozen=True)
class PopulationTransferValidation:
    status: str
    reason: str
    event_id: str | None
    population_residual: torch.Tensor | None
    independent_mass_residual: torch.Tensor | None
    population_machine_zero: bool
    independent_mass_machine_zero: bool
    exact_float32_closure_claimed: bool


@dataclass(frozen=True)
class PopulationTransferPlan:
    """R1 output: evidence result only; ``operations`` is intentionally empty."""

    status: str
    reason: str
    validation: PopulationTransferValidation
    operations: tuple[object, ...] = ()


def _valid_cell(cell: object) -> bool:
    return isinstance(cell, tuple) and len(cell) == 3 and all(
        isinstance(value, int) and not isinstance(value, bool) for value in cell
    )


def _withheld(event_id: str | None, status: str, reason: str) -> PopulationTransferValidation:
    return PopulationTransferValidation(status, reason, event_id, None, None, False, False, False)


def _scalar(value: object, name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise PopulationTransferPlanError(f"{name} must be a scalar tensor")
    if value.dtype != dtype or value.device != device or not bool(torch.isfinite(value)):
        raise PopulationTransferPlanError(f"{name} must be finite and match population dtype/device")
    return value


def validate_i_to_g_population_transfer(
    event: object, evidence: object,
) -> PopulationTransferValidation:
    """Validate explicit partitions, f residual, and independent mass evidence.

    No tolerance is used.  For float32, even a bitwise-zero reduction is only a
    machine result and ``exact_float32_closure_claimed`` remains false.  R1
    therefore never promotes it to mathematical exactness or an executable
    policy.
    """
    event_id = event.event_id if isinstance(event, IToGPopulationTransferEvent) else None
    if not isinstance(event, IToGPopulationTransferEvent):
        return _withheld(None, "WITHHELD_INVALID_TRANSFER_EVENT", "IToGPopulationTransferEvent is required")
    if not isinstance(event.event_id, str) or not event.event_id.strip():
        return _withheld(event_id, "WITHHELD_MISSING_EVENT_ID", "event id must be non-empty")
    if event.lattice != D3Q19:
        return _withheld(event_id, "WITHHELD_D3Q19_ONLY", "R1 accepts D3Q19 only")
    partitions = (event.converting_donors, event.source_partition, event.destination_partition)
    if any(not isinstance(partition, tuple) or not partition for partition in partitions):
        return _withheld(event_id, "WITHHELD_MISSING_PARTITION", "all event partitions must be non-empty tuples")
    if any(any(not _valid_cell(cell) for cell in partition) for partition in partitions):
        return _withheld(event_id, "WITHHELD_INVALID_PARTITION_CELL", "partitions require integer (z, y, x) cells")
    if len(set(event.source_partition)) != len(event.source_partition) or len(set(event.destination_partition)) != len(event.destination_partition):
        return _withheld(event_id, "WITHHELD_DUPLICATE_PARTITION_CELL", "source and destination partitions must be unique")
    if set(event.source_partition) & set(event.destination_partition):
        return _withheld(event_id, "WITHHELD_OVERLAPPING_PARTITIONS", "source and destination partitions must not overlap")
    if set(event.converting_donors) != set(event.source_partition):
        return _withheld(event_id, "WITHHELD_SOURCE_CONVERSION_MISMATCH", "every and only converting I→G donor must be a source")
    if (event.source_phase_before, event.source_phase_after, event.destination_phase_before, event.destination_phase_after) != (Phase.I, Phase.G, Phase.I, Phase.I):
        return _withheld(event_id, "WITHHELD_INVALID_OWNERSHIP_PARTITION", "R1 requires I→G sources and surviving I destinations")
    if not isinstance(evidence, PopulationTransferEvidence):
        return _withheld(event_id, "WITHHELD_MISSING_POPULATION_EVIDENCE", "explicit pre/post donor and receiver f tensors are required")
    populations = (evidence.donor_before, evidence.donor_after, evidence.receiver_before, evidence.receiver_after)
    if any(not isinstance(value, torch.Tensor) for value in populations):
        return _withheld(event_id, "WITHHELD_INVALID_POPULATION_EVIDENCE", "population evidence must be tensors")
    first = evidence.donor_before
    expected_donor, expected_receiver = (len(event.source_partition), Q), (len(event.destination_partition), Q)
    if first.shape != expected_donor or evidence.donor_after.shape != expected_donor or evidence.receiver_before.shape != expected_receiver or evidence.receiver_after.shape != expected_receiver:
        return _withheld(event_id, "WITHHELD_POPULATION_SHAPE_MISMATCH", "population tensors must be (partition-size, 19)")
    if any(value.dtype != first.dtype or value.device != first.device or not bool(torch.isfinite(value).all()) for value in populations):
        return _withheld(event_id, "WITHHELD_INVALID_POPULATION_VALUES", "populations must be finite and share dtype/device")
    try:
        ledger = evidence.independent_mass
        debit = _scalar(ledger.debit, "independent_mass.debit", first.dtype, first.device)
        credit = _scalar(ledger.credit, "independent_mass.credit", first.dtype, first.device)
        recorded_mass_residual = _scalar(ledger.residual, "independent_mass.residual", first.dtype, first.device)
    except (AttributeError, PopulationTransferPlanError) as error:
        return _withheld(event_id, "WITHHELD_INVALID_INDEPENDENT_MASS_LEDGER", str(error))
    actual_mass_residual = debit + credit
    if not torch.equal(recorded_mass_residual, actual_mass_residual):
        return _withheld(event_id, "WITHHELD_INDEPENDENT_MASS_RESIDUAL_TAMPERED", "independent-mass residual must equal separate debit plus credit")
    population_residual = (evidence.donor_after - evidence.donor_before).sum() + (evidence.receiver_after - evidence.receiver_before).sum()
    population_zero = bool(population_residual == 0)
    mass_zero = bool(actual_mass_residual == 0)
    if not population_zero:
        return PopulationTransferValidation("WITHHELD_POPULATION_RESIDUAL_NONZERO", "post-minus-pre f population sum is nonzero; direct copy/reset is not conservative evidence", event_id, population_residual, actual_mass_residual, False, mass_zero, False)
    if not mass_zero:
        return PopulationTransferValidation("WITHHELD_INDEPENDENT_MASS_RESIDUAL_NONZERO", "independent mass debit/credit does not close", event_id, population_residual, actual_mass_residual, True, False, False)
    return PopulationTransferValidation(WITHHELD_UNSPECIFIED_TRANSFER_POLICY, "R1 verified explicit evidence but defines no f transfer operator; no copy, overwrite, or magic correction is authorized", event_id, population_residual, actual_mass_residual, True, True, False)


def plan_i_to_g_population_transfer(
    event: object, evidence: object, *, policy: Literal["unspecified"] | None = None,
) -> PopulationTransferPlan:
    """Return an empty, fail-closed plan unless a future policy is defined and tested."""
    validation = validate_i_to_g_population_transfer(event, evidence)
    if policy not in (None, "unspecified"):
        validation = PopulationTransferValidation("WITHHELD_UNSUPPORTED_TRANSFER_POLICY", "R1 has no validated executable f transfer policy", validation.event_id, validation.population_residual, validation.independent_mass_residual, validation.population_machine_zero, validation.independent_mass_machine_zero, False)
    return PopulationTransferPlan(validation.status, validation.reason, validation)
