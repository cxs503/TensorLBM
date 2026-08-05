"""Fail-closed ownership observations for one D3Q19 free-surface step.

R1 is deliberately a cold diagnostic adapter.  It turns facts already emitted
by ``free_surface_step`` into immutable records, but never changes populations,
fill, flags, or tracked mass.  ``OBSERVED_NOT_PHYSICAL_CLOSURE`` means exactly
that: tracked-state ownership evidence exists; physical/PV closure is not
claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .d3q19 import C

GAS = 0
LIQUID = 1
INTERFACE = 2


class OwnershipLedgerError(ValueError):
    """An asserted paired ownership transaction is incomplete or invalid."""


@dataclass(frozen=True)
class CellOwner:
    cell: tuple[int, int, int]
    owner_phase: str


@dataclass(frozen=True)
class LiquidInterfaceTransferRecord:
    direction: int
    source: CellOwner
    target: CellOwner
    credit: float
    debit: float | None
    net: float | None
    ownership: str


@dataclass(frozen=True)
class RedistributionRecord:
    donor: CellOwner
    receiver: CellOwner
    mass_delta: float
    ownership: str


@dataclass(frozen=True)
class ConversionRecord:
    cell: tuple[int, int, int]
    before_owner_phase: str
    after_owner_phase: str
    mass_before: float
    mass_after: float
    mass_delta: float
    ownership: str


@dataclass(frozen=True)
class ABBRecord:
    population_delta: float
    population_only: bool
    inventory_owner_status: str


@dataclass(frozen=True)
class OwnershipLedgerState:
    """Immutable per-step tracked-state ownership observation.

    This is not a physical/PV closure state, including when all presently
    observable transfer pairs balance.
    """

    status: str
    unresolved_categories: tuple[str, ...]
    liquid_interface_transfers: tuple[LiquidInterfaceTransferRecord, ...]
    redistributions: tuple[RedistributionRecord, ...]
    conversions: tuple[ConversionRecord, ...]
    abb_records: tuple[ABBRecord, ...]


def _cell(raw: object, name: str) -> tuple[int, int, int]:
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        raise OwnershipLedgerError(f"{name} must be a (z, y, x) cell identity")
    return tuple(int(value) for value in raw)  # type: ignore[return-value]


def _phase(flag: object) -> str:
    value = int(flag)
    if value == LIQUID:
        return "LIQUID"
    if value == INTERFACE:
        return "INTERFACE"
    if value == GAS:
        return "GAS"
    return "OTHER"


def _source(target: tuple[int, int, int], q: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    # C is ordered (x, y, z), whereas field identities are (z, y, x).
    return (
        (target[0] - int(C[q, 2])) % shape[0],
        (target[1] - int(C[q, 1])) % shape[1],
        (target[2] - int(C[q, 0])) % shape[2],
    )


def _liquid_interface_records(
    flags: torch.Tensor,
    mass_delta_liquid: torch.Tensor | None,
    liquid_interface_mask: torch.Tensor | None,
    paired: bool,
) -> tuple[tuple[LiquidInterfaceTransferRecord, ...], tuple[str, ...]]:
    if mass_delta_liquid is None and liquid_interface_mask is None:
        return (), ()
    if mass_delta_liquid is None or liquid_interface_mask is None:
        raise OwnershipLedgerError("L/I ownership requires both mass_delta_liquid and liquid_interface_mask")
    if tuple(mass_delta_liquid.shape) != (19, *flags.shape) or liquid_interface_mask.shape != mass_delta_liquid.shape:
        raise OwnershipLedgerError("L/I ownership facts must have shape (19, *flags.shape)")

    records: list[LiquidInterfaceTransferRecord] = []
    shape = tuple(int(value) for value in flags.shape)
    for raw in torch.nonzero(liquid_interface_mask, as_tuple=False).tolist():
        q, z, y, x = (int(value) for value in raw)
        target = (z, y, x)
        source = _source(target, q, shape)
        source_phase = _phase(flags[source].item())
        target_phase = _phase(flags[target].item())
        if target_phase != "INTERFACE":
            raise OwnershipLedgerError("L/I paired transfer requires an INTERFACE target owner")
        if paired and source_phase != "LIQUID":
            raise OwnershipLedgerError("L/I paired transfer requires a LIQUID source owner")
        credit = float(mass_delta_liquid[q, z, y, x])
        if paired:
            records.append(LiquidInterfaceTransferRecord(
                q, CellOwner(source, "LIQUID"), CellOwner(target, "INTERFACE"),
                credit, -credit, 0.0, "PAIRED",
            ))
        else:
            records.append(LiquidInterfaceTransferRecord(
                q, CellOwner(source, source_phase), CellOwner(target, "INTERFACE"),
                credit, None, None, "UNPAIRED/WITHHELD",
            ))
    return tuple(records), (() if paired else ("unpaired_liquid_interface_debit",))


def _evidence_records(
    evidence: Mapping[str, object] | None, flags: torch.Tensor,
) -> tuple[tuple[RedistributionRecord, ...], tuple[ConversionRecord, ...], tuple[str, ...]]:
    if evidence is None:
        return (), (), ()
    redistributions: list[RedistributionRecord] = []
    conversions: list[ConversionRecord] = []
    unresolved: list[str] = []
    raw_links = evidence.get("redistribution_links", ())
    raw_cells = evidence.get("conversion_cells", ())
    if not isinstance(raw_links, (tuple, list)) or not isinstance(raw_cells, (tuple, list)):
        raise OwnershipLedgerError("conversion evidence must contain sequence records")
    for raw in raw_links:
        if not isinstance(raw, Mapping):
            raise OwnershipLedgerError("redistribution evidence record must be a mapping")
        donor = _cell(raw.get("donor"), "redistribution donor")
        receiver = _cell(raw.get("receiver"), "redistribution receiver")
        donor_phase = _phase(flags[donor].item())
        # A valid same-step receiver can be GAS in the pre-topology snapshot
        # and promoted to INTERFACE by the staged plan.  Evidence records that
        # post-conversion owner explicitly; never infer it from a population.
        receiver_phase = _phase(raw.get("receiver_flag_after", flags[receiver].item()))
        if receiver_phase != "INTERFACE":
            raise OwnershipLedgerError("redistribution receiver must be an INTERFACE owner")
        if donor_phase not in {"LIQUID", "INTERFACE"}:
            raise OwnershipLedgerError("redistribution donor must have a tracked-state owner")
        redistributions.append(RedistributionRecord(
            CellOwner(donor, donor_phase), CellOwner(receiver, "INTERFACE"),
            float(raw.get("mass_delta", 0.0)), "OBSERVED_TRACKED_STATE",
        ))
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise OwnershipLedgerError("conversion evidence record must be a mapping")
        before = _phase(raw.get("flag_before"))
        after = _phase(raw.get("flag_after"))
        if before not in {"LIQUID", "INTERFACE"} or after not in {"LIQUID", "INTERFACE", "GAS"}:
            raise OwnershipLedgerError("conversion evidence has an invalid tracked-state owner")
        conversions.append(ConversionRecord(
            _cell(raw.get("cell"), "conversion cell"), before, after,
            float(raw.get("mass_before", 0.0)), float(raw.get("mass_after", 0.0)),
            float(raw.get("mass_delta", 0.0)), "OBSERVED_TRACKED_STATE",
        ))
    return tuple(redistributions), tuple(conversions), tuple(dict.fromkeys(unresolved))


def build_ownership_ledger(
    *, flags: torch.Tensor, mass_delta_liquid: torch.Tensor | None = None,
    liquid_interface_mask: torch.Tensor | None = None,
    paired_liquid_interface_debit: bool = False,
    conversion_evidence: Mapping[str, object] | None = None,
    abb_population_delta: float = 0.0,
) -> OwnershipLedgerState:
    """Build an immutable, fail-closed ownership observation from step facts."""
    links, link_unresolved = _liquid_interface_records(
        flags, mass_delta_liquid, liquid_interface_mask, paired_liquid_interface_debit,
    )
    redistributions, conversions, evidence_unresolved = _evidence_records(conversion_evidence, flags)
    abb = (ABBRecord(float(abb_population_delta), True, "WITHHELD"),)
    unresolved = list(link_unresolved) + list(evidence_unresolved)
    unresolved.append("abb_population_inventory_owner_withheld")
    return OwnershipLedgerState(
        status="OBSERVED_NOT_PHYSICAL_CLOSURE",
        unresolved_categories=tuple(dict.fromkeys(unresolved)),
        liquid_interface_transfers=links,
        redistributions=redistributions,
        conversions=conversions,
        abb_records=abb,
    )
