"""Pure conservative inventory reference for free-surface diagnostics.

This module deliberately has no population or solver dependency.  It defines
ownership for the liquid inventory represented by bulk LIQUID density plus
INTERFACE mass and records only zero-sum internal transactions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

# Keep these local so this reference cannot acquire a solver import dependency.
GAS = 0
LIQUID = 1
INTERFACE = 2


class InventoryTopologyError(ValueError):
    """A requested transaction has no conservative, local ownership."""


@dataclass(frozen=True)
class SolverMappingGap:
    code: str
    detail: str


@dataclass(frozen=True)
class InventoryState:
    """Physical liquid inventory, separated from LBM populations.

    ``bulk_liquid`` is valid only at LIQUID sites and may have arbitrary
    density. ``interface_mass`` is valid only at INTERFACE sites and is bounded
    by ``rho_liquid``.  GAS sites own neither quantity.
    """

    flags: torch.Tensor
    bulk_liquid: torch.Tensor
    interface_mass: torch.Tensor
    rho_liquid: float = 1.0

    @classmethod
    def from_fields(
        cls,
        flags: torch.Tensor,
        *,
        bulk_liquid: torch.Tensor,
        interface_mass: torch.Tensor,
        rho_liquid: float = 1.0,
    ) -> "InventoryState":
        state = cls(flags.clone(), bulk_liquid.clone(), interface_mass.clone(), rho_liquid)
        state.validate()
        return state

    @property
    def total(self) -> float:
        return float((self.bulk_liquid + self.interface_mass).sum())

    def validate(self) -> None:
        if self.flags.shape != self.bulk_liquid.shape or self.flags.shape != self.interface_mass.shape:
            raise InventoryTopologyError("flags and inventory fields must have identical shapes")
        liquid = self.flags == LIQUID
        interface = self.flags == INTERFACE
        invalid_bulk = (~liquid) & (self.bulk_liquid != 0)
        invalid_interface = (~interface) & (self.interface_mass != 0)
        if bool(invalid_bulk.any()) or bool(invalid_interface.any()):
            raise InventoryTopologyError("inventory must be owned by its declared cell phase")
        if bool((self.bulk_liquid < 0).any()) or bool((self.interface_mass < 0).any()):
            raise InventoryTopologyError("inventory cannot be negative")



@dataclass
class InventoryLedger:
    start_total: float
    entries: list[tuple[str, float]]

    @classmethod
    def start(cls, state: InventoryState) -> "InventoryLedger":
        return cls(state.total, [])

    def record(self, name: str, before: InventoryState, after: InventoryState) -> None:
        self.entries.append((name, after.total - before.total))

    def delta(self, name: str) -> float:
        return sum(delta for entry, delta in self.entries if entry == name)

    @property
    def total_delta(self) -> float:
        return sum(delta for _, delta in self.entries)


def _replace(
    state: InventoryState, *, flags: torch.Tensor | None = None,
    bulk: torch.Tensor | None = None, interface: torch.Tensor | None = None,
) -> InventoryState:
    return InventoryState.from_fields(
        state.flags if flags is None else flags,
        bulk_liquid=state.bulk_liquid if bulk is None else bulk,
        interface_mass=state.interface_mass if interface is None else interface,
        rho_liquid=state.rho_liquid,
    )


def apply_frozen_step(
    state: InventoryState,
    *,
    link_exchanges: Iterable[tuple[int, int, float]] = (),
    conversions: Iterable[tuple[int, int]] = (),
    redistributions: Iterable[tuple[int, Sequence[tuple[int, float]]]] = (),
) -> tuple[InventoryState, InventoryLedger]:
    """Apply a closed topology transaction sequence without population mutation.

    Each link tuple is ``(liquid_index, interface_index, amount)``. Positive
    amount moves inventory from LIQUID to INTERFACE; negative reverses it.
    Conversion is ``(interface_index, LIQUID)`` and consumes exactly one
    nominal liquid density, leaving its positive overflow for an explicit
    redistribution owned by that converting site.
    """
    current = state
    ledger = InventoryLedger.start(state)
    for liquid_index, interface_index, amount in link_exchanges:
        if int(current.flags[liquid_index]) != LIQUID or int(current.flags[interface_index]) != INTERFACE:
            raise InventoryTopologyError("link exchange requires a LIQUID and INTERFACE endpoint")
        bulk = current.bulk_liquid.clone()
        interface = current.interface_mass.clone()
        bulk[liquid_index] -= amount
        interface[interface_index] += amount
        before = current
        current = _replace(current, bulk=bulk, interface=interface)
        ledger.record("liquid_interface_exchange", before, current)

    pending: dict[int, float] = {}
    for index, target in conversions:
        if target != LIQUID or int(current.flags[index]) != INTERFACE:
            raise InventoryTopologyError("only INTERFACE to LIQUID conversion is defined by this reference")
        owned = float(current.interface_mass[index])
        if owned < current.rho_liquid:
            raise InventoryTopologyError("conversion requires at least rho_liquid inventory")
        overflow = owned - current.rho_liquid
        flags, bulk, interface = current.flags.clone(), current.bulk_liquid.clone(), current.interface_mass.clone()
        flags[index] = LIQUID
        bulk[index] = current.rho_liquid
        interface[index] = 0.0
        before = current
        current = _replace(current, flags=flags, bulk=bulk, interface=interface)
        ledger.record("interface_to_liquid", before, current)
        pending[index] = overflow

    for donor, receivers in redistributions:
        if donor not in pending:
            raise InventoryTopologyError("redistribution requires overflow owned by a conversion")
        total = sum(amount for _, amount in receivers)
        if abs(total - pending[donor]) > 1.0e-6:
            raise InventoryTopologyError("overflow must be redistributed exactly; no clamp or global rescale")
        interface = current.interface_mass.clone()
        for receiver, amount in receivers:
            if int(current.flags[receiver]) != INTERFACE:
                raise InventoryTopologyError("overflow receivers must be INTERFACE cells")
            interface[receiver] += amount
        before = current
        current = _replace(current, interface=interface)
        ledger.record("redistribution", before, current)
        del pending[donor]
    if pending:
        raise InventoryTopologyError("overflow requires an explicit redistribution transaction")
    if bool((current.interface_mass > current.rho_liquid).any()):
        raise InventoryTopologyError("interface mass overflow requires an explicit conversion transaction")
    if abs(current.total - ledger.start_total) > 1.0e-6:
        raise InventoryTopologyError("closed transaction sequence is not conservative")
    return current, ledger


def solver_mapping_gaps(solver_ledger: Mapping[str, float]) -> list[SolverMappingGap]:
    """Name unowned solver effects that cannot map to this pure ledger."""
    gaps: list[SolverMappingGap] = []
    if solver_ledger.get("exchange_liquid_delta", 0.0):
        gaps.append(SolverMappingGap("unpaired_liquid_interface_exchange", "solver records only the interface endpoint; no bulk-liquid debit is exposed"))
    if solver_ledger.get("abb_population_delta", 0.0):
        gaps.append(SolverMappingGap("abb_population_inventory_ownership", "ABB changes populations but provides no declared inventory owner"))
    if "conversion" in solver_ledger:
        gaps.append(SolverMappingGap("conversion_transaction_ownership", "solver exposes only aggregate post-conversion mass, not per-cell conversion ownership"))
    if "redistribution" in solver_ledger:
        gaps.append(SolverMappingGap("redistribution_transaction_ownership", "solver exposes only aggregate redistribution mass, not donor/receiver transfers"))
    return gaps
