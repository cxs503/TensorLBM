"""Pure ABB reconstruction-to-inventory reference for the D3Q19 free surface.

This diagnostic is intentionally not called by :func:`free_surface_step`.
It maps its current anti-bounce-back (ABB) population assignment exactly and
makes the otherwise missing inventory decision explicit: an ABB population
change at an INTERFACE cell can enter an inventory ledger only if a caller
names a LIQUID bulk owner for the equal and opposite transfer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .d3q19 import C, OPPOSITE, equilibrium3d
from .free_surface_lbm import GAS, INTERFACE, LIQUID


class ABBInventoryOwnershipError(ValueError):
    """An ABB population reconstruction lacks a valid explicit bulk owner."""


@dataclass(frozen=True)
class ABBReconstructionDensityChange:
    """Exact D3Q19 per-link population replacement performed by solver ABB.

    ``per_link_delta[q, x]`` is ``f_abb[q, x] - f_streamed[q, x]`` precisely
    where an INTERFACE target ``x`` pull-streams from a GAS source ``x-c_q``.
    Its sum is a population-density change, not an implicit liquid inventory
    transfer.
    """

    link_mask: torch.Tensor
    per_link_delta: torch.Tensor
    interface_delta: torch.Tensor
    population_delta: float


@dataclass(frozen=True)
class ABBInventoryState:
    """Separate, phase-owned inventory fields used only by this reference."""

    flags: torch.Tensor
    bulk_liquid: torch.Tensor
    interface_inventory: torch.Tensor

    @classmethod
    def from_flags_and_values(
        cls, flags: torch.Tensor, *, bulk_liquid: torch.Tensor,
        interface_inventory: torch.Tensor,
    ) -> "ABBInventoryState":
        state = cls(flags.clone(), bulk_liquid.clone(), interface_inventory.clone())
        state.validate()
        return state

    @property
    def total(self) -> float:
        return float((self.bulk_liquid + self.interface_inventory).sum())

    def validate(self) -> None:
        if self.flags.shape != self.bulk_liquid.shape or self.flags.shape != self.interface_inventory.shape:
            raise ABBInventoryOwnershipError("flags and inventory fields must have identical shapes")
        if bool(((self.flags != LIQUID) & (self.bulk_liquid != 0)).any()):
            raise ABBInventoryOwnershipError("bulk inventory must be owned by LIQUID cells")
        if bool(((self.flags != INTERFACE) & (self.interface_inventory != 0)).any()):
            raise ABBInventoryOwnershipError("interface inventory must be owned by INTERFACE cells")


@dataclass(frozen=True)
class ABBInventoryTransaction:
    """The explicit two-sided ledger transaction for one frozen ABB update."""

    interface_delta: float
    bulk_delta: float
    total_delta: float


def _pull_flags(flags: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        flags.roll((int(C[q, 2]), int(C[q, 1]), int(C[q, 0])), (0, 1, 2))
        for q in range(19)
    ])


def abb_reconstruction_density_change(
    f_post: torch.Tensor, f_streamed: torch.Tensor, flags: torch.Tensor, *,
    rho_gas: float, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
) -> ABBReconstructionDensityChange:
    """Map the exact ABB assignment currently in ``free_surface_step``.

    The solver computes ``feq_g[q]+feq_g[bar(q)]-f_post[bar(q)]`` and replaces
    the pull-streamed population only on GAS-facing INTERFACE links.  This
    function returns that replacement delta without changing populations.
    """
    expected = (19, *flags.shape)
    if tuple(f_post.shape) != expected or tuple(f_streamed.shape) != expected:
        raise ValueError("f_post and f_streamed must have shape (19, *flags.shape)")
    if any(field.shape != flags.shape for field in (ux, uy, uz)):
        raise ValueError("velocity fields must match flags shape")
    source_flags = _pull_flags(flags)
    link_mask = (flags == INTERFACE).unsqueeze(0) & (source_flags == GAS)
    f_eq_gas = equilibrium3d(torch.full_like(ux, float(rho_gas)), ux, uy, uz)
    f_abb = f_eq_gas + f_eq_gas[OPPOSITE.to(f_post.device)] - f_post[OPPOSITE.to(f_post.device)]
    per_link_delta = torch.where(link_mask, f_abb - f_streamed, torch.zeros_like(f_post))
    interface_delta = per_link_delta.sum(dim=0)
    return ABBReconstructionDensityChange(
        link_mask=link_mask,
        per_link_delta=per_link_delta,
        interface_delta=interface_delta,
        population_delta=float(per_link_delta.sum()),
    )


def apply_closed_abb_inventory_transaction(
    state: ABBInventoryState, change: ABBReconstructionDensityChange, *,
    bulk_owner: torch.Tensor | None = None,
) -> tuple[ABBInventoryState, ABBInventoryTransaction]:
    """Credit ABB reconstruction to INTERFACE and debit named LIQUID owners.

    ``bulk_owner`` has the D3Q19 link shape and supplies a flattened spatial
    LIQUID-cell index for every active ABB link; ``-1`` is allowed only on
    inactive links.  The API rejects an omitted, non-liquid, or incomplete
    owner rather than silently treating ABB as an inventory source.
    """
    if tuple(change.per_link_delta.shape) != (19, *state.flags.shape):
        raise ABBInventoryOwnershipError("ABB change shape must match inventory state")
    if bulk_owner is None:
        raise ABBInventoryOwnershipError("ABB reconstruction requires an explicit LIQUID bulk owner per link")
    if bulk_owner.shape != change.per_link_delta.shape:
        raise ABBInventoryOwnershipError("bulk_owner must have the D3Q19 link shape")
    active = change.link_mask
    if bool((bulk_owner[active] < 0).any()):
        raise ABBInventoryOwnershipError("every active ABB link requires an explicit LIQUID bulk owner")
    spatial_count = state.flags.numel()
    if bool((bulk_owner[active] >= spatial_count).any()):
        raise ABBInventoryOwnershipError("ABB bulk owner index is outside the inventory domain")
    owner_flags = state.flags.flatten()[bulk_owner[active].to(torch.long)]
    if bool((owner_flags != LIQUID).any()):
        raise ABBInventoryOwnershipError("ABB bulk owner must name a LIQUID cell")

    interface = state.interface_inventory + change.interface_delta
    bulk = state.bulk_liquid.clone()
    # scatter_add preserves a separate debit for every declared link, including
    # multiple reconstruction links explicitly assigned to the same bulk cell.
    debit = torch.zeros_like(bulk).flatten()
    debit.scatter_add_(0, bulk_owner[active].to(torch.long), -change.per_link_delta[active])
    bulk = bulk + debit.reshape_as(bulk)
    after = ABBInventoryState.from_flags_and_values(
        state.flags, bulk_liquid=bulk, interface_inventory=interface,
    )
    interface_delta = float(change.interface_delta.sum())
    bulk_delta = float(debit.sum())
    transaction = ABBInventoryTransaction(interface_delta, bulk_delta, interface_delta + bulk_delta)
    if abs(transaction.total_delta) > 5.0e-5 or abs(after.total - state.total) > 5.0e-5:
        raise ABBInventoryOwnershipError("closed ABB inventory transaction is not conservative")
    return after, transaction
