"""Pure temporal 2:1 reflux bookkeeping for a D3Q27 nested interface.

This module records *integrated*, oriented D3Q27 interface-flux packets only;
it does not discover patches, reconstruct faces, stream/collide populations, or
mutate a solver.  A packet's 27 entries must already include its appropriate
face-area and time-step factors, and all packets must use the same interface
orientation.  Thus one coarse-step packet and the sum of its two fine-substep
packets have common units.

Ownership is deliberately explicit.  ``coarse_correction`` belongs to the
coarse interface-flux ledger, while each member of ``fine_corrections`` belongs
to the corresponding fine-substep ledger.  The helper returns values rather
than applying them.  With ``mismatch = coarse - (fine_0 + fine_1)``, it assigns
``-mismatch / 2`` to coarse and splits ``+mismatch / 2`` equally across the two
fine owners. Consequently either corrected ledger agrees with the other and the
sum of all owner corrections is zero direction-by-direction (hence for mass
and all D3Q27 momentum moments).  Equal splitting is temporal bookkeeping,
not a claim about an intra-step physical flux profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .d3q27 import C

_Q = 27
_SUBSTEPS = frozenset((0, 1))


@dataclass(frozen=True)
class D3Q27InterfaceFluxPacket:
    """One source-owned, oriented, integrated D3Q27 interface-flux packet.

    ``substep`` is ``None`` exclusively for the coarse full-step packet, and
    must be 0 or 1 for a fine packet. Validation of the flux tensor occurs at
    reflux time so packet creation remains a simple immutable record.
    """

    flux: torch.Tensor
    substep: int | None


@dataclass(frozen=True)
class D3Q27TemporalRefluxResult:
    """Unapplied 2:1 reflux data, with corrections owned by their input ledgers."""

    mismatch: torch.Tensor
    coarse_correction: torch.Tensor
    fine_corrections: tuple[torch.Tensor, torch.Tensor]
    corrected_coarse_flux: torch.Tensor
    corrected_fine_fluxes: tuple[torch.Tensor, torch.Tensor]
    mismatch_mass: torch.Tensor
    mismatch_momentum: torch.Tensor


def _validate_flux(packet: D3Q27InterfaceFluxPacket, *, name: str) -> None:
    if not isinstance(packet, D3Q27InterfaceFluxPacket):
        raise TypeError(f"{name} must be a D3Q27InterfaceFluxPacket")
    if not isinstance(packet.flux, torch.Tensor):
        raise TypeError(f"{name}.flux must be a torch.Tensor")
    if packet.flux.shape != (_Q,):
        raise ValueError(f"{name}.flux must have shape (27,)")
    if not packet.flux.is_floating_point():
        raise TypeError(f"{name}.flux must have a floating-point dtype")
    if not bool(torch.isfinite(packet.flux).all().item()):
        raise ValueError(f"{name}.flux must be finite")


def _validate_fine_packets(fine_packets: Sequence[D3Q27InterfaceFluxPacket]) -> tuple[D3Q27InterfaceFluxPacket, D3Q27InterfaceFluxPacket]:
    if not isinstance(fine_packets, Sequence) or len(fine_packets) != 2:
        raise ValueError("fine_packets must contain exactly two fine substep packets")
    first, second = fine_packets
    _validate_flux(first, name="fine_packets[0]")
    _validate_flux(second, name="fine_packets[1]")
    substeps = (first.substep, second.substep)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in substeps) or set(substeps) != _SUBSTEPS:
        raise ValueError("fine_packets must provide exactly substeps 0 and 1")
    ordered = {first.substep: first, second.substep: second}
    return ordered[0], ordered[1]


def reflux_d3q27_2to1(
    coarse_packet: D3Q27InterfaceFluxPacket,
    fine_packets: Sequence[D3Q27InterfaceFluxPacket],
) -> D3Q27TemporalRefluxResult:
    """Compare one coarse packet with two fine packets and return conservative corrections.

    No supplied tensor is modified. ``coarse_packet`` must be the sole full
    coarse-step packet (``substep=None``); ``fine_packets`` must contain exactly
    one packet for each ordered fine substep 0 and 1. All flux tensors must be
    finite, shape ``(27,)``, and share dtype/device. The returned fine tuple is
    ordered by substep regardless of input order.
    """
    _validate_flux(coarse_packet, name="coarse_packet")
    if coarse_packet.substep is not None:
        raise ValueError("coarse_packet.substep must be None for the full coarse step")
    fine_0, fine_1 = _validate_fine_packets(fine_packets)
    if any(packet.flux.dtype != coarse_packet.flux.dtype or packet.flux.device != coarse_packet.flux.device for packet in (fine_0, fine_1)):
        raise ValueError("coarse and fine packet fluxes must share dtype and device")

    fine_total = fine_0.flux + fine_1.flux
    mismatch = coarse_packet.flux - fine_total
    # Symmetric ownership places both corrected ledgers at their common
    # midpoint. Fine corrections therefore sum to -coarse_correction.
    coarse_correction = -mismatch / 2.0
    fine_correction = mismatch / 4.0
    corrected_coarse = coarse_packet.flux + coarse_correction
    corrected_fine = (fine_0.flux + fine_correction, fine_1.flux + fine_correction)
    directions = C.to(device=mismatch.device, dtype=mismatch.dtype)
    mismatch_mass = mismatch.sum()
    mismatch_momentum = (mismatch[:, None] * directions).sum(dim=0)
    return D3Q27TemporalRefluxResult(
        mismatch=mismatch,
        coarse_correction=coarse_correction,
        fine_corrections=(fine_correction, fine_correction),
        corrected_coarse_flux=corrected_coarse,
        corrected_fine_fluxes=corrected_fine,
        mismatch_mass=mismatch_mass,
        mismatch_momentum=mismatch_momentum,
    )


__all__ = [
    "D3Q27InterfaceFluxPacket",
    "D3Q27TemporalRefluxResult",
    "reflux_d3q27_2to1",
]
