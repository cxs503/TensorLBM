"""Face-local conservative coarse/fine flux accounting for lattice Boltzmann AMR.

The register observes populations *after collision and before streaming* on
links that cross a refinement boundary.  Each discrete link is counted once,
including diagonal links leaving through an edge or corner.  Fine transfers
are integrated over substeps and scaled by the fine-cell physical volume.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _lattice_velocities(q: int, device: torch.device) -> torch.Tensor:
    if q == 19:
        from .d3q19 import C
    elif q == 27:
        from .d3q27 import C
    else:
        raise ValueError("only D3Q19 and D3Q27 are supported")
    return C.to(device=device)


@dataclass(frozen=True)
class KineticInterfaceLinks:
    """Origin-cell masks for every population crossing one closed interface."""

    inside: torch.Tensor
    outgoing_origins: torch.Tensor
    incoming_origins: torch.Tensor

    @property
    def q(self) -> int:
        return int(self.outgoing_origins.shape[0])


@dataclass(frozen=True)
class KineticInterfaceTransfer:
    """Population inventory transported across a closed interface."""

    outgoing: torch.Tensor
    incoming: torch.Tensor

    @property
    def net_outgoing(self) -> torch.Tensor:
        return self.outgoing - self.incoming

    def __add__(self, other: object) -> "KineticInterfaceTransfer":
        if not isinstance(other, KineticInterfaceTransfer):
            return NotImplemented
        return KineticInterfaceTransfer(
            self.outgoing + other.outgoing,
            self.incoming + other.incoming,
        )

    def scaled(self, factor: float) -> "KineticInterfaceTransfer":
        return KineticInterfaceTransfer(
            self.outgoing * factor, self.incoming * factor,
        )


@dataclass(frozen=True)
class FaceLocalRefluxReport:
    requested_inventory_correction: torch.Tensor
    applied_inventory_correction: torch.Tensor
    residual: torch.Tensor
    corrected_links: int
    limited_directions: int

    @property
    def mass_residual(self) -> float:
        return float(self.residual.sum().item())


def build_kinetic_interface_links(
    inside: torch.Tensor,
    *,
    q: int,
) -> KineticInterfaceLinks:
    """Build non-periodic link masks for a strictly interior owned volume."""
    if inside.ndim != 3 or inside.dtype is not torch.bool:
        raise ValueError("inside must be a 3-D boolean tensor")
    if not bool(inside.any()) or bool(inside[0].any()) or bool(inside[-1].any()):
        raise ValueError("owned volume must be non-empty and strictly interior")
    if bool(inside[:, 0].any()) or bool(inside[:, -1].any()):
        raise ValueError("owned volume must be strictly interior")
    if bool(inside[:, :, 0].any()) or bool(inside[:, :, -1].any()):
        raise ValueError("owned volume must be strictly interior")
    c = _lattice_velocities(q, inside.device)
    outgoing = torch.zeros((q, *inside.shape), dtype=torch.bool, device=inside.device)
    incoming = torch.zeros_like(outgoing)
    for direction in range(1, q):
        cx, cy, cz = (int(value) for value in c[direction].tolist())
        destination_inside = torch.roll(
            inside, shifts=(-cz, -cy, -cx), dims=(0, 1, 2),
        )
        outgoing[direction] = inside & ~destination_inside
        incoming[direction] = ~inside & destination_inside
    return KineticInterfaceLinks(inside, outgoing, incoming)


def observe_kinetic_interface_transfer(
    post_collision: torch.Tensor,
    links: KineticInterfaceLinks,
    *,
    cell_volume: float = 1.0,
) -> KineticInterfaceTransfer:
    """Integrate pre-stream population transfer across the interface."""
    if post_collision.shape != links.outgoing_origins.shape:
        raise ValueError("post_collision and interface links must share shape")
    if cell_volume <= 0.0:
        raise ValueError("cell_volume must be positive")
    outgoing = (post_collision * links.outgoing_origins).sum(dim=(1, 2, 3))
    incoming = (post_collision * links.incoming_origins).sum(dim=(1, 2, 3))
    return KineticInterfaceTransfer(
        outgoing * cell_volume, incoming * cell_volume,
    )


def _apply_population_total(
    populations: torch.Tensor,
    mask: torch.Tensor,
    requested: torch.Tensor,
    *,
    maximum_removal_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Apply one requested total per direction, proportionally on link cells."""
    applied = torch.zeros_like(requested)
    limited = 0
    corrected_links = 0
    for direction in range(populations.shape[0]):
        selected = mask[direction]
        count = int(selected.sum().item())
        if count == 0:
            continue
        corrected_links += count
        values = populations[direction, selected]
        inventory = values.sum()
        desired = requested[direction]
        factor = desired / inventory.clamp_min(1e-30)
        limited_factor = factor.clamp_min(-maximum_removal_fraction)
        if bool(limited_factor != factor):
            limited += 1
        delta = values * limited_factor
        populations[direction, selected] = values + delta
        applied[direction] = delta.sum()
    return populations, applied, corrected_links, limited


def apply_face_local_reflux(
    coarse_populations: torch.Tensor,
    coarse_links: KineticInterfaceLinks,
    coarse_transfer: KineticInterfaceTransfer,
    fine_transfer: KineticInterfaceTransfer,
    *,
    maximum_removal_fraction: float = 0.2,
) -> tuple[torch.Tensor, FaceLocalRefluxReport]:
    """Correct only exterior coarse cells receiving/supplying interface links."""
    if not 0.0 < maximum_removal_fraction <= 1.0:
        raise ValueError("maximum_removal_fraction must lie in (0,1]")
    if coarse_populations.shape != coarse_links.outgoing_origins.shape:
        raise ValueError("coarse populations and links must share shape")
    # Fine-minus-coarse net outward transport is exactly the inventory that
    # the coarse exterior must gain after the fine-owned patch replaces its
    # coarse representation.
    delta_out = fine_transfer.outgoing - coarse_transfer.outgoing
    delta_in = fine_transfer.incoming - coarse_transfer.incoming
    c = _lattice_velocities(coarse_links.q, coarse_populations.device)
    receiving = torch.zeros_like(coarse_links.outgoing_origins)
    for direction in range(1, coarse_links.q):
        cx, cy, cz = (int(value) for value in c[direction].tolist())
        receiving[direction] = torch.roll(
            coarse_links.outgoing_origins[direction],
            shifts=(cz, cy, cx), dims=(0, 1, 2),
        )
    corrected, applied_out, count_out, limited_out = _apply_population_total(
        coarse_populations, receiving, delta_out,
        maximum_removal_fraction=maximum_removal_fraction,
    )
    corrected, applied_in, count_in, limited_in = _apply_population_total(
        corrected, coarse_links.incoming_origins, -delta_in,
        maximum_removal_fraction=maximum_removal_fraction,
    )
    requested = delta_out - delta_in
    applied = applied_out + applied_in
    residual = requested - applied
    return corrected, FaceLocalRefluxReport(
        requested, applied, residual, count_out + count_in,
        limited_out + limited_in,
    )


__all__ = [
    "FaceLocalRefluxReport",
    "KineticInterfaceLinks",
    "KineticInterfaceTransfer",
    "apply_face_local_reflux",
    "build_kinetic_interface_links",
    "observe_kinetic_interface_transfer",
]
