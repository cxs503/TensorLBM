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

    def __add__(self, other: object) -> KineticInterfaceTransfer:
        if not isinstance(other, KineticInterfaceTransfer):
            return NotImplemented
        return KineticInterfaceTransfer(
            self.outgoing + other.outgoing,
            self.incoming + other.incoming,
        )

    def scaled(self, factor: float) -> KineticInterfaceTransfer:
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
    maximum_correction_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Apply a bounded requested total per direction on interface links."""
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
        limited_factor = factor.clamp(
            -maximum_correction_fraction, maximum_correction_fraction,
        )
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
    maximum_correction_fraction: float = 0.2,
) -> tuple[torch.Tensor, FaceLocalRefluxReport]:
    """Correct exterior links without unbounded removal or injection.

    A large positive fine/coarse mismatch is as destabilising as excessive
    removal: it can multiply one directional population in a single step.
    Both signs are therefore capped relative to the selected link inventory;
    any unapplied amount remains explicit in ``report.residual`` and fails the
    production conservation gate.
    """
    if not 0.0 < maximum_correction_fraction <= 1.0:
        raise ValueError("maximum_correction_fraction must lie in (0,1]")
    if coarse_populations.shape != coarse_links.outgoing_origins.shape:
        raise ValueError("coarse populations and links must share shape")
    # Fine-minus-coarse *net* outward transport is exactly the inventory that
    # the coarse exterior must gain after the fine-owned patch replaces its
    # coarse representation.  Outgoing and incoming quadratures must be
    # combined before correction: at edges/corners their fine/coarse link
    # counts differ even for a uniform equilibrium, while their net transfer
    # is exactly zero (free-stream preservation).
    requested = fine_transfer.net_outgoing - coarse_transfer.net_outgoing
    c = _lattice_velocities(coarse_links.q, coarse_populations.device)
    receiving = torch.zeros_like(coarse_links.outgoing_origins)
    for direction in range(1, coarse_links.q):
        cx, cy, cz = (int(value) for value in c[direction].tolist())
        receiving[direction] = torch.roll(
            coarse_links.outgoing_origins[direction],
            shifts=(cz, cy, cx), dims=(0, 1, 2),
        )
    exterior_links = receiving | coarse_links.incoming_origins
    corrected, applied, corrected_links, limited = _apply_population_total(
        coarse_populations, exterior_links, requested,
        maximum_correction_fraction=maximum_correction_fraction,
    )
    residual = requested - applied
    return corrected, FaceLocalRefluxReport(
        requested, applied, residual, corrected_links, limited,
    )


__all__ = [
    "FaceLocalRefluxReport",
    "KineticInterfaceLinks",
    "KineticInterfaceTransfer",
    "apply_face_local_reflux",
    "build_kinetic_interface_links",
    "observe_kinetic_interface_transfer",
]
