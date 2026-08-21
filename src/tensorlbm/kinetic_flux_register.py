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


def _lattice_weights(q: int, device: torch.device) -> torch.Tensor:
    if q == 19:
        from .d3q19 import W
    elif q == 27:
        from .d3q27 import W
    else:
        raise ValueError("only D3Q19 and D3Q27 are supported")
    return W.to(device=device)


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
            self.outgoing * factor,
            self.incoming * factor,
        )


@dataclass(frozen=True)
class FaceLocalRefluxReport:
    raw_kinetic_mismatch: torch.Tensor
    requested_inventory_correction: torch.Tensor
    applied_inventory_correction: torch.Tensor
    residual: torch.Tensor
    corrected_links: int
    limited_directions: int
    maximum_applied_correction_fraction: float

    @property
    def mass_residual(self) -> float:
        return float(self.residual.sum().item())


def conserved_population_moments(
    population_inventory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mass and momentum represented by a Q-population inventory."""
    if (
        not isinstance(population_inventory, torch.Tensor)
        or population_inventory.ndim != 1
        or population_inventory.numel() not in (19, 27)
    ):
        raise ValueError("population_inventory must be a D3Q19 or D3Q27 vector")
    if not population_inventory.is_floating_point():
        raise TypeError("population_inventory must be floating point")
    c = _lattice_velocities(
        int(population_inventory.numel()),
        population_inventory.device,
    ).to(dtype=population_inventory.dtype)
    return (
        population_inventory.sum(),
        (population_inventory[:, None] * c).sum(dim=0),
    )


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
            inside,
            shifts=(-cz, -cy, -cx),
            dims=(0, 1, 2),
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
        outgoing * cell_volume,
        incoming * cell_volume,
    )


def _apply_population_total(
    populations: torch.Tensor,
    mask: torch.Tensor,
    requested: torch.Tensor,
    *,
    maximum_correction_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int, float]:
    """Apply a bounded requested total per direction on interface links."""
    applied = torch.zeros_like(requested)
    limited = 0
    corrected_links = 0
    maximum_applied_fraction = 0.0
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
            -maximum_correction_fraction,
            maximum_correction_fraction,
        )
        if bool(limited_factor != factor):
            limited += 1
        maximum_applied_fraction = max(
            maximum_applied_fraction,
            abs(float(limited_factor.item())),
        )
        delta = values * limited_factor
        populations[direction, selected] = values + delta
        applied[direction] = delta.sum()
    return (
        populations,
        applied,
        corrected_links,
        limited,
        maximum_applied_fraction,
    )


def project_onto_conserved_moments(
    kinetic_mismatch: torch.Tensor,
) -> torch.Tensor:
    """Discard non-conserved kinetic modes while preserving mass/momentum.

    Isothermal LBM collision conserves only density and the three momentum
    components, not each directional population.  Coarse/fine reflux must
    therefore match those four moments and must not inject the remaining
    kinetic-mode mismatch into the exterior flow.
    """
    if kinetic_mismatch.ndim != 1 or kinetic_mismatch.numel() not in (19, 27):
        raise ValueError("kinetic_mismatch must be a D3Q19 or D3Q27 vector")
    c = _lattice_velocities(
        int(kinetic_mismatch.numel()),
        kinetic_mismatch.device,
    ).to(dtype=kinetic_mismatch.dtype)
    w = _lattice_weights(
        int(kinetic_mismatch.numel()),
        kinetic_mismatch.device,
    ).to(dtype=kinetic_mismatch.dtype)
    mass, momentum = conserved_population_moments(kinetic_mismatch)
    projected = w * (mass + 3.0 * (c * momentum).sum(dim=1))
    # Lattice constants are commonly stored in FP32 even for an FP64 state.
    # Close the four conserved moments algebraically so their reflux accuracy
    # is set by the state dtype, not by rounded tabulated weights.
    projected[0] += mass - projected.sum()
    momentum_residual = momentum - (projected[:, None] * c).sum(dim=0)
    for axis, (positive_index, negative_index) in enumerate(
        ((1, 2), (3, 4), (5, 6)),
    ):
        half = 0.5 * momentum_residual[axis]
        projected[positive_index] += half
        projected[negative_index] -= half
    return projected


def project_onto_active_conserved_moments(
    kinetic_mismatch: torch.Tensor,
    active_directions: torch.Tensor,
) -> torch.Tensor:
    """Represent mass/momentum using only directions with crossing links.

    A stream-register correction must not modify the rest population or a
    direction that did not cross the coarse/fine boundary.  The weighted
    minimum-norm projection below solves the four conserved constraints on
    the active velocity subset.  It is the active-link analogue of
    :func:`project_onto_conserved_moments`.
    """
    if kinetic_mismatch.ndim != 1 or kinetic_mismatch.numel() not in (19, 27):
        raise ValueError("kinetic_mismatch must be a D3Q19 or D3Q27 vector")
    if (
        not isinstance(active_directions, torch.Tensor)
        or active_directions.shape != kinetic_mismatch.shape
        or active_directions.dtype is not torch.bool
    ):
        raise ValueError("active_directions must be a boolean Q-vector")
    if active_directions.device != kinetic_mismatch.device:
        raise ValueError("active_directions and kinetic_mismatch must share device")
    c = _lattice_velocities(
        int(kinetic_mismatch.numel()),
        kinetic_mismatch.device,
    ).to(dtype=kinetic_mismatch.dtype)
    w = _lattice_weights(
        int(kinetic_mismatch.numel()),
        kinetic_mismatch.device,
    ).to(dtype=kinetic_mismatch.dtype)
    active_weights = torch.where(active_directions, w, torch.zeros_like(w))
    if not bool(active_directions[1:7].all()):
        raise ValueError("active crossing directions omit an axial pair")
    basis = torch.cat(
        (
            torch.ones((1, c.shape[0]), device=c.device, dtype=c.dtype),
            c.T,
        ),
        dim=0,
    )
    gram = (basis * active_weights.unsqueeze(0)) @ basis.T
    mass, momentum = conserved_population_moments(kinetic_mismatch)
    target = torch.cat((mass.reshape(1), momentum))
    coefficients = torch.linalg.solve(gram, target)
    projected = active_weights * (basis.T @ coefficients)
    # Close residual roundoff on active opposite-axis pairs.  A valid closed
    # 3-D interface necessarily has all six axial directions available.
    projected[1] += target[0] - projected.sum()
    momentum_residual = target[1:] - (projected[:, None] * c).sum(dim=0)
    for axis, (positive_index, negative_index) in enumerate(
        ((1, 2), (3, 4), (5, 6)),
    ):
        half = 0.5 * momentum_residual[axis]
        projected[positive_index] += half
        projected[negative_index] -= half
    return projected


def apply_face_local_reflux(
    coarse_populations: torch.Tensor,
    coarse_links: KineticInterfaceLinks,
    coarse_transfer: KineticInterfaceTransfer,
    fine_transfer: KineticInterfaceTransfer,
    *,
    maximum_correction_fraction: float = 0.2,
    correction_stencil: str = "exterior_cells",
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
    if correction_stencil not in ("exterior_cells", "crossing_links"):
        raise ValueError(
            "correction_stencil must be exterior_cells or crossing_links",
        )
    if coarse_populations.shape != coarse_links.outgoing_origins.shape:
        raise ValueError("coarse populations and links must share shape")
    # Fine-minus-coarse net transport is first reduced to the four moments
    # conserved by isothermal collision.  Matching all Q populations would
    # incorrectly reflux non-conserved stress/higher-order kinetic modes and
    # can generate O(1) corrections in strong-gradient flows.
    raw_mismatch = fine_transfer.net_outgoing - coarse_transfer.net_outgoing
    c = _lattice_velocities(coarse_links.q, coarse_populations.device)
    receiving = torch.zeros_like(coarse_links.outgoing_origins)
    for direction in range(1, coarse_links.q):
        cx, cy, cz = (int(value) for value in c[direction].tolist())
        receiving[direction] = torch.roll(
            coarse_links.outgoing_origins[direction],
            shifts=(cz, cy, cx),
            dims=(0, 1, 2),
        )
    exterior_links = receiving | coarse_links.incoming_origins
    if correction_stencil == "crossing_links":
        active_directions = exterior_links.reshape(
            coarse_links.q,
            -1,
        ).any(dim=1)
        requested = project_onto_active_conserved_moments(
            raw_mismatch,
            active_directions,
        )
        correction_mask = exterior_links
    else:
        requested = project_onto_conserved_moments(raw_mismatch)
        exterior_cells = exterior_links.any(dim=0)
        correction_mask = exterior_cells.unsqueeze(0).expand_as(exterior_links)
    (
        corrected,
        applied,
        corrected_links,
        limited,
        maximum_applied_fraction,
    ) = _apply_population_total(
        coarse_populations,
        correction_mask,
        requested,
        maximum_correction_fraction=maximum_correction_fraction,
    )
    residual = requested - applied
    return corrected, FaceLocalRefluxReport(
        raw_mismatch,
        requested,
        applied,
        residual,
        corrected_links,
        limited,
        maximum_applied_fraction,
    )


__all__ = [
    "FaceLocalRefluxReport",
    "KineticInterfaceLinks",
    "KineticInterfaceTransfer",
    "apply_face_local_reflux",
    "build_kinetic_interface_links",
    "conserved_population_moments",
    "observe_kinetic_interface_transfer",
    "project_onto_active_conserved_moments",
    "project_onto_conserved_moments",
]
