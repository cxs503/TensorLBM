"""Fail-closed D3Q19 CH adapter streaming and phase-boundary contract.

This module is deliberately separate from ``multiphase3d.free_energy_step_3d``:
that production routine is collision-only and uses periodic differential
operators internally.  The stream here is an explicit adapter operation, not a
claim that the current production CH path is a complete physical solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..d3q19 import C, OPPOSITE

D3Q19_POPULATIONS = 19
ADAPTER_STREAM_STAGE = "collision_then_adapter_stream"
PHASE_FLUX_WITHHELD = "withheld"
BoundaryPolicy = Literal["periodic", "no_flux"]


@dataclass(frozen=True)
class PhaseBoundaryContract:
    """Declared adapter boundary semantics, intentionally not a wetting model.

    ``periodic`` wraps every D3Q19 population.  ``no_flux`` uses link-wise
    reflection at each exterior crossing, so no population is wrapped across a
    domain face.  Neither policy defines a continuum phase-flux observable;
    consequently ``phase_flux`` is ``None`` and its status is ``withheld``.
    """

    boundary: BoundaryPolicy
    stage: str = ADAPTER_STREAM_STAGE
    physical: bool = False
    phase_flux_status: str = PHASE_FLUX_WITHHELD
    phase_flux: None = None
    wetting: bool = False

    def __post_init__(self) -> None:
        if self.boundary not in ("periodic", "no_flux"):
            raise ValueError("boundary must be either 'periodic' or 'no_flux'")


@dataclass(frozen=True)
class FreeEnergyAdapterStreamResult:
    """Adapter-streamed coupled CH populations plus fail-closed metadata."""

    f: torch.Tensor
    g: torch.Tensor
    contract: PhaseBoundaryContract

    @property
    def stage(self) -> str:
        return self.contract.stage

    @property
    def physical(self) -> bool:
        return self.contract.physical

    @property
    def phase_flux_status(self) -> str:
        return self.contract.phase_flux_status


def _validate_distribution(distribution: torch.Tensor) -> None:
    if not isinstance(distribution, torch.Tensor):
        raise TypeError("distribution must be a torch.Tensor")
    if distribution.ndim != 4 or distribution.shape[0] != D3Q19_POPULATIONS:
        raise ValueError("distribution must have shape (19, nz, ny, nx)")
    if any(size <= 0 for size in distribution.shape[1:]):
        raise ValueError("distribution spatial dimensions must be positive")
    if not distribution.is_floating_point():
        raise TypeError("distribution must have a floating-point dtype")


def _periodic_stream(distribution: torch.Tensor) -> torch.Tensor:
    """Pull-stream using authoritative ``C=(cx, cy, cz)`` on ``(z, y, x)``."""
    output = torch.empty_like(distribution)
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        output[q] = torch.roll(distribution[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return output


def _no_flux_stream(distribution: torch.Tensor) -> torch.Tensor:
    """Push-stream with exterior crossings reflected into their opposite link.

    The assignment is formulated without ``roll``: a population at source
    ``x`` moves to ``x + c_q`` when that destination is in bounds; otherwise it
    remains at ``x`` but changes to the D3Q19 opposite direction.  This makes
    wraparound impossible and preserves the distribution inventory exactly.
    """
    output = torch.zeros_like(distribution)
    nz, ny, nx = distribution.shape[1:]
    z, y, x = torch.meshgrid(
        torch.arange(nz, device=distribution.device),
        torch.arange(ny, device=distribution.device),
        torch.arange(nx, device=distribution.device),
        indexing="ij",
    )
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        destination_z, destination_y, destination_x = z + cz, y + cy, x + cx
        inside = (
            (destination_z >= 0)
            & (destination_z < nz)
            & (destination_y >= 0)
            & (destination_y < ny)
            & (destination_x >= 0)
            & (destination_x < nx)
        )
        output[q].index_put_(
            (destination_z[inside], destination_y[inside], destination_x[inside]),
            distribution[q][inside],
        )
        output[int(OPPOSITE[q].item())][~inside] = distribution[q][~inside]
    return output


def stream_d3q19_adapter(
    distribution: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
) -> torch.Tensor:
    """Stream a D3Q19 distribution under explicit adapter-only boundary rules.

    This is not wired into production CH collision and does not calculate phase
    flux or wetting physics.  Callers must retain the accompanying
    :class:`PhaseBoundaryContract` if they need to report those withheld terms.
    """
    _validate_distribution(distribution)
    if not isinstance(boundary, str):
        raise TypeError("boundary must be either 'periodic' or 'no_flux'")
    if boundary == "periodic":
        return _periodic_stream(distribution)
    if boundary == "no_flux":
        return _no_flux_stream(distribution)
    raise ValueError("boundary must be either 'periodic' or 'no_flux'")


def stream_free_energy_adapter(
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
) -> FreeEnergyAdapterStreamResult:
    """Adapter-stream a coupled post-collision CH ``(f, g)`` state.

    This pure operation intentionally accepts post-collision populations rather
    than invoking production collision itself.  It therefore cannot change the
    collision owner or imply that the combined sequence is physically complete.
    """
    _validate_distribution(f)
    _validate_distribution(g)
    if f.shape != g.shape:
        raise ValueError("f and g must have the same shape")
    if f.device != g.device:
        raise ValueError("f and g must be on the same device")
    if f.dtype != g.dtype:
        raise TypeError("f and g must have the same dtype")
    contract = PhaseBoundaryContract(boundary=boundary)
    return FreeEnergyAdapterStreamResult(
        f=stream_d3q19_adapter(f, boundary=boundary),
        g=stream_d3q19_adapter(g, boundary=boundary),
        contract=contract,
    )


__all__ = [
    "ADAPTER_STREAM_STAGE",
    "BoundaryPolicy",
    "D3Q19_POPULATIONS",
    "FreeEnergyAdapterStreamResult",
    "PHASE_FLUX_WITHHELD",
    "PhaseBoundaryContract",
    "stream_d3q19_adapter",
    "stream_free_energy_adapter",
]
