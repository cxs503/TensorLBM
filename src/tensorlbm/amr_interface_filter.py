"""Moment-preserving kinetic filter for coarse/fine transition shells.

Abrupt resolution changes can reflect unresolved non-hydrodynamic modes back
into a fine block.  This module damps only the kinetic residual *above* the
resolved second-order viscous stress in a thin physical shell.  Density,
momentum and the complete symmetric stress tensor remain unchanged; no
empirical body force or geometry modification is introduced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .amr_population_transfer import regularize_nonequilibrium_second_order


@dataclass(frozen=True)
class InterfaceFilterControlVolumeClearance:
    """Geometric separation between a box CV stencil and an AMR filter.

    ``minimum_physical_interface_clearance_cells`` counts cells between the
    physical fine-block boundary and the half-open control-volume box.  The
    streaming force observer also samples source cells immediately outside
    that box, so the usable guard is reduced by both the filter width and the
    streaming stencil radius.
    """

    minimum_physical_interface_clearance_cells: int
    minimum_unfiltered_source_guard_cells: int
    required_streaming_source_guard_cells: int
    flux_stencil_outside_filter: bool

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "minimum_physical_interface_clearance_cells": (
                self.minimum_physical_interface_clearance_cells
            ),
            "minimum_unfiltered_source_guard_cells": (
                self.minimum_unfiltered_source_guard_cells
            ),
            "required_streaming_source_guard_cells": (
                self.required_streaming_source_guard_cells
            ),
            "flux_stencil_outside_filter": self.flux_stencil_outside_filter,
        }


def assess_interface_filter_control_volume_clearance(
    shape: tuple[int, int, int],
    *,
    bounds_xyz: tuple[int, int, int, int, int, int],
    ghost: int,
    filter_width: int,
    streaming_stencil_radius: int = 1,
) -> InterfaceFilterControlVolumeClearance:
    """Assess whether a Cartesian CV flux stencil is outside a filter shell.

    ``shape`` is ordered ``(nz, ny, nx)`` while ``bounds_xyz`` is the
    half-open box ``(x0, x1, y0, y1, z0, z1)``.  A radius-one streaming
    stencil is required by D3Q19/D3Q27 control-volume momentum fluxes.
    """
    if ghost < 1:
        raise ValueError("ghost width must be positive")
    if filter_width < 0:
        raise ValueError("filter width must be non-negative")
    if streaming_stencil_radius < 0:
        raise ValueError("streaming stencil radius must be non-negative")
    if len(shape) != 3 or min(shape) <= 2 * ghost:
        raise ValueError("shape must contain a physical fine-block interior")

    x0, x1, y0, y1, z0, z1 = bounds_xyz
    bounds_by_storage_axis = ((z0, z1), (y0, y1), (x0, x1))
    clearances: list[int] = []
    for size, (lower, upper) in zip(
        shape, bounds_by_storage_axis, strict=True,
    ):
        if not ghost <= lower < upper <= size - ghost:
            raise ValueError(
                "control-volume bounds must lie in the physical fine block",
            )
        clearances.extend((lower - ghost, size - ghost - upper))

    minimum_clearance = min(clearances)
    unfiltered_source_guard = minimum_clearance - filter_width
    return InterfaceFilterControlVolumeClearance(
        minimum_physical_interface_clearance_cells=minimum_clearance,
        minimum_unfiltered_source_guard_cells=unfiltered_source_guard,
        required_streaming_source_guard_cells=streaming_stencil_radius,
        flux_stencil_outside_filter=(
            unfiltered_source_guard >= streaming_stencil_radius
        ),
    )


def interface_shell_blend(
    shape: tuple[int, int, int],
    *,
    ghost: int,
    width: int,
    strength: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a raised-cosine damping field on the physical interface shell."""
    if ghost < 1:
        raise ValueError("ghost width must be positive")
    if width < 0:
        raise ValueError("interface filter width must be non-negative")
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("interface filter strength must lie in [0,1]")
    if min(shape) <= 2 * ghost:
        raise ValueError("population block has no physical interior")
    blend = torch.zeros(shape, device=device, dtype=dtype)
    if width == 0 or strength == 0.0:
        return blend
    physical_shape = tuple(size - 2 * ghost for size in shape)
    if min(physical_shape) <= 2 * width:
        raise ValueError("interface filter must leave an unfiltered physical core")

    coordinates = torch.meshgrid(
        *(torch.arange(size, device=device) for size in shape),
        indexing="ij",
    )
    distances = [
        torch.minimum(coordinate - ghost, size - ghost - 1 - coordinate)
        for coordinate, size in zip(coordinates, shape, strict=True)
    ]
    distance = torch.minimum(torch.minimum(distances[0], distances[1]), distances[2])
    active = (distance >= 0) & (distance < width)
    phase = distance.to(dtype).clamp(min=0) / float(width)
    profile = 0.5 * (1.0 + torch.cos(math.pi * phase))
    blend[active] = strength * profile[active]
    return blend


def _macroscopic_and_equilibrium(f: torch.Tensor) -> torch.Tensor:
    if f.shape[0] == 19:
        from .d3q19 import equilibrium3d, macroscopic3d

        rho, ux, uy, uz = macroscopic3d(f)
        return equilibrium3d(rho, ux, uy, uz, device=f.device)
    if f.shape[0] == 27:
        from .d3q27 import equilibrium27, macroscopic27

        rho, ux, uy, uz = macroscopic27(f)
        return equilibrium27(rho, ux, uy, uz, device=f.device)
    raise ValueError("only D3Q19 and D3Q27 are supported")


def _remove_conserved_roundoff(non_equilibrium: torch.Tensor) -> torch.Tensor:
    if non_equilibrium.shape[0] == 19:
        from .d3q19 import C
    else:
        from .d3q27 import C
    c = C.to(device=non_equilibrium.device, dtype=non_equilibrium.dtype)
    result = non_equilibrium.clone()
    result[0] -= result.sum(dim=0)
    for negative, positive, axis in ((1, 2, 0), (3, 4, 1), (5, 6, 2)):
        momentum = (
            result * c[:, axis, None, None, None]
        ).sum(dim=0)
        result[negative] -= 0.5 * momentum
        result[positive] += 0.5 * momentum
    return result


def _damp_core(f: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Damp kinetic modes on one population tensor (blend broadcast over the
    trailing spatial axes, which may be a packed ``(1, 1, n)`` view)."""
    equilibrium = _macroscopic_and_equilibrium(f)
    non_equilibrium = _remove_conserved_roundoff(f - equilibrium)
    resolved_stress = regularize_nonequilibrium_second_order(non_equilibrium)
    kinetic_residual = _remove_conserved_roundoff(
        non_equilibrium - resolved_stress,
    )
    return f - blend.unsqueeze(0) * kinetic_residual


def damp_interface_nonequilibrium(
    f: torch.Tensor,
    blend: torch.Tensor,
) -> torch.Tensor:
    """Damp kinetic modes while retaining each cell's density and momentum.

    The per-cell work is identical for every spatial location, so when the
    blend is sparse (the usual interface-shell case) the active cells are
    gathered into one packed tensor, processed with the same batched tensor
    kernels, and scattered back — no Python loop over cells or directions,
    and bitwise-identical results for every filtered cell (the elementwise
    arithmetic and the ``sum(dim=0)`` reductions over ``q`` are unchanged).
    """
    if not isinstance(f, torch.Tensor) or f.ndim != 4 or f.shape[0] not in (19, 27):
        raise ValueError("f must have shape (19|27,nz,ny,nx)")
    if not f.is_floating_point():
        raise TypeError("f must be floating point")
    if blend.shape != f.shape[1:] or blend.device != f.device:
        raise ValueError("blend must match the population spatial shape and device")
    if blend.dtype != f.dtype:
        raise TypeError("blend and populations must have the same dtype")
    if bool(((blend < 0.0) | (blend > 1.0)).any()):
        raise ValueError("blend values must lie in [0,1]")
    if not bool(blend.any()):
        return f

    mask = blend > 0
    if mask.sum() * 4 < mask.numel():
        # Sparse interface shell: process only the active cells as one
        # packed batch.  The reductions are still along ``q`` (dim 0), so
        # every active cell sees exactly the old arithmetic.
        indices = torch.nonzero(mask, as_tuple=False)  # (n_active, 3)
        zi, yi, xi = indices[:, 0], indices[:, 1], indices[:, 2]
        packed = f[:, zi, yi, xi].reshape(f.shape[0], 1, 1, -1)
        packed_blend = blend[zi, yi, xi].reshape(1, 1, -1)
        filtered = _damp_core(packed, packed_blend)
        result = f.clone()
        result[:, zi, yi, xi] = filtered.reshape(f.shape[0], -1)
        return result
    return _damp_core(f, blend)


__all__ = [
    "InterfaceFilterControlVolumeClearance",
    "assess_interface_filter_control_volume_clearance",
    "damp_interface_nonequilibrium",
    "interface_shell_blend",
]
