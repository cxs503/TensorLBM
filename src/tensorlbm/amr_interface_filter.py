"""Moment-preserving kinetic filter for coarse/fine transition shells.

Abrupt resolution changes can reflect unresolved non-hydrodynamic modes back
into a fine block.  This module damps only the kinetic residual *above* the
resolved second-order viscous stress in a thin physical shell.  Density,
momentum and the complete symmetric stress tensor remain unchanged; no
empirical body force or geometry modification is introduced.
"""
from __future__ import annotations

import math

import torch

from .amr_population_transfer import regularize_nonequilibrium_second_order


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


def damp_interface_nonequilibrium(
    f: torch.Tensor,
    blend: torch.Tensor,
) -> torch.Tensor:
    """Damp kinetic modes while retaining each cell's density and momentum."""
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

    equilibrium = _macroscopic_and_equilibrium(f)
    non_equilibrium = _remove_conserved_roundoff(f - equilibrium)
    resolved_stress = regularize_nonequilibrium_second_order(non_equilibrium)
    kinetic_residual = _remove_conserved_roundoff(
        non_equilibrium - resolved_stress,
    )
    return f - blend.unsqueeze(0) * kinetic_residual


__all__ = ["damp_interface_nonequilibrium", "interface_shell_blend"]
