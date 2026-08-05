"""Pure, reusable ship-hydrodynamics reference utilities.

The functions in this module have no solver, geometry-builder, or benchmark
runtime dependencies.  They can therefore be used by hull cases independently
of the SUBOFF benchmark implementation.
"""
from __future__ import annotations

import math

import torch


def ittc57_friction_coefficient(reynolds: float) -> float:
    """Return the ITTC-1957 turbulent friction coefficient for ``reynolds``.

    The correlation is valid only above the low-Reynolds-number cutoff used by
    the existing benchmark implementation.
    """
    if reynolds <= 100.0:
        raise ValueError("Reynolds number too low for ITTC-1957 formula")
    return 0.075 / (math.log10(reynolds) - 2.0) ** 2


def voxel_wetted_area(mask: torch.Tensor, dx: float) -> float:
    """Return exposed voxel-face area of a 3-D solid ``mask``.

    Each solid/fluid transition and physical-domain boundary face contributes
    one face of area ``dx**2``.  This preserves the established SUBOFF voxel
    area convention while remaining independent of SUBOFF-specific code.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D tensor")

    m = mask
    area_faces = torch.tensor(0, dtype=torch.int64, device=m.device)
    area_faces += m[:, :, 0].sum()
    area_faces += m[:, :, -1].sum()
    area_faces += m[:, 0, :].sum()
    area_faces += m[:, -1, :].sum()
    area_faces += m[0, :, :].sum()
    area_faces += m[-1, :, :].sum()
    area_faces += (m[:, :, 1:] != m[:, :, :-1]).sum()
    area_faces += (m[:, 1:, :] != m[:, :-1, :]).sum()
    area_faces += (m[1:, :, :] != m[:-1, :, :]).sum()
    return float(area_faces.item()) * dx * dx
