"""Orientation-aware local surface-area weights for BFL boundary nodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SurfaceAreaWeightDiagnostics:
    boundary_nodes: int
    unweighted_nodes: int
    raw_area: float
    calibrated_area: float
    calibration_factor: float


def bfl_surface_area_weights(
    fluid_boundary_mask: torch.Tensor,
    normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    reference_area: float | None = None,
    calibration_factor: float | None = None,
    boundary_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SurfaceAreaWeightDiagnostics]:
    """Estimate a local patch area from axial lattice-face projections.

    For a surface patch, the sum of its projections on the three coordinate
    planes is ``(|nx|+|ny|+|nz|) dA``.  The number of crossed axial BFL faces
    approximates that projected sum, hence the local estimator
    ``dA = N_axial / ||n||_1``.  An optional analytical total area calibrates
    only the global scale and retains the orientation-dependent distribution.
    """
    if fluid_boundary_mask.ndim != 4 or fluid_boundary_mask.shape[0] not in (19, 27):
        raise ValueError("fluid_boundary_mask must have shape (19|27,nz,ny,nx)")
    shape = fluid_boundary_mask.shape[1:]
    if any(component.shape != shape for component in normals):
        raise ValueError("normal fields must share the spatial grid shape")
    if reference_area is not None and reference_area <= 0.0:
        raise ValueError("reference_area must be positive")
    if calibration_factor is not None and calibration_factor <= 0.0:
        raise ValueError("calibration_factor must be positive")
    if reference_area is not None and calibration_factor is not None:
        raise ValueError("provide reference_area or calibration_factor, not both")
    if boundary_mask is not None and (
        boundary_mask.shape != shape or boundary_mask.dtype is not torch.bool
    ):
        raise ValueError("boundary_mask must be bool with the spatial grid shape")
    if fluid_boundary_mask.shape[0] == 19:
        from .d3q19 import C
    else:
        from .d3q27 import C
    dtype = normals[0].dtype
    device = fluid_boundary_mask.device
    c = C.to(device=device)
    axial = c.abs().sum(dim=1) == 1
    axial[0] = False
    axial_count = fluid_boundary_mask[axial].sum(dim=0).to(dtype=dtype)
    nx_n, ny_n, nz_n = (component.to(device=device, dtype=dtype) for component in normals)
    normal_l1 = nx_n.abs() + ny_n.abs() + nz_n.abs()
    boundary = fluid_boundary_mask.any(dim=0)
    if boundary_mask is not None:
        boundary = boundary & boundary_mask.to(device=device)
    valid = boundary & (axial_count > 0.0) & (normal_l1 > 1e-12)
    raw = torch.where(
        valid,
        axial_count / normal_l1.clamp_min(1e-12),
        torch.zeros_like(normal_l1),
    )
    raw_area = float(raw.sum().item())
    if reference_area is not None:
        if raw_area <= 0.0:
            raise ValueError("cannot calibrate an empty BFL surface")
        scale = reference_area / raw_area
    elif calibration_factor is not None:
        scale = calibration_factor
    else:
        scale = 1.0
    weights = raw * scale
    diagnostics = SurfaceAreaWeightDiagnostics(
        boundary_nodes=int(boundary.sum().item()),
        unweighted_nodes=int((boundary & ~valid).sum().item()),
        raw_area=raw_area,
        calibrated_area=float(weights.sum().item()),
        calibration_factor=float(scale),
    )
    return weights, diagnostics


__all__ = [
    "SurfaceAreaWeightDiagnostics",
    "bfl_surface_area_weights",
]
