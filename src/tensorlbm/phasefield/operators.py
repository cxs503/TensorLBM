"""Explicit-boundary finite-difference operators for 3-D scalar phase fields."""

from __future__ import annotations

from typing import Literal

import torch

BoundaryPolicy = Literal["periodic", "no_flux"]


def _validate(field: torch.Tensor, boundary: BoundaryPolicy, solid_mask: torch.Tensor | None) -> None:
    if field.ndim != 3:
        raise ValueError("phase-field operators require a 3-D scalar tensor shaped (z, y, x)")
    if boundary not in ("periodic", "no_flux"):
        raise ValueError("boundary must be either 'periodic' or 'no_flux'")
    if solid_mask is not None:
        if solid_mask.shape != field.shape or solid_mask.dtype != torch.bool:
            raise ValueError("solid_mask must be a bool tensor with the same (z, y, x) shape")


def _neighbor(field: torch.Tensor, dim: int, direction: int, boundary: BoundaryPolicy, solid_mask: torch.Tensor | None) -> torch.Tensor:
    rolled = torch.roll(field, shifts=-direction, dims=dim)
    if boundary == "periodic" and solid_mask is None:
        return rolled
    source_mask = None if solid_mask is None else torch.roll(solid_mask, shifts=-direction, dims=dim)
    invalid = torch.zeros_like(field, dtype=torch.bool)
    if boundary == "no_flux":
        edge = [slice(None)] * 3
        edge[dim] = -1 if direction > 0 else 0
        invalid[tuple(edge)] = True
    if solid_mask is not None:
        invalid = invalid | solid_mask | source_mask
    return torch.where(invalid, field, rolled)


def central_gradient_3d(
    field: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return central gradients ``(d/dx, d/dy, d/dz)`` for ``(z, y, x)`` data.

    ``boundary`` is intentionally mandatory.  ``no_flux`` mirrors the adjacent
    interior value at domain edges and at solid neighbours, never wraps them.
    """
    _validate(field, boundary, solid_mask)
    grad_z = 0.5 * (_neighbor(field, 0, 1, boundary, solid_mask) - _neighbor(field, 0, -1, boundary, solid_mask))
    grad_y = 0.5 * (_neighbor(field, 1, 1, boundary, solid_mask) - _neighbor(field, 1, -1, boundary, solid_mask))
    grad_x = 0.5 * (_neighbor(field, 2, 1, boundary, solid_mask) - _neighbor(field, 2, -1, boundary, solid_mask))
    if solid_mask is not None:
        zero = torch.zeros((), dtype=field.dtype, device=field.device)
        grad_x = torch.where(solid_mask, zero, grad_x)
        grad_y = torch.where(solid_mask, zero, grad_y)
        grad_z = torch.where(solid_mask, zero, grad_z)
    return grad_x, grad_y, grad_z


def laplacian_3d(
    field: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the seven-point Laplacian for 3-D scalar ``(z, y, x)`` data."""
    _validate(field, boundary, solid_mask)
    # Keep this addition order aligned with the legacy D3Q19 FE stencil so its
    # FP32 results remain bitwise stable when it delegates here.
    result = (
        _neighbor(field, 2, -1, boundary, solid_mask)
        + _neighbor(field, 2, 1, boundary, solid_mask)
        + _neighbor(field, 1, -1, boundary, solid_mask)
        + _neighbor(field, 1, 1, boundary, solid_mask)
        + _neighbor(field, 0, -1, boundary, solid_mask)
        + _neighbor(field, 0, 1, boundary, solid_mask)
        - 6.0 * field
    )
    if solid_mask is not None:
        result = torch.where(solid_mask, torch.zeros((), dtype=field.dtype, device=field.device), result)
    return result
