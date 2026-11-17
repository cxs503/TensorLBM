"""Explicit-boundary finite-difference operators for scalar phase fields."""

from __future__ import annotations

from typing import Literal

import torch

BoundaryPolicy = Literal["periodic", "no_flux"]


def _validate(
    field: torch.Tensor, boundary: BoundaryPolicy, solid_mask: torch.Tensor | None
) -> None:
    if field.ndim not in (2, 3):
        raise ValueError("phase-field operators require a 2-D or 3-D scalar tensor")
    if boundary not in ("periodic", "no_flux"):
        raise ValueError("boundary must be either 'periodic' or 'no_flux'")
    if solid_mask is not None:
        if solid_mask.shape != field.shape or solid_mask.dtype != torch.bool:
            raise ValueError("solid_mask must be a bool tensor with the same (z, y, x) shape")


def _neighbor(
    field: torch.Tensor,
    dim: int,
    direction: int,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None,
) -> torch.Tensor:
    rolled = torch.roll(field, shifts=-direction, dims=dim)
    if boundary == "periodic" and solid_mask is None:
        return rolled
    source_mask = (
        None if solid_mask is None else torch.roll(solid_mask, shifts=-direction, dims=dim)
    )
    invalid = torch.zeros_like(field, dtype=torch.bool)
    if boundary == "no_flux":
        edge = [slice(None)] * field.ndim
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
    if field.ndim != 3:
        raise ValueError("central_gradient_3d requires a 3-D scalar tensor")
    _validate(field, boundary, solid_mask)
    grad_z = 0.5 * (
        _neighbor(field, 0, 1, boundary, solid_mask) - _neighbor(field, 0, -1, boundary, solid_mask)
    )
    grad_y = 0.5 * (
        _neighbor(field, 1, 1, boundary, solid_mask) - _neighbor(field, 1, -1, boundary, solid_mask)
    )
    grad_x = 0.5 * (
        _neighbor(field, 2, 1, boundary, solid_mask) - _neighbor(field, 2, -1, boundary, solid_mask)
    )
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
    if field.ndim != 3:
        raise ValueError("laplacian_3d requires a 3-D scalar tensor")
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
        result = torch.where(
            solid_mask, torch.zeros((), dtype=field.dtype, device=field.device), result
        )
    return result


def central_gradient_2d(
    field: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central gradients (x, y); no-flux edges use the adjacent fluid value."""
    if field.ndim != 2:
        raise ValueError("central_gradient_2d requires a 2-D scalar tensor")
    _validate(field, boundary, solid_mask)
    return tuple(
        0.5
        * (
            _neighbor(field, dim, 1, boundary, solid_mask)
            - _neighbor(field, dim, -1, boundary, solid_mask)
        )
        for dim in (1, 0)
    )


def laplacian_2d(
    field: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Five-point Laplacian with explicit domain and solid boundaries."""
    if field.ndim != 2:
        raise ValueError("laplacian_2d requires a 2-D scalar tensor")
    _validate(field, boundary, solid_mask)
    result = (
        _neighbor(field, 1, -1, boundary, solid_mask)
        + _neighbor(field, 1, 1, boundary, solid_mask)
        + _neighbor(field, 0, -1, boundary, solid_mask)
        + _neighbor(field, 0, 1, boundary, solid_mask)
        - 4.0 * field
    )
    return result if solid_mask is None else result.masked_fill(solid_mask, 0.0)
