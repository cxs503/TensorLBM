"""Immersed-boundary operators shared by the three-way coupling chain.

Provides the two interface quadratures that the acceptance gates compare:

* **Eulerian → Lagrangian interpolation** of any grid field onto Lagrangian
  markers with the 4-point smoothed delta (:func:`interpolate`),
* **Lagrangian → Eulerian spreading** of marker forces onto the grid
  (:func:`spread`) — the reaction side of the LBM↔DEM hand-off,
* **surface traction integral** ``F = ∮ (−p n + σ·n) dA`` (:func:`traction_integral`)
  which is the independent route used by gate (a),
* **wall/body momentum exchange** helpers (:func:`reaction_force`).

The 4-point kernel is the one used elsewhere in TensorLBM
(:func:`tensorlbm.ibm.ibm_delta_4pt`); :func:`phi4` is the float64 twin and a
unit test asserts the two agree, so the coupling does not silently re-invent a
different regularisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..ibm import ibm_delta_4pt

CS2 = 1.0 / 3.0


def phi4(r: torch.Tensor) -> torch.Tensor:
    """Four-point smoothed delta function (Peskin), elementwise."""
    a = r.abs()
    inner = (3.0 - 2.0 * a + torch.sqrt(torch.clamp(1.0 + 4.0 * a - 4.0 * a * a, min=0.0))) / 8.0
    outer = (
        5.0 - 2.0 * a - torch.sqrt(torch.clamp(-7.0 + 12.0 * a - 4.0 * a * a, min=0.0))
    ) / 8.0
    return torch.where(a < 1.0, inner, torch.where(a < 2.0, outer, torch.zeros_like(a)))


# ---------------------------------------------------------------------------
# Marker sets
# ---------------------------------------------------------------------------


@dataclass
class MarkerSet:
    """Lagrangian markers of a closed 2D body."""

    x: torch.Tensor  # (M,) lattice x
    y: torch.Tensor  # (M,) lattice y
    nx: torch.Tensor  # (M,) outward normal x
    ny: torch.Tensor  # (M,) outward normal y
    dA: torch.Tensor  # (M,) arc length weight

    def __len__(self) -> int:
        return int(self.x.numel())


def circle_markers(
    cx: float,
    cy: float,
    radius: float,
    spacing: float = 0.6,
    dtype: torch.dtype = torch.float64,
) -> MarkerSet:
    """Uniform circular marker set with arc-length weights."""
    m = max(int(round(2.0 * math.pi * radius / spacing)), 16)
    theta = torch.arange(m, dtype=dtype) * (2.0 * math.pi / m)
    nx = torch.cos(theta)
    ny = torch.sin(theta)
    dA = torch.full((m,), 2.0 * math.pi * radius / m, dtype=dtype)
    return MarkerSet(x=cx + radius * nx, y=cy + radius * ny, nx=nx, ny=ny, dA=dA)


def line_markers(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    normal: Tuple[float, float],
    spacing: float = 0.5,
    dtype: torch.dtype = torch.float64,
) -> MarkerSet:
    """Straight segment of markers with a constant surface normal."""
    length = math.hypot(x1 - x0, y1 - y0)
    m = max(int(round(length / spacing)), 4)
    s = torch.arange(m, dtype=dtype) / (m - 1)
    sx = x0 + (x1 - x0) * s
    sy = y0 + (y1 - y0) * s
    nxv = torch.full((m,), float(normal[0]), dtype=dtype)
    nyv = torch.full((m,), float(normal[1]), dtype=dtype)
    dA = torch.full((m,), length / (m - 1), dtype=dtype)
    return MarkerSet(x=sx, y=sy, nx=nxv, ny=nyv, dA=dA)


# ---------------------------------------------------------------------------
# Interpolation / spreading
# ---------------------------------------------------------------------------


def phi2(r: torch.Tensor) -> torch.Tensor:
    """Two-point (hat) delta function used by the cheap coupling fringe."""
    a = r.abs()
    return torch.where(a < 1.0, 1.0 - a, torch.zeros_like(a))


def marker_stencil(
    markers: MarkerSet, ny: int, nx: int, support: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flattened ``(M, s*s)`` support indices and kernel weights."""
    if support == 2:
        return _stencil_hat(markers, ny, nx)
    return _stencil4(markers, ny, nx)


def _stencil_hat(markers: MarkerSet, ny: int, nx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = markers.x, markers.y
    m = x.numel()
    idx = torch.zeros((m, 4), dtype=torch.long)
    w = torch.zeros((m, 4), dtype=x.dtype)
    ix0 = torch.floor(x).to(torch.long)
    iy0 = torch.floor(y).to(torch.long)
    k = 0
    for a in (0, 1):
        ix_raw = ix0 + a - 1
        wx = phi2(ix_raw.to(x.dtype) - x)
        ix = ix_raw.clamp(0, nx - 1)
        for b in (0, 1):
            iy_raw = iy0 + b - 1
            wy = phi2(iy_raw.to(x.dtype) - y)
            iy = iy_raw.clamp(0, ny - 1)
            idx[:, k] = iy * nx + ix
            w[:, k] = wx * wy
            k += 1
    return idx, w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)


def _stencil4(markers: MarkerSet, ny: int, nx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = markers.x, markers.y
    ix0 = torch.floor(x).to(torch.long) - 1
    iy0 = torch.floor(y).to(torch.long) - 1
    idx = torch.zeros((x.numel(), 16), dtype=torch.long)
    w = torch.zeros((x.numel(), 16), dtype=x.dtype)
    k = 0
    for a in range(4):
        ix_raw = ix0 + a
        wx = phi4(ix_raw.to(x.dtype) - x)
        ix = ix_raw.clamp(0, nx - 1)
        for b in range(4):
            iy_raw = iy0 + b
            wy = phi4(iy_raw.to(x.dtype) - y)
            iy = iy_raw.clamp(0, ny - 1)
            idx[:, k] = iy * nx + ix
            w[:, k] = wx * wy
            k += 1
    return idx, w


def interpolate_stencil(
    field: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    solid: Optional[torch.Tensor] = None,
    renormalise: bool = True,
) -> torch.Tensor:
    ny, nx = field.shape
    flat = field.reshape(-1)
    weights = w
    if solid is not None:
        alive = (~solid.reshape(-1)[idx]).to(w.dtype)
        weights = w * alive
    out = (flat[idx] * weights).sum(dim=1)
    if renormalise:
        out = out / weights.sum(dim=1).clamp(min=1e-12)
    return out


def spread_stencil(
    values: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    ny: int,
    nx: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    out = torch.zeros(ny * nx, dtype=dtype)
    out.index_add_(0, idx.reshape(-1), (values.unsqueeze(1) * w).reshape(-1))
    return out.view(ny, nx)


def interpolate(
    field: torch.Tensor,
    markers: MarkerSet,
    solid: Optional[torch.Tensor] = None,
    renormalise: bool = True,
    support: int = 4,
) -> torch.Tensor:
    """Interpolate ``field`` (shape ``(ny, nx)``) onto the markers.

    Indices are clamped at the domain edge; cells flagged ``solid`` are
    skipped and the remaining kernel weights renormalised (so a marker next to
    a wall does not pick up the zero populations inside it).
    """
    ny, nx = field.shape
    idx, w = marker_stencil(markers, ny, nx, support=support)
    return interpolate_stencil(field, idx, w, solid, renormalise)


def spread(
    values: torch.Tensor,
    markers: MarkerSet,
    ny: int,
    nx: int,
    dtype: torch.dtype = torch.float64,
    support: int = 4,
) -> torch.Tensor:
    """Spread marker values onto the grid (transpose of :func:`interpolate`)."""
    idx, w = marker_stencil(markers, ny, nx, support=support)
    return spread_stencil(values, idx, w, ny, nx, dtype=dtype)


def interpolate_pair(
    ux: torch.Tensor, uy: torch.Tensor, markers: MarkerSet, solid: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return interpolate(ux, markers, solid), interpolate(uy, markers, solid)


# ---------------------------------------------------------------------------
# Interface closure diagnostics
# ---------------------------------------------------------------------------


def traction_integral(
    pressure: torch.Tensor,
    stress: torch.Tensor,
    markers: MarkerSet,
    solid: Optional[torch.Tensor] = None,
    support: int = 4,
) -> torch.Tensor:
    """``F = ∮ (−p n + σ·n) dA`` on the body, from Eulerian fields.

    ``stress`` is ordered ``(sigma_xx, sigma_xy, sigma_yy)`` — the ordering
    produced by :meth:`tensorlbm.three_way.core2d.Flow2D.deviatoric_stress`.
    """
    ny, nx = pressure.shape
    idx, w = marker_stencil(markers, ny, nx, support=support)
    p = interpolate_stencil(pressure, idx, w, solid)
    sxx = interpolate_stencil(stress[0], idx, w, solid)
    sxy = interpolate_stencil(stress[1], idx, w, solid)
    syy = interpolate_stencil(stress[2], idx, w, solid)
    nx, ny = markers.nx, markers.ny
    tx = -p * nx + sxx * nx + sxy * ny
    ty = -p * ny + sxy * nx + syy * ny
    return torch.stack([(tx * markers.dA).sum(), (ty * markers.dA).sum()])


def direct_forcing_force(
    u_solid_x: torch.Tensor,
    u_solid_y: torch.Tensor,
    u_fluid_x: torch.Tensor,
    u_fluid_y: torch.Tensor,
    markers: MarkerSet,
    body_area: float,
    rho: float = 1.0,
    dt: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IBM direct-forcing force on a rigid body and its grid force density.

    Returns ``(F_body, grid_force)`` where ``F_body`` is the force the fluid
    exerts on the body (reaction of the momentum injected into the fluid) and
    ``grid_force`` is the ``(nx, ny)``-shaped density to hand to the LBM.
    """
    n_markers = float(len(markers))
    dA = markers.dA
    area_per_marker = body_area / n_markers
    # velocity deficit at the markers, area-weighted by the marker quadrature
    dux = (u_solid_x - u_fluid_x) * dA / dA.mean()
    duy = (u_solid_y - u_fluid_y) * dA / dA.mean()
    f_marker_x = rho * area_per_marker * dux / dt
    f_marker_y = rho * area_per_marker * duy / dt
    return torch.stack([f_marker_x.sum(), f_marker_y.sum()]), (f_marker_x, f_marker_y)


def reaction_force(grid_force: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Force on the body = −(force injected into the fluid)."""
    return -torch.stack([grid_force[0].sum(), grid_force[1].sum()])