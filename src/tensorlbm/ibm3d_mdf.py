"""Immersed-boundary 3D diffuse-interface coupling with multi-direct forcing.

This module provides the *diffuse-interface* (kernel-smeared) immersed-boundary
operator with **multi-direct forcing (MDF)** for three-dimensional grids, the
combination used by the OSS reference ``haimingz/vivsim`` (``vivsim.ib3d`` +
``vivsim.ib.mdf``, MIT, commit ``fb84b12f``).

Method (Wang, Breugem & co-workers; as implemented by vivsim)
-------------------------------------------------------------
A closed triangulated body surface carries ``N`` Lagrangian markers with
surface-area weights ``ds_j``.  For each marker a tensor-product discrete delta
stencil maps the marker to ``4**3 = 64`` Eulerian cells (``radius=2``).

One *direct-forcing* iteration relaxes the marker velocity ``u_m`` towards its
target ``u_t``::

    g_j      = 2 · ds_j · (u_t − u_m)          # marker force (ρ=1, Δt=1)
    G        = Σ_j g_j δ(x − x_j)              # diffuse onto the grid
    u_m     += Σ 0.5 · G δ(x − x_j)            # Δu = F/(2ρ) predicted by the grid update

and the iteration is repeated ``n_iter`` times (MDF).  Only the *incremental*
force of each iteration is used to update ``u_m``; the grid force finally
diffused is the accumulated marker force, so the momentum handed to the fluid
equals the total marker reaction.

Why this matters numerically: a single pass (``n_iter = 1``, the classical
Uhlmann direct forcing) leaves a residual slip ``(I − ds·K Kᵀ)`` at the
markers, because the interpolation/spreading pair is not an exact inverse.
Iterating drives that residual down; with area weights ``ds`` the fixed point
is the physical one (force per unit area), which an unweighted marker force
cannot produce.

Provenance / licence
--------------------
The stencil, delta kernels, MDF loop and surface-area weighting are a port of
``haimingz/vivsim`` (MIT, (c) 2024 Haiming Zhu):

* ``vivsim/ib3d/stencil.py``   → :func:`build_stencil`, :func:`interpolate`, :func:`spread`
* ``vivsim/ib/kernels.py``     → :func:`kernel_peskin_4pt`, :func:`kernel_peskin_3pt`, :func:`kernel_cosine_4pt`
* ``vivsim/ib/mdf.py``         → :func:`multi_direct_forcing`
* ``vivsim/ib3d/geometry.py``  → :func:`triangle_areas`, :func:`surface_area`, :func:`surface_weights`

Port differences (deliberate, see ``docs/ibm3d_mdf/DIFFERENCES_VS_VIVSIM.md``):
tensor backend is PyTorch (``index_add_`` instead of JAX ``.at[].add``), the
grid layout is TensorLBM's ``(nz, ny, nx)`` with flat index
``iz·ny·nx + iy·nx + ix``, markers are ``(x, y, z)``, and out-of-range stencil
entries are dropped (OSS relies on JAX scatter semantics) rather than wrapping.

Public API
----------
``kernel_peskin_4pt(r)`` / ``kernel_peskin_3pt(r)`` / ``kernel_cosine_4pt(r)``
    1-D discrete delta kernels.
``build_stencil(marker_coords, shape, kernel=..., radius=2)``
    ``(weights, flat_indices)`` of shape ``(N, S)``.
``interpolate(grid, weights, indices)`` / ``spread(marker_values, weights, indices, shape)``
    Lagrangian ↔ Eulerian transfer.
``surface_weights(vertices, faces)`` / ``sphere_surface(n_theta, n_phi, ...)``
    Marker set with lumped area weights ``ds`` (Σds = triangulated area).
``multi_direct_forcing(grid_u, weights, indices, u_target, ds, n_iter=5)``
    The MDF loop; returns ``(grid_force, marker_reaction)``.
``drag_coefficient(force, rho, speed, diameter)``
    ``C_d = 2 F / (ρ U² A)`` with ``A = π D²/4``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "kernel_peskin_4pt",
    "kernel_peskin_3pt",
    "kernel_cosine_4pt",
    "KERNELS",
    "build_stencil",
    "interpolate",
    "spread",
    "triangle_areas",
    "surface_area",
    "surface_weights",
    "sphere_surface",
    "multi_direct_forcing",
    "mdf_residuals",
    "drag_coefficient",
    "MDFResult",
]

# --------------------------------------------------------------------------- #
# Discrete delta kernels (diffuse interface)
# --------------------------------------------------------------------------- #


def kernel_peskin_4pt(r: torch.Tensor) -> torch.Tensor:
    """Peskin 4-point delta (support |r| ≤ 2), Peskin (2002) eq. (4).

    ``φ = (3 − 2|r| + √(1 + 4|r| − 4r²))/8`` for ``|r| ≤ 1`` and
    ``φ = (5 − 2|r| − √(−7 + 12|r| − 4r²))/8`` for ``1 < |r| ≤ 2``.

    Identical (to the last bit of the formula) to ``tensorlbm.ibm.ibm_delta_4pt``
    and to ``vivsim.ib.kernels.kernel_peskin_4pt``.
    """
    ra = r.abs()
    disc1 = (1.0 + 4.0 * ra - 4.0 * ra * ra).clamp(min=0.0)
    phi1 = (3.0 - 2.0 * ra + torch.sqrt(disc1)) / 8.0
    disc2 = (-7.0 + 12.0 * ra - 4.0 * ra * ra).clamp(min=0.0)
    phi2 = (5.0 - 2.0 * ra - torch.sqrt(disc2)) / 8.0
    return torch.where(
        ra <= 1.0,
        phi1,
        torch.where(ra <= 2.0, phi2, torch.zeros_like(ra)),
    )


def kernel_peskin_3pt(r: torch.Tensor) -> torch.Tensor:
    """Peskin 3-point delta (support |r| ≤ 1.5)."""
    ra = r.abs()
    disc = (1.0 - 3.0 * ra * ra).clamp(min=0.0)
    near = (1.0 + torch.sqrt(disc)) / 3.0
    disc2 = (-2.0 + 6.0 * ra - 3.0 * ra * ra).clamp(min=0.0)
    far = (5.0 - 3.0 * ra - torch.sqrt(disc2)) / 6.0
    return torch.where(
        ra < 0.5,
        near,
        torch.where(ra <= 1.5, far, torch.zeros_like(ra)),
    )


def kernel_cosine_4pt(r: torch.Tensor) -> torch.Tensor:
    """4-point cosine delta ``φ = (1 + cos(π|r|/2))/4`` for ``|r| ≤ 2``."""
    ra = r.abs()
    val = (1.0 + torch.cos(math.pi * ra * 0.5)) * 0.25
    return torch.where(ra <= 2.0, val, torch.zeros_like(ra))


KERNELS = {
    "peskin4pt": kernel_peskin_4pt,
    "4pt": kernel_peskin_4pt,
    "peskin3pt": kernel_peskin_3pt,
    "cosine4pt": kernel_cosine_4pt,
}


def _resolve_kernel(kernel) -> tuple[callable, int]:
    """Return ``(kernel_fn, default_stencil_radius)`` for a name or callable."""
    if callable(kernel):
        return kernel, 2
    key = str(kernel).lower()
    if key not in KERNELS:
        raise ValueError(f"unknown IBM kernel {kernel!r}; use one of {sorted(KERNELS)}")
    return KERNELS[key], (2 if "4" in key else 2)


# --------------------------------------------------------------------------- #
# Eulerian layout helpers
# --------------------------------------------------------------------------- #
#
# TensorLBM solves on ``(nz, ny, nx)`` arrays; flat index =
# ``iz * (ny * nx) + iy * nx + ix``.  Markers are given in physical
# ``(x, y, z)`` order, which is the OSS convention.


def _axis_stencil(coord: torch.Tensor, n: int, radius: int, wrap: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D stencil for one axis: ``(indices, valid_mask)`` of shape ``(N, 2*radius)``."""
    offsets = torch.arange(-radius + 1, radius + 1, device=coord.device, dtype=coord.dtype)
    ref = torch.floor(coord)
    raw = ref.unsqueeze(1) + offsets.unsqueeze(0)
    if wrap:
        idx = torch.remainder(raw, n).long()
        valid = torch.ones_like(raw, dtype=torch.bool)
    else:
        valid = (raw >= 0) & (raw <= n - 1)
        idx = raw.clamp(0, n - 1).long()
    return idx, valid


def build_stencil(
    marker_coords: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    kernel="peskin4pt",
    radius: int | None = None,
    wrap: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-product delta stencil for every marker.

    Args:
        marker_coords: ``(N, 3)`` marker positions in lattice units, physical
            ``(x, y, z)`` order.
        shape: Eulerian grid shape ``(nz, ny, nx)``.
        kernel: kernel name (see :data:`KERNELS`) or callable.
        radius: stencil half width; ``None`` → 2 (i.e. 4 points per axis).
        wrap: periodic wrap-around.  ``False`` reproduces the OSS behaviour
            (entries outside the grid are dropped, weight zero).

    Returns:
        ``(weights, flat_indices)``, both ``(N, S)`` with ``S = (2·radius)³``.
        ``weights`` sums to 1 per marker for a partition-of-unity kernel when
        the marker is at least ``radius`` cells from every boundary.
    """
    nz, ny, nx = shape
    kfn, default_radius = _resolve_kernel(kernel)
    rad = default_radius if radius is None else int(radius)
    coords = marker_coords
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError("marker_coords must have shape (N, 3)")
    ix, vx = _axis_stencil(coords[:, 0], nx, rad, wrap)
    iy, vy = _axis_stencil(coords[:, 1], ny, rad, wrap)
    iz, vz = _axis_stencil(coords[:, 2], nz, rad, wrap)

    wx = kfn(ix.to(coords.dtype) - coords[:, 0:1])
    wy = kfn(iy.to(coords.dtype) - coords[:, 1:2])
    wz = kfn(iz.to(coords.dtype) - coords[:, 2:3])

    s = ix.shape[1]
    w = (wx.unsqueeze(2).unsqueeze(3) * wy.unsqueeze(1).unsqueeze(3) * wz.unsqueeze(1).unsqueeze(2))
    valid = (vx.unsqueeze(2).unsqueeze(3) & vy.unsqueeze(1).unsqueeze(3) & vz.unsqueeze(1).unsqueeze(2))
    w = torch.where(valid, w, torch.zeros_like(w))

    # broadcast to (N, sx, sy, sz) and flatten in (x, y, z) stencil order
    flat = (
        iz.unsqueeze(1).unsqueeze(1) * (ny * nx)
        + iy.unsqueeze(1).unsqueeze(3) * nx
        + ix.unsqueeze(2).unsqueeze(3)
    ).reshape(coords.shape[0], s * s * s)
    return w.reshape(coords.shape[0], s * s * s), flat


def interpolate(
    grid: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Interpolate an Eulerian field onto the markers.

    Args:
        grid: ``(n_comp, nz, ny, nx)`` or ``(nz, ny, nx)``.
        weights: ``(N, S)`` from :func:`build_stencil`.
        indices: ``(N, S)`` flat cell indices (same layout as ``grid``).

    Returns:
        ``(N, n_comp)`` marker values.
    """
    squeezed = grid.dim() == 3
    if squeezed:
        grid = grid.unsqueeze(0)
    n_comp = grid.shape[0]
    flat = grid.reshape(n_comp, -1)
    n_markers, n_stencil = indices.shape
    sampled = flat[:, indices.reshape(-1)].reshape(n_comp, n_markers, n_stencil)
    out = torch.einsum("cms,ms->mc", sampled, weights)
    return out[:, 0] if squeezed else out


def spread(
    marker_values: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Spread marker values onto the Eulerian grid (accumulate, no wrap).

    Args:
        marker_values: ``(N, n_comp)`` or ``(N,)``.
        weights: ``(N, S)``.
        indices: ``(N, S)`` flat cell indices.
        shape: ``(nz, ny, nx)``.

    Returns:
        ``(n_comp, nz, ny, nx)`` or ``(nz, ny, nx)`` grid field.
    """
    squeezed = marker_values.dim() == 1
    if squeezed:
        marker_values = marker_values.unsqueeze(1)
    n_comp = marker_values.shape[1]
    n_stencil = indices.shape[1]
    n_markers = indices.shape[0]
    # (N, C, S) -> (C, N*S); the permute is required so that the flat order is
    # component-major (reshaping straight from (N, C, S) would interleave them).
    contrib = (
        marker_values.unsqueeze(2) * weights.unsqueeze(1)
    ).permute(1, 0, 2).reshape(n_comp, n_markers * n_stencil)
    out = torch.zeros((n_comp, int(torch.tensor(shape).prod())), dtype=marker_values.dtype, device=marker_values.device)
    out.index_add_(1, indices.reshape(-1), contrib)
    out = out.reshape(n_comp, *shape)
    return out[0] if squeezed else out


# --------------------------------------------------------------------------- #
# Geometry: triangulated surface, lumped area weights
# --------------------------------------------------------------------------- #


def triangle_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Per-face areas of a triangulated surface, shape ``(n_faces,)``."""
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return 0.5 * torch.linalg.norm(torch.cross(e1, e2, dim=1), dim=1)


def surface_area(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Total area of a triangulated surface (scalar)."""
    return triangle_areas(vertices, faces).sum()


def surface_weights(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Lumped per-vertex area weights ``ds`` (each triangle's area split in 3).

    ``Σ ds`` equals :func:`surface_area`, i.e. the force ``Σ_j 2·ds_j·Δu_j`` is a
    physical surface integral (ρ=1, Δt=1).
    """
    areas = triangle_areas(vertices, faces) / 3.0
    ds = torch.zeros(vertices.shape[0], dtype=vertices.dtype, device=vertices.device)
    ds.index_add_(0, faces.reshape(-1), areas.repeat_interleave(3))
    return ds


def sphere_surface(
    center: tuple[float, float, float],
    radius: float,
    *,
    n_theta: int = 32,
    n_phi: int = 16,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical spherical surface mesh (UV sphere) with `faces`.

    Args:
        center: sphere centre in lattice units ``(x, y, z)``.
        radius: radius in lattice units.
        n_theta: points along the equator (longitude).
        n_phi: points from pole to pole (latitude, ≥3).
        dtype: floating point type.

    Returns:
        ``(vertices (n_vertices, 3), faces (n_faces, 3))``.
    """
    n_phi = max(3, int(n_phi))
    n_theta = max(3, int(n_theta))
    phi = torch.linspace(0.0, math.pi, n_phi + 1, dtype=dtype)[1:-1]
    theta = torch.linspace(0.0, 2.0 * math.pi, n_theta + 1, dtype=dtype)[:-1]
    # poles
    north = torch.tensor([0.0, 0.0, radius], dtype=dtype)
    south = torch.tensor([0.0, 0.0, -radius], dtype=dtype)
    ring = torch.stack(
        torch.meshgrid(phi, theta, indexing="ij"), dim=-1
    )  # (n_phi-1, n_theta, 2)
    ph = ring[..., 0]
    th = ring[..., 1]
    x = radius * torch.sin(ph) * torch.cos(th)
    y = radius * torch.sin(ph) * torch.sin(th)
    z = radius * torch.cos(ph)
    body = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=1)
    verts = torch.cat([north.unsqueeze(0), body, south.unsqueeze(0)], dim=0)
    verts = verts + torch.tensor(center, dtype=dtype).unsqueeze(0)

    faces: list[list[int]] = []
    n_rings = n_phi - 1
    # north cap
    for j in range(n_theta):
        faces.append([0, 1 + j, 1 + (j + 1) % n_theta])
    # quads between rings
    for i in range(n_rings - 1):
        base0 = 1 + i * n_theta
        base1 = 1 + (i + 1) * n_theta
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            faces.append([base0 + j, base1 + j, base1 + jn])
            faces.append([base0 + j, base1 + jn, base0 + jn])
    # south cap
    south_idx = verts.shape[0] - 1
    base = 1 + (n_rings - 1) * n_theta
    for j in range(n_theta):
        faces.append([south_idx, base + (j + 1) % n_theta, base + j])
    return verts, torch.tensor(faces, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Multi-direct forcing
# --------------------------------------------------------------------------- #


@dataclass
class MDFResult:
    """Outcome of :func:`multi_direct_forcing`."""

    grid_force: torch.Tensor      # (3, nz, ny, nx) force density handed to the fluid
    marker_reaction: torch.Tensor  # (N, 3) force ON the body (−marker force)
    marker_velocity: torch.Tensor  # (N, 3) marker velocity after the iterations
    marker_force_total: torch.Tensor  # (N, 3) force applied to the fluid
    residuals: list[float]         # ‖u_t − u_m‖∞/U_scale after each iteration


def multi_direct_forcing(
    grid_u: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    u_target: torch.Tensor,
    ds: torch.Tensor | float,
    *,
    n_iter: int = 5,
    accumulate: bool = False,
    return_result: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | MDFResult:
    """Multi-direct forcing: relax marker velocities onto ``u_target``.

    Args:
        grid_u: ``(3, nz, ny, nx)`` or ``(3, n_flat)`` fluid velocity.
        weights, indices: stencil from :func:`build_stencil`.
        u_target: ``(N, 3)`` target marker velocity.
        ds: ``(N,)`` marker surface-area weights, or a scalar.
        n_iter: number of direct-forcing iterations (1 = Uhlmann single pass).
        accumulate: ``False`` = OSS recipe (each iteration's ``u_m`` update uses
            only that iteration's incremental grid force).  ``True`` re-evaluates
            ``u_m`` from the *accumulated* force, which converges in one
            iteration for a consistent stencil but is not what vivsim does; it is
            kept for the sensitivity study only.
        return_result: return the :class:`MDFResult` (default) or the legacy
            ``(grid_force, marker_reaction)`` tuple.

    Returns:
        :class:`MDFResult` (or ``(grid_force, marker_reaction)``).

    Notes:
        ``n_iter = 1`` reproduces classical direct forcing, but *with* the
        area weighting and the factor ``2`` (``F = 2ρAΔu``).  TensorLBM's
        historical :func:`tensorlbm.ibm.ibm_direct_forcing_3d` omits both.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    squeezed = grid_u.dim() == 2
    if squeezed:
        raise ValueError("grid_u must be (3, nz, ny, nx)")
    nz, ny, nx = grid_u.shape[1:]
    shape = (nz, ny, nx)
    ds_t = torch.as_tensor(ds, dtype=grid_u.dtype, device=grid_u.device)
    if ds_t.dim() == 0:
        ds_t = ds_t.expand(u_target.shape[0])
    ds_col = ds_t.reshape(-1, 1)

    u_target = u_target.to(grid_u.dtype)
    if accumulate:
        marker_force_total = torch.zeros_like(u_target)
        force_grid_total = torch.zeros_like(grid_u)
        residuals: list[float] = []
        scale = _slip_scale(grid_u, weights, indices, u_target)
        for _ in range(n_iter):
            u_m = interpolate(grid_u, weights, indices)
            step = (u_target - u_m) * ds_col * 2.0
            marker_force_total = marker_force_total + step
            force_grid_total = force_grid_total + spread(step, weights, indices, shape)
            grid_u = grid_u + 0.5 * force_grid_total
            residuals.append(float((u_target - interpolate(grid_u, weights, indices)).abs().max() / scale))
        grid_force = force_grid_total
        marker_u = interpolate(grid_u, weights, indices)
    else:
        marker_u = interpolate(grid_u, weights, indices)
        marker_force_total = torch.zeros_like(u_target)
        residuals = []
        scale = _slip_scale(grid_u, weights, indices, u_target)
        for _ in range(n_iter):
            step = (u_target - marker_u) * ds_col * 2.0
            marker_force_total = marker_force_total + step
            grid_step = spread(step, weights, indices, shape)
            marker_u = marker_u + interpolate(grid_step * 0.5, weights, indices)
            residuals.append(float((u_target - marker_u).abs().max() / scale))
        grid_force = spread(marker_force_total, weights, indices, shape)

    result = MDFResult(
        grid_force=grid_force,
        marker_reaction=-marker_force_total,
        marker_velocity=marker_u,
        marker_force_total=marker_force_total,
        residuals=residuals,
    )
    if return_result:
        return result
    return result.grid_force, result.marker_reaction



def _slip_scale(
    grid_u: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    u_target: torch.Tensor,
) -> torch.Tensor:
    """Normalisation for the slip residual ``‖u_t − u_m‖``.

    ``max(‖u_t‖∞, ‖u_m(0)‖∞, 1e-30)`` — for a *stationary* body ``u_t = 0`` and
    the natural scale is the initial slip, so the residual reported is the
    fraction of the initial velocity mismatch that survives.
    """
    u_m0 = interpolate(grid_u, weights, indices)
    return torch.maximum(
        u_target.abs().max(), u_m0.abs().max()
    ).clamp(min=1e-30)


def mdf_residuals(
    grid_u: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    u_target: torch.Tensor,
    ds: torch.Tensor | float,
    *,
    max_iter: int = 10,
) -> list[float]:
    """Slip residual ``‖u_t − u_m‖∞ / ‖u_t‖∞`` after each MDF iteration."""
    res = multi_direct_forcing(
        grid_u, weights, indices, u_target, ds, n_iter=max_iter, return_result=True
    )
    return res.residuals


def drag_coefficient(
    force: torch.Tensor,
    rho: float,
    speed: float,
    diameter: float,
) -> float:
    """``C_d = 2 F_x / (ρ U² A)`` with the frontal area ``A = π D²/4``."""
    area = math.pi * diameter * diameter / 4.0
    return float(2.0 * float(force[0]) / (rho * speed * speed * area))


def drag_coefficient_2d(force: torch.Tensor, rho: float, speed: float, diameter: float) -> float:
    """``C_d = 2 F_x / (ρ U² D)`` for a 2D (unit-depth) body."""
    return float(2.0 * float(force[0]) / (rho * speed * speed * diameter))