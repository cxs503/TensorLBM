"""Sub-voxel surface measure for voxelised LBM geometry (P248-N02).

Why this module exists
----------------------
The legacy wetted-area estimator in :mod:`tensorlbm.hydrodynamics`
(:func:`~tensorlbm.hydrodynamics.voxel_wetted_area`) counts exposed
solid/fluid *faces* and multiplies by ``dx**2``.  That "staircase" count
converges only like ``O(h)`` (worse for grid-aligned surfaces: a
digitised circular cross-section over-counts its perimeter by ~27 %
because the Manhattan perimeter of a disc is ``8R`` instead of
``2*pi*R``).  Any wetted-area derived quantity (friction, form factor,
``C_F``) therefore inherits a 5-30 % geometry error that no amount of
turbulence modelling can fix.

This module replaces the face count with a **sub-voxel measure**: the
interface inside every cut cell is reconstructed as a *plane* (the
un-truncated level-set plane through the cell centre, normal
``grad(phi)/|grad(phi)|``), and the plane/cell intersection is computed
**exactly** (polygon clipping, no sampling).  The two outputs per cell
are

* ``solid_fraction`` -- the exact volume fraction of the box on the
  ``phi <= 0`` side of the plane (exact for a planar interface);
* ``patch_area``    -- the exact area of ``plane ∩ box``.

Both are exact for a planar interface and therefore second order
(``O(h**2)``) for a smooth curved one, with no ``h``-dependent tuning
knob.  An optional **second-order curvature correction**

    A_patch *= 1 + h**2 / 24 * (|Hess(phi)|_F**2 - (n . Hess(phi) . n)**2)

adds back the area of the curved patch that the flat reconstruction
fits inside one cell (``kappa1**2 + kappa2**2`` is the squared
Frobenius norm of the *tangential* Hessian, i.e. the sum of the squared
principal curvatures).

Anisotropic cells are supported exactly: a box ``(hx, hy, hz)`` is
mapped affinely onto the unit cube, the plane normal becomes the
reference normal ``h * n / |h * n|`` and areas scale by
``hx*hy*hz / |h * n|`` (see :func:`plane_box_measure`).  This is what
the "vertical-tangent" (keel) difficulty needs: where ``|n_z| -> 0`` a
coarse ``h_z`` costs nothing, which is the analytic basis for the
z-anisotropic refinement recommended by the P248-N02 plan.

Conventions
-----------
* Grids are ``(nz, ny, nx)`` with ``x`` on the last (fastest) axis, the
  TensorLBM lattice convention; ``spacing`` may be a scalar or a
  ``(hx, hy, hz)`` triple.
* ``phi < 0`` is solid (the SDF sign convention of
  :func:`tensorlbm.voxelize.sdf_from_mask`).

References
----------
* Plane/box clipping volume via the divergence theorem
  ``V = 1/3 * sum_faces A_f (c_f . n_f)`` (exact for any origin).
* Curvature correction: the area element of a Monge patch
  ``z = k1 x^2 / 2 + k2 y^2 / 2`` is ``sqrt(1 + (k1 x)^2 + (k2 y)^2)``,
  whose cell average over ``[-h/2, h/2]^2`` is
  ``1 + h^2 (k1^2 + k2^2) / 24 + O(h^4)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SubvoxelMeasure",
    "band_measure",
    "band_measure_nodal",
    "mask_to_levelset",
    "plane_box_measure",
    "plane_cube_measure",
    "staircase_surface_area",
    "subvoxel_measure",
]


# ---------------------------------------------------------------------------
# Fixed cube topology (unit cube [-1/2, 1/2]^3, corner index = 4*bx + 2*by + bz)
# ---------------------------------------------------------------------------

_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)],
    dtype=np.float64,
)

_EDGES = np.array(
    [(a, b) for a in range(8) for b in range(8) if bin(a ^ b).count("1") == 1 and a < b],
    dtype=np.int64,
)


def _face_topology() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(faces[6, 4], normals[6, 3])`` with outward counter-clockwise loops."""
    axis_faces = []
    for axis in range(3):
        for value, sign in ((-0.5, -1.0), (0.5, 1.0)):
            idx = [k for k in range(8) if abs(_CORNERS[k, axis] - value) < 1e-12]
            # sort the 4 loop corners by angle in the face plane
            others = [a for a in range(3) if a != axis]
            ang = np.arctan2(_CORNERS[idx][:, others[1]], _CORNERS[idx][:, others[0]])
            loop = [idx[i] for i in np.argsort(ang)]
            normal = np.zeros(3)
            normal[axis] = sign
            axis_faces.append((loop, normal))
    faces = np.array([f for f, _ in axis_faces], dtype=np.int64)
    normals = np.array([n for _, n in axis_faces], dtype=np.float64)
    return faces, normals


_FACES, _FACE_NORMALS = _face_topology()


# ---------------------------------------------------------------------------
# Vectorised polygon helpers
# ---------------------------------------------------------------------------


def _clip_halfspace(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sutherland-Hodgman clip of convex polygons against ``values <= 0``.

    ``points``/``values`` are ``(N, m, 3)`` / ``(N, m)``.  The output keeps
    the input vertex order, has ``2*m`` slots and a validity mask; invalid
    slots are copies of the first valid vertex so that downstream fan
    triangulation degenerates to zero-area triangles.
    """
    n, m = values.shape
    nxt = (np.arange(m) + 1) % m
    v_i = values
    v_n = values[:, nxt]
    p_i = points
    p_n = points[:, nxt]
    keep_i = v_i <= 0.0
    keep_n = v_n <= 0.0
    cross = keep_i != keep_n
    den = v_i - v_n
    safe = np.where(den == 0.0, 1.0, den)
    t = np.where(cross, v_i / safe, 0.0)
    inter = p_i + t[..., None] * (p_n - p_i)

    slot_pts = np.concatenate([p_i, inter], axis=1)
    slot_ok = np.concatenate([keep_i, cross], axis=1)
    emit_key = np.concatenate([2 * np.arange(m), 2 * np.arange(m) + 1])[None, :]
    keys = np.where(slot_ok, emit_key, 2 * m)
    order = np.argsort(keys, axis=1, kind="stable")
    pts = np.take_along_axis(slot_pts, order[..., None], axis=1)
    ok = np.take_along_axis(slot_ok, order, axis=1)
    count = ok.sum(axis=1)
    first = pts[:, 0:1, :]
    pts = np.where(ok[..., None], pts, first)
    return pts, count


def _polygon_area_centroid(
    points: np.ndarray, count: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Area and centroid of the (padded) convex polygons via fan triangulation."""
    k = points.shape[1]
    normal = np.broadcast_to(np.asarray(normal, dtype=np.float64), (points.shape[0], 3))
    if k < 3:
        z = np.zeros(points.shape[0])
        return z, np.zeros((points.shape[0], 3))
    p0 = points[:, 0:1, :]
    a = points[:, 1 : k - 1, :] - p0
    b = points[:, 2:k, :] - p0
    tri_cross = np.cross(a, b)
    signed = 0.5 * np.einsum("nkj,nj->nk", tri_cross, normal)
    weight = np.abs(signed)
    centroid_tri = (p0 + points[:, 1 : k - 1, :] + points[:, 2:k, :]) / 3.0
    wsum = weight.sum(axis=1)
    area = np.abs(signed.sum(axis=1))
    area = np.where(count >= 3, area, 0.0)
    wsum = np.where((count >= 3) & (wsum > 1e-300), wsum, 1.0)
    centroid = (weight[..., None] * centroid_tri).sum(axis=1) / wsum[:, None]
    return area, centroid


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal in-plane basis ``(e1, e2)`` for each unit ``normal``."""
    idx = np.argmin(np.abs(normal), axis=1)
    helper = np.zeros_like(normal)
    helper[np.arange(normal.shape[0]), idx] = 1.0
    e1 = np.cross(normal, helper)
    norm = np.linalg.norm(e1, axis=1)
    norm = np.where(norm < 1e-12, 1.0, norm)
    e1 = e1 / norm[:, None]
    e2 = np.cross(normal, e1)
    return e1, e2


# ---------------------------------------------------------------------------
# Exact plane / cube measure
# ---------------------------------------------------------------------------


def plane_cube_measure(
    normal: np.ndarray, offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact solid fraction and cut area for planes through the unit cube.

    Args:
        normal: ``(N, 3)`` reference-space unit normals.
        offset: ``(N,)`` signed plane offsets ``n . xi = offset``; the solid
            side is ``n . xi <= offset``.

    Returns:
        ``(fraction, cut_area)`` -- the volume fraction of the unit cube on
        the solid side and the area of ``{n . xi = offset}`` inside the cube
        (both dimensionless, cube volume = 1, cube face area = 1).
    """
    normal = np.atleast_2d(np.asarray(normal, dtype=np.float64))
    offset = np.atleast_1d(np.asarray(offset, dtype=np.float64))
    if normal.shape[1] != 3:
        raise ValueError("normal must have shape (N, 3)")
    if offset.shape[0] != normal.shape[0]:
        raise ValueError("offset must have one entry per normal")
    n_cells = normal.shape[0]
    corner_val = _CORNERS @ normal.T - offset[None, :]  # (8, N)
    corner_val = corner_val.T  # (N, 8)

    # ---- cut polygon: order the 12 edge crossings by angle in the plane ----
    e1, e2 = _plane_basis(normal)
    centre = offset[:, None] * normal  # a point on the plane
    a = corner_val[:, _EDGES[:, 0]]
    b = corner_val[:, _EDGES[:, 1]]
    neg_a = a < 0.0
    neg_b = b < 0.0
    cross = neg_a != neg_b
    den = np.where((a - b) == 0.0, 1.0, a - b)
    t = np.where(cross, a / den, 0.0)
    pa = _CORNERS[_EDGES[:, 0]]
    pb = _CORNERS[_EDGES[:, 1]]
    pts = pa[None, :, :] + t[..., None] * (pb - pa)[None, :, :]
    rel = pts - centre[:, None, :]  # angle must be measured about the plane point
    ang = np.einsum("nke,ne->nk", rel, e2)
    ang2 = np.einsum("nke,ne->nk", rel, e1)
    angle = np.arctan2(ang, ang2)
    keys = np.where(cross, angle, np.pi * 4.0)  # invalid -> far angle
    order = np.argsort(keys, axis=1, kind="stable")
    cut_pts = np.take_along_axis(pts, order[..., None], axis=1)[:, :6, :]
    cut_ok = np.take_along_axis(cross, order, axis=1)[:, :6]
    cut_count = np.minimum(cut_ok.sum(axis=1), 6)
    first = cut_pts[:, 0:1, :]
    cut_pts = np.where(cut_ok[..., None], cut_pts, first)
    cut_area, cut_centroid = _polygon_area_centroid(cut_pts, cut_count, normal)

    # ---- volume: clipped box faces + cut face via the divergence theorem ----
    total = np.zeros(n_cells)
    for face, face_normal in zip(_FACES, _FACE_NORMALS):
        fpts = np.broadcast_to(_CORNERS[face], (n_cells, 4, 3)).copy()
        fval = corner_val[:, face]
        cpts, ccount = _clip_halfspace(fpts, fval)
        # a square clipped by a half-plane has at most 5 vertices
        f_area, f_centroid = _polygon_area_centroid(cpts[:, :5, :], ccount, face_normal)
        total += f_area * (f_centroid @ face_normal)
    cut_contrib = cut_area * np.einsum("nk,nk->n", cut_centroid, normal)
    fraction = (total + cut_contrib) / 3.0
    fraction = np.clip(fraction, 0.0, 1.0)
    return fraction, cut_area


def plane_box_measure(
    normal: np.ndarray, distance: np.ndarray, cell_size: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact solid fraction and patch area for planes cutting rectangular cells.

    Args:
        normal: ``(N, 3)`` physical-space unit normals (``grad(phi)/|grad(phi)|``).
        distance: ``(N,)`` physical signed distance from the cell centre to the
            interface (``phi / |grad(phi)|``); solid is ``distance <= 0``.
        cell_size: scalar, ``(3,)`` or ``(N, 3)`` box edge lengths.

    Returns:
        ``(solid_fraction, patch_area, inside_plane)``; ``patch_area`` is a
        physical area, ``inside_plane`` marks cells the plane actually cuts.
    """
    normal = np.atleast_2d(np.asarray(normal, dtype=np.float64))
    distance = np.atleast_1d(np.asarray(distance, dtype=np.float64))
    n_cells = normal.shape[0]
    size = np.asarray(cell_size, dtype=np.float64)
    if size.ndim == 0:
        size = np.full((n_cells, 3), float(size))
    elif size.ndim == 1:
        size = np.broadcast_to(size, (n_cells, 3)).copy()
    if size.shape != (n_cells, 3):
        raise ValueError("cell_size must be scalar, (3,) or (N, 3)")

    scaled = size * normal  # |J^T n|
    scale = np.linalg.norm(scaled, axis=1)
    scale = np.where(scale < 1e-300, 1e-300, scale)
    ref_normal = scaled / scale[:, None]
    ref_offset = distance / scale
    fraction, ref_area = plane_cube_measure(ref_normal, ref_offset)
    jacobian = np.prod(size, axis=1)
    area = jacobian * ref_area / scale
    support = 0.5 * np.abs(ref_normal).sum(axis=1)
    inside = np.abs(ref_offset) <= support + 1e-12
    return fraction, area, inside


# ---------------------------------------------------------------------------
# Grid-level driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubvoxelMeasure:
    """Grid-level sub-voxel measurement result."""

    area: float
    volume: float
    n_cut_cells: int
    n_solid_cells: int
    spacing: tuple[float, float, float]

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.spacing))


def _as_spacing(spacing: float | tuple[float, float, float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        value = float(spacing)
        if not value > 0.0:
            raise ValueError("spacing must be positive")
        return (value, value, value)
    hx, hy, hz = (float(v) for v in spacing)
    if min(hx, hy, hz) <= 0.0:
        raise ValueError("spacing components must be positive")
    return (hx, hy, hz)


def subvoxel_measure(
    phi: np.ndarray,
    spacing: float | tuple[float, float, float] = 1.0,
    *,
    gradient: np.ndarray | None = None,
    curvature_correction: bool = False,
    band_factor: float = 1.0,
) -> SubvoxelMeasure:
    """Sub-voxel surface area and solid volume of a sampled level set.

    Args:
        phi: ``(nz, ny, nx)`` level set, ``phi < 0`` solid.
        spacing: scalar or ``(hx, hy, hz)``.
        gradient: optional ``(3, nz, ny, nx)`` gradient (``d/dx, d/dy, d/dz``);
            by default second-order central differences of ``phi`` are used.
        curvature_correction: apply the ``h**2/24 * (k1**2 + k2**2)`` term.
        band_factor: widen (``>1``) the cut-cell search band; only affects
            curved interfaces whose true surface leaves a cell that the
            centre plane misses.

    Returns:
        :class:`SubvoxelMeasure` with the total patch area, the solid volume
        and cell counters.
    """
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"phi must be 3-D (nz, ny, nx), got shape {arr.shape}")
    hx, hy, hz = _as_spacing(spacing)
    if gradient is None:
        gz, gy, gx = np.gradient(arr, hz, hy, hx)
        grad = np.stack([gx, gy, gz])
    else:
        grad = np.asarray(gradient, dtype=np.float64)
        if grad.shape != (3,) + arr.shape:
            raise ValueError("gradient must have shape (3, nz, ny, nx)")

    n_solid = int(np.count_nonzero(arr < 0.0))
    cell_vol = hx * hy * hz
    area, band_volume, n_cut, n_solid_band = band_measure(
        arr, grad, (hx, hy, hz), curvature=curvature_correction, band_factor=band_factor
    )
    # Cell the band does not touch is entirely solid or entirely void.
    volume = band_volume + cell_vol * float(n_solid - n_solid_band)
    return SubvoxelMeasure(
        area=area,
        volume=volume,
        n_cut_cells=n_cut,
        n_solid_cells=n_solid,
        spacing=(hx, hy, hz),
    )


def band_measure(
    phi: np.ndarray,
    grad: np.ndarray,
    spacing: float | tuple[float, float, float] = 1.0,
    *,
    curvature: bool = False,
    band_factor: float = 1.0,
) -> tuple[float, float, int, int]:
    """Measure the cut-cell band of a level set on an arbitrary (sub-)grid.

    This is the slab kernel behind :func:`subvoxel_measure`: the caller
    supplies ``phi`` and its gradient on any block of cells (a z-slab, a
    chunk, ...) and accumulates the returns.  Cells outside the band are
    guaranteed to be entirely solid or entirely void.

    Args:
        phi: ``(nz, ny, nx)`` level set, ``phi < 0`` solid.
        grad: ``(3, nz, ny, nx)`` gradient in ``(d/dx, d/dy, d/dz)`` order.
        spacing: scalar or ``(hx, hy, hz)``.
        curvature: apply the second-order curvature correction.
        band_factor: widen the cut-cell search band.

    Returns:
        ``(area, band_solid_volume, n_cut_cells, n_band_solid_cells)``.
    """
    arr = np.asarray(phi, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    if arr.ndim != 3 or grad.shape != (3,) + arr.shape:
        raise ValueError("phi must be (nz, ny, nx) and grad (3, nz, ny, nx)")
    hx, hy, hz = _as_spacing(spacing)
    gmag = np.sqrt(np.einsum("k...,k...->...", grad, grad))
    safe = np.where(gmag < 1e-300, 1e-300, gmag)
    normal = grad / safe[None, ...]
    dist = arr / safe
    support = band_factor * 0.5 * (
        np.abs(normal[0]) * hx + np.abs(normal[1]) * hy + np.abs(normal[2]) * hz
    )
    band = np.abs(dist) <= support
    if not band.any():
        return 0.0, 0.0, 0, 0

    flat = np.flatnonzero(band.ravel())
    idx = np.unravel_index(flat, arr.shape)
    nrm = np.stack([normal[0][idx], normal[1][idx], normal[2][idx]], axis=1)
    dst = dist[idx]
    fraction, area, inside = plane_box_measure(nrm, dst, np.array([hx, hy, hz]))
    if curvature:
        area = area * _curvature_factor(arr, grad, nrm, idx, (hx, hy, hz))
    area_total = float(area[inside].sum())
    band_volume = float(fraction.sum()) * hx * hy * hz
    n_band_solid = int(np.count_nonzero(dst < 0.0))
    return area_total, band_volume, int(np.count_nonzero(inside)), n_band_solid


def _curvature_factor(
    phi: np.ndarray,
    grad: np.ndarray,
    normal: np.ndarray,
    idx: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """``1 + h^2/24 * (k1^2 + k2^2)`` for the cut cells (second-order term)."""
    hx, hy, hz = spacing
    h_eff2 = (hx * hx + hy * hy + hz * hz) / 3.0
    gx, gy, gz = grad[0], grad[1], grad[2]

    def central(field: np.ndarray, axis: int, h: float) -> np.ndarray:
        return np.gradient(field, h, axis=axis)

    hxx = central(gx, 2, hx)[idx]
    hyy = central(gy, 1, hy)[idx]
    hzz = central(gz, 0, hz)[idx]
    hxy = central(gx, 1, hy)[idx]
    hxz = central(gx, 0, hz)[idx]
    hyz = central(gy, 0, hz)[idx]

    frob2 = hxx**2 + hyy**2 + hzz**2 + 2.0 * (hxy**2 + hxz**2 + hyz**2)
    nxx = (
        normal[:, 0] * hxx * normal[:, 0]
        + normal[:, 1] * hyy * normal[:, 1]
        + normal[:, 2] * hzz * normal[:, 2]
    ) + 2.0 * (
        normal[:, 0] * hxy * normal[:, 1]
        + normal[:, 0] * hxz * normal[:, 2]
        + normal[:, 1] * hyz * normal[:, 2]
    )
    # squared sum of the two principal curvatures
    k2 = np.maximum(frob2 - nxx**2, 0.0)
    return 1.0 + h_eff2 * k2 / 24.0


_NODAL_SHIFTS = np.array(
    [[sx, sy, sz] for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)], dtype=np.int64
)


def band_measure_nodal(
    phi: np.ndarray,
    spacing: float | tuple[float, float, float] = 1.0,
    *,
    curvature: bool = False,
    sign_tol: float = 0.0,
) -> tuple[float, float, int, int]:
    """Cut-cell measure driven by the 8 **corner values** of each cell.

    ``phi`` is read as a *nodal* field: the cell ``(i, j, k)`` spans the nodes
    ``i..i+1`` x ``j..j+1`` x ``k..k+1``, so the array must be one wider than
    the cell count in each direction.  A cell contains the interface iff its
    8 corner values do not all share a sign; for those cells the interface is
    the least-squares plane through the 8 corner values, whose normal and
    offset follow from the (orthogonal) design matrix in closed form.

    Why this exists: :func:`band_measure` reconstructs the plane from
    ``phi / |grad(phi)|`` at the cell centre, which is only a first-order
    distance estimate *and* only reliable when ``|grad phi|`` is nearly
    constant inside the cell.  A level set whose gradient varies by an order
    of magnitude within one cell -- e.g. a hull half-beam varying like
    ``zeta**0.24``, where ``|grad phi|`` jumps from ~1 to ~60 across the keel
    layer -- makes that estimate over-report the distance by ~2x, and the
    cut-cell test then silently drops half of a nearly-horizontal pan.  The
    corner-value fit sees the actual variation and is immune.

    Args:
        phi: ``(nz, ny, nx)`` nodal level set, ``phi < 0`` solid.
        spacing: scalar or ``(hx, hy, hz)``.
        curvature: apply the second-order curvature correction (finite
            differences of the fitted normals, so it needs a smooth field).
        sign_tol: values with ``|phi| <= sign_tol`` are treated as on the
            surface during cell selection.

    Returns:
        ``(area, total_solid_volume, n_cut_cells, n_solid_cells)``.  Unlike
        :func:`band_measure`, the volume already includes the fully solid
        cells, so the caller must not add them again.
    """
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("phi must be 3-D (nz, ny, nx)")
    hx, hy, hz = _as_spacing(spacing)
    if min(arr.shape) < 2:
        return 0.0, 0.0, 0, 0
    shape = (arr.shape[0] - 1, arr.shape[1] - 1, arr.shape[2] - 1)
    corner = np.empty(shape + (8,), dtype=np.float64)
    for idx, (di, dj, dk) in enumerate(_NODAL_SHIFTS):
        corner[..., idx] = arr[di : di + shape[0], dj : dj + shape[1], dk : dk + shape[2]]
    neg = corner < -sign_tol
    pos = corner > sign_tol
    band = neg.any(axis=-1) & pos.any(axis=-1)
    cell_vol0 = hx * hy * hz
    n_solid_all = int(np.count_nonzero(neg.all(axis=-1)))
    if not band.any():
        return 0.0, cell_vol0 * n_solid_all, 0, n_solid_all
    a = corner.mean(axis=-1)
    # array axes are (z, y, x); the public normal convention is (x, y, z)
    h_axis = np.array([hz, hy, hx])
    grad_axis = np.empty(shape + (3,), dtype=np.float64)
    for axes in range(3):
        # corner offsets are +-(h/2); the design matrix is orthogonal, so
        # b_axis = sum(o_axis * v) / sum(o_axis^2) = sum(sign * v) / (4 h)
        sv = (2.0 * _NODAL_SHIFTS[:, axes] - 1.0)[None, None, None, :]
        grad_axis[..., axes] = (corner * sv).sum(axis=-1) / (4.0 * h_axis[axes])
    grad = grad_axis[..., ::-1]  # (dz, dy, dx) -> (dx, dy, dz)
    gnorm = np.sqrt(np.einsum("...k,...k->...", grad, grad))
    floor = 1e-14 * max(1.0, float(np.max(np.abs(arr))))
    gnorm = np.where(gnorm < floor, floor, gnorm)
    normal = grad / gnorm[..., None]
    dist = a / gnorm

    idx = np.nonzero(band)
    nrm = normal[idx]
    dst = dist[idx]
    fraction, area, inside = plane_box_measure(nrm, dst, np.array([hx, hy, hz]))
    if curvature:
        area = area * _nodal_curvature_factor(normal, band, idx, (hx, hy, hz))
    cell_vol = hx * hy * hz
    n_solid_cells = int(np.count_nonzero(neg.all(axis=-1)))
    volume = cell_vol * (float(fraction.sum()) + float(n_solid_cells))
    return (
        float(area[inside].sum()),
        float(volume),
        int(np.count_nonzero(inside)),
        n_solid_cells,
    )


def _nodal_curvature_factor(normal, band, idx, spacing) -> np.ndarray:
    """``1 + h^2/24 (k1^2 + k2^2)`` from finite differences of fitted normals."""
    hx, hy, hz = spacing
    h_eff2 = (hx * hx + hy * hy + hz * hz) / 3.0
    nrm = np.broadcast_to(normal, band.shape + (3,))
    grads = []
    for axes, h in enumerate((hz, hy, hx)):  # array axes are (z, y, x)
        d = np.zeros_like(nrm)
        sl_hi = [slice(None)] * 3
        sl_lo = [slice(None)] * 3
        sl_hi[axes] = slice(1, None)
        sl_lo[axes] = slice(None, -1)
        d[tuple(sl_hi)] = (nrm[tuple(sl_hi)] - nrm[tuple(sl_lo)]) / h
        d[tuple(sl_lo)] = d[tuple(sl_hi)]
        grads.append(d[idx][..., ::-1])  # -> (d/dx, d/dy, d/dz)
    hxx, hyy, hzz = (np.einsum("nk,nk->n", grads[i], nrm[idx]) for i in range(3))
    n = nrm[idx]
    frob2 = (
        np.einsum("nk,nk->n", grads[0], grads[0])
        + np.einsum("nk,nk->n", grads[1], grads[1])
        + np.einsum("nk,nk->n", grads[2], grads[2])
    )
    nt = hxx * n[:, 0] ** 2 + hyy * n[:, 1] ** 2 + hzz * n[:, 2] ** 2
    return 1.0 + h_eff2 * np.maximum(frob2 - nt**2, 0.0) / 24.0


# ---------------------------------------------------------------------------
# Mask-only route (level-set reconstruction from a boolean voxel mask)
# ---------------------------------------------------------------------------


def mask_to_levelset(
    mask: np.ndarray,
    spacing: float | tuple[float, float, float] = 1.0,
    *,
    sigma: float | None = 1.2,
) -> np.ndarray:
    """Reconstruct a signed level set from a boolean voxel mask.

    The occupancy field is smoothed with a Gaussian of half-cell width; the
    ``0.5`` isosurface of a smoothed step *is* the interface for a planar
    face and deviates by ``O(h**2)`` for a curved one, so dividing by
    ``|grad(smoothed)|`` recovers a signed distance to second order::

        phi = (0.5 - sigma_field) / |grad(sigma_field)| * h

    Honest limitation: the mask has already discarded sub-voxel information.
    Measured on a sphere (P248-N02): the mask route reaches a few percent
    area error where a sampled SDF reaches ``1e-4``, i.e. it does **not**
    deliver the second-order behaviour of the primary route.  Keep the SDF
    in the geometry pipeline; use this only to retrofit legacy masks.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError("mask must be 3-D (nz, ny, nx)")
    hx, hy, hz = _as_spacing(spacing)
    h_mean = (hx + hy + hz) / 3.0
    solid = arr.astype(bool)
    # Euclidean distance transform to the nearest *volume-element centre* of the
    # other phase; the true interface sits half a cell beyond that centre, hence
    # the -0.5 cell correction.  (A smoothed-occupancy ratio is not usable: the
    # gradient of the smoothed field vanishes in the bulk and the ratio blows up.)
    d_out = distance_transform_edt(~solid, sampling=(hz, hy, hx))
    d_in = distance_transform_edt(solid, sampling=(hz, hy, hx))
    phi = np.where(solid, -(d_in - 0.5 * h_mean), d_out - 0.5 * h_mean)
    if sigma is None:
        return phi
    # The transform is piecewise linear, so its finite-difference gradient is
    # noisy; a ~1 cell Gaussian removes that noise without moving the zero
    # level set (measured: 37 % -> 1.7 % area error on a sphere at R/h = 32).
    return gaussian_filter(phi, sigma=float(sigma), mode="nearest")


# ---------------------------------------------------------------------------
# Staircase baseline (negative control / legacy convention)
# ---------------------------------------------------------------------------


def staircase_surface_area(
    mask: np.ndarray, spacing: float | tuple[float, float, float] = 1.0
) -> float:
    """Exposed-face area of a boolean mask -- the legacy ``O(h)`` estimator.

    This is :func:`tensorlbm.hydrodynamics.voxel_wetted_area` generalised to
    anisotropic cells, kept here as the negative control that quantifies how
    much the sub-voxel measure buys.
    """
    m = np.asarray(mask).astype(bool)
    if m.ndim != 3:
        raise ValueError("mask must be 3-D (nz, ny, nx)")
    hx, hy, hz = _as_spacing(spacing)
    nz, ny, nx = m.shape
    faces = 0
    faces += int(m[:, :, 0].sum() + m[:, :, -1].sum()) * hy * hz
    faces += int(m[:, 0, :].sum() + m[:, -1, :].sum()) * hx * hz
    faces += int(m[0, :, :].sum() + m[-1, :, :].sum()) * hx * hy
    faces += int(np.count_nonzero(m[:, :, 1:] != m[:, :, :-1])) * hy * hz
    faces += int(np.count_nonzero(m[:, 1:, :] != m[:, :-1, :])) * hx * hz
    faces += int(np.count_nonzero(m[1:, :, :] != m[:-1, :, :])) * hx * hy
    return float(faces)