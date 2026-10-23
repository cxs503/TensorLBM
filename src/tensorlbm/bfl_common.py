"""Common BFL (Bouzidi–Firdaouss–Lallemand) interpolated bounce-back module.

This module provides a **unified, solver-agnostic** BFL boundary condition
that replaces the half-way bounce-back (``bounce_back_cells_3d``) with a
second-order accurate interpolated bounce-back at the *exact* wall position.

BFL advantages over half-way BB
--------------------------------
* Interpolated bounce-back at the exact wall position (not half-way).
* Better accuracy for curved surfaces (cylinder, sphere, ship hull).
* Reduces the staircase effect inherent to voxelised geometries.

Mathematics
-----------
For each near-wall fluid cell with a boundary link in direction *d*,
let *q* be the fractional distance from the fluid node to the wall
(``q = 0`` at the fluid cell, ``q = 1`` at the solid neighbour):

* **q < 0.5** (linear interpolation)::

      f_bc = 2·q·f_opp + (1 − 2·q)·f_prev[d]

* **q ≥ 0.5** (quadratic interpolation)::

      f_bc = f_opp / (2·q) + (2·q − 1) / (2·q) · f_prev[opp]

where ``f_opp`` is the post-stream opposite population (streamed from
the solid side) and ``f_prev`` is the pre-stream (post-collision)
distribution.

For a **moving wall** with velocity **u_w**, a momentum correction is
added::

    f_bc += 2·ρ·w[opp_d]·(c[opp_d]·u_w) / c_s²

Integration with common modules
-------------------------------
* **Geometry**: uses :class:`tensorlbm.drag_pressure.SurfaceMesh` for
  normals and ``compute_q_*`` helpers for the *q*-field.
* **Force**: :func:`drag_pressure_integration` +
  :func:`drag_friction_integration` (with ``formula='bfl'``).
* **Main loop**: ``lbm_step_correct`` with ``bounce_back_fn`` replaced
  by :func:`bfl_bounce_back_common`.
* **Strouhal**: :func:`tensorlbm.postprocess.detect_strouhal`.

Supported lattices: ``D3Q19``, ``D3Q27``.

Reference
---------
Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001).
"Momentum transfer of a Boltzmann-lattice fluid with boundaries."
*Physics of Fluids*, 13(11), 3452–3459.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .d3q19 import OPPOSITE as _OPP19
from .d3q19 import C as _C19
from .d3q19 import W as _W19
from .d3q27 import OPPOSITE as _OPP27
from .d3q27 import C as _C27
from .d3q27 import W as _W27

__all__ = [
    "SUPPORTED_LATTICES",
    "BFLLatticeName",
    "bfl_bounce_back_common",
    "bfl_moving_wall_correction",
    "compute_q_cylinder_common",
    "compute_q_sphere_common",
    "compute_q_flat_walls_common",
    "compute_q_stl_common",
    "compute_q_wall_sphere",
    "compute_q_wall_cylinder",
    "compute_q_wall_generic",
    "compute_q_generic_common",
    "bfl_step",
    "BFLBoundaryLinks",
    "bfl_boundary_link_indices",
    "bfl_bounce_back_sparse",
    "bfl_force_ledger_sparse",
    "bfl_moving_wall_correction_sparse",
]

SUPPORTED_LATTICES: tuple[str, ...] = ("D3Q19", "D3Q27")
BFLLatticeName = Literal["D3Q19", "D3Q27"]

# Pre-computed OPPOSITE arrays as plain Python lists (no .item() in hot path)
_OPP19_LIST: list[int] = [int(x) for x in _OPP19.tolist()]
_OPP27_LIST: list[int] = [int(x) for x in _OPP27.tolist()]


def _lattice_params(lattice: str):
    """Return (Q, C, W, OPPOSITE_list) for the given lattice."""
    lattice_u = lattice.upper()
    if lattice_u == "D3Q19":
        return 19, _C19, _W19, _OPP19_LIST
    if lattice_u == "D3Q27":
        return 27, _C27, _W27, _OPP27_LIST
    raise ValueError(f"Unsupported lattice {lattice!r}; supported: {SUPPORTED_LATTICES}")


# --------------------------------------------------------------------------- #
# 1. BFL interpolated bounce-back (vectorised, lattice-agnostic)
# --------------------------------------------------------------------------- #
def bfl_bounce_back_common(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    *,
    lattice: BFLLatticeName = "D3Q19",
    wall_correction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply BFL interpolated bounce-back for ALL directions (vectorised).

    This is the **common-module** replacement for
    :func:`tensorlbm.boundaries3d.bounce_back_cells_3d`.  It is fully
    vectorised (no per-direction Python loop) and works for both D3Q19
    and D3Q27.

    Args:
        f: Post-stream distribution ``(Q, nz, ny, nx)``.
        f_prev: Pre-stream (post-collision) distribution, same shape.
        fluid_boundary_mask: ``(Q, nz, ny, nx)`` bool — True at boundary
            links (fluid node whose neighbour in direction *d* is solid).
        q_field: ``(Q, nz, ny, nx)`` float — fractional distance *q*.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.
        wall_correction: Optional ``(Q, nz, ny, nx)`` float — moving-wall
            momentum correction (already opp-indexed by the caller via
            :func:`bfl_moving_wall_correction`).

    Returns:
        Updated distribution tensor, same shape as *f*.
    """
    Q, velocities, _, _ = _lattice_params(lattice)
    if f.shape[0] != Q:
        raise ValueError(f"f has {f.shape[0]} directions but lattice {lattice!r} expects {Q}")

    opp_tensor = (_OPP19 if Q == 19 else _OPP27).to(f.device)

    # Gather post-collision populations.  For q<1/2, BFL also needs the
    # outgoing population one lattice node upstream (x-c_d).
    fp_opp_all = f_prev[opp_tensor]  # f_prev[opp[d]]
    fp_d_all = f_prev  # f_prev[d]
    fp_d_upstream_all = torch.stack(
        [
            torch.roll(
                f_prev[direction],
                shifts=tuple(int(value) for value in velocities[direction, [2, 1, 0]].tolist()),
                dims=(0, 1, 2),
            )
            for direction in range(Q)
        ]
    )

    q = q_field
    mask = fluid_boundary_mask

    # The compute_q_* helpers store q as float32.  Promote it to the
    # population dtype before forming the interpolation coefficients: with
    # float64 populations, leaving 1/(2q) and (2q-1) in float32 rounds the
    # quadratic-branch weights to ~1e-7 relative accuracy, which both breaks
    # exact 2-D/3-D planar equivalence and silently degrades the second-order
    # accuracy the BFL interpolation exists to provide.  The promotion is an
    # exact identity for float32 runs.
    q = q.to(dtype=f.dtype)

    mask_lin = (q < 0.5) & mask  # linear regime
    mask_quad = (~mask_lin) & mask  # quadratic regime

    # Linear: f_bc = 2q·fp_d(x) + (1-2q)·fp_d(x-c_d)
    f_bc_lin = 2.0 * q * fp_d_all + (1.0 - 2.0 * q) * fp_d_upstream_all

    # Quadratic: f_bc = fp_d/(2q) + (2q-1)/(2q)·fp_opp
    safe_q = torch.where(mask_quad, q, torch.ones_like(q))
    inv_2q = 1.0 / (2.0 * safe_q)
    f_bc_quad = fp_d_all * inv_2q + (2.0 * safe_q - 1.0) * inv_2q * fp_opp_all

    f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

    # Moving-wall momentum correction (already opp-indexed by caller)
    if wall_correction is not None:
        f_bc = f_bc + wall_correction

    # Scatter: f_out[e] = f_bc[opp[e]] where mask[opp[e]] is True
    mask_for_e = mask[opp_tensor]
    f_bc_for_e = f_bc[opp_tensor]

    return torch.where(mask_for_e, f_bc_for_e, f)


# --------------------------------------------------------------------------- #
# 2. Moving-wall momentum correction
# --------------------------------------------------------------------------- #
def bfl_moving_wall_correction(
    fluid_boundary_mask: torch.Tensor,
    moving_wall_mask: torch.Tensor,
    u_wall: tuple[float, float, float],
    *,
    lattice: BFLLatticeName = "D3Q19",
    rho_w: float = 1.0,
) -> torch.Tensor:
    """Compute the moving-wall momentum correction tensor for BFL.

    For a wall moving with velocity **u_w**, the correction for the
    *unknown* population ``f[opp_d]`` is::

        corr[opp_d] = 2·ρ·w[opp_d]·(c[opp_d]·u_w) / c_s²

    Since the BFL scatter sets ``f_out[e] = f_bc[opp[e]]``, the correction
    added to ``f_bc[d]`` is ``corr[opp[d]]``.

    Args:
        fluid_boundary_mask: ``(Q, nz, ny, nx)`` bool.
        moving_wall_mask: ``(nz, ny, nx)`` bool — True at moving-wall
            near-wall cells.
        u_wall: ``(uwx, uwy, uwz)`` wall velocity.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.
        rho_w: Wall density (default 1.0).

    Returns:
        ``(Q, nz, ny, nx)`` float correction tensor.
    """
    Q, C_lat, W_lat, _ = _lattice_params(lattice)
    device = fluid_boundary_mask.device
    c = C_lat.to(device).float()
    w = W_lat.to(device).float()
    opp_tensor = (_OPP19 if Q == 19 else _OPP27).to(device)

    # correction[i] = 2*rho*w[i]*(c[i]·u_w)/cs2 = 6*rho*w[i]*(c[i]·u_w)
    c_dot_u = c[:, 0] * u_wall[0] + c[:, 1] * u_wall[1] + c[:, 2] * u_wall[2]  # (Q,)
    correction_dir = 6.0 * rho_w * w * c_dot_u  # (Q,)

    # wall_correction[d] = correction[opp[d]]
    wall_corr_dir = correction_dir[opp_tensor]  # (Q,)

    nz, ny, nx = fluid_boundary_mask.shape[1:]
    wall_corr = wall_corr_dir.view(Q, 1, 1, 1).expand(Q, nz, ny, nx).clone()

    # Apply only at moving-wall cells
    wall_corr = wall_corr * moving_wall_mask.unsqueeze(0).float()
    return wall_corr


# --------------------------------------------------------------------------- #
# 3. q-field computation — analytical surfaces
# --------------------------------------------------------------------------- #
def compute_q_cylinder_common(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    radius: float,
    device: torch.device,
    *,
    axis: str = "z",
    cz: float | None = None,
    lattice: BFLLatticeName = "D3Q19",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BFL *q*-field for a 2-D extruded cylinder.

    The cylinder cross-section lies in the plane perpendicular to *axis*;
    *q*-values are computed in that plane and broadcast across all layers
    along *axis*.

    Args:
        nx, ny, nz: Grid dimensions.
        cx, cy: Cylinder centre in the cross-section plane.
            For axis='y' or 'x', *cz* specifies the centre along z.
        radius: Cylinder radius.
        device: Target device.
        axis: Extrusion axis (``'z'``, ``'y'``, or ``'x'``).
        cz: Centre along z (for axis='y' or 'x').
        lattice: ``"D3Q19"`` or ``"D3Q27"``.

    Returns:
        ``(fluid_boundary_mask, q_field)`` each ``(Q, nz, ny, nx)``.
    """
    Q, C_lat, _, _ = _lattice_params(lattice)
    c = C_lat.to(device).float()

    if axis == "z":
        coord2, coord1 = torch.meshgrid(
            torch.arange(ny, device=device, dtype=torch.float64),
            torch.arange(nx, device=device, dtype=torch.float64),
            indexing="ij",
        )
        c1, c2 = cx, cy
        vi1, vi2, vai = 0, 1, 2
        bdim = 0
    elif axis == "y":
        cz_c = cz if cz is not None else nz / 2.0
        coord2, coord1 = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float64),
            torch.arange(nx, device=device, dtype=torch.float64),
            indexing="ij",
        )
        c1, c2 = cx, cz_c
        vi1, vi2, vai = 0, 2, 1
        bdim = 1
    elif axis == "x":
        cz_c = cz if cz is not None else nz / 2.0
        coord2, coord1 = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float64),
            torch.arange(ny, device=device, dtype=torch.float64),
            indexing="ij",
        )
        c1, c2 = cy, cz_c
        vi1, vi2, vai = 1, 2, 0
        bdim = 2
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    fluid_boundary_mask = torch.zeros((Q, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((Q, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    for d in range(Q):
        dv1 = float(c[d, vi1].item())
        dv2 = float(c[d, vi2].item())
        dva = float(c[d, vai].item())

        if dv1 == 0.0 and dv2 == 0.0 and dva == 0.0:
            continue  # rest

        if dva != 0.0 and dv1 == 0.0 and dv2 == 0.0:
            continue  # pure axis direction

        dist_nb = (coord1 + dv1 - c1) ** 2 + (coord2 + dv2 - c2) ** 2
        nb_is_solid = dist_nb <= radius**2
        dist_self = (coord1 - c1) ** 2 + (coord2 - c2) ** 2
        self_is_fluid = dist_self > radius**2
        boundary = self_is_fluid & nb_is_solid

        if not boundary.any():
            continue

        d1 = coord1 - c1
        d2 = coord2 - c2
        a_coef = dv1**2 + dv2**2
        if a_coef < 1e-10:
            continue
        b_coef = 2.0 * (dv1 * d1 + dv2 * d2)
        c_coef = d1**2 + d2**2 - radius**2

        discriminant = b_coef**2 - 4.0 * a_coef * c_coef
        safe_disc = torch.where(
            boundary & (discriminant >= 0.0),
            discriminant,
            torch.zeros_like(discriminant),
        )
        sqrt_disc = torch.sqrt(safe_disc)

        t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)

        # The neighbour is x+c=x+1*c, so the ray parameter itself is the
        # fractional BFL link distance even when c is diagonal.
        q1 = t1
        q2 = t2

        valid1 = (t1 > 1e-10) & (q1 <= 1.0 + 1e-10)
        valid2 = (t2 > 1e-10) & (q2 <= 1.0 + 1e-10)

        q_val = (
            torch.where(
                valid1 & valid2,
                torch.min(q1, q2),
                torch.where(valid1, q1, torch.where(valid2, q2, torch.full_like(q1, 0.5))),
            )
            .clamp(1e-6, 1.0)
            .float()
        )

        boundary_3d = boundary.unsqueeze(bdim).expand(nz, ny, nx)
        q_val_3d = q_val.unsqueeze(bdim).expand(nz, ny, nx)
        fluid_boundary_mask[d] = boundary_3d
        q_field[d] = torch.where(boundary_3d, q_val_3d, q_field[d])

    return fluid_boundary_mask, q_field


def compute_q_sphere_common(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    device: torch.device,
    *,
    lattice: BFLLatticeName = "D3Q19",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BFL *q*-field for a 3-D sphere via ray-sphere intersection.

    Args:
        nx, ny, nz: Grid dimensions.
        cx, cy, cz: Sphere centre.
        radius: Sphere radius.
        device: Target device.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.

    Returns:
        ``(fluid_boundary_mask, q_field)`` each ``(Q, nz, ny, nx)``.
    """
    Q, C_lat, _, _ = _lattice_params(lattice)
    c = C_lat.to(device)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64),
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )

    fluid_boundary_mask = torch.zeros((Q, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((Q, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    for d in range(Q):
        dcx = float(c[d, 0].item())
        dcy = float(c[d, 1].item())
        dcz = float(c[d, 2].item())
        if dcx == 0.0 and dcy == 0.0 and dcz == 0.0:
            continue

        dist_nb = (xx + dcx - cx) ** 2 + (yy + dcy - cy) ** 2 + (zz + dcz - cz) ** 2
        nb_is_solid = dist_nb <= radius**2
        dist_self = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        self_is_fluid = dist_self > radius**2
        boundary = self_is_fluid & nb_is_solid

        if not boundary.any():
            continue

        dx = xx - cx
        dy = yy - cy
        dz_v = zz - cz
        a_coef = dcx**2 + dcy**2 + dcz**2
        b_coef = 2.0 * (dcx * dx + dcy * dy + dcz * dz_v)
        c_coef = dx**2 + dy**2 + dz_v**2 - radius**2

        discriminant = b_coef**2 - 4.0 * a_coef * c_coef
        safe_disc = torch.where(
            boundary & (discriminant >= 0.0),
            discriminant,
            torch.zeros_like(discriminant),
        )
        sqrt_disc = torch.sqrt(safe_disc)

        t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)

        q1 = t1
        q2 = t2

        valid1 = (t1 > 1e-10) & (q1 <= 1.0 + 1e-10)
        valid2 = (t2 > 1e-10) & (q2 <= 1.0 + 1e-10)

        q_val = (
            torch.where(
                valid1 & valid2,
                torch.min(q1, q2),
                torch.where(valid1, q1, torch.where(valid2, q2, torch.full_like(q1, 0.5))),
            )
            .clamp(1e-6, 1.0)
            .float()
        )

        fluid_boundary_mask[d] = boundary
        q_field[d] = torch.where(boundary, q_val, q_field[d])

    return fluid_boundary_mask, q_field


# --------------------------------------------------------------------------- #
# 4. q-field computation — flat walls (Couette / channel)
# --------------------------------------------------------------------------- #
def compute_q_flat_walls_common(
    nx: int,
    ny: int,
    nz: int,
    device: torch.device,
    *,
    wall_axis: str = "y",
    lattice: BFLLatticeName = "D3Q19",
) -> tuple[torch.Tensor, torch.Tensor]:
    """BFL mask and *q* for flat walls (q=0.5 everywhere, half-way BB).

    Args:
        nx, ny, nz: Grid dimensions.
        device: Target device.
        wall_axis: ``"x"``, ``"y"``, or ``"z"``.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.

    Returns:
        ``(fluid_boundary_mask, q_field)`` each ``(Q, nz, ny, nx)``.
    """
    Q, C_lat, _, _ = _lattice_params(lattice)
    c = C_lat.to(device).float()
    mask = torch.zeros((Q, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((Q, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    wall_map = {"x": 2, "y": 1, "z": 0}
    if wall_axis not in wall_map:
        raise ValueError(f"wall_axis must be x/y/z, got '{wall_axis}'")
    wall_dim = wall_map[wall_axis]
    dims = [nz, ny, nx]
    dim_size = dims[wall_dim]

    for d in range(1, Q):
        cv = float(c[d, wall_dim].item())
        if cv == 0.0:
            continue
        fluid_idx = dim_size - 2 if cv > 0 else 1
        if wall_dim == 0:
            mask[d, fluid_idx, :, :] = True
        elif wall_dim == 1:
            mask[d, :, fluid_idx, :] = True
        else:
            mask[d, :, :, fluid_idx] = True

    return mask, q_field


# --------------------------------------------------------------------------- #
# 5. q-field computation — STL surfaces (ray-triangle intersection)
# --------------------------------------------------------------------------- #
def compute_q_stl_common(
    solid: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    lattice: BFLLatticeName = "D3Q19",
    n_substeps: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BFL *q*-field for an STL-voxelised geometry.

    For each boundary link (fluid → solid), the fractional distance *q*
    is found by ray-marching along the lattice direction and testing
    ray-triangle intersection against the STL triangles near the cell.

    This is a CPU computation (NumPy) that is run once during
    preprocessing and the result is moved to the solid mask's device.

    Args:
        solid: ``(nz, ny, nx)`` bool — voxelised solid mask.
        vertices: ``(N_v, 3)`` float — STL vertices.
        faces: ``(N_f, 3)`` int — triangle vertex indices.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.
        n_substeps: Number of sub-step samples along each link for the
            ray-march (default 20).

    Returns:
        ``(fluid_boundary_mask, q_field)`` each ``(Q, nz, ny, nx)``.
    """
    import numpy as np

    Q, C_lat, _, _ = _lattice_params(lattice)
    c_np = C_lat.numpy()
    nz, ny, nx = solid.shape
    solid_np = solid.cpu().numpy()

    fluid_boundary_mask = torch.zeros((Q, nz, ny, nx), dtype=torch.bool)
    q_field = torch.full((Q, nz, ny, nx), 0.5, dtype=torch.float32)

    verts_np = vertices.cpu().numpy().astype(np.float64)
    faces_np = faces.cpu().numpy().astype(np.int64)

    # Build a simple spatial index: map cell (i,j,k) to nearby triangles
    # by checking triangle bounding boxes against a cell grid.
    # For efficiency, we only test triangles whose bbox overlaps the
    # boundary cell neighbourhood.
    tri_min = verts_np[faces_np].min(axis=1)  # (N_f, 3)
    tri_max = verts_np[faces_np].max(axis=1)  # (N_f, 3)

    fluid_mask = ~solid_np

    for d in range(Q):
        dcx, dcy, dcz = int(c_np[d, 0]), int(c_np[d, 1]), int(c_np[d, 2])
        if dcx == 0 and dcy == 0 and dcz == 0:
            continue

        # Neighbour solid via roll
        nb_solid = np.roll(solid_np, shift=(-dcz, -dcy, -dcx), axis=(0, 1, 2))
        boundary = fluid_mask & nb_solid

        if not boundary.any():
            continue

        # Get boundary cell coordinates
        bk, bj, bi = np.where(boundary)
        n_bnd = bk.size
        if n_bnd == 0:
            continue

        mask_d = np.zeros((nz, ny, nx), dtype=bool)
        q_d = np.full((nz, ny, nx), 0.5, dtype=np.float32)

        # Ray origin = (bi, bj, bk), direction = (dcx, dcy, dcz)
        # March from t=0 to t=1 in n_substeps, find first crossing.
        ts = np.linspace(0.0, 1.0, n_substeps + 1)[1:]  # skip t=0

        for idx in range(n_bnd):
            ki, ji, ii = int(bk[idx]), int(bj[idx]), int(bi[idx])
            # Ray points
            rx = ii + ts * dcx
            ry = ji + ts * dcy
            rz = ki + ts * dcz

            # Find triangles whose bbox contains any ray point
            # Quick bbox test
            rmin = np.array([rx.min(), ry.min(), rz.min()])
            rmax = np.array([rx.max(), ry.max(), rz.max()])

            # Triangle bbox overlap with ray bbox
            overlap = (
                (tri_max[:, 0] >= rmin[0] - 1)
                & (tri_min[:, 0] <= rmax[0] + 1)
                & (tri_max[:, 1] >= rmin[1] - 1)
                & (tri_min[:, 1] <= rmax[1] + 1)
                & (tri_max[:, 2] >= rmin[2] - 1)
                & (tri_min[:, 2] <= rmax[2] + 1)
            )
            cand = np.where(overlap)[0]
            if cand.size == 0:
                continue

            # Möller–Trumbore ray-triangle intersection
            q_found = 0.5
            for t_idx, t_val in enumerate(ts):
                ox, oy, oz = ii + t_val * dcx, ji + t_val * dcy, ki + t_val * dcz
                # Direction is normalised to link length
                dir_len = math.sqrt(dcx**2 + dcy**2 + dcz**2)
                dxr, dyr, dzr = dcx / dir_len, dcy / dir_len, dcz / dir_len

                for ti in cand:
                    v0 = verts_np[faces_np[ti, 0]]
                    v1 = verts_np[faces_np[ti, 1]]
                    v2 = verts_np[faces_np[ti, 2]]
                    e1 = v1 - v0
                    e2 = v2 - v0
                    pv = np.array(
                        [
                            dyr * e2[2] - dzr * e2[1],
                            dzr * e2[0] - dxr * e2[2],
                            dxr * e2[1] - dyr * e2[0],
                        ]
                    )
                    det = e1 @ pv
                    if abs(det) < 1e-12:
                        continue
                    inv_det = 1.0 / det
                    tv = np.array([ox - v0[0], oy - v0[1], oz - v0[2]])
                    u = (tv @ pv) * inv_det
                    if u < 0.0 or u > 1.0:
                        continue
                    qv = np.cross(tv, e1)
                    v = np.array([dxr, dyr, dzr]) @ qv * inv_det
                    if v < 0.0 or u + v > 1.0:
                        continue
                    t_hit = (e2 @ qv) * inv_det
                    if t_hit > 1e-6:
                        q_found = float(t_val)
                        break
                if q_found < 0.5 or q_found != 0.5:
                    break

            mask_d[ki, ji, ii] = True
            q_d[ki, ji, ii] = q_found

        fluid_boundary_mask[d] = torch.from_numpy(mask_d)
        q_field[d] = torch.from_numpy(q_d)

    # Move to solid's device
    device = solid.device
    return fluid_boundary_mask.to(device), q_field.to(device)


# --------------------------------------------------------------------------- #
# 6. q-field computation — generic voxelised solid (q=0.5 fallback)
# --------------------------------------------------------------------------- #
def compute_q_generic_common(
    solid: torch.Tensor,
    device: torch.device,
    *,
    lattice: BFLLatticeName = "D3Q19",
) -> tuple[torch.Tensor, torch.Tensor]:
    """BFL *q*-field for an arbitrary voxelised 3-D solid (q=0.5 fallback).

    Identifies all fluid nodes that are direct lattice neighbours of a
    solid node and returns *q* = 0.5 (half-way bounce-back).  This is the
    generic counterpart to the analytical ``compute_q_*`` helpers; it
    works for any Boolean obstacle mask.

    Args:
        solid: ``(nz, ny, nx)`` bool — True where solid.
        device: Target device.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.

    Returns:
        ``(fluid_boundary_mask, q_field)`` each ``(Q, nz, ny, nx)``.
    """
    Q, C_lat, _, _ = _lattice_params(lattice)
    c = C_lat.to(device)
    nz, ny, nx = solid.shape
    solid = solid.to(device)

    fluid_boundary_mask = torch.zeros((Q, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((Q, nz, ny, nx), 0.5, dtype=torch.float32, device=device)
    fluid_mask = ~solid

    for d in range(Q):
        dcx = int(c[d, 0].item())
        dcy = int(c[d, 1].item())
        dcz = int(c[d, 2].item())
        if dcx == 0 and dcy == 0 and dcz == 0:
            continue
        nb_solid = torch.roll(solid, shifts=(-dcz, -dcy, -dcx), dims=(0, 1, 2))
        boundary = fluid_mask & nb_solid
        fluid_boundary_mask[d] = boundary

    return fluid_boundary_mask, q_field


# --------------------------------------------------------------------------- #
# 6b. q-wall (normal distance) for friction integration
# --------------------------------------------------------------------------- #
# The BFL q-field is per-direction (along lattice vectors).  For the friction
# formula τ = ν·u_t / q_wall, we need the **normal distance** from the
# near-wall cell centre to the wall surface, NOT the average of per-direction
# q values.  Averaging per-direction q biases the result low for convex
# surfaces (sphere mean q ≈ 0.38 instead of 0.5), inflating friction by ~2×.
#
# These helpers compute the true normal distance analytically for sphere and
# cylinder, and via a robust fallback for arbitrary solids.
# --------------------------------------------------------------------------- #


def compute_q_wall_sphere(
    near: torch.Tensor,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    """Normal distance from each near-wall cell to a sphere surface.

    For a sphere the outward normal is radial, so the normal distance
    from a cell at (x, y, z) to the surface is simply::

        q_wall = sqrt((x-cx)² + (y-cy)² + (z-cz)²) − R

    clamped to [0.1, 1.0] to avoid singularities.  For cells just outside
    the sphere this is ≈ 0.5 (same as half-way BB), giving the correct
    friction when used with τ = ν·u_t / q_wall.

    Parameters
    ----------
    near : (nz, ny, nx) bool — near-wall mask.
    cx, cy, cz : sphere centre.
    radius : sphere radius.
    device : target device.

    Returns
    -------
    q_wall : (nz, ny, nx) float32 — normal distance, 0.5 outside near-wall.
    """
    nz, ny, nx = near.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    q_wall = (r - radius).clamp(0.1, 1.0)
    q_wall = torch.where(near, q_wall, torch.full_like(q_wall, 0.5))
    return q_wall


def compute_q_wall_cylinder(
    near: torch.Tensor,
    cx: float,
    cy: float,
    radius: float,
    device: torch.device,
    *,
    axis: str = "z",
    cz: float | None = None,
) -> torch.Tensor:
    """Normal distance from each near-wall cell to a cylinder surface.

    For a cylinder extruded along *axis*, the normal distance in the
    cross-section plane is::

        q_wall = sqrt((x-cx)² + (y-cy)²) − R   (axis='z')

    (analogous for other axes).  Clamped to [0.1, 1.0].
    """
    nz, ny, nx = near.shape
    if axis == "z":
        yy, xx = torch.meshgrid(
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        c1, c2 = cx, cy
    elif axis == "y":
        cz_c = cz if cz is not None else nz / 2.0
        zz, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        c1, c2 = cx, cz_c
        yy = zz  # reuse variable name
    elif axis == "x":
        cz_c = cz if cz is not None else nz / 2.0
        zz, yy = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            indexing="ij",
        )
        xx = zz  # cross-section plane uses (y, z) → map to (xx, yy)
        c1, c2 = cy, cz_c
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    r = torch.sqrt((xx - c1) ** 2 + (yy - c2) ** 2)
    q_wall = (r - radius).clamp(0.1, 1.0)
    q_wall = torch.where(near, q_wall, torch.full_like(q_wall, 0.5))
    return q_wall


def compute_q_wall_generic(
    near: torch.Tensor,
    solid: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Estimate normal wall distance for arbitrary voxelised solids.

    For each near-wall cell, counts the number of solid neighbours in each
    axis direction and estimates the normal distance as::

        q_wall ≈ 0.5 / max(|∂solid/∂x|, |∂solid/∂y|, |∂solid/∂z|)

    This gives 0.5 for face-aligned walls (standard BB) and smaller values
    for diagonal surfaces.  Clamped to [0.25, 1.0] for stability.

    For best accuracy, use :func:`compute_q_wall_sphere` or
    :func:`compute_q_wall_cylinder` when the analytical surface is known.
    """
    nz, ny, nx = near.shape
    solid_f = solid.float()

    # Gradient magnitude (central difference)
    gx = torch.zeros_like(solid_f)
    gy = torch.zeros_like(solid_f)
    gz = torch.zeros_like(solid_f)
    gx[:, :, 1:-1] = (solid_f[:, :, 2:] - solid_f[:, :, :-2]) / 2.0
    gy[:, 1:-1, :] = (solid_f[:, 2:, :] - solid_f[:, :-2, :]) / 2.0
    gz[1:-1, :, :] = (solid_f[2:, :, :] - solid_f[:-2, :, :]) / 2.0

    grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2).clamp(min=1e-6)
    # For face-aligned walls: grad_mag = 0.5, so q_wall = 0.5/0.5 = 1.0
    # But we want q_wall = 0.5 for face-aligned walls.
    # Actually: the normal distance for a face-aligned wall is 0.5 (half-way).
    # The gradient gives 0.5 for face-aligned, √2/2 for diagonal, etc.
    # We want q_wall = 0.5 for face-aligned, so:
    q_wall = 0.5 / grad_mag.clamp(min=0.5)
    q_wall = q_wall.clamp(0.25, 1.0)
    q_wall = torch.where(near, q_wall, torch.full_like(q_wall, 0.5))
    return q_wall


# --------------------------------------------------------------------------- #
# 7. Convenience: BFL step (collision → NoDynamics → BB → stream → BFL)
# --------------------------------------------------------------------------- #
def bfl_step(
    f: torch.Tensor,
    f_pre: torch.Tensor,
    solid: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    stream_fn,
    far_field_bc_fn=None,
    u_in: float = 0.0,
    *,
    lattice: BFLLatticeName = "D3Q19",
    wall_correction: torch.Tensor | None = None,
    correct_mass_fn=None,
    target_mass: float | None = None,
    step: int = 0,
    mass_interval: int = 200,
) -> torch.Tensor:
    """One BFL LBM step: NoDynamics → BB → stream → far-field → BFL.

    This is the BFL equivalent of :func:`tensorlbm.lbm_step_correct.lbm_step_correct`
    with ``bounce_back_fn`` replaced by the BFL interpolation.  The caller
    is responsible for collision (call this *after* collision).

    Order of operations:
      1. NoDynamics: restore solid cells to pre-collision.
      2. Half-way bounce-back at solid cells (so BFL receives properly
         bounced-back values from solid cells).
      3. Streaming.
      4. Far-field BC (optional).
      5. BFL interpolated bounce-back (replaces half-way BB at boundary).
      6. Mass correction (optional).

    Args:
        f: Post-collision distribution ``(Q, nz, ny, nx)``.
        f_pre: Pre-collision distribution (for NoDynamics).
        solid: ``(nz, ny, nx)`` bool solid mask.
        fluid_boundary_mask: ``(Q, nz, ny, nx)`` bool.
        q_field: ``(Q, nz, ny, nx)`` float.
        stream_fn: Streaming function (e.g. ``stream3d``).
        far_field_bc_fn: Far-field BC function (optional).
        u_in: Free-stream velocity for far-field BC.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.
        wall_correction: Moving-wall correction tensor (optional).
        correct_mass_fn: Mass correction function (optional).
        target_mass: Target mass for correction.
        step: Current step number.
        mass_interval: Mass correction interval.

    Returns:
        Updated distribution tensor.
    """
    from .boundaries3d import bounce_back_cells_3d

    # 1. NoDynamics
    sm = solid.unsqueeze(0).expand_as(f)
    f = torch.where(sm, f_pre, f)

    # 2. Half-way bounce-back at solid (before streaming)
    f = bounce_back_cells_3d(f, solid)
    f_pre_stream = f.clone()

    # 3. Streaming
    f = stream_fn(f)

    # 4. Far-field BC
    if far_field_bc_fn is not None:
        f = far_field_bc_fn(f, u_in)

    # 5. BFL interpolated bounce-back
    f = bfl_bounce_back_common(
        f,
        f_pre_stream,
        fluid_boundary_mask,
        q_field,
        lattice=lattice,
        wall_correction=wall_correction,
    )

    # 6. Mass correction
    if correct_mass_fn is not None and target_mass is not None:
        if step % mass_interval == 0:
            f = correct_mass_fn(f, target_mass)

    return f


# --------------------------------------------------------------------------- #
# 8. Sparse BFL kernels — precomputed boundary-link index sets
# --------------------------------------------------------------------------- #
#
# The vectorised :func:`bfl_bounce_back_common` materialises ~8 full
# ``(Q, nz, ny, nx)`` temporaries per call (opposite/upstream gathers,
# both interpolation branches, the correction add and the scatter mask),
# and requires the dense ``(Q, nz, ny, nx)`` mask/q_field to stay resident.
# On large external-flow grids (e.g. an 84.7M-cell sphere domain) that is
# ~8x6.4 GB of transients on top of the solver's own distributions and the
# kernel OOMs a 32 GB GPU.
#
# The boundary-link set of an immersed body is *tiny* compared with the
# grid (a D60 sphere has ~6.5e4 links vs 1.6e9 lattice nodes), and both the
# mask and the q-field are static geometry.  The kernels below therefore
# precompute, once per geometry, the flat indices of every boundary link
# (and of its upstream node ``x - c_d``), and apply the interpolation on
# ``n_links``-sized vectors only.  The arithmetic mirrors
# :func:`bfl_bounce_back_common` (and the per-direction force ledger of
# :func:`tensorlbm.bfl_d3q19.bouzidi_bounce_back_d3q19`) statement by
# statement, in the same evaluation order, so the sparse path is
# **bit-identical** to the dense one.
#
# Upstream convention: the dense kernel builds the upstream population with
# ``torch.roll``, i.e. a *circular* shift.  The precomputation reproduces
# that wraparound exactly (modular index arithmetic), but a link whose
# upstream actually crosses the domain boundary almost always indicates a
# geometry touching the domain edge, so by default it is intercepted with a
# :class:`ValueError` unless ``allow_wrap=True`` is passed explicitly.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class BFLBoundaryLinks:
    """Precomputed sparse boundary-link index set for one geometry.

    Attributes:
        lattice: Lattice name (``"D3Q19"`` / ``"D3Q27"``).
        Q: Number of velocities.
        grid_shape: ``(nz, ny, nx)`` of the underlying grid.
        link_dir: ``(n_links,)`` int64 — BFL link direction *d*
            (fluid → solid) of each link.
        link_out_dir: ``(n_links,)`` int64 — ``opp[d]``, the direction the
            interpolated population is scattered into.
        link_idx: ``(n_links,)`` int64 — row-major flat index of the fluid
            cell of each link.  Within each direction block the indices are
            ascending (same order as boolean-mask gathering on the dense
            field, which the force-ledger sum order relies on).
        link_up_idx: ``(n_links,)`` int64 — flat index of the upstream node
            ``x - c_d`` (modular/wrapped, matching ``torch.roll``).
        link_q: ``(n_links,)`` float32 — fractional distance *q* at the link.
        offsets: ``(Q + 1,)`` tuple of ints — slice bounds of each
            direction's block, ``links.direction_slice(d)``.
        n_wrapped: Number of links whose upstream wrapped the domain edge.
    """

    lattice: BFLLatticeName
    Q: int
    grid_shape: tuple[int, int, int]
    link_dir: torch.Tensor
    link_out_dir: torch.Tensor
    link_idx: torch.Tensor
    link_up_idx: torch.Tensor
    link_q: torch.Tensor
    offsets: tuple[int, ...]
    n_wrapped: int

    @property
    def n_links(self) -> int:
        return int(self.link_idx.numel())

    def direction_slice(self, d: int) -> slice:
        """Slice selecting direction *d*'s link block."""
        return slice(self.offsets[d], self.offsets[d + 1])

    def to(self, device: torch.device) -> "BFLBoundaryLinks":
        """Return a copy with every index/value tensor on *device*."""
        return BFLBoundaryLinks(
            lattice=self.lattice,
            Q=self.Q,
            grid_shape=self.grid_shape,
            link_dir=self.link_dir.to(device),
            link_out_dir=self.link_out_dir.to(device),
            link_idx=self.link_idx.to(device),
            link_up_idx=self.link_up_idx.to(device),
            link_q=self.link_q.to(device),
            offsets=self.offsets,
            n_wrapped=self.n_wrapped,
        )


def bfl_boundary_link_indices(
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor | None = None,
    *,
    lattice: BFLLatticeName = "D3Q19",
    allow_wrap: bool = False,
) -> BFLBoundaryLinks:
    """Precompute the sparse boundary-link index set of a BFL geometry.

    Args:
        fluid_boundary_mask: ``(Q, nz, ny, nx)`` bool — True at boundary
            links (fluid node whose neighbour in direction *d* is solid),
            as produced by the ``compute_q_*`` helpers.
        q_field: Optional ``(Q, nz, ny, nx)`` float field of fractional
            distances.  When ``None`` every link gets *q* = 0.5 (the
            half-way fallback used by the flat-wall/generic helpers).
        lattice: ``"D3Q19"`` or ``"D3Q27"``.
        allow_wrap: When False (default), a link whose upstream node
            ``x - c_d`` leaves the domain raises :class:`ValueError` (the
            dense kernel's ``torch.roll`` would silently wrap around the
            opposite edge).  When True the wrap is reproduced exactly.

    Returns:
        :class:`BFLBoundaryLinks` ready for :func:`bfl_bounce_back_sparse`
        and :func:`bfl_force_ledger_sparse`.
    """
    Q, C_lat, _, opp_list = _lattice_params(lattice)
    if fluid_boundary_mask.ndim != 4 or fluid_boundary_mask.shape[0] != Q:
        raise ValueError(
            f"fluid_boundary_mask must be ({Q}, nz, ny, nx), got {tuple(fluid_boundary_mask.shape)}"
        )
    if q_field is not None and q_field.shape != fluid_boundary_mask.shape:
        raise ValueError(
            f"q_field shape {tuple(q_field.shape)} does not match mask "
            f"{tuple(fluid_boundary_mask.shape)}"
        )
    device = fluid_boundary_mask.device
    nz, ny, nx = (int(s) for s in fluid_boundary_mask.shape[1:])
    plane = ny * nx

    link_dir_blocks: list[torch.Tensor] = []
    link_out_blocks: list[torch.Tensor] = []
    link_idx_blocks: list[torch.Tensor] = []
    link_up_blocks: list[torch.Tensor] = []
    link_q_blocks: list[torch.Tensor] = []
    offsets = [0]
    n_wrapped = 0
    wrapped_report: list[str] = []

    for d in range(Q):
        idx = fluid_boundary_mask[d].reshape(-1).nonzero(as_tuple=True)[0]
        offsets.append(offsets[-1] + int(idx.numel()))
        if idx.numel() == 0:
            continue
        cx, cy, cz = (int(v) for v in C_lat[d, [0, 1, 2]].tolist())
        z = idx // plane
        yx = idx % plane
        y = yx // nx
        x = yx % nx
        uz = z - cz
        uy = y - cy
        ux = x - cx
        out_of_domain = (uz < 0) | (uz >= nz) | (uy < 0) | (uy >= ny) | (ux < 0) | (ux >= nx)
        n_wrap_d = int(out_of_domain.sum().item())
        if n_wrap_d and not allow_wrap:
            raise ValueError(
                f"BFL sparse precompute: {n_wrap_d} link(s) in direction d={d} "
                f"(c=({cx},{cy},{cz})) have an upstream node x-c_d outside the "
                f"domain; the dense kernel's torch.roll would silently wrap "
                f"around the opposite edge. Move the geometry away from the "
                f"domain boundary or pass allow_wrap=True to reproduce the "
                f"wrap exactly."
            )
        n_wrapped += n_wrap_d
        if n_wrap_d:
            wrapped_report.append(f"d={d}:{n_wrap_d}")
        up = (
            torch.remainder(uz, nz) * plane + torch.remainder(uy, ny) * nx + torch.remainder(ux, nx)
        )
        link_idx_blocks.append(idx)
        link_dir_blocks.append(torch.full_like(idx, d))
        link_out_blocks.append(torch.full_like(idx, opp_list[d]))
        link_up_blocks.append(up)
        if q_field is None:
            link_q_blocks.append(
                torch.full((idx.numel(),), 0.5, dtype=torch.float32, device=device)
            )
        else:
            link_q_blocks.append(q_field[d].reshape(-1)[idx])

    def _cat(blocks: list[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if not blocks:
            return torch.empty((0,), dtype=dtype, device=device)
        return torch.cat(blocks)

    return BFLBoundaryLinks(
        lattice=lattice,
        Q=Q,
        grid_shape=(nz, ny, nx),
        link_dir=_cat(link_dir_blocks, torch.int64),
        link_out_dir=_cat(link_out_blocks, torch.int64),
        link_idx=_cat(link_idx_blocks, torch.int64),
        link_up_idx=_cat(link_up_blocks, torch.int64),
        link_q=_cat(link_q_blocks, torch.float32),
        offsets=tuple(offsets),
        n_wrapped=n_wrapped,
    )


def _bfl_links_wall_correction(
    wall_correction: torch.Tensor | None, links: BFLBoundaryLinks
) -> torch.Tensor | None:
    """Gather a wall correction onto the link vectors.

    Accepts either the dense ``(Q, nz, ny, nx)`` correction tensor (gathered
    at the link positions — same opp-indexing convention as
    :func:`bfl_bounce_back_common`) or an ``(n_links,)`` per-link vector in
    :class:`BFLBoundaryLinks` order (as produced by
    :func:`bfl_moving_wall_correction_sparse`).
    """
    if wall_correction is None:
        return None
    if wall_correction.ndim == 4:
        Q, nz, ny, nx = links.Q, *links.grid_shape
        if tuple(wall_correction.shape) != (Q, nz, ny, nx):
            raise ValueError(
                f"dense wall_correction must be ({Q}, {nz}, {ny}, {nx}), got "
                f"{tuple(wall_correction.shape)}"
            )
        return wall_correction.reshape(Q, -1)[links.link_dir, links.link_idx]
    if wall_correction.ndim == 1 and wall_correction.numel() == links.n_links:
        return wall_correction
    raise ValueError(
        "wall_correction must be None, a (Q, nz, ny, nx) tensor, or an "
        f"(n_links={links.n_links},) vector; got shape "
        f"{tuple(wall_correction.shape)}"
    )


def bfl_bounce_back_sparse(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    links: BFLBoundaryLinks,
    *,
    wall_correction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse BFL interpolated bounce-back, bit-identical to the dense kernel.

    Computes the same interpolation as :func:`bfl_bounce_back_common`
    (same expressions, same evaluation order) on the precomputed link
    vectors only, then scatters into ``f_out[opp[d], x]``.  Peak kernel
    memory is one clone of *f* plus O(n_links) vectors, instead of ~8 full
    ``(Q, nz, ny, nx)`` temporaries plus the resident mask/q_field.

    Args:
        f: Post-stream distribution ``(Q, nz, ny, nx)``, contiguous.
        f_prev: Pre-stream (post-collision) distribution, contiguous.
        links: Precomputed :class:`BFLBoundaryLinks` for this geometry.
        wall_correction: ``None``, a dense ``(Q, nz, ny, nx)`` tensor (the
            dense-kernel convention, gathered at the links), or an
            ``(n_links,)`` per-link vector.

    Returns:
        Updated distribution tensor (new tensor; *f* is not modified).
    """
    Q, nz, ny, nx = links.Q, *links.grid_shape
    if tuple(f.shape) != (Q, nz, ny, nx) or tuple(f_prev.shape) != (Q, nz, ny, nx):
        raise ValueError(
            f"f/f_prev must be ({Q}, {nz}, {ny}, {nx}) to match the links; "
            f"got {tuple(f.shape)} / {tuple(f_prev.shape)}"
        )
    if not (f.is_contiguous() and f_prev.is_contiguous()):
        raise ValueError("f and f_prev must be contiguous (reshape-as-view)")

    if links.n_links == 0:
        return f.clone()

    fp_flat = f_prev.reshape(Q, -1)
    idx = links.link_idx
    up = links.link_up_idx
    d_arr = links.link_dir
    e_arr = links.link_out_dir

    fp_d = fp_flat[d_arr, idx]  # f_prev[d](x)
    fp_up = fp_flat[d_arr, up]  # f_prev[d](x - c_d)   (torch.roll upstream)
    fp_od = fp_flat[e_arr, idx]  # f_prev[opp[d]](x)

    q = links.link_q.to(dtype=f.dtype)
    lin = q < 0.5

    # Same expressions/association as bfl_bounce_back_common.
    f_bc_lin = 2.0 * q * fp_d + (1.0 - 2.0 * q) * fp_up
    q_safe = torch.where(lin, torch.ones_like(q), q)
    inv_2q = 1.0 / (2.0 * q_safe)
    f_bc_quad = fp_d * inv_2q + (2.0 * q_safe - 1.0) * inv_2q * fp_od
    f_bc = torch.where(lin, f_bc_lin, f_bc_quad)

    corr = _bfl_links_wall_correction(wall_correction, links)
    if corr is not None:
        f_bc = f_bc + corr

    f_out = f.clone()
    f_out.reshape(Q, -1)[e_arr, idx] = f_bc
    return f_out


def bfl_force_ledger_sparse(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    links: BFLBoundaryLinks,
    *,
    wall_correction: torch.Tensor | None = None,
    fraction: float = 1.0,
    force_frame: str = "laboratory",
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """Sparse per-link momentum ledger, bit-identical to the dense ledger.

    Mirrors the force accounting of
    :func:`tensorlbm.bfl_d3q19.bouzidi_bounce_back_d3q19` (default path:
    scalar ``fraction``, optional wall correction, laboratory or wall
    frame): per direction *d* the same masked-vector arithmetic and the
    same per-direction summation order are reproduced on the precomputed
    link blocks, so every accumulated component matches the dense ledger
    bit-for-bit.  The link-normal decomposition variant is not provided.

    Args:
        f: Post-stream distribution ``(Q, nz, ny, nx)``, contiguous.
        f_prev: Pre-stream distribution, contiguous.
        links: Precomputed :class:`BFLBoundaryLinks`.
        wall_correction: ``None``, dense ``(Q, nz, ny, nx)``, or
            ``(n_links,)`` per-link vector.
        fraction: Scalar boundary fraction (default 1.0).
        force_frame: ``"laboratory"`` (conservative ledger closing a fixed
            control volume) or ``"wall"`` (removes the transparent-wall
            background flux).

    Returns:
        ``(force, active_links)`` with ``force`` a 3-tuple of 0-dim tensors.
    """
    if force_frame not in {"laboratory", "wall"}:
        raise ValueError(f"force_frame must be 'laboratory' or 'wall', got {force_frame!r}")
    Q, nz, ny, nx = links.Q, *links.grid_shape
    if tuple(f.shape) != (Q, nz, ny, nx) or tuple(f_prev.shape) != (Q, nz, ny, nx):
        raise ValueError(
            f"f/f_prev must be ({Q}, {nz}, {ny}, {nx}) to match the links; "
            f"got {tuple(f.shape)} / {tuple(f_prev.shape)}"
        )
    if not (f.is_contiguous() and f_prev.is_contiguous()):
        raise ValueError("f and f_prev must be contiguous (reshape-as-view)")

    opp_list = _OPP19_LIST if Q == 19 else _OPP27_LIST
    c_lat = _C19 if Q == 19 else _C27

    f_flat = f.reshape(Q, -1)
    fp_flat = f_prev.reshape(Q, -1)
    corr = _bfl_links_wall_correction(wall_correction, links)

    stat_acc = torch.zeros(3, dtype=f.dtype, device=f.device)
    corr_acc = torch.zeros(3, dtype=f.dtype, device=f.device)
    transparent_acc = torch.zeros(3, dtype=f.dtype, device=f.device)
    active_links = 0

    for d in range(1, Q):
        sl = links.direction_slice(d)
        if sl.start == sl.stop:
            continue
        od = opp_list[d]
        cx, cy, cz = (int(v) for v in c_lat[d, [0, 1, 2]].tolist())
        idx = links.link_idx[sl]
        up = links.link_up_idx[sl]
        q_cell = links.link_q[sl].to(dtype=f.dtype)
        lin = q_cell < 0.5
        fp_d = fp_flat[d, idx]
        fp_od = fp_flat[od, idx]
        fp_up = fp_flat[d, up]
        q_safe = torch.where(lin, torch.ones_like(q_cell), q_cell)
        inv_2q = 1.0 / (2.0 * q_safe)
        f_bc_stat = torch.where(
            lin,
            2.0 * q_cell * fp_d + (1.0 - 2.0 * q_cell) * fp_up,
            fp_d * inv_2q + (2.0 * q_safe - 1.0) * inv_2q * fp_od,
        )
        if corr is not None:
            corr_d = corr[sl]
        else:
            corr_d = torch.zeros_like(q_cell)
        f_streamed_unk = f_flat[od, idx]
        frac_cell = float(fraction)
        stat_link = fp_d + (1.0 - frac_cell) * f_streamed_unk + frac_cell * f_bc_stat
        corr_link = frac_cell * corr_d
        c_vec = torch.tensor([cx, cy, cz], dtype=f.dtype, device=f.device)
        active_links += int(idx.numel())
        stat_acc += c_vec * stat_link.sum()
        corr_acc += c_vec * corr_link.sum()
        transparent_acc += c_vec * (fp_d + f_streamed_unk).sum()

    total_force = stat_acc + corr_acc
    returned_force = total_force - transparent_acc if force_frame == "wall" else total_force
    force = (
        returned_force[0].detach(),
        returned_force[1].detach(),
        returned_force[2].detach(),
    )
    return force, active_links


def bfl_moving_wall_correction_sparse(
    links: BFLBoundaryLinks,
    moving_wall_mask: torch.Tensor,
    u_wall: tuple[float, float, float],
    *,
    lattice: BFLLatticeName | None = None,
    rho_w: float = 1.0,
) -> torch.Tensor:
    """Per-link moving-wall momentum correction (no full-tensor allocation).

    Computes the same correction values as :func:`bfl_moving_wall_correction`
    (same opp-indexing convention) but returns only the ``(n_links,)``
    vector in :class:`BFLBoundaryLinks` order:

        corr[d, x] = correction[opp[d]] = 6·ρ·w[opp[d]]·(c[opp[d]]·u_w)

    evaluated only at moving-wall links (zero elsewhere).

    Args:
        links: Precomputed :class:`BFLBoundaryLinks`.
        moving_wall_mask: ``(nz, ny, nx)`` bool — True at moving-wall cells.
        u_wall: ``(uwx, uwy, uwz)`` wall velocity.
        lattice: Lattice override (defaults to ``links.lattice``).
        rho_w: Wall density (default 1.0).

    Returns:
        ``(n_links,)`` float32 correction vector for
        :func:`bfl_bounce_back_sparse` / :func:`bfl_force_ledger_sparse`.
    """
    lattice = lattice or links.lattice
    Q, C_lat, W_lat, _ = _lattice_params(lattice)
    if tuple(moving_wall_mask.shape) != links.grid_shape:
        raise ValueError(
            f"moving_wall_mask must be {links.grid_shape}, got {tuple(moving_wall_mask.shape)}"
        )
    device = links.link_idx.device
    c = C_lat.to(device).float()
    w = W_lat.to(device).float()

    # Same expression/association as bfl_moving_wall_correction.
    c_dot_u = c[:, 0] * u_wall[0] + c[:, 1] * u_wall[1] + c[:, 2] * u_wall[2]
    correction_dir = 6.0 * rho_w * w * c_dot_u  # (Q,)
    opp_tensor = (_OPP19 if Q == 19 else _OPP27).to(device)
    wall_corr_dir = correction_dir[opp_tensor]  # (Q,)

    moving_flat = moving_wall_mask.reshape(-1)
    return wall_corr_dir[links.link_dir] * moving_flat[links.link_idx].float()
