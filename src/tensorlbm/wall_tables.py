"""Indirect-addressing wall tables (MF-LBM style) for TensorLBM.

The repo already owns *one* half of the indirect-addressing machinery: the
sparse boundary-fluid-link (BFL) tables of :mod:`tensorlbm.bfl_common` (one row
per fluid->solid link, carrying the interpolation distance ``q``), which drive
the momentum bounce-back.  The second half -- the tables MF-LBM builds in
``Geometry_preprocessing.F90`` and consumes in ``Phase_gradient.F90`` -- is a
*node-indexed* pair of lists:

* ``solid_boundary_nodes``: every solid cell with at least one fluid neighbour,
  carrying the directions that point at a fluid cell and
  ``la_weight = sum of the D3Q19 weights of those directions``;
* ``fluid_boundary_nodes``: every fluid cell with at least one solid neighbour,
  carrying the smoothed-wall normal (and, in MF-LBM, the local contact angle).

Neither table stores a ``q``: MF-LBM never interpolates the momentum PDF.
Walls are handled implicitly by the AA-pattern full-way bounce-back (the
kernels are guarded by ``if(walls(i,j,k)==0)``, ``Kernel_multiphase.F90:44`` and
``:408``; there is no bounce-back routine anywhere in the tree).  The tables
exist so that per-step near-wall *field* operations (phase-field continuation
into the solid, gradient/normal correction, contact-angle rotation) touch only
``n_boundary`` cells instead of ``n_cells``.

This module transcribes those rules into TensorLBM with two deviations that are
documented and tested rather than silent:

1. the 18-slot fixed row ``neighbor_list(1:18)`` (int32 per slot, 72 B/node) is
   stored as a condensed CSR list (int8 direction ids + int64 offsets);
2. an extra link view (:class:`WallLinkTable`) is produced from the same
   traversal so the geometry can also feed the BFL momentum path -- the OSS
   table carries no link/gap information at all.

Provenance (lanl/MF-LBM @ 11be324a97fe6331b2271102da370460715355e8):

* types           ``Module.F90:89-103``
* D3Q19 weights   ``Module.F90:127`` (``w_equ(0:18)``)
* ISO8 coeffs     ``Module.F90:228``
* cell tagging    ``Geometry_preprocessing.F90:126-141``
                  (1 solid, 0 fluid, 2 solid-boundary, -1 fluid-boundary)
* wall smoothing  ``Geometry_preprocessing.F90:52`` (``i_max_iteration = 4``)
                  and ``:143-160`` (D3Q27-weighted 27-point passes)
* table build     ``Geometry_preprocessing.F90:196-217``
* wall normal     ``Geometry_preprocessing.F90:230-383`` (ISO8 shells)
* consumers       ``Phase_gradient.F90:14-28`` (phi), ``:86-107`` (cn),
                  ``:222-265`` (contact-angle rotation)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import torch

from tensorlbm.d3q19 import C, W_EXACT64

__all__ = [
    "MF_LBM_N_SMOOTH_ITERATIONS",
    "MF_LBM_ISO8_COEFFS",
    "SolidBoundaryTable",
    "FluidBoundaryTable",
    "WallLinkTable",
    "WallTables",
    "build_wall_tables",
    "iso8_wall_normal",
    "la_weight_extrapolate",
    "dense_la_weight_extrapolate",
    "table_footprint_bytes",
    "dense_footprint_bytes",
]

#: ``Geometry_preprocessing.F90:52`` -- ``i_max_iteration = 4`` smoothing passes.
MF_LBM_N_SMOOTH_ITERATIONS = 4

#: ``Module.F90:228`` -- ``ISO8(1:7)``, one coefficient per offset shell below.
MF_LBM_ISO8_COEFFS: tuple[float, ...] = (
    4.0 / 45.0,
    1.0 / 21.0,
    2.0 / 105.0,
    5.0 / 504.0,
    1.0 / 315.0,
    1.0 / 630.0,
    1.0 / 5040.0,
)

#: D3Q27 shell weights of the wall-indicator smoothing
#: (``Geometry_preprocessing.F90:143-160``; ``we(0:3)`` = 8/27, 2/27, 1/54, 1/216).
_D3Q27_SHELL_WEIGHTS: dict[int, float] = {
    0: 8.0 / 27.0,
    1: 2.0 / 27.0,
    2: 1.0 / 54.0,
    3: 1.0 / 216.0,
}

#: Offset shells, one representative per +-pair, ordered like
#: :data:`MF_LBM_ISO8_COEFFS` (``Geometry_preprocessing.F90:230-383``).
_ISO8_SHELLS: tuple[tuple[tuple[int, int, int], ...], ...] = (
    # (1,0,0) class -- 3 representatives
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    # (1,1,0) class -- 6
    ((1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)),
    # (1,1,1) class -- 4
    ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)),
    # (2,0,0) class -- 3
    ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
    # (2,1,0) class -- 12
    ((0, 1, 2), (0, 1, -2), (0, 2, 1), (0, 2, -1),
     (1, 0, 2), (1, 0, -2), (1, 2, 0), (1, -2, 0),
     (2, 0, 1), (2, 0, -1), (2, 1, 0), (2, -1, 0)),
    # (2,1,1) class -- 12
    ((1, 1, 2), (1, 1, -2), (1, -1, 2), (1, -1, -2),
     (1, 2, 1), (1, 2, -1), (1, -2, 1), (1, -2, -1),
     (2, 1, 1), (2, 1, -1), (2, -1, 1), (2, -1, -1)),
    # (2,2,0) class -- 6
    ((0, 2, 2), (0, 2, -2), (2, 0, 2), (2, 0, -2), (2, 2, 0), (2, -2, 0)),
)


def _flat_grid(shape: Sequence[int]) -> torch.Tensor:
    """Row-major flat index grid for ``shape = (n0, n1, n2)``."""
    n0, n1, n2 = (int(s) for s in shape)
    i0 = torch.arange(n0).view(-1, 1, 1)
    i1 = torch.arange(n1).view(1, -1, 1)
    i2 = torch.arange(n2).view(1, 1, -1)
    return (i0 * n1 + i1) * n2 + i2


def _shift(field: torch.Tensor, offset: Sequence[int], *, fill: float | bool) -> torch.Tensor:
    """``out[x] = field[x + offset]``; sites outside the domain take ``fill``."""
    o = tuple(int(v) for v in offset)
    if o == (0, 0, 0):
        return field
    if field.dtype == torch.bool:
        return _shift(field.to(torch.uint8), offset, fill=int(bool(fill))).to(torch.bool)
    pad = tuple(max(abs(v) for v in o) for _ in range(3))
    pads: list[int] = []
    for axis in reversed(range(3)):
        pads.extend((pad[axis], pad[axis]))
    padded = torch.nn.functional.pad(field, pads, mode="constant", value=float(fill))
    n0, n1, n2 = (int(s) for s in field.shape)
    o0, o1, o2 = o
    return padded[
        pad[0] + o0: pad[0] + o0 + n0,
        pad[1] + o1: pad[1] + o1 + n1,
        pad[2] + o2: pad[2] + o2 + n2,
    ]


@dataclass(frozen=True, eq=False)
class SolidBoundaryTable:
    """MF-LBM ``solid_boundary_nodes`` (``Module.F90:89-96``), CSR-condensed.

    Attributes:
        ijk: ``(N, 3)`` int64 ``(i, j, k)`` of each solid boundary cell.
        flat: ``(N,)`` int64 row-major flat index of those cells.
        nbr_dir: ``(n_nbr,)`` int8 direction ids (1..18, indexing as in
            :data:`tensorlbm.d3q19.C`) pointing at a *fluid* neighbour;
            condensed replacement of ``neighbor_list(1:18)``.
        nbr_offset: ``(N + 1,)`` int64 CSR offsets into ``nbr_dir``.
        n_fluid: ``(N,)`` int64 ``i_fluid_num`` (number of listed neighbours).
        la_weight: ``(N,)`` float64 ``sum w_equ(n)`` over the listed directions.
    """

    ijk: torch.Tensor
    flat: torch.Tensor
    nbr_dir: torch.Tensor
    nbr_offset: torch.Tensor
    n_fluid: torch.Tensor
    la_weight: torch.Tensor

    def __len__(self) -> int:
        return int(self.flat.numel())

    def neighbor_rows(self, width: int = 18, pad: int = -1) -> torch.Tensor:
        """Dense ``(N, width)`` row view in MF-LBM ``neighbor_list(1:18)`` layout."""
        out = torch.full((len(self), width), pad, dtype=torch.int64)
        n_fluid = self.n_fluid.to(torch.int64)
        for row in range(len(self)):
            count = int(n_fluid[row])
            if count:
                start = int(self.nbr_offset[row])
                out[row, :count] = self.nbr_dir[start: start + count].to(torch.int64)
        return out


@dataclass(frozen=True, eq=False)
class FluidBoundaryTable:
    """MF-LBM ``fluid_boundary_nodes`` (``Module.F90:98-103``).

    ``theta`` is carried per node by MF-LBM (``Geometry_preprocessing.F90:505``);
    it is not stored here because the contact-angle model lives in TensorLBM's
    free-surface module.
    """

    ijk: torch.Tensor
    flat: torch.Tensor
    normal: torch.Tensor

    def __len__(self) -> int:
        return int(self.flat.numel())


@dataclass(frozen=True, eq=False)
class WallLinkTable:
    """Fluid->solid link view of the same geometry (BFL-compatible).

    MF-LBM's tables carry no link information at all; this view is what
    TensorLBM's momentum path needs, so building both from one traversal keeps
    the two halves of the indirect addressing consistent.
    """

    fluid_flat: torch.Tensor
    solid_flat: torch.Tensor
    direction: torch.Tensor
    q: torch.Tensor

    def __len__(self) -> int:
        return int(self.fluid_flat.numel())


@dataclass(frozen=True, eq=False)
class WallTables:
    """Container for the node-indexed wall tables plus the link view."""

    shape: tuple[int, int, int]
    solid_boundary: SolidBoundaryTable
    fluid_boundary: FluidBoundaryTable
    links: WallLinkTable
    meta: dict

    @property
    def n_solid_boundary(self) -> int:
        return len(self.solid_boundary)

    @property
    def n_fluid_boundary(self) -> int:
        return len(self.fluid_boundary)

    @property
    def n_links(self) -> int:
        return len(self.links)


def _smooth_wall_indicator(solid: torch.Tensor, iterations: int) -> torch.Tensor:
    """``Geometry_preprocessing.F90:143-160``: D3Q27-weighted 27-point smoothing."""
    smooth = solid.to(torch.float64)
    if iterations <= 0:
        return smooth
    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float64)
    for o in itertools.product((-1, 0, 1), repeat=3):
        kernel[0, 0, o[0] + 1, o[1] + 1, o[2] + 1] = _D3Q27_SHELL_WEIGHTS[sum(v * v for v in o)]
    weight = kernel.sum()
    view = smooth.view(1, 1, *smooth.shape)
    for _ in range(int(iterations)):
        view = torch.nn.functional.conv3d(view, kernel, padding=1) / weight
    return view.view(*smooth.shape)


def _iso8_component(
    smooth: torch.Tensor, alpha: int, variant: str = "projected"
) -> torch.Tensor:
    """ISO8 central-difference sum for one component.

    ``Geometry_preprocessing.F90:236-378``: component ``alpha`` sums
    ``f(x+o) - f(x-o)`` over every shell representative with ``o[alpha] != 0``,
    weighted by the shell coefficient.  Outside the domain ``f`` is taken as 0
    (the OSS code would read wall-filled ghost layers).

    Two variants:

    * ``"oss_literal"`` -- the OSS expression verbatim: the shell coefficient
      multiplies the *unweighted* difference, so a representative with
      ``|o_alpha| = 2`` contributes exactly as much as one with
      ``|o_alpha| = 1``.  This is not a projected gradient and it mixes
      Cartesian components (see :func:`iso8_wall_normal`).
    * ``"projected"`` (default) -- the difference is projected onto the
      component, ``o_alpha * (f(x+o) - f(x-o))``.  With the OSS coefficients
      this is unit-normalised (``2 * sum_s c_s sum_o o_x^2 = 0.99683`` for the
      x component, i.e. the coefficient set is the projected-gradient set) and
      it returns the exact normal of a plane wall.
    """
    if variant not in ("oss_literal", "projected"):
        raise ValueError("variant must be 'oss_literal' or 'projected'")
    grad = torch.zeros_like(smooth)
    for coeff, shell in zip(MF_LBM_ISO8_COEFFS, _ISO8_SHELLS):
        for o in shell:
            if o[alpha] == 0:
                continue
            diff = _shift(smooth, o, fill=0.0) - _shift(smooth, tuple(-v for v in o), fill=0.0)
            grad = grad + coeff * (diff if variant == "oss_literal" else o[alpha] * diff)
    return grad


def _pad_axis(field: torch.Tensor, pad: int, axis: int, periodic: bool) -> torch.Tensor:
    """Pad one axis: wrap if periodic, else fill with solid (``1.0``)."""
    n = int(field.shape[axis])
    idx = torch.arange(-pad, n + pad)
    if periodic:
        idx = torch.remainder(idx, n)
    out = field.index_select(axis, idx.clamp(0, n - 1).to(torch.int64)).clone()
    if not periodic:
        head = [slice(None)] * 3
        tail = [slice(None)] * 3
        head[axis] = slice(0, pad)
        tail[axis] = slice(n + pad, n + 2 * pad)
        out[tuple(head)] = 1.0
        out[tuple(tail)] = 1.0
    return out


def _pad_walls(solid: torch.Tensor, pad: int, periodic_axes: Sequence[bool]) -> torch.Tensor:
    """Ghost layers for the smoothing/ISO8 stencils.

    Non-periodic faces are filled with solid (``True``), exactly like the
    wall-filled ghost layers of ``Geometry_preprocessing.F90``; periodic axes
    are wrapped (MF-LBM's x direction is periodic, y/z are walls).
    """
    out = solid.to(torch.float64)
    for axis in (2, 1, 0):
        out = _pad_axis(out, pad, axis, bool(periodic_axes[axis]))
    return out


def iso8_wall_normal(
    solid: torch.Tensor,
    *,
    smooth_iterations: int = MF_LBM_N_SMOOTH_ITERATIONS,
    periodic_axes: Sequence[bool] = (False, False, False),
    variant: str = "projected",
) -> torch.Tensor:
    """ISO8 wall normal of the smoothed solid indicator.

    Args:
        solid: ``(nz, ny, nx)`` bool/0-1 wall indicator (1 = solid).
        smooth_iterations: smoothing passes applied before the ISO8 stencil.
        periodic_axes: ``(z, y, x)`` periodic flags used when building the ghost
            layers of the stencil; non-periodic faces are wall-filled.
        variant: ``"projected"`` (default, unit-normalised gradient) or
            ``"oss_literal"`` (the MF-LBM expression verbatim, see
            :func:`_iso8_component`).

    Returns:
        ``(nz, ny, nx, 3)`` float64 unit normals, oriented along *increasing
        solid fraction* (fluid -> wall), MF-LBM's ``nwx,nwy,nwz`` sign
        convention; exactly zero where the smoothed gradient vanishes.
    """
    solid = torch.as_tensor(solid)
    if solid.dim() != 3:
        raise ValueError("solid must be a 3-D (nz, ny, nx) array")
    periodic = tuple(bool(v) for v in periodic_axes)
    if len(periodic) != 3:
        raise ValueError("periodic_axes must have three entries")
    pad = int(smooth_iterations) + 3
    padded = _pad_walls(solid.bool(), pad, periodic)
    smooth = _smooth_wall_indicator(padded, smooth_iterations)
    normal = torch.stack([_iso8_component(smooth, a, variant) for a in range(3)], dim=-1)
    n0, n1, n2 = (int(v) for v in solid.shape)
    normal = normal[pad: pad + n0, pad: pad + n1, pad: pad + n2]
    norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    safe = torch.where(norm > 0, norm, torch.ones_like(norm))
    return torch.where(norm > 0, normal / safe, torch.zeros_like(normal))


def _classify(
    solid: torch.Tensor, periodic_axes: Sequence[bool] = (False, False, False)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``Geometry_preprocessing.F90:126-141`` cell tagging, vectorised.

    Returns ``(solid_boundary, fluid_boundary, nb_is_fluid)`` where
    ``nb_is_fluid[d-1]`` is the fluidity of the neighbour ``x + c_d``.  The
    neighbour lookup wraps on ``periodic_axes`` and is wall-filled elsewhere,
    mirroring the OSS ghost layers.
    """
    n0, n1, n2 = (int(v) for v in solid.shape)
    padded = _pad_walls(solid, 1, periodic_axes) > 0.5

    def _nb(offset: Sequence[int]) -> torch.Tensor:
        o = tuple(int(v) for v in offset)
        return padded[
            1 + o[0]: 1 + o[0] + n0,
            1 + o[1]: 1 + o[1] + n1,
            1 + o[2]: 1 + o[2] + n2,
        ]

    nb_is_fluid = torch.stack(
        [~_nb(C[d].tolist()) for d in range(1, C.shape[0])]
    )
    solid_boundary = solid & nb_is_fluid.any(dim=0)
    fluid_boundary = (~solid) & (~nb_is_fluid).any(dim=0)
    return solid_boundary, fluid_boundary, nb_is_fluid


def build_wall_tables(
    walls: torch.Tensor,
    *,
    smooth_iterations: int = MF_LBM_N_SMOOTH_ITERATIONS,
    q_field: torch.Tensor | None = None,
    periodic_axes: Sequence[bool] = (False, False, False),
    normal_variant: str = "projected",
) -> WallTables:
    """Build MF-LBM-style indirect-addressing wall tables.

    Args:
        walls: ``(nz, ny, nx)`` bool (``True`` = solid) voxel geometry.  Cells
            outside the domain are treated as solid, reproducing the ghost-layer
            convention of ``Geometry_preprocessing.F90``.
        smooth_iterations: wall-indicator smoothing passes for the normal.
        q_field: optional ``(nz, ny, nx)`` fractional wall distance per fluid
            cell for the link view.  ``None`` (default) marks the geometry as
            half-way (``q = 0.5`` on every link), which is exactly the regime
            MF-LBM lives in (it does not interpolate at all).

    Returns:
        :class:`WallTables`.

    Raises:
        ValueError: if ``walls`` is not 3-D or ``q_field`` does not match it.
    """
    walls = torch.as_tensor(walls)
    if walls.dim() != 3:
        raise ValueError("walls must be a 3-D (nz, ny, nx) array")
    if q_field is not None and tuple(q_field.shape) != tuple(walls.shape):
        raise ValueError("q_field must have the same shape as walls")
    solid = walls.bool()
    shape = tuple(int(v) for v in solid.shape)
    q_count = int(C.shape[0])
    flat_grid = _flat_grid(shape)
    sb_mask, fb_mask, nb_is_fluid = _classify(solid, periodic_axes)

    # ---- solid_boundary_nodes (Geometry_preprocessing.F90:196-217) -------
    sb_idx = sb_mask.nonzero(as_tuple=False).to(torch.int64)
    dirs_per_cell: list[list[int]] = []
    la_weight = torch.zeros(sb_idx.shape[0], dtype=torch.float64)
    for row, (i, j, k) in enumerate(sb_idx.tolist()):
        listed = []
        for d in range(1, q_count):
            if bool(nb_is_fluid[d - 1, i, j, k]):
                listed.append(d)
                la_weight[row] += float(W_EXACT64[d])
        dirs_per_cell.append(listed)
    n_fluid = torch.tensor([len(x) for x in dirs_per_cell], dtype=torch.int64)
    nbr_offset = torch.zeros(sb_idx.shape[0] + 1, dtype=torch.int64)
    if sb_idx.shape[0]:
        nbr_offset[1:] = torch.cumsum(n_fluid, dim=0)
    nbr_dir = torch.tensor([d for row in dirs_per_cell for d in row], dtype=torch.int8)
    solid_boundary = SolidBoundaryTable(
        ijk=sb_idx,
        flat=flat_grid[sb_mask].to(torch.int64),
        nbr_dir=nbr_dir,
        nbr_offset=nbr_offset,
        n_fluid=n_fluid,
        la_weight=la_weight,
    )

    # ---- fluid_boundary_nodes (Geometry_preprocessing.F90:230-383) -------
    fb_idx = fb_mask.nonzero(as_tuple=False).to(torch.int64)
    normals = iso8_wall_normal(
        solid,
        smooth_iterations=smooth_iterations,
        periodic_axes=periodic_axes,
        variant=normal_variant,
    )
    fluid_boundary = FluidBoundaryTable(
        ijk=fb_idx,
        flat=flat_grid[fb_mask].to(torch.int64),
        normal=normals[fb_mask] if fb_idx.numel() else torch.zeros((0, 3), dtype=torch.float64),
    )

    # ---- link view (fluid -> solid), TensorLBM extension ------------------
    fluid_idx_parts: list[torch.Tensor] = []
    solid_idx_parts: list[torch.Tensor] = []
    dir_parts: list[torch.Tensor] = []
    link_solid = ~nb_is_fluid  # (18, nz, ny, nx)
    fluid_cells = ~solid
    base_flat = flat_grid.clone()
    n_outside = 0
    periodic = tuple(bool(v) for v in periodic_axes)
    for d in range(1, q_count):
        mask = fluid_cells & link_solid[d - 1]
        if not bool(mask.any()):
            continue
        cells = mask.nonzero(as_tuple=False).to(torch.int64)
        step = C[d].to(torch.int64)
        nb_ijk = cells + step.view(1, 3)
        wrapped = nb_ijk.clone()
        for axis, per in enumerate(periodic):
            if per:
                wrapped[:, axis] = torch.remainder(nb_ijk[:, axis], shape[axis])
        # a wall placed on a non-periodic domain face has no in-domain solid
        # cell; such links are reported in meta instead of being indexed out of
        # range (TensorLBM's BFL path expects both endpoints to exist).
        inside = (
            (wrapped[:, 0] >= 0) & (wrapped[:, 0] < shape[0])
            & (wrapped[:, 1] >= 0) & (wrapped[:, 1] < shape[1])
            & (wrapped[:, 2] >= 0) & (wrapped[:, 2] < shape[2])
        )
        n_outside += int((~inside).sum())
        if not bool(inside.any()):
            continue
        cells = cells[inside]
        nb_ijk = wrapped[inside]
        fluid_mask = torch.zeros_like(mask)
        fluid_mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
        fluid_idx_parts.append(flat_grid[fluid_mask].to(torch.int64))
        solid_idx_parts.append(((nb_ijk[:, 0] * shape[1] + nb_ijk[:, 1]) * shape[2] + nb_ijk[:, 2]).to(torch.int64))
        dir_parts.append(torch.full((cells.shape[0],), d, dtype=torch.int64))
    if fluid_idx_parts:
        fluid_flat = torch.cat(fluid_idx_parts)
        solid_flat = torch.cat(solid_idx_parts)
        direction = torch.cat(dir_parts)
    else:
        fluid_flat = torch.zeros(0, dtype=torch.int64)
        solid_flat = torch.zeros(0, dtype=torch.int64)
        direction = torch.zeros(0, dtype=torch.int64)
    if q_field is None:
        q_vals = torch.full((fluid_flat.shape[0],), 0.5, dtype=torch.float32)
        q_source = "voxel_halfway"
    else:
        q_vals = torch.as_tensor(q_field).reshape(-1)[fluid_flat].to(torch.float32)
        q_source = "supplied_q_field"
    links = WallLinkTable(
        fluid_flat=fluid_flat, solid_flat=solid_flat, direction=direction, q=q_vals
    )
    del base_flat

    meta = {
        "shape": list(shape),
        "n_cells": int(solid.numel()),
        "n_solid": int(solid.sum()),
        "solid_fraction": float(solid.sum()) / float(solid.numel()),
        "n_solid_boundary": len(solid_boundary),
        "n_fluid_boundary": len(fluid_boundary),
        "n_links": len(links),
        "n_links_outside_domain": n_outside,
        "nbr_entries": int(nbr_dir.numel()),
        "smooth_iterations": int(smooth_iterations),
        "periodic_axes": [bool(v) for v in periodic_axes],
        "iso8_coeffs": list(MF_LBM_ISO8_COEFFS),
        "q_source": q_source,
        "normal_variant": normal_variant,
        "provenance": "lanl/MF-LBM 11be324a: Module.F90:89-103, Geometry_preprocessing.F90:126-217,230-383",
    }
    return WallTables(
        shape=shape,
        solid_boundary=solid_boundary,
        fluid_boundary=fluid_boundary,
        links=links,
        meta=meta,
    )


def la_weight_extrapolate(
    field: torch.Tensor,
    tables: WallTables,
    periodic_axes: Sequence[bool] = (False, False, False),
) -> torch.Tensor:
    """MF-LBM ``la_weight`` continuation of a cell field into the wall.

    Transcribes ``Phase_gradient.F90:14-28`` (the phase field) and
    ``:86-107`` (the colour-gradient components)::

        phi[x] = sum_n w_equ(n) * phi[x + c_n] / la_weight[x]

    summed over the listed (fluid) neighbours of a solid boundary cell.  The
    result is the *weighted mean of the fluid neighbours*, i.e. the field at
    their weight centroid; it is written back onto the solid boundary cells and
    returned as a new array (the input is not modified).

    Args:
        field: ``(nz, ny, nx)`` values to continue.
        tables: geometry tables from :func:`build_wall_tables`.
        periodic_axes: ``(z, y, x)`` axes on which the neighbour lookup wraps
            (a periodic axis may hand the stencil a neighbour across the domain
            seam); non-periodic axes assume the neighbour is in-domain.

    Returns:
        Copy of ``field`` with the solid boundary cells overwritten.
    """
    field = torch.as_tensor(field)
    if tuple(field.shape) != tuple(tables.shape):
        raise ValueError("field shape does not match the wall tables")
    out = field.clone()
    sb = tables.solid_boundary
    acc = torch.zeros(len(sb), dtype=field.dtype)
    rows = sb.ijk.tolist()
    periodic = tuple(bool(v) for v in periodic_axes)
    sizes = tuple(int(v) for v in field.shape)
    for row, (i, j, k) in enumerate(rows):
        start = int(sb.nbr_offset[row])
        count = int(sb.n_fluid[row])
        total = acc.new_zeros(())
        for pos in range(start, start + count):
            d = int(sb.nbr_dir[pos])
            step = C[d].tolist()
            nb = [i + step[0], j + step[1], k + step[2]]
            nb = [n % sizes[a] if periodic[a] else n for a, n in enumerate(nb)]
            total = total + float(W_EXACT64[d]) * field[nb[0], nb[1], nb[2]]
        acc[row] = total / sb.la_weight[row]
    if len(sb):
        flat = sb.flat.to(torch.int64)
        out.reshape(-1)[flat] = acc
    return out


def dense_la_weight_extrapolate(field: torch.Tensor, walls: torch.Tensor) -> torch.Tensor:
    """Brute-force reference implementation of :func:`la_weight_extrapolate`.

    Written as an independent site-by-site double loop (no shift helpers, no
    table objects) so it can serve as the reference arm of the equivalence
    test.  Sites outside the domain are solid.
    """
    field = torch.as_tensor(field)
    walls = torch.as_tensor(walls).bool()
    if tuple(field.shape) != tuple(walls.shape):
        raise ValueError("field and walls shapes differ")
    nz, ny, nx = (int(v) for v in walls.shape)
    weights = [float(w) for w in W_EXACT64]
    dirs = [tuple(int(v) for v in C[d].tolist()) for d in range(C.shape[0])]
    out = field.clone()
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                if not bool(walls[i, j, k]):
                    continue
                total = 0.0
                weight = 0.0
                any_fluid = False
                for d in range(1, len(dirs)):
                    i2, j2, k2 = i + dirs[d][0], j + dirs[d][1], k + dirs[d][2]
                    inside = 0 <= i2 < nz and 0 <= j2 < ny and 0 <= k2 < nx
                    if inside and not bool(walls[i2, j2, k2]):
                        any_fluid = True
                        total += weights[d] * float(field[i2, j2, k2])
                        weight += weights[d]
                if any_fluid:
                    out[i, j, k] = total / weight
    return out


def table_footprint_bytes(tables: WallTables, *, dir_bytes: int = 1, offset_bytes: int = 8) -> dict:
    """Byte footprint of the table pair (as implemented) and of the list payload."""
    sb = tables.solid_boundary
    n_sb = len(sb)
    n_nbr = int(sb.nbr_dir.numel())
    bytes_csr = n_nbr * dir_bytes + (n_sb + 1) * offset_bytes + n_sb * 8
    # MF-LBM's fixed row: 4 int32 (ix,iy,iz,i_fluid_num) + 18 int32 + 1 double
    bytes_oss_row = n_sb * (4 * 4 + 18 * 4 + 8)
    fb = tables.fluid_boundary
    bytes_fb = len(fb) * (3 * 4 + 3 * 8)
    bytes_oss_fb = len(fb) * (3 * 4 + 4 * 8)
    return {
        "n_solid_boundary": n_sb,
        "n_fluid_boundary": len(fb),
        "nbr_entries": n_nbr,
        "bytes_csr_implemented": bytes_csr,
        "bytes_mf_lbm_fixed_row": bytes_oss_row,
        "bytes_fluid_boundary_implemented": bytes_fb,
        "bytes_fluid_boundary_mf_lbm": bytes_oss_fb,
        "bytes_total_implemented": bytes_csr + bytes_fb,
        "bytes_total_mf_lbm_equivalent": bytes_oss_row + bytes_oss_fb,
    }


def dense_footprint_bytes(shape: Sequence[int], *, q_count: int = 19, mask_bytes: int = 1, q_bytes: int = 4) -> dict:
    """Byte footprint of the dense representation the tables replace."""
    n = 1
    for s in shape:
        n *= int(s)
    return {
        "n_cells": n,
        "bytes_mask_dense": n * q_count * mask_bytes,
        "bytes_qfield_dense": n * q_bytes + n * q_count * mask_bytes,
        "bytes_mask_plus_q": n * q_count * mask_bytes + n * q_bytes,
    }