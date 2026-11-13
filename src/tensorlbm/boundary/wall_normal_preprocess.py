"""Platform-side entry for the ISO8 wall-normal table (XVAL-LBM-ISO8-WALL-NORMAL-TABLE).

This is the caller the 3-D solver pipeline uses: it takes a voxelised solid
mask in the *platform* convention and returns the wall-normal table in the
*platform* convention, so downstream 3-D boundary conditions (geometric
wetting / contact angle, Laplace-pressure closures) can consume it directly
without knowing the upstream (lanl/MF-LBM) Fortran conventions.

Convention translation (all verified against the compiled OSS preprocessor,
see ``artifacts/xval_iso8_wall_normal_table/``):

* input  ``solid_mask``: ``(nx, ny, nz)`` integer array, 1 = solid, 0 = fluid
  (plain NumPy array; the OSS preprocessor reads the same values).
* output indices: **0-based** (the raw table reports Fortran 1-based indices).
* output normals: **out of the solid, into the fluid** (the raw table points
  into the solid -- MF-LBM's convention, D9 in the anchor doc).
* output rows: physical-domain nodes only by default (the raw table also lists
  the ``overlap`` band outside the box, needed only by MPI subdomains).

The raw OSS-convention table stays available via
:func:`tensorlbm.boundary.wall_normal_iso8.wall_normal_table`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .wall_normal_iso8 import (
    DEFAULT_GHOST_LAYERS,
    DEFAULT_OVERLAP,
    in_physical_domain,
    wall_normal_table,
)

__all__ = ["WallNormalTable", "build_wall_normal_table"]


@dataclass
class WallNormalTable:
    """Wall normals in the platform convention.

    Attributes:
        indices:  ``(N, 3)`` int64, 0-based ``(i, j, k)`` of fluid boundary nodes.
        normals:  ``(N, 3)`` float64, unit normals pointing **out of the solid**.
        grad_mag: ``(N,)`` float64, |grad| of the smoothed indicator (pre-normalisation).
        n_nodes:  number of rows.
    """

    indices: np.ndarray
    normals: np.ndarray
    grad_mag: np.ndarray
    n_nodes: int

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(indices, normals)`` for direct use in a boundary-condition kernel."""
        return self.indices, self.normals


def build_wall_normal_table(
    solid_mask: np.ndarray,
    *,
    periodic: Sequence[int] = (0, 0, 0),
    domain_walls: Sequence[int] = (0, 0, 0, 0, 0, 0),
    n_smooth: int | None = None,
    ghost: int = DEFAULT_GHOST_LAYERS,
    overlap: int = DEFAULT_OVERLAP,
    outward: bool = True,
    physical_only: bool = True,
) -> WallNormalTable:
    """Build the wall-normal table for a voxelised solid mask.

    Args:
        solid_mask: ``(nx, ny, nz)`` integer array, 1 = solid, 0 = fluid.
        periodic: per-axis periodic flags (upstream ``iper/jper/kper``).
        domain_walls: ``(xmin,xmax,ymin,ymax,zmin,zmax)``; 1 = domain face is a wall.
        n_smooth: D3Q27 smoothing iterations (default = upstream value, 4).
        ghost: ghost layers for the smoothing (upstream 6).
        overlap: node-listing overlap (upstream 4).
        outward: return normals pointing out of the solid (platform convention);
            ``False`` keeps the raw OSS sign (into the solid).
        physical_only: keep only nodes inside the physical box (default True).

    Returns:
        :class:`WallNormalTable` with 0-based indices and unit normals.
    """
    walls = np.asarray(solid_mask)
    if walls.ndim != 3:
        raise ValueError(f"solid_mask must be 3-D, got shape {walls.shape}")
    if not np.issubdtype(walls.dtype, np.integer):
        raise TypeError("solid_mask must be an integer array (1 = solid, 0 = fluid)")
    if not set(np.unique(walls)).issubset({0, 1}):
        raise ValueError("solid_mask values must be 0 (fluid) or 1 (solid)")

    raw = wall_normal_table(
        walls,
        periodic=tuple(int(p) for p in periodic),
        domain_walls=tuple(int(d) for d in domain_walls),
        n_smooth=n_smooth,
        ghost=ghost,
        overlap=overlap,
    )
    idx = raw["indices"] - 1                      # Fortran 1-based -> 0-based
    nrm = -raw["normals"] if outward else raw["normals"]
    mag = raw["grad_mag"]
    if physical_only:
        keep = in_physical_domain(raw["indices"], walls.shape)
        idx, nrm, mag = idx[keep], nrm[keep], mag[keep]
    return WallNormalTable(
        indices=idx.astype(np.int64),
        normals=np.ascontiguousarray(nrm),
        grad_mag=np.ascontiguousarray(mag),
        n_nodes=int(len(idx)),
    )
