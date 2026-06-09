"""Parametric offshore structure CAD module for TensorLBM.

Provides parametric geometry generators for common offshore and ocean
engineering structures.  All generators produce boolean 3-D solid masks
compatible with the LBM obstacle representation used by TensorLBM.

Supported structure types
-------------------------
- **Monopile** – vertical circular cylinder; standard offshore wind turbine
  foundation and near-bed pile/jacket leg primitive.
- **Jacket** – 4-leg inclined tubular steel jacket; simplified as four
  uniform-diameter cylinder legs, triangular bracing omitted (conservative
  drag model for LBM).
- **Spar** – deep-draft floating spar-buoy platform; multi-section cylinder
  (hull section + tapered keel + upper column).
- **Semi-sub** – semi-submersible: two rectangular pontoons with four
  cylindrical columns connecting to a deck box (simplified geometry).

Public API
----------
- :class:`OffshoreStructureType`    – structure family enum.
- :func:`monopile_mask`             – 3-D boolean voxel mask for a monopile.
- :func:`jacket_mask`               – 3-D boolean voxel mask for a jacket.
- :func:`spar_mask`                 – 3-D boolean voxel mask for a spar.
- :func:`semi_sub_mask`             – 3-D boolean voxel mask for a semi-sub.
- :func:`build_offshore_mask`       – unified dispatch + statistics.
- :func:`offshore_statistics`       – geometric statistics dictionary.
- :func:`generate_offshore_previews`– matplotlib multi-view figure.
- :func:`export_offshore_stl`       – ASCII STL surface mesh export.
"""
from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import matplotlib.figure

__all__ = [
    "OffshoreStructureType",
    "monopile_mask",
    "jacket_mask",
    "spar_mask",
    "semi_sub_mask",
    "build_offshore_mask",
    "offshore_statistics",
    "generate_offshore_previews",
    "export_offshore_stl",
]


# ---------------------------------------------------------------------------
# Structure type enum
# ---------------------------------------------------------------------------


class OffshoreStructureType(str, Enum):  # noqa: UP042
    """Supported offshore structure families."""

    MONOPILE = "monopile"
    """Single vertical cylinder (wind turbine foundation, pile)."""

    JACKET = "jacket"
    """Four-leg jacket structure (tubular steel space frame)."""

    SPAR = "spar"
    """Deep-draft spar-buoy floating platform."""

    SEMI_SUB = "semi_sub"
    """Semi-submersible (pontoons + columns + deck box)."""


_STRUCTURE_LABELS = {
    OffshoreStructureType.MONOPILE: "Monopile / Vertical Cylinder",
    OffshoreStructureType.JACKET: "Jacket Structure (4-leg)",
    OffshoreStructureType.SPAR: "Spar Floating Platform",
    OffshoreStructureType.SEMI_SUB: "Semi-Submersible Platform",
}


# ---------------------------------------------------------------------------
# Internal voxel primitives
# ---------------------------------------------------------------------------


def _cylinder_mask(
    grid: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    z_bot: float,
    z_top: float,
    *,
    incline_x: float = 0.0,
    incline_y: float = 0.0,
) -> np.ndarray:
    """Mark voxels belonging to an optionally inclined cylinder.

    Parameters
    ----------
    grid : boolean array (nx, ny, nz) to update in-place.
    cx, cy : axis centroid at the mid-height (z = (z_bot+z_top)/2).
    r  : cylinder radius (lattice units).
    z_bot, z_top : vertical extent (lattice indices, float).
    incline_x, incline_y : horizontal offset per unit height (taper in x/y).
    """
    nx, ny, nz = grid.shape
    nz_f = float(nz)
    z_mid = 0.5 * (z_bot + z_top)

    for iz in range(int(math.floor(z_bot)), min(int(math.ceil(z_top)) + 1, nz)):
        if iz < 0 or iz >= nz:
            continue
        dz = float(iz) - z_mid
        axis_x = cx + incline_x * dz
        axis_y = cy + incline_y * dz
        # Build x/y index arrays efficiently
        x_idx = np.arange(nx, dtype=np.float32)
        y_idx = np.arange(ny, dtype=np.float32)
        xx, yy = np.meshgrid(x_idx, y_idx, indexing="ij")
        inside = (xx - axis_x) ** 2 + (yy - axis_y) ** 2 <= r**2
        grid[:, :, iz] |= inside
    return grid


def _box_mask(
    grid: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> np.ndarray:
    """Mark voxels belonging to an axis-aligned box (in-place)."""
    nx, ny, nz = grid.shape
    ix0 = max(0, int(math.floor(x0)))
    ix1 = min(nx, int(math.ceil(x1)) + 1)
    iy0 = max(0, int(math.floor(y0)))
    iy1 = min(ny, int(math.ceil(y1)) + 1)
    iz0 = max(0, int(math.floor(z0)))
    iz1 = min(nz, int(math.ceil(z1)) + 1)
    grid[ix0:ix1, iy0:iy1, iz0:iz1] = True
    return grid


# ---------------------------------------------------------------------------
# Public mask generators
# ---------------------------------------------------------------------------


def monopile_mask(
    nx: int,
    ny: int,
    nz: int,
    diameter: float,
    *,
    cx: float | None = None,
    cy: float | None = None,
    z_bot: float = 0.0,
    z_top: float | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a 3-D voxel mask for a vertical monopile cylinder.

    Parameters
    ----------
    nx, ny, nz : Grid dimensions.
    diameter   : Pile diameter (lattice units).
    cx, cy     : Horizontal centroid (default: grid centre).
    z_bot      : Bottom of pile in lattice units (default 0).
    z_top      : Top of pile in lattice units (default nz).
    device     : Torch device string.

    Returns
    -------
    torch.BoolTensor of shape (nx, ny, nz).
    """
    cx = float(cx) if cx is not None else nx / 2.0
    cy = float(cy) if cy is not None else ny / 2.0
    z_top = float(z_top) if z_top is not None else float(nz)
    r = diameter / 2.0

    grid = np.zeros((nx, ny, nz), dtype=bool)
    _cylinder_mask(grid, cx, cy, r, z_bot, z_top)
    return torch.as_tensor(grid, device=device)


def jacket_mask(
    nx: int,
    ny: int,
    nz: int,
    leg_diameter: float,
    foot_spread: float,
    head_spread: float,
    *,
    cx: float | None = None,
    cy: float | None = None,
    z_bot: float = 0.0,
    z_top: float | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a 3-D voxel mask for a 4-leg jacket structure.

    The jacket has four cylindrical legs, each inclined from a wide foot
    (``foot_spread`` between legs) at ``z_bot`` to a narrow head
    (``head_spread`` between legs) at ``z_top``.

    Parameters
    ----------
    nx, ny, nz    : Grid dimensions.
    leg_diameter  : Leg outer diameter (lattice units).
    foot_spread   : Distance between leg pairs at the seabed (lu).
    head_spread   : Distance between leg pairs at the top (lu).
    cx, cy        : Plan centroid (default: grid centre).
    z_bot, z_top  : Vertical extent (lu).
    device        : Torch device string.

    Returns
    -------
    torch.BoolTensor of shape (nx, ny, nz).
    """
    cx = float(cx) if cx is not None else nx / 2.0
    cy = float(cy) if cy is not None else ny / 2.0
    z_top = float(z_top) if z_top is not None else float(nz)
    r = leg_diameter / 2.0
    h = z_top - z_bot
    if h <= 0.0:
        raise ValueError("z_top must be greater than z_bot")

    # Four corners at mid-height
    z_mid = 0.5 * (z_bot + z_top)
    foot_h = foot_spread / 2.0
    head_h = head_spread / 2.0
    # Average centroid spread at mid-height
    mid_h = 0.5 * (foot_h + head_h)
    # Inclination per unit height (positive z = upward, legs converge)
    taper = (head_h - foot_h) / h  # < 0 means narrowing upward

    corners = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    grid = np.zeros((nx, ny, nz), dtype=bool)
    for sx, sy in corners:
        lx = cx + sx * mid_h
        ly = cy + sy * mid_h
        # incline toward centre as z increases
        inc_x = -sx * taper
        inc_y = -sy * taper
        _cylinder_mask(grid, lx, ly, r, z_bot, z_top, incline_x=inc_x, incline_y=inc_y)
    return torch.as_tensor(grid, device=device)


def spar_mask(
    nx: int,
    ny: int,
    nz: int,
    hull_diameter: float,
    keel_diameter: float,
    column_diameter: float,
    *,
    cx: float | None = None,
    cy: float | None = None,
    z_keel: float = 0.0,
    z_hull_top: float | None = None,
    z_column_top: float | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a 3-D voxel mask for a spar floating platform.

    The spar is modelled as three coaxial cylinders stacked vertically:

    1. **Keel** (``z_keel`` → ``z_keel + keel_height``): wide, heavy cylinder.
    2. **Hull** (``keel_top`` → ``z_hull_top``): main buoyancy cylinder.
    3. **Column** (``z_hull_top`` → ``z_column_top``): narrow upper column.

    Default heights: keel = 15 % nz, hull = 60 % nz, column = 25 % nz.

    Parameters
    ----------
    nx, ny, nz          : Grid dimensions.
    hull_diameter       : Main hull diameter (lu).
    keel_diameter       : Keel section diameter (lu, ≥ hull_diameter typical).
    column_diameter     : Upper column diameter (lu, ≤ hull_diameter).
    cx, cy              : Plan centroid (default: grid centre).
    z_keel              : Bottom of keel (lu, default 0).
    z_hull_top          : Top of hull section (lu, default 0.75*nz).
    z_column_top        : Top of column (lu, default nz).
    device              : Torch device string.

    Returns
    -------
    torch.BoolTensor of shape (nx, ny, nz).
    """
    cx = float(cx) if cx is not None else nx / 2.0
    cy = float(cy) if cy is not None else ny / 2.0
    z_hull_top = float(z_hull_top) if z_hull_top is not None else 0.75 * nz
    z_column_top = float(z_column_top) if z_column_top is not None else float(nz)
    keel_height = 0.15 * nz
    z_keel_top = z_keel + keel_height

    grid = np.zeros((nx, ny, nz), dtype=bool)
    # Keel section (widest)
    _cylinder_mask(grid, cx, cy, keel_diameter / 2.0, z_keel, z_keel_top)
    # Main hull cylinder
    _cylinder_mask(grid, cx, cy, hull_diameter / 2.0, z_keel_top, z_hull_top)
    # Upper column
    _cylinder_mask(grid, cx, cy, column_diameter / 2.0, z_hull_top, z_column_top)
    return torch.as_tensor(grid, device=device)


def semi_sub_mask(
    nx: int,
    ny: int,
    nz: int,
    pontoon_length: float,
    pontoon_width: float,
    pontoon_height: float,
    column_diameter: float,
    column_height: float,
    *,
    cx: float | None = None,
    cy: float | None = None,
    z_keel: float = 0.0,
    deck_thickness: float | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a 3-D voxel mask for a semi-submersible platform.

    Geometry (plan view of 4-column configuration):

    ::

        Col(P,F) ---deck--- Col(S,F)
             |                   |
           pontoon (fore)       (fore brace omitted)
             |                   |
        Col(P,A) ---deck--- Col(S,A)
             |                   |
           pontoon (port)  pontoon (stbd)

    Simplified model: 2 rectangular pontoons (port & starboard) running
    fore-aft, plus 4 cylindrical columns rising from pontoon corners, plus
    an optional flat deck box connecting the column tops.

    Parameters
    ----------
    nx, ny, nz         : Grid dimensions.
    pontoon_length     : Fore-aft length of each pontoon (lu).
    pontoon_width      : Width of each pontoon (lu).
    pontoon_height     : Height of each pontoon (lu).
    column_diameter    : Diameter of each column (lu).
    column_height      : Height of each column above pontoon top (lu).
    cx, cy             : Plan centroid (default: grid centre).
    z_keel             : Bottom of pontoons (lu, default 0).
    deck_thickness     : Deck slab thickness (lu, default 0.1*column_height).
    device             : Torch device string.

    Returns
    -------
    torch.BoolTensor of shape (nx, ny, nz).
    """
    cx = float(cx) if cx is not None else nx / 2.0
    cy = float(cy) if cy is not None else ny / 2.0
    deck_thickness = deck_thickness if deck_thickness is not None else max(1.0, 0.1 * column_height)

    # Pontoon separation (centre-to-centre) = beam / 2
    pontoon_sep = ny / 4.0  # separation from centreline to each pontoon centre

    grid = np.zeros((nx, ny, nz), dtype=bool)

    z_pontoon_bot = z_keel
    z_pontoon_top = z_keel + pontoon_height
    z_column_top = z_pontoon_top + column_height
    z_deck_top = z_column_top + deck_thickness

    # Two rectangular pontoons (port and starboard)
    for sign in (-1.0, 1.0):
        py_centre = cy + sign * pontoon_sep
        py0 = py_centre - pontoon_width / 2.0
        py1 = py_centre + pontoon_width / 2.0
        px0 = cx - pontoon_length / 2.0
        px1 = cx + pontoon_length / 2.0
        _box_mask(grid, px0, px1, py0, py1, z_pontoon_bot, z_pontoon_top)

    # Four columns at pontoon corners
    col_x_offsets = [-pontoon_length / 4.0, pontoon_length / 4.0]
    for sign in (-1.0, 1.0):
        py_col = cy + sign * pontoon_sep
        for dx in col_x_offsets:
            _cylinder_mask(
                grid,
                cx + dx,
                py_col,
                column_diameter / 2.0,
                z_pontoon_top,
                z_column_top,
            )

    # Deck box connecting column tops
    deck_x0 = cx - pontoon_length / 4.0 - column_diameter / 2.0
    deck_x1 = cx + pontoon_length / 4.0 + column_diameter / 2.0
    deck_y0 = cy - pontoon_sep - pontoon_width / 2.0
    deck_y1 = cy + pontoon_sep + pontoon_width / 2.0
    _box_mask(grid, deck_x0, deck_x1, deck_y0, deck_y1, z_column_top, z_deck_top)

    return torch.as_tensor(grid, device=device)


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict[OffshoreStructureType, dict] = {
    OffshoreStructureType.MONOPILE: {
        "diameter": None,  # computed from grid
    },
    OffshoreStructureType.JACKET: {
        "leg_diameter": None,
        "foot_spread": None,
        "head_spread": None,
    },
    OffshoreStructureType.SPAR: {
        "hull_diameter": None,
        "keel_diameter": None,
        "column_diameter": None,
    },
    OffshoreStructureType.SEMI_SUB: {
        "pontoon_length": None,
        "pontoon_width": None,
        "pontoon_height": None,
        "column_diameter": None,
        "column_height": None,
    },
}


def _auto_params(struct_type: OffshoreStructureType, nx: int, ny: int, nz: int, **kwargs) -> dict:
    """Fill in sensible default dimensions scaled to grid size."""
    ref = min(nx, ny, nz)
    if struct_type == OffshoreStructureType.MONOPILE:
        defaults = {"diameter": ref * 0.15}
    elif struct_type == OffshoreStructureType.JACKET:
        defaults = {
            "leg_diameter": ref * 0.04,
            "foot_spread": min(nx, ny) * 0.5,
            "head_spread": min(nx, ny) * 0.25,
        }
    elif struct_type == OffshoreStructureType.SPAR:
        defaults = {
            "hull_diameter": ref * 0.18,
            "keel_diameter": ref * 0.22,
            "column_diameter": ref * 0.10,
        }
    elif struct_type == OffshoreStructureType.SEMI_SUB:
        defaults = {
            "pontoon_length": nx * 0.6,
            "pontoon_width": min(nx, ny) * 0.08,
            "pontoon_height": nz * 0.10,
            "column_diameter": min(nx, ny) * 0.10,
            "column_height": nz * 0.50,
        }
    else:
        raise ValueError(f"Unknown structure type: {struct_type}")
    defaults.update({k: v for k, v in kwargs.items() if v is not None})
    return defaults


def build_offshore_mask(
    struct_type: OffshoreStructureType | str,
    nx: int,
    ny: int,
    nz: int,
    *,
    device: str = "cpu",
    **kwargs,
) -> dict:
    """Build a 3-D LBM obstacle mask for an offshore structure.

    Parameters
    ----------
    struct_type : Structure family (``OffshoreStructureType`` or its value).
    nx, ny, nz  : Grid dimensions.
    device      : Torch device string.
    **kwargs    : Structure-specific geometry overrides (passed to the
                  relevant mask function; any ``None`` value is replaced
                  by the auto-computed default).

    Returns
    -------
    dict with keys:

    - ``"mask"``       : torch.BoolTensor (nx, ny, nz)
    - ``"stats"``      : dict from :func:`offshore_statistics`
    """
    if isinstance(struct_type, str):
        struct_type = OffshoreStructureType(struct_type)

    params = _auto_params(struct_type, nx, ny, nz, **kwargs)

    if struct_type == OffshoreStructureType.MONOPILE:
        mask = monopile_mask(nx, ny, nz, **params, device=device)
    elif struct_type == OffshoreStructureType.JACKET:
        mask = jacket_mask(nx, ny, nz, **params, device=device)
    elif struct_type == OffshoreStructureType.SPAR:
        mask = spar_mask(nx, ny, nz, **params, device=device)
    elif struct_type == OffshoreStructureType.SEMI_SUB:
        mask = semi_sub_mask(nx, ny, nz, **params, device=device)
    else:
        raise ValueError(f"Unknown structure type: {struct_type}")

    stats = offshore_statistics(struct_type, nx, ny, nz, mask)
    return {"mask": mask, "stats": stats}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def offshore_statistics(
    struct_type: OffshoreStructureType | str,
    nx: int,
    ny: int,
    nz: int,
    mask: torch.Tensor | np.ndarray,
) -> dict:
    """Return geometric statistics for an offshore structure mask.

    Parameters
    ----------
    struct_type : Structure family.
    nx, ny, nz  : Grid dimensions.
    mask        : Boolean 3-D mask (nx, ny, nz).

    Returns
    -------
    dict with solid_cells, fluid_cells, solid_fraction, grid, label.
    """
    if isinstance(struct_type, str):
        struct_type = OffshoreStructureType(struct_type)
    if isinstance(mask, torch.Tensor):
        arr = mask.cpu().numpy()
    else:
        arr = np.asarray(mask)

    solid = int(arr.sum())
    total = int(arr.size)
    fluid = total - solid
    return {
        "structure_type": struct_type.value,
        "label": _STRUCTURE_LABELS[struct_type],
        "solid_cells": solid,
        "fluid_cells": fluid,
        "solid_fraction": round(solid / total, 6) if total > 0 else 0.0,
        "grid": f"{nx}×{ny}×{nz}",
    }


# ---------------------------------------------------------------------------
# 2-D preview figure
# ---------------------------------------------------------------------------


def generate_offshore_previews(
    struct_type: OffshoreStructureType | str,
    nx: int = 80,
    ny: int = 80,
    nz: int = 80,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """Render top-view, side-view, and front-view projections.

    Returns a matplotlib Figure with three subplots showing XY, XZ and YZ
    projections of the offshore structure voxel mask.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(struct_type, str):
        struct_type = OffshoreStructureType(struct_type)

    result = build_offshore_mask(struct_type, nx, ny, nz, **kwargs)
    mask_np = result["mask"].cpu().numpy().astype(np.float32)
    stats = result["stats"]

    label = _STRUCTURE_LABELS[struct_type]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Offshore Structure Preview – {label}", fontsize=12)

    # XY projection (top view)
    axes[0].imshow(mask_np.max(axis=2).T, origin="lower", aspect="equal", cmap="Blues")
    axes[0].set_title("Top View (XY)")
    axes[0].set_xlabel("x (lu)")
    axes[0].set_ylabel("y (lu)")

    # XZ projection (side view)
    axes[1].imshow(mask_np.max(axis=1).T, origin="lower", aspect="equal", cmap="Blues")
    axes[1].set_title("Side View (XZ)")
    axes[1].set_xlabel("x (lu)")
    axes[1].set_ylabel("z (lu)")

    # YZ projection (front view)
    axes[2].imshow(mask_np.max(axis=0).T, origin="lower", aspect="equal", cmap="Blues")
    axes[2].set_title("Front View (YZ)")
    axes[2].set_xlabel("y (lu)")
    axes[2].set_ylabel("z (lu)")

    solid = stats["solid_cells"]
    fluid = stats["fluid_cells"]
    fig.text(
        0.5,
        0.01,
        f"Grid {stats['grid']}  |  Solid {solid:,}  |  Fluid {fluid:,}",
        ha="center",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    return fig


# ---------------------------------------------------------------------------
# STL export (marching-cubes-free: face enumeration of voxel surfaces)
# ---------------------------------------------------------------------------


def _write_stl_facet(lines: list[str], n: tuple, v0: tuple, v1: tuple, v2: tuple) -> None:
    lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
    lines.append("    outer loop")
    for v in (v0, v1, v2):
        lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
    lines.append("    endloop")
    lines.append("  endfacet")


def export_offshore_stl(
    struct_type: OffshoreStructureType | str,
    output_path: str | Path,
    nx: int = 60,
    ny: int = 60,
    nz: int = 60,
    **kwargs,
) -> Path:
    """Export an offshore structure as an ASCII STL voxel surface mesh.

    Parameters
    ----------
    struct_type : Structure family.
    output_path : Destination file path.
    nx, ny, nz  : Grid resolution (lower values → faster export).
    **kwargs    : Geometry parameter overrides.

    Returns
    -------
    Resolved Path to the written STL file.
    """
    if isinstance(struct_type, str):
        struct_type = OffshoreStructureType(struct_type)

    result = build_offshore_mask(struct_type, nx, ny, nz, **kwargs)
    mask_np = result["mask"].cpu().numpy().astype(np.int8)

    # Face enumeration: for each solid voxel, emit exposed faces.
    normals = [
        ((-1, 0, 0), (0, -1, 1), (0, 0, 1), "x-"),
        ((1, 0, 0), (0, 0, 1), (0, 1, 1), "x+"),
        ((0, -1, 0), (0, 0, 1), (1, 0, 1), "y-"),
        ((0, 1, 0), (1, 0, 1), (0, 0, 1), "y+"),
        ((0, 0, -1), (0, 1, 0), (1, 1, 0), "z-"),
        ((0, 0, 1), (1, 0, 1), (0, 1, 1), "z+"),
    ]

    lines: list[str] = [f"solid {struct_type.value}"]
    padded = np.pad(mask_np, 1, constant_values=0)

    # Expose XY faces
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if not padded[ix + 1, iy + 1, iz + 1]:
                    continue
                x, y, z = float(ix), float(iy), float(iz)
                # -X face
                if not padded[ix, iy + 1, iz + 1]:
                    _write_stl_facet(lines, (-1, 0, 0), (x, y, z), (x, y + 1, z), (x, y + 1, z + 1))
                    _write_stl_facet(lines, (-1, 0, 0), (x, y, z), (x, y + 1, z + 1), (x, y, z + 1))
                # +X face
                if not padded[ix + 2, iy + 1, iz + 1]:
                    _write_stl_facet(lines, (1, 0, 0), (x + 1, y, z), (x + 1, y + 1, z + 1), (x + 1, y + 1, z))
                    _write_stl_facet(lines, (1, 0, 0), (x + 1, y, z), (x + 1, y, z + 1), (x + 1, y + 1, z + 1))
                # -Y face
                if not padded[ix + 1, iy, iz + 1]:
                    _write_stl_facet(lines, (0, -1, 0), (x, y, z), (x + 1, y, z), (x + 1, y, z + 1))
                    _write_stl_facet(lines, (0, -1, 0), (x, y, z), (x + 1, y, z + 1), (x, y, z + 1))
                # +Y face
                if not padded[ix + 1, iy + 2, iz + 1]:
                    _write_stl_facet(lines, (0, 1, 0), (x, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y + 1, z))
                    _write_stl_facet(lines, (0, 1, 0), (x, y + 1, z), (x, y + 1, z + 1), (x + 1, y + 1, z + 1))
                # -Z face
                if not padded[ix + 1, iy + 1, iz]:
                    _write_stl_facet(lines, (0, 0, -1), (x, y, z), (x + 1, y + 1, z), (x + 1, y, z))
                    _write_stl_facet(lines, (0, 0, -1), (x, y, z), (x, y + 1, z), (x + 1, y + 1, z))
                # +Z face
                if not padded[ix + 1, iy + 1, iz + 2]:
                    _write_stl_facet(lines, (0, 0, 1), (x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1))
                    _write_stl_facet(lines, (0, 0, 1), (x, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1))

    lines.append(f"endsolid {struct_type.value}")

    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
