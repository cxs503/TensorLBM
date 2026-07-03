"""Parametric SUBOFF submarine hull CAD module for TensorLBM.

Implements a DARPA SUBOFF-inspired axisymmetric body of revolution with
an ellipsoidal bow, cylindrical parallel midbody, and polynomial stern
taper.  Three model variants are supported:

- **BARE_HULL** – naked axisymmetric body (AFF-1 equivalent).
- **WITH_SAIL**  – bare hull plus a conning-tower sail (AFF-3 equivalent).
- **FULL**       – bare hull, sail, and four cruciform stern appendages
                   (AFF-8 equivalent).

Reference geometry: Groves, N.C., Huang, T.T., Chang, M.S. (1989),
"Geometric Characteristics of DARPA SUBOFF Models", DTRC/SHD-1298-01.

Normalized geometry (default, matching DARPA SUBOFF AFF-1):
  - Hull length L (user-supplied lattice units)
  - Max radius  R = L / (2 × 8.57) ≈ 0.0583 L  (L/D ≈ 8.57)
  - Bow nose fraction: 0.233 L
  - Parallel midbody: 0.233 L – 0.748 L
  - Stern taper: 0.748 L – 1.0 L

Public API
----------
- :class:`SuboffHullType`         – model variant enum.
- :class:`SuboffConfig`           – parametric geometry configuration.
- :func:`suboff_radius_profile`   – normalized radius r(xi)/R_max.
- :func:`suboff_hull_mask`        – 3-D boolean mask (bare hull only).
- :func:`build_suboff_mask`       – convenience wrapper (mask + statistics).
- :func:`suboff_statistics`       – hull form coefficient dictionary.
- :func:`generate_suboff_previews` – multi-view matplotlib figure.
- :func:`export_suboff_stl`       – ASCII STL surface mesh export.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import matplotlib.figure

__all__ = [
    "SuboffHullType",
    "SuboffConfig",
    "suboff_radius_profile",
    "suboff_hull_mask",
    "build_suboff_mask",
    "suboff_statistics",
    "generate_suboff_previews",
    "export_suboff_stl",
    "suboff_mesh_data",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SuboffHullType(str, Enum):  # noqa: UP042
    """DARPA SUBOFF model variant."""

    BARE_HULL = "bare_hull"
    """Axisymmetric body of revolution only (AFF-1)."""

    WITH_SAIL = "with_sail"
    """Bare hull with conning-tower sail (AFF-3)."""

    FULL = "full"
    """Bare hull, sail, and four cruciform stern appendages (AFF-8)."""


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SuboffConfig:
    """Parametric SUBOFF submarine hull geometry configuration.

    All fractional parameters are normalized to hull length *L* (1 = full
    hull length).  Default values match the DARPA SUBOFF proportions.

    Parameters
    ----------
    r_over_l :
        Maximum radius / hull length.  Default 1/8.57 ≈ 0.1167 (DARPA SUBOFF).
    bow_fraction :
        Fraction of L used for the ellipsoidal nose.  Default 0.233.
    stern_fraction :
        Fraction of L used for the polynomial stern taper.  Default 0.252.
    stern_exponent :
        Polynomial exponent controlling stern taper sharpness.  ``n=2``
        gives a circular (semi-ellipsoidal) cross-section; larger values
        give a more blunt transition.  Default 2.0.
    sail_x_frac :
        Axial centre of the conning-tower sail as fraction of L from bow.
        Default 0.44.
    sail_length_frac :
        Axial length of the sail as fraction of L.  Default 0.12.
    sail_height_frac :
        Sail height (above hull surface) as fraction of L.  Default 0.14.
    sail_halfwidth_frac :
        Sail half-width as fraction of L.  Default 0.025.
    fin_x_frac :
        Axial centre of the cruciform stern fins as fraction of L.
        Default 0.87.
    fin_length_frac :
        Axial length of each fin as fraction of L.  Default 0.10.
    fin_span_frac :
        Radial span of each fin (from hull surface) as fraction of L.
        Default 0.12.
    fin_thickness_frac :
        Thickness of each fin as fraction of L.  Default 0.015.
    """

    # --- Main body ---
    r_over_l: float = 1.0 / (2.0 * 8.57)   # R/L ≈ 0.0583 (L/D ≈ 8.57)
    bow_fraction: float = 0.233
    stern_fraction: float = 0.252
    stern_exponent: float = 2.0

    # --- Sail (conning tower) ---
    sail_x_frac: float = 0.44
    sail_length_frac: float = 0.12
    sail_height_frac: float = 0.14
    sail_halfwidth_frac: float = 0.025

    # --- Cruciform stern appendages ---
    fin_x_frac: float = 0.87
    fin_length_frac: float = 0.10
    fin_span_frac: float = 0.12
    fin_thickness_frac: float = 0.015

    # --- Metadata (read-only) ---
    _label: str = field(default="DARPA SUBOFF-inspired", init=False, repr=False)


# ---------------------------------------------------------------------------
# Internal profile helpers
# ---------------------------------------------------------------------------


def suboff_radius_profile(
    xi: np.ndarray | float,
    config: SuboffConfig | None = None,
) -> np.ndarray:
    """Normalised hull radius *r(xi) / R_max* for the SUBOFF bare hull.

    Parameters
    ----------
    xi :
        Normalised axial coordinate ∈ [0, 1], where 0 = bow-tip and
        1 = stern-tip.
    config :
        Geometry configuration; uses :class:`SuboffConfig` defaults when
        *None*.

    Returns
    -------
    np.ndarray
        Normalised radius ∈ [0, 1] (same shape as *xi*).
    """
    if config is None:
        config = SuboffConfig()

    alpha = config.bow_fraction     # bow end (normalised)
    beta = config.stern_fraction    # stern start from aft
    n = config.stern_exponent

    xi = np.asarray(xi, dtype=float)
    r = np.zeros_like(xi)

    # Bow section: quarter-ellipse profile (r = 0 at tip, r = R_max at alpha)
    bow = (xi >= 0.0) & (xi < alpha)
    xi_bow = np.clip(xi[bow] / alpha, 0.0, 1.0)
    # quarter-ellipse: r = sqrt(1 - (1-t)^2) = sqrt(2t - t^2)
    r[bow] = np.sqrt(np.clip(2.0 * xi_bow - xi_bow**2, 0.0, 1.0))

    # Parallel midbody
    mid = (xi >= alpha) & (xi <= 1.0 - beta)
    r[mid] = 1.0

    # Stern taper: generalised-ellipse taper from R_max to 0
    stern = (xi > 1.0 - beta) & (xi <= 1.0)
    eta = np.clip((xi[stern] - (1.0 - beta)) / beta, 0.0, 1.0)
    if n == 2.0:
        # Circular quarter-ellipse: r = sqrt(1 - eta^2)
        r[stern] = np.sqrt(np.clip(1.0 - eta**2, 0.0, 1.0))
    else:
        # Generalised: r = (1 - eta^n)^(1/n)
        r[stern] = np.clip(1.0 - eta**n, 0.0, 1.0) ** (1.0 / n)

    return np.clip(r, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public mask generators
# ---------------------------------------------------------------------------


def suboff_hull_mask(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    length: float,
    radius: float,
    device: torch.device,
    config: SuboffConfig | None = None,
) -> torch.Tensor:
    """Boolean solid mask for the SUBOFF bare hull (axisymmetric body).

    The hull axis runs along the x-direction.  The body is a surface of
    revolution about the point ``(cx, cy, cz)`` in the y-z plane.

    Parameters
    ----------
    nx, ny, nz :
        Grid dimensions (x = axial / flow, y = transverse, z = vertical).
    cx :
        x-coordinate of the hull midship point (cells).
    cy :
        y-coordinate of the hull axis (cells).
    cz :
        z-coordinate of the hull axis (cells).
    length :
        Total hull length (lattice units).
    radius :
        Maximum hull radius (lattice units).  If ≤ 0, derived from
        ``config.r_over_l * length``.
    device :
        PyTorch device for the output tensor.
    config :
        Geometry configuration.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``(nz, ny, nx)``, *True* = solid cell.
    """
    if config is None:
        config = SuboffConfig()
    if radius <= 0.0:
        radius = config.r_over_l * length

    # Build coordinate grids
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Normalised axial position: xi = 0 at bow, 1 at stern
    # Bow is at cx - length/2, stern at cx + length/2
    x_bow = cx - length / 2.0
    xi_t = (xx - x_bow) / length  # 0 at bow, 1 at stern

    xi_np = xi_t.cpu().numpy()
    r_norm = suboff_radius_profile(xi_np, config)  # normalised [0, 1]

    # Actual radius in lattice units
    r_lu = torch.tensor(r_norm * radius, device=device, dtype=torch.float32)

    # Radial distance from axis in y-z plane
    r_grid = torch.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)

    # Solid where inside radius and within hull axial extent
    in_axial = (xi_t >= 0.0) & (xi_t <= 1.0)
    mask = in_axial & (r_grid <= r_lu)
    return mask


def _add_sail_mask(
    mask: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    length: float,
    radius: float,
    config: SuboffConfig,
    device: torch.device,
) -> torch.Tensor:
    """Add the conning-tower sail to an existing hull mask (in-place OR)."""
    x_bow = cx - length / 2.0

    # Sail axial extents
    sail_x_c = x_bow + config.sail_x_frac * length
    sail_half_len = config.sail_length_frac * length / 2.0

    # Sail y extents (centred on hull axis)
    sail_half_w = config.sail_halfwidth_frac * length

    # Sail z extents: from hull top (cz + radius) upward
    sail_z_bottom = cz + radius  # top of hull at axial centre of sail
    sail_z_top = sail_z_bottom + config.sail_height_frac * length

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    sail = (
        (xx >= sail_x_c - sail_half_len)
        & (xx <= sail_x_c + sail_half_len)
        & (yy >= cy - sail_half_w)
        & (yy <= cy + sail_half_w)
        & (zz >= sail_z_bottom)
        & (zz <= sail_z_top)
    )
    return mask | sail


def _add_fin_masks(
    mask: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    length: float,
    radius: float,
    config: SuboffConfig,
    device: torch.device,
) -> torch.Tensor:
    """Add cruciform stern appendages (4 fins) to an existing mask."""
    x_bow = cx - length / 2.0
    fin_x_c = x_bow + config.fin_x_frac * length
    fin_half_len = config.fin_length_frac * length / 2.0
    fin_span = config.fin_span_frac * length
    fin_half_t = config.fin_thickness_frac * length / 2.0

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    in_axial_fin = (xx >= fin_x_c - fin_half_len) & (xx <= fin_x_c + fin_half_len)

    # Top fin (z+): from cz+radius upward
    in_y_fin = (yy >= cy - fin_half_t) & (yy <= cy + fin_half_t)
    in_z_fin = (zz >= cz - fin_half_t) & (zz <= cz + fin_half_t)
    top = in_axial_fin & in_y_fin & (zz >= cz + radius) & (zz <= cz + radius + fin_span)
    # Bottom fin (z-)
    bot = in_axial_fin & in_y_fin & (zz <= cz - radius) & (zz >= cz - radius - fin_span)
    # Port fin (y+): from cy+radius outward
    port = in_axial_fin & in_z_fin & (yy >= cy + radius) & (yy <= cy + radius + fin_span)
    # Starboard fin (y-)
    stbd = in_axial_fin & in_z_fin & (yy <= cy - radius) & (yy >= cy - radius - fin_span)

    return mask | top | bot | port | stbd


def build_suboff_mask(
    hull_type: SuboffHullType | str = SuboffHullType.BARE_HULL,
    nx: int = 200,
    ny: int = 80,
    nz: int = 80,
    cx: float | None = None,
    cy: float | None = None,
    cz: float | None = None,
    length: float | None = None,
    radius: float | None = None,
    config: SuboffConfig | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Build a SUBOFF solid mask and return it with form statistics.

    Default placement: hull axis at grid centre, bow at ``cx - length/2``.
    Default hull length: ``0.6 * nx``; default radius from ``config.r_over_l``.

    Parameters
    ----------
    hull_type :
        Model variant: ``"bare_hull"``, ``"with_sail"``, or ``"full"``.
    nx, ny, nz :
        Grid dimensions.
    cx, cy, cz :
        Axis midpoint (cells).  Defaults to grid centre.
    length :
        Hull length (lattice units).  Defaults to ``0.6 * nx``.
    radius :
        Maximum hull radius (lattice units).  Defaults to
        ``config.r_over_l * length``.
    config :
        Parametric geometry; uses :class:`SuboffConfig` defaults when
        *None*.
    device :
        PyTorch device string.

    Returns
    -------
    mask : torch.Tensor, shape ``(nz, ny, nx)``, bool
    stats : dict
    """
    if isinstance(hull_type, str):
        hull_type = SuboffHullType(hull_type)
    if config is None:
        config = SuboffConfig()

    dev = torch.device(device)

    cx = float(cx) if cx is not None else nx / 2.0
    cy = float(cy) if cy is not None else ny / 2.0
    cz = float(cz) if cz is not None else nz / 2.0
    length = float(length) if length is not None else nx * 0.6
    radius = float(radius) if radius is not None else config.r_over_l * length

    # Build bare hull mask
    mask = suboff_hull_mask(nx, ny, nz, cx, cy, cz, length, radius, dev, config)

    # Add sail
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        mask = _add_sail_mask(mask, nx, ny, nz, cx, cy, cz, length, radius, config, dev)

    # Add cruciform fins
    if hull_type == SuboffHullType.FULL:
        mask = _add_fin_masks(mask, nx, ny, nz, cx, cy, cz, length, radius, config, dev)

    total = nx * ny * nz
    solid = int(mask.sum().item())

    stats_form = suboff_statistics(hull_type, length, radius, config)
    stats = {
        **stats_form,
        "solid_cells": solid,
        "fluid_cells": total - solid,
        "total_cells": total,
        "nx": nx, "ny": ny, "nz": nz,
        "cx": cx, "cy": cy, "cz": cz,
        "length": length, "radius": radius,
    }
    return mask, stats


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def suboff_statistics(
    hull_type: SuboffHullType | str,
    length: float,
    radius: float,
    config: SuboffConfig | None = None,
) -> dict:
    """Return hull form statistics for a SUBOFF model.

    Parameters
    ----------
    hull_type :
        Model variant.
    length :
        Hull length (any consistent unit).
    radius :
        Maximum hull radius.
    config :
        Parametric geometry.

    Returns
    -------
    dict
        Keys: hull_type, label, L_D_ratio, r_over_l, bow_fraction,
        stern_fraction, displacement_lu3, wetted_area_lu2,
        prismatic_coefficient.
    """
    if isinstance(hull_type, str):
        hull_type = SuboffHullType(hull_type)
    if config is None:
        config = SuboffConfig()

    diameter = 2.0 * radius
    l_d = length / diameter if diameter > 0 else float("nan")

    # Volume of bare hull (numerical integration)
    xi_int = np.linspace(0.0, 1.0, 2000)
    r_norm = suboff_radius_profile(xi_int, config)
    # V = pi * R_max^2 * L * integral of rho^2 dxi over [0,1]
    vol_bare = math.pi * radius**2 * length * float(np.trapz(r_norm**2, xi_int))

    # Wetted area of bare hull (surface of revolution)
    # A = 2*pi * R_max * L * integral of rho * sqrt(1 + (d rho/d xi * L/R)^2) dxi
    # Simplified without the derivative correction:
    circ_integral = float(np.trapz(r_norm, xi_int))
    wetted_bare = 2.0 * math.pi * radius * length * circ_integral

    # Prismatic coefficient (Cp = V / (A_max * L))
    a_max = math.pi * radius**2
    cp = vol_bare / (a_max * length) if (a_max * length) > 0 else float("nan")

    _labels = {
        SuboffHullType.BARE_HULL: "SUBOFF Bare Hull (AFF-1 inspired)",
        SuboffHullType.WITH_SAIL: "SUBOFF + Sail (AFF-3 inspired)",
        SuboffHullType.FULL: "SUBOFF Full Appendage (AFF-8 inspired)",
    }

    return {
        "hull_type": hull_type.value,
        "label": _labels[hull_type],
        "L_D_ratio": round(l_d, 3),
        "r_over_l": round(radius / length, 5) if length > 0 else None,
        "bow_fraction": config.bow_fraction,
        "stern_fraction": config.stern_fraction,
        "displacement_lu3": round(vol_bare, 2),
        "wetted_area_lu2": round(wetted_bare, 2),
        "prismatic_coefficient": round(float(cp), 4),
    }


# ---------------------------------------------------------------------------
# Preview figure
# ---------------------------------------------------------------------------


def generate_suboff_previews(
    hull_type: SuboffHullType | str = SuboffHullType.BARE_HULL,
    length: float = 100.0,
    radius: float | None = None,
    config: SuboffConfig | None = None,
) -> "matplotlib.figure.Figure":  # noqa: UP037
    """Generate a multi-view matplotlib figure for the SUBOFF model.

    The figure contains three subplots:

    1. **Side profile** – normalised radius vs. axial position.
    2. **Cross-sections** – circular cross-sections at several stations.
    3. **Top view** – plan view of the hull + sail outline.

    Parameters
    ----------
    hull_type :
        Model variant.
    length :
        Hull length (lattice units, for axis labels).
    radius :
        Maximum hull radius.  Derived from ``config.r_over_l * length``
        when *None*.
    config :
        Parametric geometry.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if isinstance(hull_type, str):
        hull_type = SuboffHullType(hull_type)
    if config is None:
        config = SuboffConfig()
    if radius is None:
        radius = config.r_over_l * length

    n_pts = 400
    xi = np.linspace(0.0, 1.0, n_pts)
    r_norm = suboff_radius_profile(xi, config)
    r_abs = r_norm * radius
    x_abs = xi * length

    stats = suboff_statistics(hull_type, length, radius, config)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Submarine Preview – {stats['label']}\n"
        f"L={length:.0f}  R_max={radius:.2f}  L/D={stats['L_D_ratio']:.2f}"
        f"  Cp={stats['prismatic_coefficient']:.3f}  (lattice units)",
        fontsize=10,
    )

    # --- Side profile ---
    ax = axes[0]
    ax.set_title("Side Profile")
    ax.set_xlabel("Axial position x (lu)")
    ax.set_ylabel("Radius r (lu)")
    ax.fill_between(x_abs, r_abs, -r_abs, alpha=0.35, color="#4472C4", label="Hull cross-section")
    ax.plot(x_abs, r_abs, "b-", linewidth=1.5)
    ax.plot(x_abs, -r_abs, "b-", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(0, length * 1.02)
    ax.set_ylim(-radius * 1.6, radius * 1.6)
    ax.set_aspect("equal")

    # Add sail in side profile
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        sail_xc = config.sail_x_frac * length
        sail_x0 = sail_xc - config.sail_length_frac * length / 2.0
        sail_w = config.sail_length_frac * length
        sail_h = config.sail_height_frac * length
        sail_z0 = radius  # top of hull at sail location (approx)
        ax.add_patch(mpatches.Rectangle(
            (sail_x0, sail_z0), sail_w, sail_h,
            color="#70AD47", alpha=0.6, label="Sail",
        ))

    # Add fin silhouettes
    if hull_type == SuboffHullType.FULL:
        fin_xc = config.fin_x_frac * length
        fin_x0 = fin_xc - config.fin_length_frac * length / 2.0
        fin_w = config.fin_length_frac * length
        fin_span = config.fin_span_frac * length
        # Top fin
        ax.add_patch(mpatches.Rectangle(
            (fin_x0, radius), fin_w, fin_span,
            color="#ED7D31", alpha=0.6, label="Fin",
        ))
        # Bottom fin
        ax.add_patch(mpatches.Rectangle(
            (fin_x0, -radius - fin_span), fin_w, fin_span,
            color="#ED7D31", alpha=0.6,
        ))

    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.3)

    # --- Cross-sections (body plan) ---
    ax = axes[1]
    ax.set_title("Cross-Sections (Body Plan)")
    ax.set_xlabel("y (lu)")
    ax.set_ylabel("z (lu)")
    stations = np.linspace(0.05, 0.95, 9)
    cmap = plt.get_cmap("RdYlGn", len(stations))
    for i, xi_s in enumerate(stations):
        r_s = float(suboff_radius_profile(np.array([xi_s]), config)[0]) * radius
        theta = np.linspace(0.0, 2 * math.pi, 120)
        ys = r_s * np.cos(theta)
        zs = r_s * np.sin(theta)
        ax.plot(ys, zs, color=cmap(i / max(len(stations) - 1, 1)), linewidth=1.0,
                label=f"x={xi_s * length:.0f}")
    ax.set_aspect("equal")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(-radius * 1.6, radius * 1.6)
    ax.set_ylim(-radius * 1.6, radius * 1.6)
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=6, loc="upper right")

    # Add sail cross-section in body plan (top half)
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        sail_hw = config.sail_halfwidth_frac * length
        sail_zbot = radius
        sail_ztop = radius + config.sail_height_frac * length
        ax.add_patch(mpatches.Rectangle(
            (-sail_hw, sail_zbot), 2 * sail_hw, sail_ztop - sail_zbot,
            color="#70AD47", alpha=0.5, label="Sail section",
        ))

    # Add fin cross-sections
    if hull_type == SuboffHullType.FULL:
        fin_span = config.fin_span_frac * length
        fin_ht = config.fin_thickness_frac * length
        # Vertical fins (top/bottom)
        ax.add_patch(mpatches.Rectangle(
            (-fin_ht / 2, radius), fin_ht, fin_span,
            color="#ED7D31", alpha=0.5,
        ))
        ax.add_patch(mpatches.Rectangle(
            (-fin_ht / 2, -radius - fin_span), fin_ht, fin_span,
            color="#ED7D31", alpha=0.5,
        ))
        # Horizontal fins (port/starboard)
        ax.add_patch(mpatches.Rectangle(
            (radius, -fin_ht / 2), fin_span, fin_ht,
            color="#ED7D31", alpha=0.5,
        ))
        ax.add_patch(mpatches.Rectangle(
            (-radius - fin_span, -fin_ht / 2), fin_span, fin_ht,
            color="#ED7D31", alpha=0.5,
        ))

    # --- Top view (plan view, y-x plane) ---
    ax = axes[2]
    ax.set_title("Top View (Plan)")
    ax.set_xlabel("Axial position x (lu)")
    ax.set_ylabel("Half-breadth y (lu)")
    ax.fill_between(x_abs, r_abs, -r_abs, alpha=0.35, color="#4472C4", label="Hull waterplane")
    ax.plot(x_abs, r_abs, "b-", linewidth=1.5)
    ax.plot(x_abs, -r_abs, "b-", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlim(0, length * 1.02)
    ax.set_ylim(-radius * 1.6, radius * 1.6)

    # Add sail top view
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        sail_xc = config.sail_x_frac * length
        sail_x0 = sail_xc - config.sail_length_frac * length / 2.0
        sail_w = config.sail_length_frac * length
        sail_hw = config.sail_halfwidth_frac * length
        ax.add_patch(mpatches.Rectangle(
            (sail_x0, -sail_hw), sail_w, 2 * sail_hw,
            color="#70AD47", alpha=0.6, label="Sail",
        ))

    # Add fin top view (port/starboard fins are visible from top)
    if hull_type == SuboffHullType.FULL:
        fin_xc = config.fin_x_frac * length
        fin_x0 = fin_xc - config.fin_length_frac * length / 2.0
        fin_w = config.fin_length_frac * length
        fin_span = config.fin_span_frac * length
        fin_ht = config.fin_thickness_frac * length
        ax.add_patch(mpatches.Rectangle(
            (fin_x0, radius), fin_w, fin_span,
            color="#ED7D31", alpha=0.5, label="Fin",
        ))
        ax.add_patch(mpatches.Rectangle(
            (fin_x0, -radius - fin_span), fin_w, fin_span,
            color="#ED7D31", alpha=0.5,
        ))

    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------------


def export_suboff_stl(
    hull_type: SuboffHullType | str = SuboffHullType.BARE_HULL,
    length: float = 100.0,
    radius: float | None = None,
    n_axial: int = 80,
    n_circ: int = 60,
    config: SuboffConfig | None = None,
    output_path: str | Path = "suboff.stl",
) -> Path:
    """Export a triangulated SUBOFF surface mesh as ASCII STL.

    The bare hull is tessellated as a surface of revolution.  The sail and
    fins are approximated as closed triangulated boxes.

    Parameters
    ----------
    hull_type :
        Model variant.
    length :
        Hull length (any consistent unit).
    radius :
        Maximum hull radius.  Derived from ``config.r_over_l * length``
        when *None*.
    n_axial :
        Number of axial sampling points on the hull surface.
    n_circ :
        Number of circumferential sampling points per cross-section.
    config :
        Parametric geometry.
    output_path :
        Destination STL file path.

    Returns
    -------
    Path
        Absolute path to the written STL file.
    """
    if isinstance(hull_type, str):
        hull_type = SuboffHullType(hull_type)
    if config is None:
        config = SuboffConfig()
    if radius is None:
        radius = config.r_over_l * length

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    triangles = _build_suboff_triangles(hull_type, length, radius, n_axial, n_circ, config)

    # ---- Write STL ----
    def _normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        n = np.cross(v1 - v0, v2 - v0)
        mag = float(np.linalg.norm(n))
        return n / mag if mag > 1e-12 else n

    solid_name = f"suboff_{hull_type.value}"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"solid {solid_name}\n")
        for v0, v1, v2 in triangles:
            n = _normal(np.asarray(v0), np.asarray(v1), np.asarray(v2))
            f.write(
                f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n"
                f"    outer loop\n"
                f"      vertex {float(v0[0]):.6e} {float(v0[1]):.6e} {float(v0[2]):.6e}\n"
                f"      vertex {float(v1[0]):.6e} {float(v1[1]):.6e} {float(v1[2]):.6e}\n"
                f"      vertex {float(v2[0]):.6e} {float(v2[1]):.6e} {float(v2[2]):.6e}\n"
                f"    endloop\n"
                f"  endfacet\n"
            )
        f.write(f"endsolid {solid_name}\n")

    return output_path


# ---------------------------------------------------------------------------
# Internal STL box helper
# ---------------------------------------------------------------------------


def _box_triangles(
    tris: list,
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
) -> None:
    """Append 12 triangles for a closed box to *tris* (x/y/z extents)."""
    # 8 corners
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # top
    ])
    # 6 faces (outward normals)
    faces = [
        (0, 2, 1), (0, 3, 2),   # bottom (-z)
        (4, 5, 6), (4, 6, 7),   # top (+z)
        (0, 1, 5), (0, 5, 4),   # front (-y)
        (2, 3, 7), (2, 7, 6),   # back (+y)
        (0, 4, 7), (0, 7, 3),   # left (-x)
        (1, 2, 6), (1, 6, 5),   # right (+x)
    ]
    for f in faces:
        tris.append((corners[f[0]], corners[f[1]], corners[f[2]]))


def _box_triangles_yz(
    tris: list,
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
) -> None:
    """Alias of ``_box_triangles`` for fins oriented in the y-direction."""
    _box_triangles(tris, x0, x1, y0, y1, z0, z1)


# ---------------------------------------------------------------------------
# Shared triangle builder (used by both STL export and mesh3d)
# ---------------------------------------------------------------------------


def _build_suboff_triangles(
    hull_type: SuboffHullType,
    length: float,
    radius: float,
    n_axial: int,
    n_circ: int,
    config: SuboffConfig,
) -> list:
    """Build the complete list of triangles for a SUBOFF model.

    Returns a list of ``(v0, v1, v2)`` tuples where each vertex is a
    1-D numpy array of shape ``(3,)``.  This list is consumed by both
    :func:`export_suboff_stl` and :func:`suboff_mesh_data`.
    """
    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # ---- Hull surface of revolution ----
    xi_arr = np.linspace(0.0, 1.0, n_axial)
    r_arr = suboff_radius_profile(xi_arr, config) * radius
    theta_arr = np.linspace(0.0, 2 * math.pi, n_circ, endpoint=False)

    X = xi_arr[:, None] * length * np.ones((1, n_circ))
    Y = r_arr[:, None] * np.cos(theta_arr[None, :])
    Z = r_arr[:, None] * np.sin(theta_arr[None, :])

    for i in range(n_axial - 1):
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            p00 = np.array([X[i, j],         Y[i, j],         Z[i, j]])
            p10 = np.array([X[i + 1, j],     Y[i + 1, j],     Z[i + 1, j]])
            p01 = np.array([X[i, j_next],     Y[i, j_next],     Z[i, j_next]])
            p11 = np.array([X[i + 1, j_next], Y[i + 1, j_next], Z[i + 1, j_next]])
            triangles.append((p00, p10, p11))
            triangles.append((p00, p11, p01))

    # Bow cap
    bow_tip = np.array([0.0, 0.0, 0.0])
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        triangles.append((bow_tip,
                           np.array([X[0, j],      Y[0, j],      Z[0, j]]),
                           np.array([X[0, j_next], Y[0, j_next], Z[0, j_next]])))

    # Stern cap
    stern_tip = np.array([length, 0.0, 0.0])
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        triangles.append((stern_tip,
                           np.array([X[-1, j_next], Y[-1, j_next], Z[-1, j_next]]),
                           np.array([X[-1, j],      Y[-1, j],      Z[-1, j]])))

    # ---- Sail (conning tower) ----
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        sail_xc = config.sail_x_frac * length
        sail_x0 = sail_xc - config.sail_length_frac * length / 2.0
        sail_x1 = sail_xc + config.sail_length_frac * length / 2.0
        sail_hw = config.sail_halfwidth_frac * length
        r_at_sail = float(suboff_radius_profile(np.array([config.sail_x_frac]), config)[0]) * radius
        sail_z0 = r_at_sail
        sail_z1 = r_at_sail + config.sail_height_frac * length
        _box_triangles(triangles, sail_x0, sail_x1, -sail_hw, sail_hw, sail_z0, sail_z1)

    # ---- Cruciform fins ----
    if hull_type == SuboffHullType.FULL:
        fin_xc = config.fin_x_frac * length
        fin_x0 = fin_xc - config.fin_length_frac * length / 2.0
        fin_x1 = fin_xc + config.fin_length_frac * length / 2.0
        fin_span = config.fin_span_frac * length
        fin_ht = config.fin_thickness_frac * length
        r_at_fin = float(suboff_radius_profile(np.array([config.fin_x_frac]), config)[0]) * radius
        _box_triangles(triangles, fin_x0, fin_x1,
                       -fin_ht / 2, fin_ht / 2,
                       r_at_fin, r_at_fin + fin_span)
        _box_triangles(triangles, fin_x0, fin_x1,
                       -fin_ht / 2, fin_ht / 2,
                       -(r_at_fin + fin_span), -r_at_fin)
        _box_triangles_yz(triangles, fin_x0, fin_x1,
                          r_at_fin, r_at_fin + fin_span,
                          -fin_ht / 2, fin_ht / 2)
        _box_triangles_yz(triangles, fin_x0, fin_x1,
                          -(r_at_fin + fin_span), -r_at_fin,
                          -fin_ht / 2, fin_ht / 2)

    return triangles


# ---------------------------------------------------------------------------
# Three.js mesh data export
# ---------------------------------------------------------------------------


def suboff_mesh_data(
    hull_type: SuboffHullType | str = SuboffHullType.BARE_HULL,
    length: float = 100.0,
    radius: float | None = None,
    n_axial: int = 60,
    n_circ: int = 48,
    config: SuboffConfig | None = None,
) -> dict:
    """Return SUBOFF mesh data as a dict suitable for Three.js rendering.

    The returned dictionary contains:

    ``positions``
        A flat Python list of ``float`` values representing triangle vertex
        positions in interleaved XYZ order: ``[x0,y0,z0, x1,y1,z1, x2,y2,z2,
        x3,y3,z3, …]``.  Each consecutive group of 9 values is one triangle.
        This maps directly to a Three.js ``Float32Array`` / ``BufferGeometry``
        ``position`` attribute (non-indexed).

    ``n_triangles``
        Total number of triangles.

    ``hull_type``
        The hull variant string (``"bare_hull"``, ``"with_sail"``, ``"full"``).

    Parameters
    ----------
    hull_type :
        SUBOFF model variant.
    length :
        Hull length in lattice units.
    radius :
        Maximum hull radius.  Auto-derived from ``config.r_over_l * length``
        when *None*.
    n_axial :
        Axial resolution of the surface-of-revolution tessellation.
    n_circ :
        Circumferential resolution.
    config :
        Parametric geometry overrides.

    Returns
    -------
    dict
        ``{"positions": [...], "n_triangles": int, "hull_type": str}``
    """
    if isinstance(hull_type, str):
        hull_type = SuboffHullType(hull_type)
    if config is None:
        config = SuboffConfig()
    if radius is None:
        radius = config.r_over_l * length

    triangles = _build_suboff_triangles(hull_type, length, radius, n_axial, n_circ, config)

    positions: list[float] = []
    for v0, v1, v2 in triangles:
        for v in (v0, v1, v2):
            positions.extend([float(v[0]), float(v[1]), float(v[2])])

    return {
        "positions": positions,
        "n_triangles": len(triangles),
        "hull_type": hull_type.value,
    }
