"""Parametric SUBOFF submarine hull CAD module for TensorLBM.

Implements the **real DARPA SUBOFF geometry** from the actual offset
tables (Groves, Huang, Chang 1989, DTRC/SHD-1298-01), ported from
``suboff8.py``.  The hull uses the 4-segment DARPA polynomial equations
(bow, parallel midbody, stern taper, stern cap).  The sail uses the
3-segment polynomial profile with a semi-elliptical cap.  The stern
appendages use swept NACA 4-digit airfoils with SUBOFF-specific
coefficients ``(0.2969, 0.126, 0.3516, 0.2852, 0.1045)``.

Three model variants are supported:

- **BARE_HULL** – naked axisymmetric body (AFF-1 equivalent).
- **WITH_SAIL**  – bare hull plus sail (AFF-3 equivalent).
- **FULL**       – bare hull, sail, and four cruciform stern appendages
                   (AFF-8 equivalent).

Real SUBOFF dimensions:
  - L = 14.292 ft = 4.356 m
  - D = 0.508 m (R = 0.254 m)
  - L/D = 8.57
  - Cp ≈ 0.79 (prismatic coefficient)

The geometry works internally in feet (matching the DARPA offset tables)
and converts to lattice units via ``ft_per_lu = L_ft / length``.

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


# --------------------------------------------------------------------------- #
# Real DARPA SUBOFF geometry constants (feet)
# --------------------------------------------------------------------------- #
# Source: Groves, Huang, Chang (1989), DTRC/SHD-1298-01.
# These are the actual offset-table dimensions used by suboff8.py.

_SUBOFF_L_FT = 14.291667  # Total hull length (ft)  = 4.356 m
_SUBOFF_RMAX_FT = 0.8333333  # Max hull radius (ft)    = 0.254 m
_M_TO_FT = 3.2808399  # metres → feet

# Hull segment boundaries (ft)
_BOW_END_FT = 3.333333  # Bow end / parallel midbody start
_MID_END_FT = 10.645833  # Parallel midbody end / stern taper start
_STERN_END_FT = 13.979167  # Stern taper end / stern cap start

# Sail (conning tower) constants (ft)
_SAIL_ZMAX = 0.109375  # Max half-thickness (transverse) of sail
_SAIL_YTMP = 1.507813  # Sail height boundary (vertical, up)
_SAIL_X1_START = 3.032986  # Segment 1 (entrance) start
_SAIL_X1_END = 3.358507  # Segment 1 end / segment 2 start
_SAIL_X2_END = 3.559028  # Segment 2 (middle) end / segment 3 start
_SAIL_X3_END = 4.241319  # Segment 3 (exit) end

# Stern appendage (fin) constants (ft)
_FIN_H = 13.146284  # Fin root chord axial position
_FIN_R_INNER = 0.075  # Fin inner radius (from axis)
_FIN_R_OUTER = 0.825  # Fin outer radius
_FIN_SWEEP_K = -0.466308  # Sweep slope
_FIN_SWEEP_C = 0.88859  # Sweep intercept

# NACA 4-digit thickness coefficients (SUBOFF variant, slightly different
# from the standard 0.2843/0.1015 closure)
_NACA_COEFFS = (0.2969, 0.126, 0.3516, 0.2852, 0.1045)


def _ft_per_lu(length: float) -> float:
    """Feet-per-lattice-unit conversion factor.

    ``dx`` (metres per lattice unit) = L_m / length, and
    ``ft_per_lu`` = dx × 3.2808399 = L_ft / length.
    """
    return _SUBOFF_L_FT / length


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


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
    r_over_l: float = 1.0 / (2.0 * 8.57)  # R/L ≈ 0.0583 (L/D ≈ 8.57)
    bow_fraction: float = 0.233
    stern_fraction: float = 0.252
    stern_exponent: float = 2.0

    # --- Sail (conning tower) ---
    sail_x_frac: float = 0.254  # real SUBOFF: centre at ~25.4% L
    sail_length_frac: float = 0.085  # real: 0.369 ft / 4.356 m
    sail_height_frac: float = 0.106  # real: 1.508 ft above centreline
    sail_halfwidth_frac: float = 0.008  # real: Zmax = 0.109 ft half-width

    # --- Cruciform stern appendages ---
    fin_x_frac: float = 0.890  # real SUBOFF: trailing edge at 92% L, centre ~89%
    fin_length_frac: float = 0.060  # real: chord 0.504–0.854 ft ≈ 0.06 L
    fin_span_frac: float = 0.052  # real: 0.075–0.825 ft radial span ≈ 0.052 L
    fin_thickness_frac: float = 0.008  # real: max NACA 0015 thickness ≈ 0.15*c/L

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

    **真实 DARPA SUBOFF 4段公式** (Groves et al. 1989, from suboff8.py):
      段1 船首 [0, 0.2333]:  多项式 + 指数 1/2.1
      段2 平行中体 (0.2333, 0.7417]:  R = R_max
      段3 尾锥 (0.7417, 0.9748]:  6阶多项式
      段4 尾尖 (0.9748, 1.0]:  椭圆收缩

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

    xi = np.asarray(xi, dtype=float)
    r = np.zeros_like(xi)

    # SUBOFF 真实参数 (ft → normalized)
    L_ft = 14.291667  # 总长 (ft)
    Rmax_ft = 0.8333333  # 最大半径 (ft)
    BOW_END = 3.333333 / L_ft  # 0.2333
    MID_END = 10.645833 / L_ft  # 0.7449
    STERN_END = 13.979167 / L_ft  # 0.9781

    # 段1: 船首 [0, 0.2333]
    bow = (xi >= 0.0) & (xi < BOW_END)
    if np.any(bow):
        x_ft = xi[bow] * L_ft
        tmp = 0.3 * x_ft - 1.0
        tmp2 = tmp * tmp
        tmp4 = tmp2 * tmp2
        a1 = (
            1.126395101 * x_ft * tmp4
            + 0.442874707 * x_ft * x_ft * (tmp2 * tmp)
            + 1.0
            - tmp4 * (1.2 * x_ft + 1.0)
        )
        a1 = np.maximum(a1, 0.0)
        r[bow] = np.power(a1, 1.0 / 2.1)

    # 段2: 平行中体 (0.2333, 0.7449]
    mid = (xi >= BOW_END) & (xi <= MID_END)
    r[mid] = 1.0

    # 段3: 尾锥 (0.7449, 0.9781]
    stern_taper = (xi > MID_END) & (xi <= STERN_END)
    if np.any(stern_taper):
        r1 = 0.1175
        k0 = 10.0
        k1 = 44.6244
        x_ft = xi[stern_taper] * L_ft
        ksi = (13.979167 - x_ft) / 3.333333
        ksi2 = ksi * ksi
        ksi3 = ksi2 * ksi
        ksi4 = ksi3 * ksi
        ksi5 = ksi4 * ksi
        ksi6 = ksi5 * ksi

        a3 = (
            r1 * r1
            + r1 * k0 * ksi2
            + (20.0 - 20.0 * r1 * r1 - 4.0 * r1 * k0 - k1 / 3.0) * ksi3
            + (-45.0 + 45.0 * r1 * r1 + 6.0 * r1 * k0 + k1) * ksi4
            + (36.0 - 36.0 * r1 * r1 - 4.0 * r1 * k0 - k1) * ksi5
            + (-10.0 + 10.0 * r1 * r1 + r1 * k0 + k1 / 3.0) * ksi6
        )
        a3 = np.maximum(a3, 0.0)
        r[stern_taper] = np.sqrt(a3)

    # 段4: 尾尖 (0.9781, 1.0]
    tail = (xi > STERN_END) & (xi <= 1.0)
    if np.any(tail):
        x_ft = xi[tail] * L_ft
        val = 1.0 - (3.2 * x_ft - 44.733333) ** 2
        val = np.maximum(val, 0.0)
        r[tail] = np.sqrt(val) * 0.1175

    return np.clip(r, 0.0, 1.0)


# ---------------------------------------------------------------------------
# NACA 0015 airfoil thickness helper
# ---------------------------------------------------------------------------

# Precompute the NACA 0015 maximum half-thickness for normalisation.
_x_sample = np.linspace(0.0, 1.0, 10000)
_t_sample = 0.15  # NACA 0015 thickness parameter
_yt_sample = (_t_sample / 0.2) * (
    0.2969 * np.sqrt(_x_sample)
    - 0.1260 * _x_sample
    - 0.3516 * _x_sample**2
    + 0.2843 * _x_sample**3
    - 0.1015 * _x_sample**5
)
_NACA0015_MAX_THICKNESS: float = float(np.max(np.maximum(_yt_sample, 0.0)))


def _naca0015_thickness(x_norm: np.ndarray | float) -> np.ndarray:
    """Normalised NACA 0015 airfoil half-thickness, ∈ [0, 1].

    Uses the symmetric NACA 4-digit thickness equation with ``t = 0.15``
    (NACA 0015).  The result is normalised so the maximum half-thickness
    (at *x* ≈ 0.30) equals 1.0.

    Parameters
    ----------
    x_norm :
        Normalised chord position ∈ [0, 1] (0 = leading edge,
        1 = trailing edge).  Values outside this range are clipped.

    Returns
    -------
    np.ndarray
        Normalised half-thickness ∈ [0, 1].
    """
    x = np.asarray(x_norm, dtype=float)
    x = np.clip(x, 0.0, 1.0)
    t = 0.15  # NACA 0015 thickness parameter
    yt = (t / 0.2) * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**5
    )
    yt = np.maximum(yt, 0.0)
    return yt / _NACA0015_MAX_THICKNESS


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
    """Add the real DARPA SUBOFF sail (3-segment polynomial + arc top).

    The sail uses the actual DARPA SUBOFF offset-table geometry from
    ``suboff8.py`` / Groves et al. (1989).  Three axial segments are
    defined:

    - **Entrance** (3.033–3.359 ft): 4th-power polynomial z-profile.
    - **Middle**  (3.359–3.559 ft): constant ``Zmax = 0.109375`` ft.
    - **Exit**   (3.559–4.241 ft): 4th-power polynomial z-profile.

    Each segment has a rectangular body (height ``y_tmp = 1.507813`` ft
    in the vertical direction, half-thickness ``z_tmp`` in the
    transverse direction) topped by a semi-elliptical cap.

    Coordinate convention (target): *z* = vertical (up), *y* =
    transverse.  This is a y↔z swap of suboff8.py's convention.
    """
    ftlu = _ft_per_lu(length)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    x_bow = cx - length / 2.0
    # Centered coords → feet.  z = up (sail on +z), y = transverse.
    x_ft = (xx - x_bow) * ftlu
    y_ft = (yy - cy) * ftlu  # transverse
    z_ft = (zz - cz) * ftlu  # vertical (up)

    Zmax = _SAIL_ZMAX  # 0.109375 ft — max half-thickness
    y_tmp = _SAIL_YTMP  # 1.507813 ft — sail height boundary

    # --- Segment 1: entrance (_SAIL_X1_START < x < _SAIL_X1_END) ---
    m1 = (x_ft > _SAIL_X1_START) & (x_ft < _SAIL_X1_END)
    D = 3.0720 * (x_ft - _SAIL_X1_START)
    C = 1.0 - torch.pow(D - 1.0, 4) * (4.0 * D + 1.0)
    B = (1.0 / 3.0) * D * D * torch.pow(D - 1.0, 3)
    A = 2.0 * D * torch.pow(D - 1.0, 4)
    z_tmp = Zmax * torch.sqrt(torch.clamp(2.094759 * A + 0.2071781 * B + C, min=0))

    sail1 = ((z_ft <= y_tmp) & (z_ft > 0) & (y_ft > -z_tmp) & (y_ft < z_tmp)) & m1
    z_upper = (z_ft > y_tmp) & (z_ft < (y_tmp + z_tmp / 2))
    z2 = torch.sqrt(torch.clamp(z_tmp * z_tmp - torch.pow(2 * (z_ft - y_tmp), 2), min=0))
    sail1_top = (z_upper & (y_ft > -z2) & (y_ft < z2)) & m1

    # --- Segment 2: middle (_SAIL_X1_END < x <= _SAIL_X2_END) ---
    m2 = (x_ft > _SAIL_X1_END) & (x_ft <= _SAIL_X2_END)
    sail2 = ((z_ft <= y_tmp) & (z_ft > 0) & (y_ft > -Zmax) & (y_ft < Zmax)) & m2
    z_upper2 = (z_ft > y_tmp) & (z_ft < (y_tmp + Zmax / 2))
    z2_2 = torch.sqrt(torch.clamp(Zmax * Zmax - torch.pow(2 * (z_ft - y_tmp), 2), min=0))
    sail2_top = (z_upper2 & (y_ft > -z2_2) & (y_ft < z2_2)) & m2

    # --- Segment 3: exit (_SAIL_X2_END < x <= _SAIL_X3_END) ---
    m3 = (x_ft <= _SAIL_X3_END) & (x_ft > _SAIL_X2_END)
    E = (_SAIL_X3_END - x_ft) / 0.6822917
    F = E - 1.0
    G = 2.238361 * E * torch.pow(F, 4)
    H = 3.106529 * (E * E) * torch.pow(F, 3)
    P = 1.0 - torch.pow(F, 4) * (4.0 * E + 1.0)
    z_tmp3 = Zmax * (G + H + P)

    sail3 = ((z_ft <= y_tmp) & (z_ft > 0) & (y_ft > -z_tmp3) & (y_ft < z_tmp3)) & m3
    z_upper3 = (z_ft > y_tmp) & (z_ft < (y_tmp + z_tmp3 / 2))
    z2_3 = torch.sqrt(torch.clamp(z_tmp3 * z_tmp3 - torch.pow(2 * (z_ft - y_tmp), 2), min=0))
    sail3_top = (z_upper3 & (y_ft > -z2_3) & (y_ft < z2_3)) & m3

    sail = sail1 | sail1_top | sail2 | sail2_top | sail3 | sail3_top
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
    """Add real DARPA SUBOFF cruciform stern appendages.

    Uses the actual NACA 4-digit thickness equation with SUBOFF-specific
    coefficients ``(0.2969, 0.126, 0.3516, 0.2852, 0.1045)`` and a sweep
    angle that shifts the chord aft as the radial position increases:

    ``cy = -0.466308 * r + 0.88859``,  ``s = (x - h) / cy + 1``

    Four fins are generated (cruciform):
    - **fin_v** (|y| ∈ [0.075, 0.825] ft): extends in y, thickness in z.
    - **fin_h** (|z| ∈ [0.075, 0.825] ft): extends in z, thickness in y.
    """
    ftlu = _ft_per_lu(length)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    x_bow = cx - length / 2.0
    x_ft = (xx - x_bow) * ftlu
    y_ft = torch.abs(yy - cy) * ftlu
    z_ft = torch.abs(zz - cz) * ftlu

    h = _FIN_H  # 13.146284 ft
    cy_val = _FIN_SWEEP_K * y_ft + _FIN_SWEEP_C  # sweep for y-fins
    cz_val = _FIN_SWEEP_K * z_ft + _FIN_SWEEP_C  # sweep for z-fins

    s = (x_ft - h) / cy_val + 1.0  # chord parameter for y-fins
    sz = (x_ft - h) / cz_val + 1.0  # chord parameter for z-fins

    # --- Fins extending in y (thickness in z) — port/starboard ---
    mask_s = (s > 0) & (s < 1)
    a = _NACA_COEFFS[0] * torch.sqrt(torch.clamp(s, min=0))
    b = _NACA_COEFFS[1] * s
    c = _NACA_COEFFS[2] * s * s
    d = _NACA_COEFFS[3] * s * s * s
    e = _NACA_COEFFS[4] * s * s * s * s
    z_suboff = a - b - c + d - e
    fin_v = (
        (y_ft > _FIN_R_INNER) & (y_ft < _FIN_R_OUTER) & (z_ft > -z_suboff) & (z_ft < z_suboff)
    ) & mask_s

    # --- Fins extending in z (thickness in y) — top/bottom ---
    mask_sz = (sz > 0) & (sz < 1)
    a_h = _NACA_COEFFS[0] * torch.sqrt(torch.clamp(sz, min=0))
    b_h = _NACA_COEFFS[1] * sz
    c_h = _NACA_COEFFS[2] * sz * sz
    d_h = _NACA_COEFFS[3] * sz * sz * sz
    e_h = _NACA_COEFFS[4] * sz * sz * sz * sz
    y_suboff = a_h - b_h - c_h + d_h - e_h
    fin_h = (
        (z_ft > _FIN_R_INNER) & (z_ft < _FIN_R_OUTER) & (y_ft > -y_suboff) & (y_ft < y_suboff)
    ) & mask_sz

    fins = fin_v | fin_h
    return mask | fins


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
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "length": length,
        "radius": radius,
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
    vol_bare = math.pi * radius**2 * length * float(np.trapezoid(r_norm**2, xi_int))

    # Wetted area of bare hull (surface of revolution)
    # A = 2*pi * R_max * L * integral of rho * sqrt(1 + (d rho/d xi * L/R)^2) dxi
    # Simplified without the derivative correction:
    circ_integral = float(np.trapezoid(r_norm, xi_int))
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

    # --- Real geometry helpers for preview ---
    _ftlu = _ft_per_lu(length)
    _inv = 1.0 / _ftlu  # ft → lu

    # Sail dimensions (lattice units)
    sail_x0 = _SAIL_X1_START * _inv
    sail_x1 = _SAIL_X3_END * _inv
    sail_body_h = _SAIL_YTMP * _inv  # rectangular body height
    sail_cap_h = (_SAIL_ZMAX / 2) * _inv  # max cap height above body

    # Fin dimensions (lattice units)
    fin_r_in = _FIN_R_INNER * _inv
    fin_r_out = _FIN_R_OUTER * _inv
    # Hull radius at fin axial location (stern taper)
    xi_fin = _FIN_H / _SUBOFF_L_FT
    r_hull_fin = float(suboff_radius_profile(np.array([xi_fin]), config)[0]) * radius

    # Add sail in side profile (real 3-segment polynomial + cap)
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        n_sp = 80
        x_ft_sp = np.linspace(_SAIL_X1_START, _SAIL_X3_END, n_sp)
        z_tmp_sp = _sail_half_thickness_np(x_ft_sp)
        x_lu_sp = x_ft_sp * _inv
        z_bot = radius  # hull surface at sail (midbody = R_max)
        z_top = z_bot + (_SAIL_YTMP + z_tmp_sp / 2) * _inv
        ax.fill_between(x_lu_sp, z_top, z_bot, alpha=0.5, color="#70AD47", label="Sail")
        ax.plot(x_lu_sp, z_top, "g-", linewidth=1.0)

    # Add fin silhouettes in side profile (swept NACA)
    if hull_type == SuboffHullType.FULL:
        n_fp = 50
        s_fp = np.linspace(0, 1, n_fp)
        # Fin leading/trailing edge x at root and tip
        for r_ft, ls in [(_FIN_R_INNER, "-"), (_FIN_R_OUTER, "--")]:
            cy = _FIN_SWEEP_K * r_ft + _FIN_SWEEP_C
            x_ft_fp = _FIN_H + (s_fp - 1) * cy
            x_lu_fp = x_ft_fp * _inv
            ax.plot(
                x_lu_fp,
                np.full_like(x_lu_fp, r_hull_fin + (r_ft - r_hull_fin)),
                color="#ED7D31",
                linewidth=0.8,
                linestyle=ls,
                alpha=0.6,
            )
        # Filled fin shape (top + bottom)
        fin_top = r_hull_fin + (fin_r_out - r_hull_fin)
        cy_root = _FIN_SWEEP_K * _FIN_R_INNER + _FIN_SWEEP_C
        cy_tip = _FIN_SWEEP_K * _FIN_R_OUTER + _FIN_SWEEP_C
        x_le_root = (_FIN_H - cy_root) * _inv
        x_te_root = _FIN_H * _inv
        x_le_tip = (_FIN_H - cy_tip) * _inv
        x_te_tip = _FIN_H * _inv
        # Top fin polygon
        ax.add_patch(
            mpatches.Polygon(
                [
                    [x_le_root, r_hull_fin],
                    [x_te_root, r_hull_fin],
                    [x_te_tip, fin_top],
                    [x_le_tip, fin_top],
                ],
                closed=True,
                color="#ED7D31",
                alpha=0.4,
                label="Fin",
            )
        )
        # Bottom fin polygon
        ax.add_patch(
            mpatches.Polygon(
                [
                    [x_le_root, -r_hull_fin],
                    [x_te_root, -r_hull_fin],
                    [x_te_tip, -fin_top],
                    [x_le_tip, -fin_top],
                ],
                closed=True,
                color="#ED7D31",
                alpha=0.4,
            )
        )

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
        ax.plot(
            ys,
            zs,
            color=cmap(i / max(len(stations) - 1, 1)),
            linewidth=1.0,
            label=f"x={xi_s * length:.0f}",
        )
    ax.set_aspect("equal")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(-radius * 1.6, radius * 1.6)
    ax.set_ylim(-radius * 1.6, radius * 1.6)
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=6, loc="upper right")

    # Add sail cross-section in body plan (real shape: rectangle + elliptical cap)
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        zmax_lu = _SAIL_ZMAX * _inv
        ytmp_lu = _SAIL_YTMP * _inv
        z_bot = radius
        z_top_body = z_bot + ytmp_lu
        z_top_cap = z_bot + ytmp_lu + zmax_lu / 2
        # Rectangular body
        ax.add_patch(
            mpatches.Rectangle(
                (-zmax_lu, z_bot),
                2 * zmax_lu,
                ytmp_lu,
                color="#70AD47",
                alpha=0.5,
                label="Sail section",
            )
        )
        # Elliptical cap (semi-ellipse, semi-axes zmax_lu in y, zmax_lu/2 in z)
        cap_theta = np.linspace(0, math.pi, 30)
        cap_y = zmax_lu * np.cos(cap_theta)
        cap_z = z_top_body + (zmax_lu / 2) * np.sin(cap_theta)
        ax.fill(cap_y, cap_z, color="#70AD47", alpha=0.4)

    # Add fin cross-sections (real NACA 4-digit airfoil)
    if hull_type == SuboffHullType.FULL:
        n_fc = 30
        s_fc = np.linspace(0, 1, n_fc)
        t_fc = _naca4_thickness_np(s_fc) * _inv  # half-thickness (lu)
        # Top fin (extends in z, thickness in y) — show airfoil cross-section
        cy_mid = _FIN_SWEEP_K * ((_FIN_R_INNER + _FIN_R_OUTER) / 2) + _FIN_SWEEP_C
        x_mid_ft = _FIN_H + (s_fc - 1) * cy_mid
        # Scale airfoil to fit in the cross-section view
        fin_scale = (fin_r_out - r_hull_fin) / max(t_fc.max(), 1e-8) * 0.3
        fin_y = t_fc * fin_scale
        fin_z = r_hull_fin + s_fc * (fin_r_out - r_hull_fin)
        ax.plot(fin_y, fin_z, color="#ED7D31", linewidth=1.0, label="Fin section")
        ax.plot(-fin_y, fin_z, color="#ED7D31", linewidth=1.0)
        ax.plot(fin_z, fin_y, color="#ED7D31", linewidth=1.0)
        ax.plot(fin_z, -fin_y, color="#ED7D31", linewidth=1.0)

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

    # Add sail top view (real width profile: 2*z_tmp along x)
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        n_st = 80
        x_ft_st = np.linspace(_SAIL_X1_START, _SAIL_X3_END, n_st)
        z_tmp_st = _sail_half_thickness_np(x_ft_st)
        x_lu_st = x_ft_st * _inv
        hw_st = z_tmp_st * _inv  # half-width in y
        ax.fill_between(x_lu_st, hw_st, -hw_st, alpha=0.5, color="#70AD47", label="Sail")
        ax.plot(x_lu_st, hw_st, "g-", linewidth=1.0)
        ax.plot(x_lu_st, -hw_st, "g-", linewidth=1.0)

    # Add fin top view (port/starboard fins visible — swept NACA)
    if hull_type == SuboffHullType.FULL:
        cy_root = _FIN_SWEEP_K * _FIN_R_INNER + _FIN_SWEEP_C
        cy_tip = _FIN_SWEEP_K * _FIN_R_OUTER + _FIN_SWEEP_C
        x_le_root = (_FIN_H - cy_root) * _inv
        x_te_root = _FIN_H * _inv
        x_le_tip = (_FIN_H - cy_tip) * _inv
        x_te_tip = _FIN_H * _inv
        # Port fin (y+)
        ax.add_patch(
            mpatches.Polygon(
                [
                    [x_le_root, r_hull_fin],
                    [x_te_root, r_hull_fin],
                    [x_te_tip, fin_r_out],
                    [x_le_tip, fin_r_out],
                ],
                closed=True,
                color="#ED7D31",
                alpha=0.4,
                label="Fin",
            )
        )
        # Starboard fin (y-)
        ax.add_patch(
            mpatches.Polygon(
                [
                    [x_le_root, -r_hull_fin],
                    [x_te_root, -r_hull_fin],
                    [x_te_tip, -fin_r_out],
                    [x_le_tip, -fin_r_out],
                ],
                closed=True,
                color="#ED7D31",
                alpha=0.4,
            )
        )

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
    fins are tessellated as NACA 0015 airfoil-profiled surfaces.

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
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> None:
    """Append 12 triangles for a closed box to *tris* (x/y/z extents)."""
    # 8 corners
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],  # bottom
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],  # top
        ]
    )
    # 6 faces (outward normals)
    faces = [
        (0, 2, 1),
        (0, 3, 2),  # bottom (-z)
        (4, 5, 6),
        (4, 6, 7),  # top (+z)
        (0, 1, 5),
        (0, 5, 4),  # front (-y)
        (2, 3, 7),
        (2, 7, 6),  # back (+y)
        (0, 4, 7),
        (0, 7, 3),  # left (-x)
        (1, 2, 6),
        (1, 6, 5),  # right (+x)
    ]
    for f in faces:
        tris.append((corners[f[0]], corners[f[1]], corners[f[2]]))


def _box_triangles_yz(
    tris: list,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> None:
    """Alias of ``_box_triangles`` for fins oriented in the y-direction."""
    _box_triangles(tris, x0, x1, y0, y1, z0, z1)


# --------------------------------------------------------------------------- #
# Real DARPA SUBOFF surface-mesh helpers (sail + fins)
# --------------------------------------------------------------------------- #


def _sail_half_thickness_np(x_ft):
    """Sail half-thickness *z_tmp* (ft) at axial position *x_ft* (ft).

    Numpy implementation of the 3-segment DARPA polynomial.
    """
    x_ft = np.asarray(x_ft, dtype=float)
    z_tmp = np.zeros_like(x_ft)
    Zmax = _SAIL_ZMAX

    # Segment 1: entrance
    m1 = (x_ft > _SAIL_X1_START) & (x_ft < _SAIL_X1_END)
    if np.any(m1):
        D = 3.0720 * (x_ft[m1] - _SAIL_X1_START)
        C = 1.0 - (D - 1.0) ** 4 * (4.0 * D + 1.0)
        B = (1.0 / 3.0) * D * D * (D - 1.0) ** 3
        A = 2.0 * D * (D - 1.0) ** 4
        z_tmp[m1] = Zmax * np.sqrt(np.maximum(2.094759 * A + 0.2071781 * B + C, 0))

    # Segment 2: middle
    m2 = (x_ft > _SAIL_X1_END) & (x_ft <= _SAIL_X2_END)
    z_tmp[m2] = Zmax

    # Segment 3: exit
    m3 = (x_ft <= _SAIL_X3_END) & (x_ft > _SAIL_X2_END)
    if np.any(m3):
        E = (_SAIL_X3_END - x_ft[m3]) / 0.6822917
        F = E - 1.0
        G = 2.238361 * E * F**4
        H = 3.106529 * E * E * F**3
        P = 1.0 - F**4 * (4.0 * E + 1.0)
        z_tmp[m3] = Zmax * (G + H + P)

    return np.maximum(z_tmp, 0.0)


def _naca4_thickness_np(s):
    """NACA 4-digit half-thickness at chord position *s* (0=LE, 1=TE).

    Uses the SUBOFF-specific coefficients ``(0.2969, 0.126, 0.3516,
    0.2852, 0.1045)`` with a 4th-power trailing-edge closure.
    """
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    a, b, c, d, e = _NACA_COEFFS
    t = a * np.sqrt(np.maximum(s, 0)) - b * s - c * s**2 + d * s**3 - e * s**4
    return np.maximum(t, 0.0)


def _sail_cross_section_pts(z_tmp: float, n_arc: int = 12) -> list[tuple[float, float]]:
    """Cross-section outline ``(y_ft, z_ft)`` for the sail at a given *z_tmp*.

    Goes from bottom-left counterclockwise to bottom-right.
    The bottom edge (inside the hull) is left open.
    """
    y_tmp = _SAIL_YTMP
    if z_tmp < 1e-8:
        return []
    pts: list[tuple[float, float]] = [(-z_tmp, 0.0), (-z_tmp, y_tmp)]
    # Semi-elliptical cap: θ from π → 0
    for j in range(1, n_arc):
        theta = math.pi * (1.0 - j / n_arc)
        y = z_tmp * math.cos(theta)
        z = y_tmp + (z_tmp / 2.0) * math.sin(theta)
        pts.append((y, z))
    pts.append((z_tmp, y_tmp))
    pts.append((z_tmp, 0.0))
    return pts


def _real_sail_triangles(
    tris: list,
    length: float,
    n_axial: int = 30,
    n_arc: int = 12,
) -> None:
    """Append surface triangles for the real SUBOFF sail.

    Works in lattice units: hull from x=0 to x=length, centred at y=z=0.
    The sail sits on +z (vertical up) with thickness in y (transverse).
    """
    ftlu = _ft_per_lu(length)
    inv = 1.0 / ftlu  # ft → lu

    x_ft_arr = np.linspace(_SAIL_X1_START, _SAIL_X3_END, n_axial)
    z_tmp_arr = _sail_half_thickness_np(x_ft_arr)
    outlines = [_sail_cross_section_pts(zt, n_arc) for zt in z_tmp_arr]

    for i in range(n_axial - 1):
        p0_list = outlines[i]
        p1_list = outlines[i + 1]
        n = min(len(p0_list), len(p1_list))
        if n < 2:
            continue
        x0 = x_ft_arr[i] * inv
        x1 = x_ft_arr[i + 1] * inv
        for j in range(n - 1):
            a = np.array([x0, p0_list[j][0] * inv, p0_list[j][1] * inv])
            b = np.array([x0, p0_list[j + 1][0] * inv, p0_list[j + 1][1] * inv])
            c = np.array([x1, p1_list[j + 1][0] * inv, p1_list[j + 1][1] * inv])
            d = np.array([x1, p1_list[j][0] * inv, p1_list[j][1] * inv])
            tris.append((a, b, c))
            tris.append((a, c, d))


def _loft_one_fin(
    tris: list,
    x_lu: np.ndarray,
    r_lu: np.ndarray,
    t_lu: np.ndarray,
    span_axis: str,
    sign: int,
) -> None:
    """Loft upper/lower surfaces + caps for one swept NACA fin.

    *x_lu* has shape ``(n_span, n_chord)``; *r_lu* is ``(n_span,)``;
    *t_lu* is ``(n_chord,)``.
    """
    n_span, n_chord = x_lu.shape

    def _pt(i: int, j: int, thick_sign: int) -> np.ndarray:
        if span_axis == "y":
            return np.array([x_lu[i, j], sign * r_lu[i], thick_sign * t_lu[j]])
        return np.array([x_lu[i, j], thick_sign * t_lu[j], sign * r_lu[i]])

    # --- Upper (+) and lower (−) surfaces ---
    for ts in (+1, -1):
        for i in range(n_span - 1):
            for j in range(n_chord - 1):
                p00 = _pt(i, j, ts)
                p10 = _pt(i, j + 1, ts)
                p11 = _pt(i + 1, j + 1, ts)
                p01 = _pt(i + 1, j, ts)
                if ts > 0:
                    tris.append((p00, p10, p11))
                    tris.append((p00, p11, p01))
                else:
                    tris.append((p00, p11, p10))
                    tris.append((p00, p01, p11))

    # --- Root cap (i=0) and tip cap (i=n_span-1) ---
    for cap_i, i in enumerate((0, n_span - 1)):
        for j in range(n_chord - 1):
            pu = _pt(i, j, +1)
            pu_n = _pt(i, j + 1, +1)
            pl = _pt(i, j, -1)
            pl_n = _pt(i, j + 1, -1)
            if cap_i == 1:  # tip
                tris.append((pu, pu_n, pl_n))
                tris.append((pu, pl_n, pl))
            else:  # root
                tris.append((pu, pl_n, pu_n))
                tris.append((pu, pl, pl_n))

    # --- Trailing-edge cap (s=1, nearly closed) ---
    te = t_lu[-1]
    if te > 1e-10:
        for i in range(n_span - 1):
            pu0 = _pt(i, -1, +1)
            pu1 = _pt(i + 1, -1, +1)
            pl0 = _pt(i, -1, -1)
            pl1 = _pt(i + 1, -1, -1)
            tris.append((pu0, pu1, pl1))
            tris.append((pu0, pl1, pl0))


def _real_fin_triangles(
    tris: list,
    length: float,
    n_chord: int = 20,
    n_span: int = 10,
) -> None:
    """Append surface triangles for the 4 real SUBOFF cruciform stern fins.

    Each fin is a swept NACA 4-digit airfoil.  Works in lattice units:
    hull from x=0 to x=length, centred at y=z=0.
    """
    ftlu = _ft_per_lu(length)
    inv = 1.0 / ftlu
    h = _FIN_H

    r_arr = np.linspace(_FIN_R_INNER, _FIN_R_OUTER, n_span)
    s_arr = np.linspace(0, 1, n_chord)
    t_arr = _naca4_thickness_np(s_arr)

    cy_arr = _FIN_SWEEP_K * r_arr + _FIN_SWEEP_C
    # x_ft[i, j] = h + (s_j - 1) * cy_i  → (n_span, n_chord)
    x_ft = h + (s_arr[None, :] - 1.0) * cy_arr[:, None]

    x_lu = x_ft * inv
    r_lu = r_arr * inv
    t_lu = t_arr * inv

    # 4 fins: (span_axis, sign)
    for span_axis, sign in (("y", +1), ("y", -1), ("z", +1), ("z", -1)):
        _loft_one_fin(tris, x_lu, r_lu, t_lu, span_axis, sign)


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
            p00 = np.array([X[i, j], Y[i, j], Z[i, j]])
            p10 = np.array([X[i + 1, j], Y[i + 1, j], Z[i + 1, j]])
            p01 = np.array([X[i, j_next], Y[i, j_next], Z[i, j_next]])
            p11 = np.array([X[i + 1, j_next], Y[i + 1, j_next], Z[i + 1, j_next]])
            triangles.append((p00, p10, p11))
            triangles.append((p00, p11, p01))

    # Bow cap
    bow_tip = np.array([0.0, 0.0, 0.0])
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        triangles.append(
            (
                bow_tip,
                np.array([X[0, j], Y[0, j], Z[0, j]]),
                np.array([X[0, j_next], Y[0, j_next], Z[0, j_next]]),
            )
        )

    # Stern cap
    stern_tip = np.array([length, 0.0, 0.0])
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        triangles.append(
            (
                stern_tip,
                np.array([X[-1, j_next], Y[-1, j_next], Z[-1, j_next]]),
                np.array([X[-1, j], Y[-1, j], Z[-1, j]]),
            )
        )

    # ---- Sail (real DARPA 3-segment polynomial + arc top) ----
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        _real_sail_triangles(triangles, length)

    # ---- Cruciform fins (real swept NACA 4-digit) ----
    if hull_type == SuboffHullType.FULL:
        _real_fin_triangles(triangles, length)

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
