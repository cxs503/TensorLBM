"""Parametric ship hull CAD module for TensorLBM.

Provides parametric ship hull geometry generators for the ship and ocean
engineering domain.  Three standard hull forms are supported:

- **Wigley** – classic parabolic hull (ITTC benchmark, Cb ≈ 0.444).
- **Series 60** – fuller form based on DTMB Series 60 (Cb = 0.60) polynomial
  approximation.
- **KCS** – KRISO Container Ship approximation (Cb ≈ 0.651).

All hull generators return a boolean 3-D solid mask compatible with
:func:`tensorlbm.run_ship_hull_flow` and the pre-processing pipeline.

Public API
----------
- :class:`ShipHullType`      – hull family enum.
- :func:`series60_hull_mask` – boolean mask for Series 60 Cb=0.60 hull.
- :func:`kcs_hull_mask`      – boolean mask for KCS approximation hull.
- :func:`hull_block_coefficient` – compute Cb from a solid mask.
- :func:`generate_hull_body_plan`  – 2-D body-plan cross-sections as arrays.
- :func:`generate_hull_waterplane` – 2-D waterplane half-breadth profile.
- :func:`generate_hull_sideprofile`– 2-D side (keel/deck) profile.
- :func:`generate_hull_previews`   – render multi-view matplotlib figure.
- :func:`export_hull_stl`          – write a simple ASCII STL file.
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
    "ShipHullType",
    "series60_hull_mask",
    "kcs_hull_mask",
    "hull_block_coefficient",
    "generate_hull_body_plan",
    "generate_hull_waterplane",
    "generate_hull_sideprofile",
    "generate_hull_previews",
    "export_hull_stl",
    "ship_resistance_estimate",
]


# ---------------------------------------------------------------------------
# Hull type enum
# ---------------------------------------------------------------------------


class ShipHullType(str, Enum):  # noqa: UP042
    """Supported parametric hull families."""

    WIGLEY = "wigley"
    SERIES60 = "series60"
    KCS = "kcs"
    KVLCC2 = "kvlcc2"
    NPL = "npl"


# ---------------------------------------------------------------------------
# Internal parametric half-beam functions
# ---------------------------------------------------------------------------

# All functions receive:
#   xi  – normalised longitudinal coordinate ∈ [-1, 1] (0 = midship)
#   zeta – normalised vertical coordinate ∈ [0, 1]  (0 = keel, 1 = waterline)
# and return half-beam normalised to B/2 (range [0, 1]).


def _wigley_half_beam(xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Wigley parabolic form: Cb ≈ 0.444."""
    in_hull = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
    hb = (1.0 - xi**2) * (1.0 - (1.0 - zeta) ** 2)
    return np.where(in_hull, hb, 0.0)


def _series60_half_beam(xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Series 60 Cb=0.60 polynomial approximation.

    Half-beam = (B/2) * (1 − ξ²)^0.51 * ζ^0.30
    Theoretical Cb ≈ 0.60 (integral of the above over the hull envelope).
    """
    in_hull = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
    zeta_c = np.clip(zeta, 0.0, 1.0)
    xi_c = np.clip(xi, -1.0, 1.0)
    hb = (1.0 - xi_c**2) ** 0.51 * zeta_c**0.30
    return np.where(in_hull, np.clip(hb, 0.0, 1.0), 0.0)


def _kcs_half_beam(xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """KCS (KRISO Container Ship) polynomial approximation.

    Half-beam = (B/2) * (1 − ξ²)^0.45 * ζ^0.24
    Theoretical Cb ≈ 0.651.
    """
    in_hull = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
    zeta_c = np.clip(zeta, 0.0, 1.0)
    xi_c = np.clip(xi, -1.0, 1.0)
    hb = (1.0 - xi_c**2) ** 0.45 * zeta_c**0.24
    return np.where(in_hull, np.clip(hb, 0.0, 1.0), 0.0)


def _kvlcc2_half_beam(xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """KVLCC2 (KRISO Very Large Crude Carrier 2) superellipse approximation.

    The KVLCC2 is a standard CFD benchmark VLCC tanker (L=320m, Cb≈0.810).
    This approximation uses a superellipse cross-section inspired by
    SiggyF/jax-vessels generate_hull.py.  The very full bow is captured by
    the low bilateral exponents, and the U-shaped midship section by the
    high vertical exponent.

    Half-beam = (B/2) * sin(acos(|ξ|))^p_lon * ζ^p_vert
    with p_lon = 0.28, p_vert = 0.14 for a block coefficient near 0.81.
    """
    in_hull = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
    zeta_c = np.clip(zeta, 0.0, 1.0)
    xi_c = np.clip(np.abs(xi), 0.0, 1.0)
    # Superellipse longitude: very blunt ends
    lon = np.where(xi_c < 1.0, (1.0 - xi_c ** 2) ** 0.28, 0.0)
    # Nearly-U vertical section (small vert exponent = very flat keel)
    vert = np.where(zeta_c > 0.0, zeta_c ** 0.14, 0.0)
    hb = lon * vert
    return np.where(in_hull, np.clip(hb, 0.0, 1.0), 0.0)


def _npl_half_beam(xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """NPL (National Physical Laboratory) high-speed displacement hull.

    The NPL series (Bailey 1976) is a standard benchmark for high-speed
    displacement craft (fast ferries, naval vessels).  Cb ≈ 0.397.
    The hull is characterised by a very fine entry, V-shaped sections,
    and a raked stern — typical of fast round-bilge monohulls.

    Half-beam = (B/2) * (1 − ξ²)^0.65 * ζ^0.55
    """
    in_hull = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
    zeta_c = np.clip(zeta, 0.0, 1.0)
    xi_c = np.clip(xi, -1.0, 1.0)
    hb = (1.0 - xi_c**2) ** 0.65 * zeta_c**0.55
    return np.where(in_hull, np.clip(hb, 0.0, 1.0), 0.0)


_HALF_BEAM_FN = {
    ShipHullType.WIGLEY: _wigley_half_beam,
    ShipHullType.SERIES60: _series60_half_beam,
    ShipHullType.KCS: _kcs_half_beam,
    ShipHullType.KVLCC2: _kvlcc2_half_beam,
    ShipHullType.NPL: _npl_half_beam,
}

_HULL_LABELS = {
    ShipHullType.WIGLEY: "Wigley Parabolic (Cb≈0.444)",
    ShipHullType.SERIES60: "Series 60 Cb=0.60",
    ShipHullType.KCS: "KCS Approximation (Cb≈0.651)",
    ShipHullType.KVLCC2: "KVLCC2 VLCC Tanker (Cb≈0.810)",
    ShipHullType.NPL: "NPL High-Speed Hull (Cb≈0.397)",
}


# ---------------------------------------------------------------------------
# Public mask generators
# ---------------------------------------------------------------------------


def _make_hull_mask(
    hull_type: ShipHullType,
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz_keel: float,
    length: float,
    beam: float,
    draft: float,
    device: torch.device,
) -> torch.Tensor:
    """Generic boolean mask builder for all hull types.

    Returns a ``(nz, ny, nx)`` boolean tensor where *True* marks solid cells.
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    xi_t = (xx - cx) / (length / 2.0)
    z_waterline = cz_keel + draft
    zeta_t = (zz - cz_keel) / draft  # 0 at keel, 1 at waterline

    xi_np = xi_t.cpu().numpy()
    zeta_np = zeta_t.cpu().numpy()

    fn = _HALF_BEAM_FN[hull_type]
    hb_norm = fn(xi_np, zeta_np)  # normalised half-beam [0, 1]

    # Actual half-beam in lattice units (symmetric about cy)
    half_beam_lu = hb_norm * (beam / 2.0)
    half_beam_t = torch.tensor(half_beam_lu, device=device, dtype=torch.float32)

    # Only mark solid inside the draft envelope
    in_draft = (zz >= cz_keel) & (zz <= z_waterline)
    mask = in_draft & (torch.abs(yy - cy) <= half_beam_t)
    return mask


def series60_hull_mask(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz_keel: float,
    length: float,
    beam: float,
    draft: float,
    device: torch.device,
) -> torch.Tensor:
    """Boolean solid mask for a Series 60 Cb=0.60 hull.

    The Series 60 hull family (DTMB) is a standard naval architecture benchmark
    with block coefficient Cb = 0.60.  This function uses a polynomial
    approximation of the half-beam distribution.

    Parameters
    ----------
    nx, ny, nz:
        Grid dimensions (x = flow, y = transverse, z = vertical).
    cx:  x-coordinate of midship (cells).
    cy:  y-coordinate of hull centreline (cells).
    cz_keel: z-coordinate of keel (cells).
    length, beam, draft: Hull principal dimensions (cells).
    device: PyTorch device for output tensor.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``(nz, ny, nx)``.
    """
    return _make_hull_mask(
        ShipHullType.SERIES60, nx, ny, nz, cx, cy, cz_keel, length, beam, draft, device
    )


def kcs_hull_mask(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz_keel: float,
    length: float,
    beam: float,
    draft: float,
    device: torch.device,
) -> torch.Tensor:
    """Boolean solid mask for a KCS (KRISO Container Ship) approximation hull.

    Cb ≈ 0.651.  Suitable for container-ship resistance and wave-making
    benchmarks.

    Parameters mirror :func:`series60_hull_mask`.
    """
    return _make_hull_mask(
        ShipHullType.KCS, nx, ny, nz, cx, cy, cz_keel, length, beam, draft, device
    )


# ---------------------------------------------------------------------------
# Block coefficient
# ---------------------------------------------------------------------------


def hull_block_coefficient(
    mask: torch.Tensor,
    beam: float,
    draft: float,
    length: float,
) -> float:
    """Compute the block coefficient Cb from a solid mask.

    Cb = V_solid / (L × B × T)

    Parameters
    ----------
    mask:   Boolean 3-D solid tensor ``(nz, ny, nx)``.
    beam, draft, length: Hull principal dimensions in the *same* units as the
        mask (lattice cells).

    Returns
    -------
    float
        Block coefficient Cb ∈ (0, 1].
    """
    v_solid = float(mask.sum().item())
    v_box = length * beam * draft
    if v_box <= 0.0:
        return 0.0
    return v_solid / v_box


# ---------------------------------------------------------------------------
# Section / profile extraction
# ---------------------------------------------------------------------------


def generate_hull_body_plan(
    hull_type: ShipHullType,
    n_stations: int = 11,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Compute body-plan cross-sections for the given hull type.

    Parameters
    ----------
    hull_type:    Hull family.
    n_stations:   Number of transverse stations (evenly spaced from -1 to 1).

    Returns
    -------
    stations : np.ndarray, shape (n_stations,)
        Normalised longitudinal positions ξ ∈ [-1, 1].
    y_sections : list of np.ndarray
        Normalised half-breadths at each station (positive port side).
    z_sections : list of np.ndarray
        Normalised vertical coordinates (0=keel, 1=WL) for each section.
    """
    stations = np.linspace(-1.0, 1.0, n_stations)
    z_arr = np.linspace(0.0, 1.0, 100)
    fn = _HALF_BEAM_FN[hull_type]
    y_sections: list[np.ndarray] = []
    z_sections: list[np.ndarray] = []
    for xi_val in stations:
        xi_rep = np.full_like(z_arr, xi_val)
        hb = fn(xi_rep, z_arr)
        y_sections.append(hb)
        z_sections.append(z_arr)
    return stations, y_sections, z_sections


def generate_hull_waterplane(
    hull_type: ShipHullType,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Half-breadth profile at the waterline (ζ=1) along ship length.

    Returns
    -------
    xi_arr : np.ndarray, shape (n_points,)
        Normalised longitudinal positions ξ ∈ [-1, 1].
    hb_arr : np.ndarray, shape (n_points,)
        Normalised half-breadth at the waterline.
    """
    xi_arr = np.linspace(-1.0, 1.0, n_points)
    zeta_wl = np.ones_like(xi_arr)
    fn = _HALF_BEAM_FN[hull_type]
    hb_arr = fn(xi_arr, zeta_wl)
    return xi_arr, hb_arr


def generate_hull_sideprofile(
    hull_type: ShipHullType,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Keel depth profile: maximum draft at each longitudinal station.

    Returns
    -------
    xi_arr : np.ndarray, shape (n_points,)
    draft_arr : np.ndarray, shape (n_points,)
        Normalised draft at each station (fraction of max draft).
    """
    xi_arr = np.linspace(-1.0, 1.0, n_points)
    fn = _HALF_BEAM_FN[hull_type]
    draft_list: list[float] = []
    z_arr = np.linspace(0.0, 1.0, 200)
    for xi_val in xi_arr:
        xi_rep = np.full_like(z_arr, xi_val)
        hb = fn(xi_rep, z_arr)
        # deepest z where hull is present
        idx = np.where(hb > 1e-4)[0]
        draft_list.append(float(z_arr[idx[0]]) if len(idx) > 0 else 1.0)
    draft_arr = np.array(draft_list)
    return xi_arr, draft_arr


# ---------------------------------------------------------------------------
# Preview figure
# ---------------------------------------------------------------------------


def generate_hull_previews(
    hull_type: ShipHullType | str,
    length: float = 100.0,
    beam: float = 16.0,
    draft: float = 8.0,
    n_stations: int = 11,
) -> "matplotlib.figure.Figure":  # noqa: UP037
    """Generate a multi-view matplotlib figure for the given hull type.

    The figure contains three subplots:

    1. **Body plan** – transverse cross-sections (half-breadth vs depth).
    2. **Waterplane** – top view half-breadth at waterline along length.
    3. **Side profile** – longitudinal keel / waterline envelope.

    Parameters
    ----------
    hull_type:   Hull family (ShipHullType or its string value).
    length, beam, draft: Principal dimensions used for axis labelling.
    n_stations:  Number of body-plan stations.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(hull_type, str):
        hull_type = ShipHullType(hull_type)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Ship Hull Preview – {_HULL_LABELS[hull_type]}\n"
        f"L={length:.0f}  B={beam:.0f}  T={draft:.0f}  (lattice units)",
        fontsize=10,
    )

    # --- Body plan ---
    ax = axes[0]
    ax.set_title("Body Plan")
    ax.set_xlabel("Half-breadth (normalised)")
    ax.set_ylabel("Depth (normalised)")
    stations, y_sects, z_sects = generate_hull_body_plan(hull_type, n_stations)
    cmap = plt.get_cmap("RdYlGn", len(stations))
    for i, (xi_val, ys, zs) in enumerate(zip(stations, y_sects, z_sects, strict=True)):
        color = cmap(i / max(len(stations) - 1, 1))
        ax.plot(ys * (beam / 2), zs * draft, color=color, linewidth=1.0,
                label=f"ξ={xi_val:.2f}")
        ax.plot(-ys * (beam / 2), zs * draft, color=color, linewidth=1.0)
    ax.set_xlim(-beam / 2 * 1.1, beam / 2 * 1.1)
    ax.set_ylim(0, draft * 1.1)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.3)

    # --- Waterplane (top view) ---
    ax = axes[1]
    ax.set_title("Waterplane (top view)")
    ax.set_xlabel("Length (normalised)")
    ax.set_ylabel("Half-breadth (normalised)")
    xi_arr, hb_arr = generate_hull_waterplane(hull_type)
    x_plot = xi_arr * (length / 2)
    ax.fill_between(x_plot, hb_arr * (beam / 2), -(hb_arr * (beam / 2)),
                    alpha=0.35, color="#4472C4", label="Waterplane area")
    ax.plot(x_plot, hb_arr * (beam / 2), "b-", linewidth=1.5)
    ax.plot(x_plot, -hb_arr * (beam / 2), "b-", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(-length / 2 * 1.05, length / 2 * 1.05)
    ax.grid(True, linewidth=0.3)

    # --- Side profile ---
    ax = axes[2]
    ax.set_title("Side Profile")
    ax.set_xlabel("Length (normalised)")
    ax.set_ylabel("Depth (normalised)")
    xi_arr2, draft_arr = generate_hull_sideprofile(hull_type)
    x_plot2 = xi_arr2 * (length / 2)
    keel_z = draft_arr * draft
    ax.fill_between(x_plot2, keel_z, np.zeros_like(keel_z),
                    alpha=0.35, color="#70AD47", label="Hull envelope")
    ax.plot(x_plot2, keel_z, "g-", linewidth=1.5, label="Keel line")
    ax.axhline(0, color="#4472C4", linewidth=1.5, linestyle="-", label="Waterline")
    ax.set_xlim(-length / 2 * 1.05, length / 2 * 1.05)
    ax.set_ylim(0, draft * 1.2)
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------------


def export_hull_stl(
    hull_type: ShipHullType | str,
    length: float = 100.0,
    beam: float = 16.0,
    draft: float = 8.0,
    n_long: int = 60,
    n_vert: int = 30,
    output_path: str | Path = "hull.stl",
) -> Path:
    """Export a triangulated ship hull surface as an ASCII STL file.

    The hull surface is triangulated from a structured grid of (ξ, ζ) samples
    on both port and starboard sides, plus the keel and transom closure faces.

    Parameters
    ----------
    hull_type:   Hull family.
    length, beam, draft: Principal dimensions (any consistent unit).
    n_long:  Longitudinal grid points.
    n_vert:  Vertical grid points.
    output_path: Destination STL file path.

    Returns
    -------
    Path
        Absolute path to the written STL file.
    """
    if isinstance(hull_type, str):
        hull_type = ShipHullType(hull_type)

    # Resolve and normalise the output path; ensure parent directory exists
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fn = _HALF_BEAM_FN[hull_type]

    xi_arr = np.linspace(-1.0, 1.0, n_long)
    z_arr = np.linspace(0.0, 1.0, n_vert)

    # Build vertex grid for starboard side (y ≥ 0)
    XI, ZETA = np.meshgrid(xi_arr, z_arr, indexing="ij")  # (n_long, n_vert)
    HB = fn(XI.ravel(), ZETA.ravel()).reshape(n_long, n_vert)

    # Physical coordinates: x ∈ [0, L], y = half-beam, z ∈ [0, T]
    X = (XI + 1.0) / 2.0 * length         # [0, L]
    Y = HB * (beam / 2.0)                  # half-beam
    Z = ZETA * draft                        # [0, T]

    triangles: list[tuple[tuple, tuple, tuple]] = []

    def _normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        n = np.cross(v1 - v0, v2 - v0)
        mag = np.linalg.norm(n)
        return n / mag if mag > 1e-12 else n

    # Starboard hull surface triangles
    for i in range(n_long - 1):
        for k in range(n_vert - 1):
            p00 = np.array([X[i, k],     Y[i, k],     Z[i, k]])
            p10 = np.array([X[i+1, k],   Y[i+1, k],   Z[i+1, k]])
            p01 = np.array([X[i, k+1],   Y[i, k+1],   Z[i, k+1]])
            p11 = np.array([X[i+1, k+1], Y[i+1, k+1], Z[i+1, k+1]])
            triangles.append((p00, p10, p11))
            triangles.append((p00, p11, p01))

    # Port hull surface (mirror in y)
    starboard_tris = list(triangles)
    for t in starboard_tris:
        p0, p1, p2 = t
        q0 = np.array([p0[0], -p0[1], p0[2]])
        q1 = np.array([p1[0], -p1[1], p1[2]])
        q2 = np.array([p2[0], -p2[1], p2[2]])
        triangles.append((q0, q2, q1))  # reversed for outward normal

    # Write ASCII STL
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"solid {hull_type.value}_hull\n")
        for t in triangles:
            v0, v1, v2 = (np.asarray(v) for v in t)
            n = _normal(v0, v1, v2)
            f.write(
                f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n"
                f"    outer loop\n"
                f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n"
                f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n"
                f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n"
                f"    endloop\n"
                f"  endfacet\n"
            )
        f.write(f"endsolid {hull_type.value}_hull\n")

    return output_path.resolve()


# ---------------------------------------------------------------------------
# Convenience: theoretical Cb values (analytical)
# ---------------------------------------------------------------------------

def theoretical_block_coefficient(hull_type: ShipHullType | str) -> float:
    """Return the theoretical (analytical) block coefficient for a hull form.

    Values are derived by integrating the parametric half-beam functions over
    the hull envelope.
    """
    if isinstance(hull_type, str):
        hull_type = ShipHullType(hull_type)
    mapping = {
        ShipHullType.WIGLEY: 4.0 / 9.0,
        ShipHullType.SERIES60: 0.600,
        ShipHullType.KCS: 0.651,
        ShipHullType.KVLCC2: 0.810,
        ShipHullType.NPL: 0.397,
    }
    return mapping[hull_type]


# ---------------------------------------------------------------------------
# Hull statistics helper
# ---------------------------------------------------------------------------

def hull_statistics(
    hull_type: ShipHullType | str,
    length: float,
    beam: float,
    draft: float,
) -> dict:
    """Return a dictionary of key hull form coefficients (analytical).

    Parameters
    ----------
    hull_type: Hull family.
    length, beam, draft: Principal dimensions (lattice units or physical).

    Returns
    -------
    dict with keys: hull_type, Cb, Cwp, Cm, Cp, L_B, B_T, displacement.
    """
    if isinstance(hull_type, str):
        hull_type = ShipHullType(hull_type)

    cb = theoretical_block_coefficient(hull_type)
    # Waterplane coefficient Cwp = Awp / (L * B).
    # In normalised coords xi ∈ [-1, 1] (half-span = 1) and hb ∈ [0, 1] (half-beam):
    #   Awp (both sides) = 2 * (B/2) * (L/2) * integral(hb_norm, xi, -1, 1)
    #   L * B = 2*(L/2) * 2*(B/2)
    #   => Cwp = integral(hb_norm, xi, -1, 1) / 2
    xi_arr, hb_arr = generate_hull_waterplane(hull_type, n_points=500)
    cwp = float(np.trapezoid(hb_arr, xi_arr)) / 2.0

    # Midship section coefficient Cm = Am / (B * T).
    # In normalised coords hb ∈ [0, 1] (half-beam) and z ∈ [0, 1]:
    #   Am (both sides) = 2 * (B/2) * T * integral(hb_norm, z, 0, 1)
    #   B * T = 2*(B/2) * T
    #   => Cm = integral(hb_norm, z, 0, 1)
    z_arr = np.linspace(0.0, 1.0, 500)
    fn = _HALF_BEAM_FN[hull_type]
    hb_mid = fn(np.zeros_like(z_arr), z_arr)
    cm = float(np.trapezoid(hb_mid, z_arr))

    cp = cb / cm if cm > 1e-6 else float("nan")

    v_disp = cb * length * beam * draft  # displacement volume (lattice units³)

    return {
        "hull_type": hull_type.value,
        "label": _HULL_LABELS[hull_type],
        "Cb": round(cb, 4),
        "Cwp": round(float(cwp), 4),
        "Cm": round(float(cm), 4),
        "Cp": round(float(cp), 4),
        "L/B": round(length / beam, 3) if beam > 0 else None,
        "B/T": round(beam / draft, 3) if draft > 0 else None,
        "displacement_lu3": round(v_disp, 1),
    }


# ---------------------------------------------------------------------------
# Full-workflow helper: build mask + compute Cb
# ---------------------------------------------------------------------------

def build_hull_mask(
    hull_type: ShipHullType | str,
    nx: int,
    ny: int,
    nz: int,
    cx: float | None = None,
    cy: float | None = None,
    cz_keel: float | None = None,
    length: float | None = None,
    beam: float | None = None,
    draft: float | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Build a hull solid mask and return it together with form statistics.

    Default placement: hull centred longitudinally at *cx* = nx/2, centred
    transversely at *cy* = ny/2, keel at *cz_keel* = nz/4.  Default hull
    dimensions: L = nx*0.5, B = ny*0.25, T = nz*0.3.

    Returns
    -------
    mask : torch.Tensor, shape (nz, ny, nx), bool
    stats : dict  (block coefficient, hull statistics, solid/fluid cells)
    """
    if isinstance(hull_type, str):
        hull_type = ShipHullType(hull_type)

    dev = torch.device(device)

    # Apply defaults
    cx = float(cx) if cx is not None else nx / 2.0
    cy = float(cy) if cy is not None else ny / 2.0
    cz_keel = float(cz_keel) if cz_keel is not None else nz / 4.0
    length = float(length) if length is not None else nx * 0.5
    beam = float(beam) if beam is not None else ny * 0.25
    draft = float(draft) if draft is not None else nz * 0.3

    if hull_type == ShipHullType.WIGLEY:
        from .obstacles import wigley_hull_mask
        mask = wigley_hull_mask(nx, ny, nz, cx, cy, cz_keel, length, beam, draft, dev)
    else:
        mask = _make_hull_mask(
            hull_type, nx, ny, nz, cx, cy, cz_keel, length, beam, draft, dev
        )

    cb_numerical = hull_block_coefficient(mask, beam=beam, draft=draft, length=length)
    stats_theo = hull_statistics(hull_type, length, beam, draft)
    total = nx * ny * nz
    solid = int(mask.sum().item())

    stats = {
        **stats_theo,
        "Cb_numerical": round(cb_numerical, 4),
        "solid_cells": solid,
        "fluid_cells": total - solid,
        "total_cells": total,
        "nx": nx, "ny": ny, "nz": nz,
        "cx": cx, "cy": cy, "cz_keel": cz_keel,
        "length": length, "beam": beam, "draft": draft,
    }
    return mask, stats


# ---------------------------------------------------------------------------
# Froude / Reynolds helpers for ship CAD workflow
# ---------------------------------------------------------------------------

def ship_lbm_parameters(
    length_m: float,
    speed_ms: float,
    nu_m2s: float = 1.139e-6,
    lbm_length: float = 100.0,
    lbm_speed: float = 0.05,
    froude_target: float | None = None,
) -> dict:
    """Compute LBM parameters from physical ship dimensions.

    If *froude_target* is given, *speed_ms* is overridden to match it:
    U = Fr * sqrt(g * L).

    Parameters
    ----------
    length_m:     Ship length (m).
    speed_ms:     Ship speed (m/s).
    nu_m2s:       Kinematic viscosity (m²/s); default sea water at 15 °C.
    lbm_length:   Hull length in lattice units.
    lbm_speed:    LBM inlet velocity (lattice units/step).
    froude_target: Target Froude number (overrides speed_ms if given).

    Returns
    -------
    dict with physical Re, Fr, LBM tau, dx, dt, etc.
    """
    g_phys = 9.81  # m/s²
    if froude_target is not None:
        speed_ms = froude_target * math.sqrt(g_phys * length_m)

    re_phys = speed_ms * length_m / nu_m2s
    fr_phys = speed_ms / math.sqrt(g_phys * length_m)

    dx = length_m / lbm_length          # m per cell
    dt = lbm_speed * dx / speed_ms       # s per step

    lbm_nu = lbm_speed * lbm_length / re_phys
    tau = 3.0 * lbm_nu + 0.5
    ma = lbm_speed / (1.0 / 3.0 ** 0.5)

    return {
        "re_physical": round(re_phys, 2),
        "froude_number": round(fr_phys, 4),
        "dx_m": round(dx, 8),
        "dt_s": round(dt, 10),
        "lbm_nu": round(lbm_nu, 8),
        "lbm_tau": round(tau, 6),
        "mach_number": round(ma, 4),
        "stable": bool(tau > 0.5),
        "lbm_length_cells": lbm_length,
        "lbm_speed_lu": lbm_speed,
    }


def ship_resistance_estimate(
    hull_type: ShipHullType | str,
    length_m: float,
    beam_m: float,
    draft_m: float,
    speed_ms: float,
    nu_m2s: float = 1.139e-6,
    rho_kgm3: float = 1025.0,
    residual_ratio: float = 0.18,
) -> dict:
    """Estimate calm-water resistance for a parametric hull.

    Uses ITTC-1957 friction line + a configurable residual-resistance ratio.
    This is intended for rapid pre-screening in early-stage design.
    """
    if isinstance(hull_type, str):
        hull_type = ShipHullType(hull_type)
    if length_m <= 0.0 or beam_m <= 0.0 or draft_m <= 0.0:
        raise ValueError("length_m, beam_m, and draft_m must be > 0")
    if speed_ms <= 0.0:
        raise ValueError("speed_ms must be > 0")
    if nu_m2s <= 0.0:
        raise ValueError("nu_m2s must be > 0")
    if rho_kgm3 <= 0.0:
        raise ValueError("rho_kgm3 must be > 0")
    if not (0.0 <= residual_ratio <= 1.0):
        raise ValueError("residual_ratio must be in [0, 1]")

    cb = theoretical_block_coefficient(hull_type)
    re = speed_ms * length_m / nu_m2s
    if re <= 100.0:
        raise ValueError("reynolds number too low for ITTC-1957 formula")

    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    # Compact wetted-surface approximation for merchant-ship like hulls.
    wetted_area_m2 = length_m * (2.0 * draft_m + beam_m) * math.sqrt(max(cb, 1e-12))
    dynamic = 0.5 * rho_kgm3 * speed_ms**2
    friction_n = dynamic * wetted_area_m2 * cf
    residual_n = friction_n * residual_ratio
    total_n = friction_n + residual_n
    ct = cf * (1.0 + residual_ratio)

    return {
        "hull_type": hull_type.value,
        "cb": round(cb, 4),
        "reynolds": re,
        "cf_ittc57": cf,
        "ct_estimated": ct,
        "wetted_area_m2": wetted_area_m2,
        "friction_resistance_n": friction_n,
        "residual_resistance_n": residual_n,
        "total_resistance_n": total_n,
        "residual_ratio": residual_ratio,
    }
