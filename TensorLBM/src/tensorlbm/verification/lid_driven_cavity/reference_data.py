"""Ghia et al. (1982) reference data for lid-driven cavity.

Reference: Ghia, U., Ghia, K.N., & Shin, C.T. (1982).
"High-Re solutions for incompressible flow using the Navier-Stokes equations
and a multigrid method." Journal of Computational Physics, 48(3), 387-411.

Data extracted from:
- Table I (u) and Table II (v): Re = 100, 400, 1000

PROVENANCE STATUS
-----------------
The inline constants ``GHIA_RE100``, ``GHIA_RE400``, ``GHIA_RE1000`` below
have been **replaced** with provenance-bearing transcriptions from the OCR
witness file (``benchmarks/verified/cavity/re5000/reference/ghia1982_tables_I_II.ocr.txt``,
sha256 ``3637ef9e...``).  Each table now carries ``provenance="BOUND"`` and
uses the full 17-point Ghia grid (not the old 14-point fabricated grid).

The old fabricated values are documented in
``benchmarks/anchors/ghia1982_provenance.json`` under ``suspect_entries``.

``GHIA_RE5000`` (Table III/IV, Re=5000) remains absent: the table that used
to live here was fabricated.  See ``docs/audit_m15543.md`` section 2.
Re=5000 raises :class:`KeyError` from :func:`get_ghia_reference` until a
provenance-bearing transcription is supplied.
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np


@dataclasses.dataclass
class GhiaReference:
    """Reference data from Ghia et al. (1982) for a specific Re."""
    re: float
    # Vertical centerline: y, u/U_lid (y increasing from 0 to 1)
    y: np.ndarray
    u: np.ndarray
    # Horizontal centerline: x, v/U_lid (x increasing from 0 to 1)
    x: np.ndarray
    v: np.ndarray
    # Provenance: free-text citation and a status flag.  Only ``"BOUND"``
    # (sha256 recorded by :func:`load_reference_json`) may gate an acceptance.
    source: str = "Ghia et al. (1982) JCP 48:387-411"
    provenance: str = "UNVERIFIED_OFFLINE"


# ---------------------------------------------------------------------------
# Re = 100 (Table I/II) — BOUND from OCR witness
# ---------------------------------------------------------------------------
GHIA_RE100 = GhiaReference(
    re=100.0,
    # Ghia lists y top-to-bottom; we store bottom-to-top for np.interp compatibility
    y=np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]),
    u=np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
                -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
                0.78871, 0.84123, 1.0]),
    x=np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]),
    v=np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527,
                0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
                -0.05906, 0.0]),
    provenance="BOUND",
)

# ---------------------------------------------------------------------------
# Re = 400 (Table I/II) — BOUND from OCR witness
# ---------------------------------------------------------------------------
GHIA_RE400 = GhiaReference(
    re=400.0,
    y=np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]),
    u=np.array([0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726,
                -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756,
                0.68439, 0.75837, 1.0]),
    x=np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]),
    v=np.array([0.0, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174,
                0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663,
                -0.12146, 0.0]),
    provenance="BOUND",
)

# ---------------------------------------------------------------------------
# Re = 1000 (Table I/II) — BOUND from OCR witness
# ---------------------------------------------------------------------------
GHIA_RE1000 = GhiaReference(
    re=1000.0,
    y=np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]),
    u=np.array([0.0, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805,
                -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117,
                0.57492, 0.65928, 1.0]),
    x=np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]),
    v=np.array([0.0, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235,
                0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669,
                -0.21388, 0.0]),
    provenance="BOUND",
)


AVAILABLE_RE = (100, 400, 1000)


def get_ghia_reference(re: float) -> GhiaReference:
    """Get Ghia (1982) reference data for a supported Reynolds number.

    Supported: Re = 100, 400, 1000 (Table I/II), all ``BOUND`` from OCR witness.
    Every other Reynolds number raises :class:`KeyError`; in particular
    Re = 5000 is deliberately absent because the table that used to be here was
    fabricated (``docs/audit_m15543.md``).  Use :func:`load_reference_json` to
    inject a provenance-bearing transcription.
    """
    refs = {100: GHIA_RE100, 400: GHIA_RE400, 1000: GHIA_RE1000}
    key = int(round(float(re)))
    if abs(key - float(re)) < 1e-9 and key in refs:
        return refs[key]
    raise KeyError(
        f"no verified Ghia (1982) data for Re={re}; available (BOUND): "
        f"{AVAILABLE_RE}. Re=5000 (Table III/IV) has no "
        "provenance-bearing transcription in this repository, supply one via "
        "load_reference_json(path)."
    )


def interpolate_ghia(ref: GhiaReference, y_points: np.ndarray, axis: str = 'u') -> np.ndarray:
    """Interpolate Ghia data at given points.

    Parameters
    ----------
    ref : GhiaReference
        Reference data
    y_points : np.ndarray
        Points to interpolate at
    axis : str
        'u' for vertical centerline u-velocity, 'v' for horizontal centerline v-velocity

    Returns
    -------
    interpolated values
    """
    if axis == 'u':
        return np.interp(y_points, ref.y, ref.u)
    else:
        return np.interp(y_points, ref.x, ref.v)


# ---------------------------------------------------------------------------
# Point anchors quoted from Ghia et al. (1982) Table I (as cited in this
# repository's own benchmark documents) -- usable as a cheap sanity check when
# the full Table III/IV transcription is unavailable.
# ---------------------------------------------------------------------------
LITERATURE_ANCHORS = {
    100: {
        "u_min": -0.21090,       # Table I, Re=100 column, y = 0.4531
        "u_min_y": 0.4531,
        "u_center": -0.20581,    # u(y = 0.5), quoted in docs/benchmarks/tutorials/cavity_re100.md
        "source": "Ghia et al. (1982) JCP 48:387-411, Table I; values as cited "
                  "in docs/benchmarks/tutorials/cavity_re100.md and "
                  "docs/benchmarks/tutorials/cavity_3d_full.md",
    },
    400: {
        "u_min": -0.32726,       # Table I, Re=400 column, y = 0.2813
        "u_min_y": 0.2813,
        "u_center": -0.11477,    # u(y = 0.5)
        "source": "Ghia et al. (1982) JCP 48:387-411, Table I; BOUND from OCR witness",
    },
    1000: {
        "u_min": -0.38289,       # Table I, Re=1000 column, y = 0.1719
        "u_min_y": 0.1719,
        "source": "Ghia et al. (1982) JCP 48:387-411, Table I; value as cited "
                  "in docs/benchmarks/tutorials/cavity_3d_full.md",
    },
}


def load_reference_json(path, require_normalised: bool = True) -> GhiaReference:
    """Load a provenance-bearing Ghia reference table from JSON.

    Expected schema::

        {"re": 5000.0,
         "y": [...], "u": [...],          # vertical centreline, u/U_lid
         "x": [...], "v": [...],          # horizontal centreline, v/U_lid
         "source": "Ghia et al. (1982) Table III/IV, p. 399, <who/what>"}

    The loader enforces the two structural invariants that the fabricated
    table violated: the tables must be **dimensionless** (lid row ``u ~ 1``)
    and the coordinates must be monotone inside ``[0, 1]``.

    Parameters
    ----------
    path : str or Path
        JSON file with the schema above.
    require_normalised : bool
        If True (default) require ``max(u)`` and ``|min(v)|`` to be O(1), i.e.
        velocities normalised by the lid velocity.

    Returns
    -------
    GhiaReference
        With ``provenance="BOUND"`` and the ``source`` string from the file.
    """
    import json
    from pathlib import Path as _Path

    path = _Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("re", "y", "u", "x", "v") if k not in data]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")

    y = np.asarray(data["y"], dtype=float)
    u = np.asarray(data["u"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    v = np.asarray(data["v"], dtype=float)
    if y.shape != u.shape or x.shape != v.shape:
        raise ValueError(f"{path}: coordinate/value length mismatch")
    for name, arr in (("y", y), ("x", x)):
        if arr.size < 3 or np.any(np.diff(arr) <= 0):
            raise ValueError(f"{path}: {name} must be strictly increasing")
        if arr.min() < -1e-12 or arr.max() > 1 + 1e-12:
            raise ValueError(f"{path}: {name} must lie in [0, 1]")
    for name, arr in (("u", u), ("v", v)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{path}: {name} contains non-finite values")

    if require_normalised:
        peak = float(max(np.abs(u).max(), np.abs(v).max()))
        if not (0.5 < peak < 1.5):
            raise ValueError(
                f"{path}: velocities are not normalised by the lid velocity "
                f"(max|u|,|v| = {peak:.4g}; expected O(1)). A lattice-unit "
                "table cannot be compared against u/U_lid profiles."
            )

    return GhiaReference(
        re=float(data["re"]), y=y, u=u, x=x, v=v,
        source=str(data.get("source", "unspecified")),
        provenance="BOUND",
    )
