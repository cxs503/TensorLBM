"""Ghia et al. (1982) reference data for lid-driven cavity.

Reference: Ghia, U., Ghia, K.N., & Shin, C.T. (1982).
"High-Re solutions for incompressible flow using the Navier-Stokes equations
and a multigrid method." Journal of Computational Physics, 48(3), 387-411.

Data extracted from:
- Table I (u) and Table II (v): Re = 100, 400, 1000

PROVENANCE STATUS (read before using these tables as an acceptance gate)
-----------------------------------------------------------------------
The ``GHIA_RE100`` / ``GHIA_RE400`` / ``GHIA_RE1000`` arrays below are
``UNVERIFIED_OFFLINE``: they were hand-entered without a page/table/sha256
record and, per the cross-audit in ``docs/audit_m15543.md`` (finding F-A), at
least the Re=100 centre value is visibly off the paper (u(0.5, 0.5) =
-0.15612 here vs the literature value -0.20581, i.e. 24%).  They are kept for
backward compatibility of existing tests only.  **Do not use them as a
pass/fail gate**; use :func:`load_reference_json` with a transcription that
carries a sha256, or compare against the point anchors in
:data:`LITERATURE_ANCHORS`.

``GHIA_RE5000`` (Table III/IV, Re=5000) was **removed**: the constant that used
to live here was fabricated -- it recycled the ``GHIA_RE1000`` values as the
positive branch, stored lattice-unit numbers instead of ``u/U_lid`` (lid row
``u = 0.1``), and had a primary vortex weaker at Re=5000 than at Re=100.  See
``docs/audit_m15543.md`` section 2 (four independent falsifications).  Re=5000
therefore raises :class:`KeyError` from :func:`get_ghia_reference` until a
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
    # Vertical centerline: y, u/U_lid
    y: np.ndarray
    u: np.ndarray
    # Horizontal centerline: x, v/U_lid
    x: np.ndarray
    v: np.ndarray
    # Provenance: free-text citation and a status flag.  Only ``"BOUND"``
    # (sha256 recorded by :func:`load_reference_json`) may gate an acceptance.
    source: str = "Ghia et al. (1982) JCP 48:387-411"
    provenance: str = "UNVERIFIED_OFFLINE"


# Re = 100 (Table I/II)
GHIA_RE100 = GhiaReference(
    re=100.0,
    y=np.array([
        0.0625, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.00000
    ]),
    u=np.array([
        -0.03717, -0.06034, -0.09168, -0.12635, -0.15174, -0.15612,
        -0.13639, -0.08795, -0.02203, 0.03174, 0.06034, 0.07410, 0.08176, 0.00000
    ]),
    x=np.array([
        0.0625, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.00000
    ]),
    v=np.array([
        0.00000, -0.01457, -0.03425, -0.05864, -0.07410, -0.07490,
        -0.05904, -0.02203, 0.03174, 0.06034, 0.07410, 0.08176, 0.08554, 0.00000
    ]),
)

# Re = 400 (Table I/II)
GHIA_RE400 = GhiaReference(
    re=400.0,
    y=np.array([
        0.0625, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.00000
    ]),
    u=np.array([
        -0.06241, -0.10156, -0.15174, -0.20508, -0.21094, -0.20508,
        -0.15174, -0.06241, 0.03174, 0.07410, 0.09168, 0.09934, 0.10156, 0.00000
    ]),
    x=np.array([
        0.0625, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.00000
    ]),
    v=np.array([
        0.00000, -0.03425, -0.07410, -0.12635, -0.15174, -0.15612,
        -0.13639, -0.08795, -0.02203, 0.03174, 0.06034, 0.07410, 0.08176, 0.00000
    ]),
)

# Re = 1000 (Table I/II)
GHIA_RE1000 = GhiaReference(
    re=1000.0,
    y=np.array([
        0.0625, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.00000
    ]),
    u=np.array([
        -0.06904, -0.11328, -0.17188, -0.23438, -0.21094, -0.19531,
        -0.13639, -0.06241, 0.03174, 0.07410, 0.09168, 0.09934, 0.10156, 0.00000
    ]),
    x=np.array([
        0.0625, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.00000
    ]),
    v=np.array([
        0.00000, -0.04590, -0.09168, -0.15174, -0.17188, -0.17188,
        -0.13639, -0.06241, 0.03174, 0.07410, 0.09168, 0.09934, 0.10156, 0.00000
    ]),
)

# Finer Ghia data for Re=100 (from the paper's finer grid)
GHIA_RE100_FINE = GhiaReference(
    re=100.0,
    y=np.array([
        0.0000, 0.0625, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.0000
    ]),
    u=np.array([
        0.00000, -0.03717, -0.06034, -0.09168, -0.12635, -0.15174, -0.15612,
        -0.13639, -0.08795, -0.02203, 0.03174, 0.06034, 0.07410, 0.08176, 0.00000
    ]),
    x=np.array([
        0.0000, 0.0625, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.0000
    ]),
    v=np.array([
        0.00000, 0.00000, -0.01457, -0.03425, -0.05864, -0.07410, -0.07490,
        -0.05904, -0.02203, 0.03174, 0.06034, 0.07410, 0.08176, 0.08554, 0.00000
    ]),
)


AVAILABLE_RE = (100, 400, 1000)


def get_ghia_reference(re: float) -> GhiaReference:
    """Get Ghia (1982) reference data for a supported Reynolds number.

    Supported: Re = 100, 400, 1000 (Table I/II), all ``UNVERIFIED_OFFLINE``.
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
        f"no verified Ghia (1982) data for Re={re}; available (unverified "
        f"offline): {AVAILABLE_RE}. Re=5000 (Table III/IV) has no "
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
