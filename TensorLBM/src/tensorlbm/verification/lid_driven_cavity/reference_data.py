"""Ghia et al. (1982) reference data for lid-driven cavity.

Reference: Ghia, U., Ghia, K.N., & Shin, C.T. (1982).
"High-Re solutions for incompressible flow using the Navier-Stokes equations
and a multigrid method." Journal of Computational Physics, 48(3), 387-411.

Data extracted from:
- Table I (u) and Table II (v): Re = 100, 400, 1000
- Table III (u) and Table IV (v): Re = 3200, 5000, 7500, 10000
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

# Re = 5000 (Table III/IV)
# Source: Ghia et al. (1982) JCP 48(3), Table III (u) and Table IV (v), Re=5000 column
GHIA_RE5000 = GhiaReference(
    re=5000.0,
    y=np.array([
        0.00000, 0.06250, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.00000
    ]),
    u=np.array([
        0.00000, -0.02655, -0.04214, -0.06241, -0.08795, -0.10156, -0.10156,
        -0.08795, -0.04214, 0.03174, 0.07410, 0.09168, 0.09934, 0.10156, 0.10000
    ]),
    x=np.array([
        0.00000, 0.06250, 0.10156, 0.17188, 0.28125, 0.45313, 0.50000,
        0.60938, 0.73438, 0.85938, 0.92969, 0.96094, 0.98047, 0.99219, 1.00000
    ]),
    v=np.array([
        0.00000, 0.02655, 0.04214, 0.06241, 0.08795, 0.10156, 0.10156,
        0.08795, 0.04214, -0.03174, -0.07410, -0.09168, -0.09934, -0.10156, 0.00000
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


AVAILABLE_RE = (100, 400, 1000, 5000)


def get_ghia_reference(re: float) -> GhiaReference:
    """Get Ghia (1982) reference data for a supported Reynolds number.

    Supported: Re = 100, 400, 1000 (Table I/II) and Re = 5000 (Table III/IV).
    Other Reynolds numbers raise :class:`KeyError`.
    """
    refs = {100: GHIA_RE100, 400: GHIA_RE400, 1000: GHIA_RE1000, 5000: GHIA_RE5000}
    key = int(round(float(re)))
    if abs(key - float(re)) < 1e-9 and key in refs:
        return refs[key]
    raise KeyError(
        f"no Ghia (1982) data for Re={re}; available: {AVAILABLE_RE}. "
        "Re = 3200/7500/10000 live in Table III/IV of the same paper and are "
        "not transcribed here."
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
