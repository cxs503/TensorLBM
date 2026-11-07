"""Machine-checkable diagnostics for shock-dominated fields.

Everything here turns a field into one number with a documented rule, so the
acceptance criteria of a verification run are reproducible instead of eyeballed:

``front_x``           position of a discontinuity from a threshold crossing
``shock_x_1d``        sub-cell shock position in a 1-D profile
``sample_state``      mean conservative state in a window
``jump_ratios``       measured rho/p/T ratios across a discontinuity
``fit_speed``         least-squares shock speed from a position history
``front_profile``     leading-front position per row (2-D)
``mach_stem_height``  stem height / triple-point height from the front profile
``standoff_distance`` shock stand-off distance ahead of a step face
``gci``               Roache grid-convergence index (Richardson based)
``relative_error``    |a-b|/|b|
"""

from __future__ import annotations

import numpy as np

from .flux import GAMMA, primitive

__all__ = [
    "front_x",
    "shock_x_1d",
    "sample_state",
    "jump_ratios",
    "fit_speed",
    "front_profile",
    "mach_stem_height",
    "standoff_distance",
    "gci",
    "relative_error",
]


def relative_error(value: float, reference: float) -> float:
    """``|value - reference| / |reference|``."""
    return float(abs(value - reference) / abs(reference))


def front_x(
    profile: np.ndarray,
    coord: np.ndarray,
    threshold: float,
    rising: bool = False,
    x_min: float | None = None,
    x_max: float | None = None,
) -> float:
    """Position where ``profile`` crosses ``threshold`` (last crossing from the right).

    The leading front is found by scanning from the large-``x`` end for the first
    cell that still meets the "behind the shock" condition, then linearly
    interpolating the crossing.  This is robust against secondary waves, slip
    lines and vortices that all sit *behind* the leading front.
    """
    x = np.asarray(coord, dtype=float)
    f = np.asarray(profile, dtype=float)
    if x_min is not None:
        mask = x >= x_min
        x, f = x[mask], f[mask]
    if x_max is not None:
        mask = x <= x_max
        x, f = x[mask], f[mask]
    behind = (f < threshold) if rising else (f > threshold)
    idx = np.flatnonzero(behind)
    if idx.size == 0:
        return float("nan")
    i = idx[-1]
    if i + 1 >= x.size:
        return float(x[i])
    f0, f1 = f[i], f[i + 1]
    denom = f1 - f0
    frac = 0.5 if denom == 0.0 else (threshold - f0) / denom
    return float(x[i] + frac * (x[i + 1] - x[i]))


def shock_x_1d(rho: np.ndarray, xc: np.ndarray, x_lo: float, x_hi: float) -> float:
    """Sub-cell position of the steepest density gradient inside a window."""
    x = np.asarray(xc)
    r = np.asarray(rho)
    m = (x >= x_lo) & (x <= x_hi)
    if m.sum() < 3:
        return float("nan")
    xs, rs = x[m], r[m]
    g = np.abs(np.diff(rs))
    i = int(np.argmax(g))
    # parabolic refinement of the gradient peak
    if 0 < i < g.size - 1:
        g0, g1, g2 = g[i - 1], g[i], g[i + 1]
        den = g0 - 2.0 * g1 + g2
        shift = 0.5 * (g0 - g2) / den if den != 0 else 0.0
    else:
        shift = 0.0
    return float(0.5 * (xs[i] + xs[i + 1]) + shift * np.mean(np.diff(xs)))


def sample_state(
    U: np.ndarray, xc: np.ndarray, x_lo: float, x_hi: float, gamma: float = GAMMA
) -> dict:
    """Mean primitive state over ``x_lo <= x <= x_hi`` (1-D field)."""
    m = (xc >= x_lo) & (xc <= x_hi)
    if not np.any(m):
        raise ValueError("empty sampling window")
    rho, u, v, p = primitive(np.asarray(U)[m], gamma)
    return {
        "rho": float(np.mean(rho)),
        "u": float(np.mean(u)),
        "v": float(np.mean(v)),
        "p": float(np.mean(p)),
        "T": float(np.mean(p / rho)),
        "a": float(np.mean(np.sqrt(gamma * p / rho))),
        "n_cells": int(m.sum()),
    }


def jump_ratios(up: dict, down: dict) -> dict:
    """Ratios downstream/upstream from two :func:`sample_state` results."""
    return {
        "rho_ratio": down["rho"] / up["rho"],
        "p_ratio": down["p"] / up["p"],
        "T_ratio": down["T"] / up["T"],
        "M2": down["u"] / down["a"],
    }


def fit_speed(t: list, x: list) -> dict:
    """Least-squares speed ``dx/dt`` plus the coefficient of determination."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size < 2:
        raise ValueError("need at least two samples")
    A = np.vstack([t, np.ones_like(t)]).T
    sol, *_ = np.linalg.lstsq(A, x, rcond=None)
    resid = x - A @ sol
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 1.0
    return {"speed": float(sol[0]), "offset": float(sol[1]), "r2": r2}


def front_profile(
    rho: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    threshold: float,
    x_min: float = 0.0,
    x_max: float | None = None,
) -> np.ndarray:
    """Leading-front ``x`` position for every row of a 2-D density field."""
    out = np.full(len(yc), np.nan)
    for j in range(len(yc)):
        out[j] = front_x(rho[j, :], xc, threshold, x_min=x_min, x_max=x_max)
    return out


def mach_stem_height(
    x_front: np.ndarray,
    yc: np.ndarray,
    slope_tol: float = 0.25,
    smooth: int = 3,
) -> dict:
    """Stem height and triple-point height from a leading-front profile.

    The Mach stem is the near-vertical part of the leading front near the wall
    (``|dx/dy| ~ 0``), the incident shock is inclined (``|dx/dy| ~ 0.58`` for a
    30 deg shock).  The triple point is where the profile switches branches, so
    it is detected as the first height (scanning upwards from the wall) at which
    the smoothed slope exceeds ``slope_tol``.  Returns ``nan`` when the pattern
    has not formed (no vertical segment / no switch).
    """
    x = np.asarray(x_front, dtype=float)
    y = np.asarray(yc, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return {
            "stem_height": float("nan"),
            "triple_point_y": float("nan"),
            "stem_x_at_contact": float("nan"),
        }
    if smooth > 1:
        k = np.ones(smooth) / smooth
        filled = np.where(ok, x, np.nanmedian(x))
        # edge padding: zero padding would fake a slope jump at the wall
        pad = np.pad(filled, (smooth, smooth), mode="edge")
        xs = np.convolve(pad, k, mode="same")[smooth:-smooth]
    else:
        xs = x
    slope = np.gradient(xs, y)
    # scan from the wall upwards: the stem must start vertical
    j0 = int(np.flatnonzero(ok)[0])
    if abs(slope[j0]) > slope_tol:
        return {
            "stem_height": float("nan"),
            "triple_point_y": float("nan"),
            "stem_x_at_contact": float(x[j0]),
        }
    triple = float("nan")
    for j in range(j0 + 2, len(y) - 1):
        if not ok[j] or not ok[j + 1]:
            continue
        # the incident branch is inclined with dx/dy > 0; require two
        # consecutive rows above the tolerance to reject single-cell noise
        if slope[j] > slope_tol and slope[j + 1] > slope_tol:
            triple = float(y[j])
            break
    return {
        "stem_height": triple - float(y[0]),
        "triple_point_y": triple,
        "stem_x_at_contact": float(x[j0]),
    }


def standoff_distance(
    rho_row: np.ndarray,
    xc: np.ndarray,
    face_x: float,
    threshold: float,
    search_lo: float,
) -> dict:
    """Shock stand-off distance ahead of a vertical face.

    Returns the front position, the stand-off ``face_x - x_front`` and the
    stand-off normalised by the step height when ``step_h`` is supplied by the
    caller.
    """
    x_f = front_x(rho_row, xc, threshold, x_min=search_lo, x_max=face_x)
    return {"front_x": float(x_f), "standoff": float(face_x - x_f)}


def gci(f_coarse: float, f_fine: float, ratio: float = 2.0, order: float = 2.0) -> dict:
    """Roache grid-convergence index for two grids (``ratio`` = refinement factor).

    ``apparent_order`` is only estimated when three grids are available; with two
    grids the formal ``order`` is used and the result is a conservative band.
    """
    if f_fine == 0.0:
        raise ValueError("reference value must be non-zero")
    eps = abs((f_fine - f_coarse) / f_fine)
    rp = ratio**order
    return {
        "relative_change": float(eps),
        "gci_fine": float(1.25 * eps / (rp - 1.0)),
        "richardson_estimate": float(f_fine + (f_fine - f_coarse) / (rp - 1.0)),
        "order_assumed": float(order),
        "refinement_ratio": float(ratio),
    }
