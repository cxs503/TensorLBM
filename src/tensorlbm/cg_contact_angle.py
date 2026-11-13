"""Contact-angle estimators for a sessile drop on a flat plate (D2Q9 CG field).

Three *independent* estimators are provided for one phase field, so that a
disagreement between them is visible instead of being hidden inside a single
number:

``circle_fit``
    Algebraic (Kasa) least-squares circle through the ``phi = level``
    interface contour, then ``theta = acos((y_wall - y_centre)/R)``.  This is
    the standard sessile-drop estimator; it assumes the contour is close to a
    circular cap and is therefore *global* (all contour points vote).

``local_normal``
    Angle between the interface normal and the wall normal, taken from the
    phase-field gradient at the contour points nearest to the wall (a band of
    ``band`` lattice units).  *Local*: it only looks at the contact line, so it
    is the estimator that isolates the wetting boundary condition from any
    global shape error, but it is sensitive to spurious currents at the wall.

``cap_inversion``
    Inverts the measured cap area ``A = integral h dx`` and contact half-width
    ``b`` through the exact circular-cap relations ``b = R sin(theta)`` and
    ``A = R^2 (theta - sin(theta) cos(theta))``.  *Integral*: independent of any
    local derivative.

Conventions
-----------
* ``phi > 0`` is the (wetting) drop phase; the vapour above it has ``phi < 0``.
* The wall surface is at ``y = y_wall`` (``0.5`` for a one-cell solid row at
  ``y = 0`` with halfway bounce-back); ``theta`` is measured *inside* the
  ``phi > 0`` phase, so ``theta < 90`` is a wetting drop.
* Angles are returned in degrees.

All routines are pure NumPy and take ``(ny, nx)`` arrays with ``y`` first, to
match the solver's layout.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "interface_contour",
    "circle_fit",
    "contact_angle_circle_fit",
    "contact_angle_local_normal",
    "contact_angle_cap_inversion",
    "measure_contact_angle",
]


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------


def interface_contour(
    phi: np.ndarray,
    *,
    level: float = 0.0,
    y_min: int = 1,
    y_max: int | None = None,
) -> dict[str, np.ndarray]:
    """Lowest ``phi = level`` crossing per column (the drop's upper interface).

    For a sessile drop sitting on a plate at the bottom, each column inside
    the contact width cuts the interface exactly once (going upward from the
    wall the phase changes from ``phi > level`` inside the drop to
    ``phi < level`` in the vapour).  Columns outside the contact width have no
    crossing and are dropped.

    Returns ``{"x": float64[m], "y": float64[m], "y_prev": ...}`` where ``y``
    is the interpolated crossing height in lattice units.
    """
    if phi.ndim != 2:
        raise ValueError(f"phi must be 2-D, got shape {phi.shape}")
    ny, nx = phi.shape
    y_max = ny - 1 if y_max is None else min(y_max, ny - 1)
    above = phi >= level

    xs: list[float] = []
    ys: list[float] = []
    for j in range(nx):
        col = above[:, j]
        idx = np.flatnonzero(col[y_min:y_max]) + y_min
        if idx.size == 0:
            continue
        i = int(idx[-1])          # topmost drop node in this column
        if i + 1 >= ny:
            continue
        p0, p1 = float(phi[i, j]), float(phi[i + 1, j])
        denom = p1 - p0
        frac = 0.0 if abs(denom) < 1e-30 else (level - p0) / denom
        frac = min(max(frac, 0.0), 1.0)
        xs.append(float(j))
        ys.append(float(i) + frac)
    return {"x": np.asarray(xs, dtype=np.float64), "y": np.asarray(ys, dtype=np.float64)}


# ---------------------------------------------------------------------------
# Estimator 1: algebraic circle fit
# ---------------------------------------------------------------------------


def circle_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Kasa algebraic least-squares circle.

    Solves ``x^2 + y^2 = 2 a x + 2 b y + c`` for the centre ``(a, b)`` and
    ``R = sqrt(c + a^2 + b^2)``.  ``rms`` is the RMS radial residual, which is
    the honest quality flag for the circle assumption.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 4:
        raise ValueError(f"need at least 4 contour points, got {x.size}")
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    rhs = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, c = (float(v) for v in sol)
    r2 = c + a * a + b * b
    if r2 <= 0.0:
        raise ValueError(f"degenerate circle fit (R^2 = {r2:.3e})")
    R = math.sqrt(r2)
    rms = float(np.sqrt(np.mean((np.hypot(x - a, y - b) - R) ** 2)))
    return {"x_centre": a, "y_centre": b, "radius": R, "rms": rms, "n_points": int(x.size)}


def contact_angle_circle_fit(
    phi: np.ndarray,
    y_wall: float,
    *,
    level: float = 0.0,
    trim: float = 0.05,
    y_min: int = 1,
) -> dict[str, Any]:
    """Contact angle from a circle fit of the interface contour.

    ``trim`` discards that fraction of the contour points at *both* ends
    (nearest to the contact line) before fitting; the circle assumption is
    worst exactly at the contact line where the wall BC acts.
    """
    contour = interface_contour(phi, level=level, y_min=y_min)
    x, y = contour["x"], contour["y"]
    if x.size < 8:
        raise ValueError(f"contour too short for a circle fit: {x.size} points")
    n_trim = int(round(trim * x.size))
    if 2 * n_trim < x.size - 6:
        x, y = x[n_trim : x.size - n_trim], y[n_trim : y.size - n_trim]
    fit = circle_fit(x, y)
    cos_t = (y_wall - fit["y_centre"]) / fit["radius"]
    theta = math.degrees(math.acos(min(max(cos_t, -1.0), 1.0)))
    out = dict(fit)
    out.update(
        {
            "theta_deg": theta,
            "cos_theta": cos_t,
            "y_wall": float(y_wall),
            "trim": trim,
            "contour_points": int(contour["x"].size),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Estimator 2: local interface normal near the contact line
# ---------------------------------------------------------------------------


def _phase_gradient(phi: np.ndarray, solid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient, one-sided where a neighbour is solid.

    A solid cell carries no physical phase value, so the wall-normal difference
    is taken from the two fluid nodes instead of straddling the wall (the
    rotated gradient of :mod:`tensorlbm.cg_wetting_geometric` is the quantity
    the contact angle must be read from in a solver run; this estimator is the
    independent *field* measurement).
    """
    gx = 0.5 * (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1))
    gy = np.zeros_like(phi)
    gy[1:-1, :] = 0.5 * (phi[2:, :] - phi[:-2, :])
    gy[0, :] = phi[1, :] - phi[0, :]
    gy[-1, :] = phi[-1, :] - phi[-2, :]
    if solid is not None:
        fluid = ~solid
        # one-sided in y where the stencil would touch a solid cell
        lower_solid = np.zeros_like(solid)
        lower_solid[1:, :] = solid[:-1, :]
        upper_solid = np.zeros_like(solid)
        upper_solid[:-1, :] = solid[1:, :]
        fix = fluid & lower_solid
        gy[fix] = phi[np.roll(np.arange(phi.shape[0]), -1)[:, None] * 0 + np.arange(phi.shape[0])[:, None] % phi.shape[0], :][fix] if False else gy[fix]
        rows = np.arange(phi.shape[0])[:, None]
        cols = np.arange(phi.shape[1])[None, :]
        i_up = np.clip(rows + 1, 0, phi.shape[0] - 1)
        i_dn = np.clip(rows - 1, 0, phi.shape[0] - 1)
        one_sided = phi[i_up, cols] - phi[rows, cols]
        gy = np.where(fix, one_sided, gy)
        fix2 = fluid & upper_solid
        one_sided2 = phi[rows, cols] - phi[i_dn, cols]
        gy = np.where(fix2, one_sided2, gy)
        gx = np.where(solid, 0.0, gx)
        gy = np.where(solid, 0.0, gy)
    return gx, gy


def contact_angle_local_normal(
    phi: np.ndarray,
    solid: np.ndarray,
    y_wall: float,
    *,
    level: float = 0.0,
    band: float = 3.0,
    wall_normal: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Contact angle from the interface normal in a band above the wall.

    At the contact line the interface normal ``n = grad(phi)/|grad(phi)|``
    (pointing into the wetting phase) makes the contact angle with the wall
    normal that points *into the fluid*, so ``cos(theta) = n . n_wall``.

    Returns the median over the sampled contour nodes plus the sample list, so
    the spread is auditable.
    """
    contour = interface_contour(phi, level=level, y_min=1)
    keep = contour["y"] <= y_wall + band
    if keep.sum() == 0:
        raise ValueError(f"no contour points within {band} lattice units of the wall")
    gx, gy = _phase_gradient(phi, solid)
    nx_w, ny_w = wall_normal
    angles: list[float] = []
    for x, y in zip(contour["x"][keep], contour["y"][keep]):
        i = int(round(y))
        j = int(round(x)) % phi.shape[1]
        i = min(max(i, 1), phi.shape[0] - 2)
        mag = math.hypot(gx[i, j], gy[i, j])
        if mag < 1e-9:
            continue
        cos_t = (gx[i, j] * nx_w + gy[i, j] * ny_w) / mag
        angles.append(math.degrees(math.acos(min(max(cos_t, -1.0), 1.0))))
    if not angles:
        raise ValueError("all sampled gradients vanished; interface too diffuse")
    arr = np.asarray(angles, dtype=np.float64)
    return {
        "theta_deg": float(np.median(arr)),
        "theta_mean": float(np.mean(arr)),
        "theta_std": float(np.std(arr)),
        "n_samples": int(arr.size),
        "samples_deg": arr.tolist(),
        "band": band,
    }


# ---------------------------------------------------------------------------
# Estimator 3: integral cap inversion
# ---------------------------------------------------------------------------


def contact_angle_cap_inversion(
    phi: np.ndarray,
    y_wall: float,
    *,
    level: float = 0.0,
) -> dict[str, Any]:
    """Contact angle from contact width and cap area (exact circular cap).

    ``b`` is the contact half-width (from the contour extents) and ``A`` the
    area of the drop cap above the wall (trapezoidal integral of the contour
    height).  For a circular cap ``b = R sin(theta)`` and
    ``A = R^2 (theta - sin(theta) cos(theta))``; eliminating ``R`` leaves a
    monotone scalar equation in ``theta`` that is solved by bisection on
    ``(1e-4, pi - 1e-4)``.
    """
    contour = interface_contour(phi, level=level, y_min=1)
    x, y = contour["x"], contour["y"]
    if x.size < 8:
        raise ValueError(f"contour too short for cap inversion: {x.size} points")

    def cap_area(theta: float) -> float:
        return theta - math.sin(theta) * math.cos(theta)

    def half_width(theta: float) -> float:
        return math.sin(theta)

    # contact width from the contour extents (the contour stops where the
    # interface meets the wall)
    b = 0.5 * float(x.max() - x.min()) + 0.5
    A = float(np.trapezoid(np.clip(y - y_wall, 0.0, None), x)) if hasattr(np, "trapezoid") else float(
        np.trapz(np.clip(y - y_wall, 0.0, None), x)
    )
    if b <= 0.0 or A <= 0.0:
        raise ValueError(f"degenerate cap (b={b:.3f}, A={A:.3f})")

    target = A / (b * b)

    def f(theta: float) -> float:
        return cap_area(theta) / (half_width(theta) ** 2) - target

    lo, hi = 1e-4, math.pi - 1e-4
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"cap inversion not bracketed: f(lo)={f_lo:.3e}, f(hi)={f_hi:.3e}, target={target:.3e}"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-14 or (hi - lo) < 1e-13:
            break
        if f_lo * fm <= 0.0:
            hi, f_hi = mid, fm
        else:
            lo, f_lo = mid, fm
    theta = 0.5 * (lo + hi)
    R = b / math.sin(theta)
    return {
        "theta_deg": math.degrees(theta),
        "contact_half_width": b,
        "cap_area": A,
        "radius": R,
        "target_A_over_b2": target,
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


def measure_contact_angle(
    phi: np.ndarray,
    solid: np.ndarray | None = None,
    y_wall: float = 0.5,
    *,
    level: float = 0.0,
    trim: float = 0.05,
    band: float = 3.0,
) -> dict[str, Any]:
    """Run all three estimators; failures are reported, never raised."""
    out: dict[str, Any] = {"theta_circle": float("nan"), "theta_local": float("nan"), "theta_cap": float("nan")}
    try:
        c = contact_angle_circle_fit(phi, y_wall, level=level, trim=trim)
        out["theta_circle"] = c["theta_deg"]
        out["circle"] = c
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        out["circle_error"] = repr(exc)
    if solid is not None:
        try:
            l = contact_angle_local_normal(phi, solid, y_wall, level=level, band=band)
            out["theta_local"] = l["theta_deg"]
            out["local"] = l
        except Exception as exc:  # noqa: BLE001
            out["local_error"] = repr(exc)
    try:
        cap = contact_angle_cap_inversion(phi, y_wall, level=level)
        out["theta_cap"] = cap["theta_deg"]
        out["cap"] = cap
    except Exception as exc:  # noqa: BLE001
        out["cap_error"] = repr(exc)
    return out