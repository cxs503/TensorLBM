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
import warnings
from typing import Any

import numpy as np

__all__ = [
    "contact_angle_geodesic",
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


def _contact_half_width(
    phi: np.ndarray, y_wall: float, level: float = 0.0, nx: int | None = None
) -> float:
    """Half contact width of a sessile drop, extrapolated to the wall plane.

    The ``phi = level`` isocontour inside the first fluid row already sits about
    half a cell above the wall, so the chord is evaluated in rows ``i0`` and
    ``i0+1`` and linearly extrapolated to ``y = y_wall``.  If the cap is thinner
    than one cell above ``i0`` the single-row chord is used (no extrapolation).
    """
    ny, ncols = phi.shape
    i0 = int(math.floor(y_wall)) + 1
    if i0 + 1 >= ny:
        raise ValueError(f"wall plane y={y_wall} leaves no fluid rows (ny={ny})")
    x_center = (ncols - 1) / 2.0 if nx is None else float(nx)

    def _chord(row: np.ndarray) -> tuple[float, float] | None:
        xs = [
            j + (level - row[j]) / (row[j + 1] - row[j])
            for j in range(ncols - 1)
            if (row[j] - level) * (row[j + 1] - level) < 0.0
        ]
        if len(xs) < 2:
            return None
        # leftmost/rightmost crossing: NOT ``min/max(key=|x - x_center|)``,
        # which ties on a symmetric cap (both sides equally far) and returns
        # the same crossing twice, giving a zero-width chord.
        return min(xs), max(xs)

    c0 = _chord(phi[i0, :])
    if c0 is None:
        raise ValueError("no interface crossing in the first fluid row")
    c1 = _chord(phi[i0 + 1, :])
    if c1 is None:
        return 0.5 * (c0[1] - c0[0])
    f = (y_wall - i0) / ((i0 + 1) - i0)
    xl = c0[0] + (c1[0] - c0[0]) * f
    xr = c0[1] + (c1[1] - c0[1]) * f
    return 0.5 * (xr - xl)


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

    # Contact half-width: chord of the phi = level isocontour in the two rows
    # adjacent to the wall, linearly extrapolated to the wall plane.  (R5
    # correction: the previous estimate 0.5*(x.max()-x.min()) + 0.5 took the
    # *widest* contour extent, which for theta > 90 deg is above the wall and
    # not the contact line, and added half a cell.  Measured bias on exact
    # diffused caps: up to -8 deg at theta = 120 deg before, <= 3 deg after; see
    # benchmarks/xval_akai2018_wetting/estimator_analytic_study.py.)
    b = _contact_half_width(phi, y_wall, level)
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

# ---------------------------------------------------------------------------
# Estimator 4: geodesic (apex height / basal width)
# ---------------------------------------------------------------------------


def _geodesic_measure_a_h(
    phi: np.ndarray,
    xi: float,
    *,
    frac: float = 0.5,
    wall_y: float = -0.5,
) -> tuple[float, float]:
    """Measure apex height h and basal half-width a for geodesic estimator."""
    ny, nx = phi.shape
    gas = 0.5 * (float(phi[-4:, :4].mean()) + float(phi[-4:, -4:].mean()))
    liquid = float(phi.max())
    level = gas + frac * (liquid - gas)

    # Crossing along centre column
    xs_grid = np.arange(nx, dtype=float)
    w = np.clip(phi - gas, 0.0, None).sum(axis=0)
    x_c = float((xs_grid * w).sum() / w.sum()) if w.sum() > 0 else (nx - 1) / 2.0
    jc = int(min(nx - 2, max(1, round(x_c))))

    # Apex height
    d = phi[:, jc] - level
    s = d[:-1] * d[1:]
    idx = np.nonzero(s < 0.0)[0]
    if idx.size == 0:
        return float("nan"), float("nan")
    i = int(idx[-1])
    height = i + float(d[i] / (d[i] - d[i + 1])) - wall_y

    # Basal half-width from contour extrapolation
    ys, xs = [], []
    for k in range(ny):
        row = phi[k]
        d_row = row - level
        for j in range(jc, nx - 1):
            if d_row[j] * d_row[j + 1] < 0.0:
                x_cross = j + float(d_row[j] / (d_row[j] - d_row[j + 1]))
                ys.append(float(k))
                xs.append(x_cross - x_c)
                break

    if len(ys) < 4:
        return float("nan"), height

    ys_a, xs_a = np.asarray(ys), np.asarray(xs)
    near = ys_a <= 3.0 * xi
    if int(near.sum()) < 4:
        near = np.ones_like(ys_a, dtype=bool)
    deg = 3 if int(near.sum()) >= 5 else (2 if int(near.sum()) >= 4 else 1)
    coef = np.polyfit(ys_a[near], xs_a[near], deg)
    a = float(np.polyval(coef, wall_y))
    return a, height


def contact_angle_geodesic(
    phi: np.ndarray,
    xi: float,
    y_wall: float = 0.5,
    *,
    frac: float = 0.5,
    lbm_convention: bool | None = None,
) -> dict[str, Any]:
    """Contact angle from apex height and basal half-width (sessile-cap geometry).

    For a cap resting on a flat wall the *exact* closed form is

        z_c = -R cos(theta)   (circle centre, measured from the wall)
        h   = R (1 - cos theta)   (apex height above the wall)
        a   = R sin(theta)        (contact half-width)

    for **every** ``theta`` in ``(0, 180)`` -- the centre is below the wall for a
    wetting cap (``theta < 90``) and above it for a non-wetting one, and the
    same algebra covers both branches.  Hence

        a / h = cot(theta / 2)   =>   theta = 2 atan(h / a)

    with no branch test and no caller-supplied convention.

    .. warning::

       The former implementation returned ``2 atan(a/h)``, which is the
       **supplement** ``180 deg - theta`` for every ``theta`` (identical only at
       90 deg).  ``lbm_convention`` selected between the two and is therefore
       deprecated: it is accepted for keyword compatibility, ignored, and emits
       a :class:`DeprecationWarning`.  Evidence and the falsifiability argument
       (analytic caps built in both templates, cross-checked against the
       independent circle fit of :func:`tensorlbm.wetting.metrics.contact_angle_circle_fit`,
       plus the ten committed round-3 golden fields) are in
       ``development/verification/contact-angle/round-5/REPORT_R5.md``.

    Parameters
    ----------
    phi : ndarray
        Phase field (ny, nx); ``phi > 0`` is the wetting (liquid) phase.
    xi : float
        Interface width in lattice units.
    y_wall : float
        Wall position in lattice units (the first cell centre is ``y_wall + 0.5``).
    frac : float
        Level set fraction (0.5 = mid-level contour).
    lbm_convention : bool | None
        Deprecated and ignored; the physical relation is used unconditionally.

    Returns
    -------
    dict with keys ``theta_deg``, ``apex_height``, ``basal_half_width``,
    ``a_over_h``, ``h_over_a``, ``convention`` (always ``"physical_sessile"``).
    """
    if lbm_convention is not None:
        warnings.warn(
            "contact_angle_geodesic: 'lbm_convention' is deprecated and ignored; "
            "theta = 2 atan(h/a) holds for every sessile cap.",
            DeprecationWarning,
            stacklevel=2,
        )
    a, h = _geodesic_measure_a_h(phi, xi, frac=frac, wall_y=y_wall - 1.0)

    out = {
        "theta_deg": float("nan"),
        "apex_height": h,
        "basal_half_width": a,
        "a_over_h": float("nan"),
        "h_over_a": float("nan"),
        "convention": "physical_sessile",
    }

    if math.isfinite(a) and math.isfinite(h) and a > 0.0 and h > 0.0:
        out["a_over_h"] = a / h
        out["h_over_a"] = h / a
        out["theta_deg"] = math.degrees(2.0 * math.atan(h / a))

    return out
