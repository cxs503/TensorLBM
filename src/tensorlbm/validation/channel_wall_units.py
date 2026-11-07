"""Wall-unit reduction of a 3-D channel profile (the ``y_w`` / ``u_tau`` layer).

What this module fixes
----------------------
A wall-resolved channel is symmetric, and the DNS it is compared against is
published *per half channel* (``y+`` from the wall to the centre line).  Turning
the solver's full-height profile into that comparison involves four decisions
that are easy to get silently wrong, and all four were wrong in the previous
campaign (``docs/reports/P20-N01-channel3d-mkm1999.md``):

1. **folding**.  Rows ``i = 1 .. ny-2`` cover *both* halves of the channel.
   Interpolating the DNS on the un-folded profile maps the second half to
   ``y+ > Re_tau`` and compares it against the centre-line value, which alone
   inflated the reported u+ error from 6.3 % to 15.2 %.
2. **wall origin**.  The effective wall of a node-based bounce-back wall is at
   ``y_w ~= 0.64`` cell, not at the textbook ``0.5``; ``y_w`` must be
   *measured* (laminar inversion / wall-stress anchoring) and it enters every
   ``y+`` as ``(i - y_w) u_tau / nu``.
3. **which ``u_tau``**.  Nondimensionalising with a wall-shear diagnostic that
   is still out of balance biases the whole profile by that amount; the
   momentum-balance ``u_tau = sqrt(a h / 2)`` is exact by construction.
4. **half-cell bulk integral**.  ``U_b`` and ``C_f`` need the wall origin in
   the quadrature as well, otherwise ``C_f`` inherits an ``O(dx+/H)`` bias.

The module is NumPy-only and side-effect free so that it can be reused by the
GPU solver, the NumPy kernel and the offline campaign scripts alike.

Estimators
----------
``momentum_balance``
    ``u_tau = sqrt(tau_w)`` with ``tau_w = a h / 2`` from the imposed body
    force.  Exact for a statistically steady channel, independent of the
    profile.
``wall_shear``
    ``u_tau = sqrt(tau_w)`` with ``tau_w`` from the bounce-back momentum
    exchange on the wall (``2 sum c_x f`` per unit wall area).  Independent of
    the mean profile but only correct once the flow is steady.
``log_law``
    Regression of the folded mean profile on ``u+ = ln(y+)/kappa + B`` in a
    window of ``y+``.  Two variants are returned: ``level`` (intercept matched,
    ``u_tau`` solved by fixed point - depends on ``kappa`` *and* ``B``) and
    ``slope`` (``u_tau = kappa dU/dln(y-y_w)`` - depends on ``kappa`` only).
    The gap between them is the *model-form* uncertainty of the law of the
    wall and is reported rather than hidden.
``total_stress``
    ``u_tau^2 = <nu dU/dy - <u'v'>> / (1 - y/h)`` averaged over the inner layer.
    A profile-based route whose only assumption is the linear total-stress
    profile of a forced channel; used as a cross-check.

References
----------
Moser, Kim & Mansour, Phys. Fluids 11, 943 (1999); Kim, Moin & Moser, JFM 177,
133 (1987).  Pope, *Turbulent Flows*, ch. 7.  Dean (1978) friction correlation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "KAPPA",
    "effective_height",
    "B_LOG",
    "WallUnits",
    "fold_channel_profile",
    "bulk_velocity",
    "friction_coefficient",
    "u_tau_momentum_balance",
    "u_tau_wall_shear",
    "u_tau_log_law",
    "u_tau_total_stress",
    "u_tau_three_way",
    "measure_wall_origin_poiseuille",
    "measure_wall_origin_viscous",
    "profile_error_vs_reference",
    "dean_channel_friction",
]

KAPPA = 0.41
B_LOG = 5.0


@dataclass
class WallUnits:
    """Folded half-channel profile expressed in wall units."""

    y_plus: np.ndarray
    u_plus: np.ndarray
    y: np.ndarray
    u: np.ndarray
    y_w: float
    u_tau: float
    nu: float
    height: float
    meta: dict = field(default_factory=dict)


def fold_channel_profile(
    profile: Iterable[float],
    *,
    y_w: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fold a full-height channel profile onto the lower half channel.

    ``profile`` holds one value per ``y`` row, including the two solid rows
    (index ``0`` and ``ny-1``).  The folded abscissa is the distance from the
    lower wall, ``y = i - y_w``, for ``i = 1 .. floor((ny-2)/2)``; the two
    half-channel samples are averaged (the channel is statistically
    symmetric, so this also halves the statistical error).

    Returns ``(y, u, info)`` where ``info`` records the folding bookkeeping so
    that a report can prove the second half was actually folded and not
    compared as a full-height profile.
    """
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1:
        raise ValueError("profile must be one value per y row")
    ny = values.size
    fluid = ny - 2
    if fluid < 4:
        raise ValueError("profile needs at least 4 fluid rows")
    half = fluid // 2
    lower = np.arange(1, 1 + half)
    upper = ny - 1 - lower
    folded = 0.5 * (values[lower] + values[upper])
    y = lower - float(y_w)
    info = {
        "rows_total": int(ny),
        "rows_folded": int(half),
        "rows_discarded_at_centre": int(fluid - 2 * half),
        "y_w": float(y_w),
        "symmetry_index_error": float(np.max(np.abs(lower + upper - (ny - 1)))),
    }
    return y, folded, info


def effective_height(ny: int, y_w: float = 0.5) -> float:
    """Wall-to-wall channel height for a given effective wall origin.

    With the two bounce-back walls at row-index ``y_w`` and ``ny-1-y_w`` the
    distance between them is ``ny-1-2 y_w``: the nominal ``ny-2`` only holds
    for the textbook ``y_w = 0.5``.  At ``y_w = 0.6447`` and ``ny = 48`` this
    is 0.71 cells narrower, i.e. ``Re_tau`` inferred from ``u_tau H / 2 nu``
    is 0.75 % lower than the nominal value - the same defect, propagated.
    """
    return float(ny - 1) - 2.0 * float(y_w)


def bulk_velocity(
    profile: Iterable[float],
    *,
    y_w: float = 0.5,
    y: Iterable[float] | None = None,
    height: float | None = None,
) -> float:
    """Bulk (cross-section averaged) velocity of the *full* channel.

    Trapezoidal quadrature with the wall origin used consistently in both the
    abscissa and the normalisation, so a laminar run reproduces the analytic
    ``2/3`` of the centre-line value exactly.

    ``y`` supplies the physical distance from the wall of each sample (needed
    for a published profile on a non-uniform mesh, e.g. the MKM table, where
    the abscissa is NOT the row index).  When it is omitted the fluid rows are
    assumed uniformly spaced at ``i - y_w`` and the height defaults to the
    number of fluid rows.
    """
    values = np.asarray(profile, dtype=float)
    if y is not None:
        yv = np.asarray(y, dtype=float)
        if yv.shape != values.shape:
            raise ValueError("y and profile must have the same length")
        span = float(height) if height is not None else float(yv[-1] - yv[0])
        return float(np.trapezoid(values, yv) / span)
    ny = values.size
    fluid = ny - 2
    if fluid < 2:
        raise ValueError("profile needs at least 2 fluid rows")
    yv = np.arange(1, ny - 1, dtype=float) - float(y_w)
    # The two effective walls are at ``y_w`` and ``ny-1-y_w`` in row-index
    # space, so the wall-to-wall height is ``ny-1-2 y_w`` - one ``2 y_w - 1``
    # narrower than the nominal fluid count.  Integrating (and normalising)
    # over the nominal height would bias ``U_b`` by O(y_w/H).
    span = float(height) if height is not None else effective_height(ny, y_w)
    # Anchor the integral on the no-slip condition: add u = 0 at both walls so
    # that the quadrature covers the whole channel, not only the cells between
    # the first and the last fluid node.
    ys = np.concatenate(([0.0], yv, [span]))
    us = np.concatenate(([0.0], values[1:-1], [0.0]))
    return float(np.trapezoid(us, ys) / span)


def friction_coefficient(u_tau: float, u_bulk: float) -> float:
    """Skin friction coefficient ``C_f = 2 u_tau^2 / U_b^2``."""
    if u_bulk <= 0.0:
        raise ValueError("bulk velocity must be positive")
    return 2.0 * u_tau**2 / (u_bulk**2)


def dean_channel_friction(re_tau: float) -> float:
    """Empirical ``C_f`` of a plane channel at the given ``Re_tau``.

    Uses the standard log-layer integration ``U_b+ ~= 2.5 ln(Re_tau) + 2.5``
    (Dean-type fit, Pope 2000, eq. 7.34 for the channel; the coefficient pair
    reproduces the Re_tau = 180 DNS bulk velocity to better than 1 %).
    """
    if re_tau <= 0.0:
        raise ValueError("Re_tau must be positive")
    u_bulk_plus = 2.5 * math.log(re_tau) + 2.5
    return 2.0 / u_bulk_plus**2


def u_tau_momentum_balance(body_force: float, height: float) -> float:
    """``u_tau`` from the exact mean momentum balance ``tau_w = rho a h / 2``."""
    value = 0.5 * float(body_force) * float(height)
    if value <= 0.0:
        raise ValueError("body force and height must give a positive wall shear")
    return math.sqrt(value)


def u_tau_wall_shear(wall_shear: float) -> float:
    """``u_tau`` from a measured wall shear stress (momentum exchange)."""
    if wall_shear <= 0.0:
        raise ValueError("wall shear stress must be positive")
    return math.sqrt(float(wall_shear))


def _log_window(y: np.ndarray, u_tau: float, nu: float, window: Sequence[float]) -> np.ndarray:
    y_plus = y * u_tau / nu
    lo, hi = float(window[0]), float(window[1])
    mask = (y_plus >= lo) & (y_plus <= hi) & (y > 0.0)
    return mask


def u_tau_log_law(
    y: Iterable[float],
    u: Iterable[float],
    nu: float,
    *,
    window: Sequence[float] = (30.0, 100.0),
    kappa: float = KAPPA,
    b_log: float = B_LOG,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Friction velocity from the log layer, both closure variants.

    ``level`` solves the implicit equation

        u(y) = u_tau [ ln((y) u_tau / nu) / kappa + B ]

    by fixed-point iteration on ``u_tau`` inside the logarithm (this is the
    physically meaningful closure: the law of the wall predicts the
    *magnitude*).  ``slope`` uses ``u_tau = kappa dU/dln y`` which follows from
    differentiating the same law and is insensitive to ``B``.

    Both are evaluated in the same window, and the window mask is rebuilt at
    every iteration because ``y+`` itself depends on ``u_tau``.  The residual
    RMS of the level fit is returned in wall units.
    """
    yv = np.asarray(y, dtype=float)
    uv = np.asarray(u, dtype=float)
    if yv.size != uv.size or yv.size < 3:
        raise ValueError("y and u must be equal-length (>=3 points)")
    # Slope route: dU / d ln y = u_tau / kappa, evaluated on the window in
    # the *current* wall units, iterated once for the window position.
    u_tau = max(float(np.max(np.abs(uv))) / 20.0, 1e-12)
    for _ in range(max_iterations):
        mask = _log_window(yv, u_tau, nu, window)
        if mask.sum() < 2:
            raise ValueError("log window contains fewer than two samples")
        slope = np.polyfit(np.log(yv[mask]), uv[mask], 1)[0]
        u_tau = kappa * float(slope)
    slope_route = u_tau
    slope_intercept = float(np.polyfit(np.log(yv[mask]), uv[mask], 1)[1])
    # Level route: fixed point on the implicit law of the wall.
    u_tau = slope_route
    for _ in range(max_iterations):
        mask = _log_window(yv, u_tau, nu, window)
        if mask.sum() < 2:
            raise ValueError("log window contains fewer than two samples")
        y_plus = yv[mask] * u_tau / nu
        predicted = u_tau * (np.log(y_plus) / kappa + b_log)
        ratio = float(np.mean(uv[mask] / predicted))
        updated = u_tau * (0.5 + 0.5 * ratio)
        if abs(updated - u_tau) <= tolerance * max(u_tau, 1e-30):
            u_tau = updated
            break
        u_tau = updated
    level_route = u_tau
    mask = _log_window(yv, level_route, nu, window)
    y_plus = yv[mask] * level_route / nu
    residual = uv[mask] - level_route * (np.log(y_plus) / kappa + b_log)
    return {
        "level": float(level_route),
        "slope": float(slope_route),
        "slope_intercept": slope_intercept,
        "model_form_gap_pct": float(
            100.0 * abs(level_route - slope_route) / max(0.5 * (level_route + slope_route), 1e-30)
        ),
        "window": [float(window[0]), float(window[1])],
        "samples": int(mask.sum()),
        "residual_rms_plus": float(np.sqrt(np.mean(residual**2)) / max(level_route, 1e-30)),
        "kappa": float(kappa),
        "b_log": float(b_log),
    }


def u_tau_total_stress(
    y: Iterable[float],
    u: Iterable[float],
    uv: Iterable[float],
    nu: float,
    height: float,
    *,
    window: Sequence[float] = (5.0, 30.0),
) -> float:
    """``u_tau`` from the linear total-stress profile of a forced channel.

    ``nu dU/dy - <u'v'> = u_tau^2 (1 - y/h)``; averaging the left-hand side
    divided by ``(1 - y/h)`` over the inner layer gives a profile-based
    estimate whose only assumption is the momentum balance itself.
    """
    yv = np.asarray(y, dtype=float)
    uv_mean = np.asarray(u, dtype=float)
    stress = np.asarray(uv, dtype=float)
    dudy = np.gradient(uv_mean, yv)
    total = nu * dudy - stress
    mask = (yv >= window[0]) & (yv <= window[1])
    if mask.sum() < 2:
        raise ValueError("total-stress window contains fewer than two samples")
    taper = 1.0 - yv[mask] / float(height)
    value = float(np.mean(total[mask] / taper))
    if not np.isfinite(value) or value <= 0.0:
        # Fail closed: a non-positive mean total stress means the profile is
        # not in momentum balance (transient), not that u_tau is imaginary.
        return float("nan")
    return float(math.sqrt(value))


def measure_wall_origin_poiseuille(
    profile: Iterable[float],
    *,
    nu: float,
    body_force: float,
    height: float | None = None,
) -> dict[str, float]:
    """Effective wall origin ``y_w`` of the discrete wall from a laminar profile.

    The discrete plane-Poiseuille solution of a bounce-back wall is a parabola
    in the *effective* coordinate ``s = y - y_w``:

        u(s) = (a / 2 nu) s (H - s) + b s ,   H = ny - 1 - 2 y_w

    ``H`` itself depends on ``y_w`` (the two effective walls are at ``y_w`` and
    ``ny-1-y_w``), so the two-parameter fit ``(y_w, b)`` is iterated to
    self-consistency.  The slip coefficient ``b`` absorbs the O(dx^2)
    truncation error of the bounce-back wall so that ``y_w`` is not biased by
    it, and the fit residual is returned as a fail-closed steady-state check.
    """
    values = np.asarray(profile, dtype=float)
    ny = values.size
    y = np.arange(1, ny - 1, dtype=float)
    u = values[1:-1]
    from scipy.optimize import least_squares  # local import: optional dependency

    span = float(height) if height is not None else float(ny - 2)
    y_w = 0.5
    slip = 0.0
    cost = float("inf")
    for _ in range(6):
        def model(params: np.ndarray, span: float = span) -> np.ndarray:
            wall, local_slip = params
            s = y - wall
            return body_force / (2.0 * nu) * s * (span - s) + local_slip * s

        best = None
        for guess in (0.5, 0.65, 0.8):
            result = least_squares(
                lambda p: model(p) - u,
                x0=np.array([guess, 0.0]),
                bounds=([-1.0, -1.0], [2.0, 1.0]),
            )
            if best is None or result.cost < best.cost:
                best = result
        new_y_w, new_slip = (float(v) for v in best.x)
        new_span = (
            float(height)
            if height is not None
            else effective_height(ny, new_y_w)
        )
        converged = abs(new_y_w - y_w) < 1e-6 and abs(new_span - span) < 1e-6
        y_w, slip, span, cost = new_y_w, new_slip, new_span, float(best.cost)
        if converged:
            break
    residual = model(np.array([y_w, slip]), span) - u
    return {
        "y_w": y_w,
        "slip": slip,
        "effective_height": span,
        "nominal_height": float(ny - 2),
        "residual_max_over_centre": float(
            np.max(np.abs(residual)) / max(np.max(np.abs(u)), 1e-30)
        ),
        "centre_line": float(np.max(u)),
    }


def measure_wall_origin_viscous(
    y: Iterable[float],
    u: Iterable[float],
    *,
    nu: float,
    wall_shear: float,
    max_y_plus: float = 4.0,
    u_tau: float | None = None,
) -> dict[str, float]:
    """Wall origin from the viscous sublayer with the wall shear *anchored*.

    In the viscous sublayer ``u = (tau_w / (rho nu)) (y - y_w)``; with ``tau_w``
    known exactly from the mean momentum balance of a forced channel the slope
    is not a free parameter, so ``y_w`` is the single unknown:

        y_w = < y - u nu / tau_w >

    This is the estimator that works on a *turbulent* run (the laminar
    inversion needs a steady laminar profile, which costs a viscous time) and
    it is the one that exposed the 0.5-cell assumption: the effective wall of a
    node-based bounce-back wall sits at ``y_w ~ 0.64`` cell.
    """
    yv = np.asarray(y, dtype=float)
    uv = np.asarray(u, dtype=float)
    if yv.size != uv.size or yv.size < 2:
        raise ValueError("y and u must be equal-length (>=2 points)")
    if wall_shear <= 0.0 or nu <= 0.0:
        raise ValueError("wall shear and viscosity must be positive")
    if u_tau is None:
        u_tau = math.sqrt(wall_shear)
    mask = (yv > 0.0) & (yv * u_tau / nu <= max_y_plus)
    if mask.sum() < 1:
        raise ValueError("no samples inside the viscous sublayer")
    offsets = yv[mask] - uv[mask] * nu / wall_shear
    slope = wall_shear / nu
    residual = uv[mask] - slope * (yv[mask] - float(np.mean(offsets)))
    return {
        "y_w": float(np.mean(offsets)),
        "y_w_spread": float(np.max(offsets) - np.min(offsets)),
        "samples": int(mask.sum()),
        "slope": float(slope),
        "residual_max": float(np.max(np.abs(residual))),
        "u_tau_implied": float(math.sqrt(wall_shear)),
        "max_y_plus": float(np.max(yv[mask] * u_tau / nu)),
    }


def u_tau_three_way(
    *,
    body_force: float,
    height: float,
    wall_shear: float | None,
    y: Iterable[float] | None = None,
    u: Iterable[float] | None = None,
    nu: float | None = None,
    window: Sequence[float] = (30.0, 100.0),
    wall_shear_uncertainty: float | None = None,
    tolerance_pct: float = 2.0,
) -> dict[str, object]:
    """Compare the three friction-velocity routes of a channel campaign.

    ``momentum_balance`` is treated as the anchor (it is exact for a steady
    forced channel), the log-law route is reported together with its own
    model-form gap, and the wall-shear route is reported with its deviation.
    ``consistent`` is True only when *every* pairwise deviation is below
    ``tolerance_pct`` **and** the log-law model-form gap is below it as well,
    so that a converged simulation is not credited when the closure itself is
    the limiting error.
    """
    routes: dict[str, float] = {}
    detail: dict[str, object] = {}
    u_tau_balance = u_tau_momentum_balance(body_force, height)
    routes["momentum_balance"] = u_tau_balance
    if wall_shear is not None:
        routes["wall_shear"] = u_tau_wall_shear(wall_shear)
        if wall_shear_uncertainty is not None:
            detail["wall_shear_uncertainty"] = float(wall_shear_uncertainty)
    if y is not None and u is not None and nu is not None:
        log = u_tau_log_law(y, u, nu, window=window)
        routes["log_law_level"] = log["level"]
        routes["log_law_slope"] = log["slope"]
        detail["log_law"] = log
        detail["log_law_model_form_gap_pct"] = log["model_form_gap_pct"]
        # Keep the required three-way set on the level closure (the standard
        # one-parameter law of the wall), the slope variant is the diagnostic.
        routes_for_consistency = {
            "momentum_balance": u_tau_balance,
            "wall_shear": routes.get("wall_shear", float("nan")),
            "log_law": log["level"],
        }
    else:
        routes_for_consistency = dict(routes)
    names = [k for k, v in routes_for_consistency.items() if np.isfinite(v)]
    deviations: dict[str, float] = {}
    max_deviation = 0.0
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            a = routes_for_consistency[name_a]
            b = routes_for_consistency[name_b]
            dev = 100.0 * abs(a - b) / max(0.5 * (a + b), 1e-30)
            deviations[f"{name_a}_vs_{name_b}"] = dev
            max_deviation = max(max_deviation, dev)
    model_gap = float(detail.get("log_law_model_form_gap_pct", 0.0) or 0.0)
    return {
        "routes": {k: float(v) for k, v in routes.items()},
        "deviations_pct": deviations,
        "max_pairwise_deviation_pct": float(max_deviation),
        "log_law_model_form_gap_pct": model_gap,
        "tolerance_pct": float(tolerance_pct),
        "consistent": bool(
            max_deviation < tolerance_pct and model_gap < tolerance_pct and len(names) >= 3
        ),
        "detail": detail,
    }


def profile_error_vs_reference(
    y_plus: Iterable[float],
    u_plus: Iterable[float],
    ref_y_plus: Iterable[float],
    ref_u_plus: Iterable[float],
    *,
    window: Sequence[float] | None = None,
) -> dict[str, float]:
    """Relative ``L2`` / ``Linf`` error of a profile against a reference.

    The reference is interpolated on the simulation abscissa (never the other
    way round, which would hide resolution effects).  Above the reference
    centre line the interpolation is clamped - the caller must fold the profile
    first (see :func:`fold_channel_profile`); the returned ``clamped_fraction``
    is the guard that makes an un-folded profile impossible to report silently.
    """
    yp = np.asarray(y_plus, dtype=float)
    up = np.asarray(u_plus, dtype=float)
    ry = np.asarray(ref_y_plus, dtype=float)
    ru = np.asarray(ref_u_plus, dtype=float)
    order = np.argsort(ry)
    ry, ru = ry[order], ru[order]
    mask = np.ones(yp.shape, dtype=bool)
    if window is not None:
        mask = (yp >= window[0]) & (yp <= window[1])
    clamped = (yp > ry[-1]).sum()
    ref = np.interp(yp[mask], ry, ru)
    diff = up[mask] - ref
    return {
        "samples": int(mask.sum()),
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-30)),
        "max_abs_deviation": float(np.max(np.abs(diff))),
        "max_relative_deviation": float(np.max(np.abs(diff) / np.maximum(np.abs(ref), 1e-30))),
        "clamped_fraction": float(clamped / max(yp.size, 1)),
        "window": [float(v) for v in window] if window is not None else None,
    }