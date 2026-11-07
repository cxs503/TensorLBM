"""Wall-origin (``y_w``) identification and three-way friction-velocity closure.

Motivation
----------
Lattice-Boltzmann wall-resolved channel analyses turn a discrete mean velocity
profile into wall units through

    y+ = (y - y_w) u_tau / nu,      u+ = u / u_tau .

The wall position ``y_w`` is *not* a free parameter of the physics: a
bounce-back wall has an effective position that depends on the collision
operator, the relaxation time and the streaming/bounce-back ordering.  Assuming
a textbook value (``y_w = 0.5`` for the halfway convention) silently biases the
whole profile by ``(y_w,measured - 0.5) u_tau / nu`` in wall units, which at
``dx+ ~ 10`` is a several-wall-unit shift of the abscissa - enough to move
``u+`` by several percent in the buffer layer and to corrupt a friction-velocity
inversion.

This module identifies ``y_w`` *from the profile itself* (measured, not
assumed) and closes the friction velocity through three independent routes:

``pressure_gradient``
    Mean momentum balance of a periodic channel driven by a uniform body force
    ``a``: ``tau_w = rho a h`` so ``u_tau = sqrt(a h)`` (``h`` = half height).
    Reference route: exact by construction, independent of the profile.

``wall_shear``
    Near-wall viscous slope ``du/dy = u_tau^2 / nu`` fitted together with the
    wall origin, ``u_tau = sqrt(a_slope nu)``.  Uses only the first few nodes,
    or the supplied wall stress (momentum exchange / wall diagnostic).

``log_law``
    Regression of ``u = (u_tau/kappa) ln((y-y_w) u_tau/nu) + B u_tau`` over a
    log-layer window, again fitting ``y_w`` jointly.

All three are compared pairwise; ``consistent`` is True when every pairwise
relative deviation is below the requested tolerance (default 3 %).

The fits are dependency-light (numpy only): the two-parameter problems are
solved by a deterministic bounded compass search, which is adequate for the
smooth two-parameter residuals involved and keeps the module usable on nodes
without scipy.

References
----------
Moser, Kim & Mansour, Phys. Fluids 11, 943 (1999) - channel DNS, Re_tau = 180,
395, 590.  Kim, Moin & Moser, JFM 177, 133 (1987).  Spalding (1961) single
wall law; Reichardt (1951).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .wall_law import B_LOG, KAPPA, spalding_u_plus

__all__ = [
    "WallOriginFit",
    "fit_wall_origin_viscous",
    "fit_wall_origin_wall_stress",
    "fit_wall_origin_loglaw",
    "u_tau_from_log_law",
    "fit_wall_origin_spalding",
    "fit_wall_origin",
    "poiseuille_wall_origin",
    "wall_offset_bias",
    "u_tau_three_way",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_profile(y: Iterable[float], u: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    yv = np.asarray(y, dtype=float)
    uv = np.asarray(u, dtype=float)
    if yv.shape != uv.shape:
        raise ValueError("y and u must have the same length")
    if yv.size < 3:
        raise ValueError("need at least 3 profile samples")
    if not np.all(np.isfinite(yv)) or not np.all(np.isfinite(uv)):
        raise ValueError("profile contains non-finite values")
    order = np.argsort(yv)
    return yv[order], uv[order]


def _compass_search(
    residual,
    x0: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    max_evaluations: int = 2000,
    tolerance: float = 1.0e-13,
) -> tuple[np.ndarray, float, int]:
    """Deterministic bounded compass (pattern) search for a smooth problem.

    Returns ``(x_best, residual_best, n_evaluations)``.  No gradients, no random
    numbers, so pytest can assert reproducible recovery tolerances.
    """
    x = np.array(
        [min(max(v, lo), hi) for v, lo, hi in zip(x0, lower, upper, strict=True)],
        dtype=float,
    )
    step = 0.25 * np.array(
        [max(hi - lo, 1.0e-12) for lo, hi in zip(lower, upper, strict=True)],
        dtype=float,
    )
    best = float(residual(x))
    evaluations = 1
    while evaluations < max_evaluations:
        improved = False
        for dim in range(x.size):
            for sign in (1.0, -1.0):
                candidate = x.copy()
                candidate[dim] = min(
                    max(candidate[dim] + sign * step[dim], lower[dim]), upper[dim]
                )
                if candidate[dim] == x[dim]:
                    continue
                value = float(residual(candidate))
                evaluations += 1
                if value < best - tolerance * max(1.0, abs(best)):
                    x, best, improved = candidate, value, True
                if evaluations >= max_evaluations:
                    break
            if evaluations >= max_evaluations:
                break
        if not improved:
            step *= 0.5
            if np.all(step <= 1.0e-13 * np.maximum(1.0, np.abs(x))):
                break
    return x, best, evaluations


def _fit_degeneracy(
    residual,
    x: np.ndarray,
    value: float,
    yv: np.ndarray,
    dy: float,
) -> dict:
    """Local conditioning of a joint ``(y_origin, u_tau)`` fit.

    Finite-difference sensitivities of the residual to each parameter, their
    dimensionless correlation, and a ridge flag.  A joint wall-origin /
    friction-velocity fit of the log law over a narrow window is a *ridge*
    problem (both parameters enter the same logarithm), so the reported
    parameters are weakly identified even when the residual is small - this
    diagnostic makes that explicit instead of hiding it in a single number.
    """
    step_origin = max(0.01 * dy, 1.0e-6)
    step_u_tau = max(1.0e-4 * abs(x[1]), 1.0e-12)
    plus_o = residual(np.array([x[0] + step_origin, x[1]]))
    minus_o = residual(np.array([x[0] - step_origin, x[1]]))
    plus_u = residual(np.array([x[0], x[1] + step_u_tau]))
    minus_u = residual(np.array([x[0], x[1] - step_u_tau]))
    g_origin = (plus_o - minus_o) / (2.0 * step_origin)
    g_u_tau = (plus_u - minus_u) / (2.0 * step_u_tau)
    normalised = math.hypot(g_origin * step_origin, g_u_tau * step_u_tau)
    ridge = bool(
        normalised <= 1.0e-6 * max(value, 1.0e-30) or min(abs(g_origin), abs(g_u_tau)) == 0.0
    )
    return {
        "gradient_origin": float(g_origin),
        "gradient_u_tau": float(g_u_tau),
        "normalised_sensitivity": float(normalised),
        "ridge_flagged": ridge,
        "note": (
            "joint (y_w, u_tau) log-law fits are weakly identified for narrow "
            "windows; prefer a measured wall origin plus a 1-D u_tau solve"
        ),
    }


@dataclass(frozen=True)
class WallOriginFit:
    """Result of a wall-origin / friction-velocity identification."""

    method: str
    y_origin: float
    u_tau: float
    n_points: int
    y_window: tuple[float, float]
    residual_rms: float
    converged: bool
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "y_origin": self.y_origin,
            "u_tau": self.u_tau,
            "n_points": self.n_points,
            "y_window": list(self.y_window),
            "residual_rms": self.residual_rms,
            "converged": self.converged,
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# 1. Viscous-sublayer fit (u+ = y+)
# ---------------------------------------------------------------------------
def fit_wall_origin_viscous(
    y: Sequence[float],
    u: Sequence[float],
    nu: float,
    *,
    y_plus_max: float = 5.0,
    y_plus_min: float = 0.5,
    iterations: int = 60,
) -> WallOriginFit:
    """Identify ``(y_w, u_tau)`` from the linear viscous sublayer.

    The law ``u = (u_tau^2/nu)(y - y_w)`` gives both unknowns from a straight
    line: ``u_tau = sqrt(a nu)`` and ``y_w = -b/a`` for ``u = a y + b``.  The
    selection window is expressed in wall units, so the fit is iterated until
    ``y_w`` and ``u_tau`` stop moving (the window depends on both).
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    yv, uv = _as_profile(y, u)
    if yv[0] < 0.0:
        raise ValueError("profile y must be non-negative")
    dy = float(yv[1] - yv[0])
    if dy <= 0.0:
        raise ValueError("profile y must be strictly increasing")

    u_tau = math.sqrt(max((uv[1] - uv[0]) / dy, 1.0e-30) * nu)
    y_origin = 0.0
    for _ in range(iterations):
        y_plus = (yv - y_origin) * u_tau / nu
        mask = (y_plus > y_plus_min) & (y_plus <= y_plus_max)
        if int(np.count_nonzero(mask)) < 3:
            mask = np.zeros_like(yv, dtype=bool)
            mask[: min(4, yv.size)] = True
        a, b = np.polyfit(yv[mask], uv[mask], 1)
        if a <= 0.0:
            break
        new_origin = -b / a
        new_u_tau = math.sqrt(a * nu)
        converged = abs(new_u_tau - u_tau) <= 1.0e-13 * max(u_tau, 1.0e-30) and abs(
            new_origin - y_origin
        ) <= 1.0e-13
        y_origin, u_tau = new_origin, new_u_tau
        if converged:
            break
    y_plus = (yv - y_origin) * u_tau / nu
    mask = (y_plus > y_plus_min) & (y_plus <= y_plus_max)
    if int(np.count_nonzero(mask)) < 3:
        mask = np.zeros_like(yv, dtype=bool)
        mask[: min(4, yv.size)] = True
    a, b = np.polyfit(yv[mask], uv[mask], 1)
    residual = uv[mask] - (a * yv[mask] + b)
    return WallOriginFit(
        method="viscous",
        y_origin=float(-b / a),
        u_tau=float(math.sqrt(max(a * nu, 0.0))),
        n_points=int(np.count_nonzero(mask)),
        y_window=(float(yv[mask].min()), float(yv[mask].max())),
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        converged=bool(a > 0.0),
        diagnostics={
            "slope": float(a),
            "intercept": float(b),
            "y_plus_max": y_plus_max,
            "y_plus_min": y_plus_min,
        },
    )


# ---------------------------------------------------------------------------
# 1b. Wall-stress-anchored fit (uses the exact momentum-balance wall stress)
# ---------------------------------------------------------------------------
def fit_wall_origin_wall_stress(
    y: Sequence[float],
    u: Sequence[float],
    nu: float,
    wall_shear: float,
    *,
    n_cells: int = 2,
    rho: float = 1.0,
) -> WallOriginFit:
    """Identify ``y_w`` from the near-wall slope fixed by the wall stress.

    In wall units the viscous sublayer obeys ``du/dy = tau_w/(rho nu)`` exactly.
    When ``tau_w`` is known independently - a channel's mean momentum balance
    ``tau_w = rho a h`` or a momentum-exchange measurement - the slope is no
    longer a fit parameter, so the wall origin follows from a one-parameter
    intercept fit over the near-wall cells:

        y_w = mean(y) - mean(u) rho nu / tau_w .

    This is a *local* estimator; it is far better conditioned than a global
    parabolic fit and is insensitive to the outer-layer profile or to
    under-converged turbulence statistics.  Being first-order accurate in
    ``dx``, it is reported alongside the parabolic estimator as an uncertainty
    band rather than as a single truth.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    if wall_shear <= 0.0:
        raise ValueError("wall_shear must be positive")
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if n_cells < 1:
        raise ValueError("n_cells must be at least 1")
    yv, uv = _as_profile(y, u)
    count = min(int(n_cells) + 1, yv.size)
    slope = wall_shear / (rho * nu)
    intercept = float(np.mean(uv[:count] - slope * yv[:count]))
    residual = uv[:count] - (slope * yv[:count] + intercept)
    return WallOriginFit(
        method="wall_stress",
        y_origin=float(-intercept / slope),
        u_tau=float(math.sqrt(wall_shear / rho)),
        n_points=int(count),
        y_window=(float(yv[0]), float(yv[count - 1])),
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        converged=True,
        diagnostics={
            "imposed_slope": float(slope),
            "wall_shear": float(wall_shear),
            "n_cells": int(count),
        },
    )


# ---------------------------------------------------------------------------
# 2. Log-law regression with joint wall origin
# ---------------------------------------------------------------------------
def fit_wall_origin_loglaw(
    y: Sequence[float],
    u: Sequence[float],
    nu: float,
    *,
    kappa: float = KAPPA,
    b: float = B_LOG,
    y_plus_window: tuple[float, float] = (30.0, 120.0),
    y_origin_initial: float = 0.0,
    u_tau_initial: float | None = None,
    y_origin_span: float | None = None,
    max_evaluations: int = 6000,
) -> WallOriginFit:
    """Joint ``(y_w, u_tau)`` fit of the log law over a wall-unit window.

        u(y) = u_tau [ ln((y - y_w) u_tau / nu) / kappa + B ]

    Two parameters from many samples; the window is expressed in wall units and
    recomputed inside the residual, so the fit is self-consistent in exactly the
    way a practitioner selecting "y+ in [30,120]" would be.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    yv, uv = _as_profile(y, u)
    dy = float(np.min(np.diff(yv)))
    if dy <= 0.0:
        raise ValueError("profile y must be strictly increasing")
    span = float(y_origin_span) if y_origin_span is not None else 10.0 * dy
    lower = [float(yv[0]) - span, 1.0e-8]
    upper = [float(yv[0]) - 1.0e-6, max(10.0 * float(uv.max()) / max(b, 1.0), 1.0e-6)]
    if u_tau_initial is None:
        u_mid = float(uv[len(uv) // 2])
        u_tau_initial = max(u_mid / 15.0, 1.0e-6)
    x0 = [min(max(y_origin_initial, lower[0]), upper[0]), u_tau_initial]
    lo_win, hi_win = y_plus_window

    def residual(x: np.ndarray) -> float:
        y_origin, u_tau = float(x[0]), float(x[1])
        wall_distance = yv - y_origin
        if np.any(wall_distance <= 0.0):
            return 1.0e12
        y_plus = wall_distance * u_tau / nu
        mask = (y_plus >= lo_win) & (y_plus <= hi_win)
        if int(np.count_nonzero(mask)) < 3:
            return 1.0e12
        model = u_tau * (np.log(y_plus[mask]) / kappa + b)
        return float(np.sqrt(np.mean((uv[mask] - model) ** 2)))

    x_best, value, evaluations = _compass_search(
        residual, x0, lower, upper, max_evaluations=max_evaluations
    )
    y_origin, u_tau = float(x_best[0]), float(x_best[1])
    y_plus = (yv - y_origin) * u_tau / nu
    mask = (y_plus >= lo_win) & (y_plus <= hi_win)
    if int(np.count_nonzero(mask)) < 3:
        mask = np.ones_like(yv, dtype=bool)
    model = u_tau * (np.log(np.maximum(y_plus[mask], 1.0e-30)) / kappa + b)
    return WallOriginFit(
        method="log_law",
        y_origin=y_origin,
        u_tau=u_tau,
        n_points=int(np.count_nonzero(mask)),
        y_window=(float(y_plus[mask].min()), float(y_plus[mask].max())),
        residual_rms=float(np.sqrt(np.mean((uv[mask] - model) ** 2))),
        converged=bool(value < 1.0e11),
        diagnostics={
            "kappa": kappa,
            "b": b,
            "y_plus_window": list(y_plus_window),
            "residual_objective": float(value),
            "evaluations": int(evaluations),
            "degeneracy": _fit_degeneracy(
                residual, np.array([y_origin, u_tau], dtype=float), value, yv, dy
            ),
        },
    )


# ---------------------------------------------------------------------------
# 2b. Friction velocity from the log law at a known wall origin (1-D solve)
# ---------------------------------------------------------------------------
def u_tau_from_log_law(
    y: Sequence[float],
    u: Sequence[float],
    nu: float,
    *,
    y_origin: float,
    kappa: float = KAPPA,
    b: float = B_LOG,
    y_plus_window: tuple[float, float] = (30.0, 120.0),
    mode: str = "level",
    iterations: int = 200,
    u_tau_initial: float | None = None,
) -> dict:
    """Friction velocity from the log law at a *known* wall origin.

    With ``y_w`` known (measured, see :func:`poiseuille_wall_origin` /
    :func:`fit_wall_origin_wall_stress`) the log law

        u(y) = (u_tau/kappa) ln((y - y_w) u_tau / nu) + B u_tau

    has a single unknown, and two standard closures are available:

    ``mode="level"``  solve the *level* of the profile with ``kappa`` and ``B``
        held at their universal values (the operational choice when the
        constants are trusted).  Monotone in ``u_tau``; located here by a
        bracketed interval search on the windowed RMS residual.
    ``mode="slope"``  use the fact that the law is linear in
        ``ln(y - y_w)`` with slope ``u_tau/kappa``: an ordinary least-squares
        slope gives ``u_tau`` with no dependence on ``nu`` or ``B`` (the
        standard choice when ``B`` is not transferable).

    The two variants bracket the model-form uncertainty of the log law; on the
    MKM Re_tau = 180 reference they differ by ~7 %, which is why a single
    "log-law friction velocity" carries a several-per-cent uncertainty band at
    low Reynolds number.  Solving ``(y_w, u_tau)`` jointly instead is a ridge
    problem: both parameters enter the same logarithm, which
    :func:`fit_wall_origin_loglaw` reports through its ``degeneracy``
    diagnostic.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    if mode not in {"level", "slope"}:
        raise ValueError("mode must be 'level' or 'slope'")
    for name, value in (("kappa", kappa), ("b", b)):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    yv, uv = _as_profile(y, u)
    wall_distance = yv - float(y_origin)
    if np.any(wall_distance <= 0.0):
        raise ValueError("all profile samples must lie above the wall origin")
    log_distance = np.log(wall_distance)

    def window_mask(u_tau: float) -> np.ndarray:
        y_plus = wall_distance * u_tau / nu
        lo, hi = y_plus_window
        mask = (y_plus >= lo) & (y_plus <= hi)
        if int(np.count_nonzero(mask)) < 3:
            mask = np.zeros_like(yv, dtype=bool)
            mask[len(yv) // 3 :] = True
        return mask

    def level_residual(u_tau: float) -> float:
        mask = window_mask(u_tau)
        y_plus = wall_distance * u_tau / nu
        model = u_tau * (np.log(y_plus[mask]) / kappa + b)
        return float(np.sqrt(np.mean((uv[mask] - model) ** 2)))

    slope_all = float(np.polyfit(log_distance, uv, 1)[0])
    if mode == "slope":
        u_tau = max(kappa * slope_all, 1.0e-12)
        mask = np.ones_like(yv, dtype=bool)
        for _ in range(40):
            candidate = window_mask(u_tau)
            slope, intercept = np.polyfit(log_distance[candidate], uv[candidate], 1)
            new_u_tau = max(kappa * float(slope), 1.0e-12)
            converged = np.array_equal(candidate, mask) and abs(
                new_u_tau - u_tau
            ) <= 1.0e-13 * max(u_tau, 1.0e-30)
            mask, u_tau = candidate, new_u_tau
            if converged:
                break
        u_plus_predicted = u_tau * (np.log(wall_distance * u_tau / nu) / kappa + b)
        residual = uv[mask] - u_plus_predicted[mask]
        return {
            "mode": "slope",
            "u_tau": float(u_tau),
            "n_points": int(np.count_nonzero(mask)),
            "y_plus_window_used": [
                float((wall_distance[mask] * u_tau / nu).min()),
                float((wall_distance[mask] * u_tau / nu).max()),
            ],
            "residual_rms": float(np.sqrt(np.mean(residual**2))),
            "residual_rms_normalised": float(
                np.sqrt(np.mean(residual**2)) / max(u_tau, 1.0e-30)
            ),
            "b_recovered": float(
                intercept / u_tau - math.log(u_tau / nu) / kappa
            ),
            "kappa": kappa,
            "b": b,
            "log_slope": float(slope),
        }

    # Level mode: bracket the minimum of the windowed residual, then refine.
    guess = (
        float(u_tau_initial)
        if u_tau_initial is not None
        else max(kappa * slope_all, max(uv.max(), 1.0e-12) / max(b, 1.0e-12))
    )
    low = max(0.2 * guess, 1.0e-12)
    high = max(5.0 * guess, 4.0 * low)
    grid = np.geomspace(low, high, 400)
    values = np.array([level_residual(float(g)) for g in grid])
    index = int(np.argmin(values))
    left = float(grid[max(index - 1, 0)])
    right = float(grid[min(index + 1, grid.size - 1)])
    for _ in range(iterations):
        third = (right - left) / 3.0
        if level_residual(left + third) <= level_residual(right - third):
            right -= third
        else:
            left += third
        if right - left <= 1.0e-13 * max(right, 1.0e-30):
            break
    u_tau = 0.5 * (left + right)
    mask = window_mask(u_tau)
    y_plus = wall_distance * u_tau / nu
    model = u_tau * (np.log(y_plus[mask]) / kappa + b)
    residual = uv[mask] - model
    return {
        "mode": "level",
        "u_tau": float(u_tau),
        "n_points": int(np.count_nonzero(mask)),
        "y_plus_window_used": [float(y_plus[mask].min()), float(y_plus[mask].max())],
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "residual_rms_normalised": float(
            np.sqrt(np.mean(residual**2)) / max(u_tau, 1.0e-30)
        ),
        "b_recovered": float(np.mean(uv[mask] / u_tau - np.log(y_plus[mask]) / kappa)),
        "kappa": kappa,
        "b": b,
    }


# ---------------------------------------------------------------------------
# 3. Spalding single-law fit (covers sublayer + buffer + log layer)
# ---------------------------------------------------------------------------
def fit_wall_origin_spalding(
    y: Sequence[float],
    u: Sequence[float],
    nu: float,
    *,
    kappa: float = KAPPA,
    b: float = B_LOG,
    y_plus_window: tuple[float, float] = (1.0, 200.0),
    y_origin_initial: float = 0.0,
    u_tau_initial: float | None = None,
    y_origin_span: float | None = None,
    max_evaluations: int = 6000,
) -> WallOriginFit:
    """Joint ``(y_w, u_tau)`` fit of Spalding's composite law.

    Spalding's law is valid from the wall through the log layer, so the fit uses
    the whole wall-resolved profile instead of a single sublayer window, which
    makes it the default ``auto`` method for data whose first cell may sit above
    the viscous sublayer.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    yv, uv = _as_profile(y, u)
    dy = float(np.min(np.diff(yv)))
    span = float(y_origin_span) if y_origin_span is not None else 10.0 * dy
    lower = [float(yv[0]) - span, 1.0e-8]
    upper = [float(yv[0]) - 1.0e-6, max(10.0 * float(uv.max()) / max(b, 1.0), 1.0e-6)]
    if u_tau_initial is None:
        u_tau_initial = max(float(uv[len(uv) // 4]) / 12.0, 1.0e-6)
    x0 = [min(max(y_origin_initial, lower[0]), upper[0]), u_tau_initial]
    lo_win, hi_win = y_plus_window

    def residual(x: np.ndarray) -> float:
        y_origin, u_tau = float(x[0]), float(x[1])
        wall_distance = yv - y_origin
        if np.any(wall_distance <= 0.0):
            return 1.0e12
        y_plus = wall_distance * u_tau / nu
        mask = (y_plus >= lo_win) & (y_plus <= hi_win)
        if int(np.count_nonzero(mask)) < 3:
            return 1.0e12
        model = np.array([spalding_u_plus(float(v), kappa, b) for v in y_plus[mask]])
        return float(np.sqrt(np.mean((uv[mask] / u_tau - model) ** 2)))

    x_best, value, evaluations = _compass_search(
        residual, x0, lower, upper, max_evaluations=max_evaluations
    )
    y_origin, u_tau = float(x_best[0]), float(x_best[1])
    y_plus = (yv - y_origin) * u_tau / nu
    mask = (y_plus >= lo_win) & (y_plus <= hi_win)
    if int(np.count_nonzero(mask)) < 3:
        mask = np.ones_like(yv, dtype=bool)
    model = np.array(
        [spalding_u_plus(float(v), kappa, b) for v in np.maximum(y_plus[mask], 1.0e-30)]
    )
    return WallOriginFit(
        method="spalding",
        y_origin=y_origin,
        u_tau=u_tau,
        n_points=int(np.count_nonzero(mask)),
        y_window=(float(y_plus[mask].min()), float(y_plus[mask].max())),
        residual_rms=float(np.sqrt(np.mean((uv[mask] / u_tau - model) ** 2))),
        converged=bool(value < 1.0e11),
        diagnostics={
            "kappa": kappa,
            "b": b,
            "y_plus_window": list(y_plus_window),
            "residual_objective": float(value),
            "evaluations": int(evaluations),
        },
    )


def fit_wall_origin(
    y: Sequence[float],
    u: Sequence[float],
    nu: float,
    *,
    method: str = "auto",
    **kwargs,
) -> WallOriginFit:
    """Dispatch to a wall-origin identification method.

    ``auto`` uses the viscous fit when the first sample lies in the viscous
    sublayer for the estimated friction velocity and Spalding's composite law
    otherwise.
    """
    if method == "viscous":
        return fit_wall_origin_viscous(y, u, nu, **kwargs)
    if method == "log_law":
        return fit_wall_origin_loglaw(y, u, nu, **kwargs)
    if method == "spalding":
        return fit_wall_origin_spalding(y, u, nu, **kwargs)
    if method != "auto":
        raise ValueError(f"unknown method {method!r}")
    yv, uv = _as_profile(y, u)
    dy = float(yv[1] - yv[0])
    u_tau_guess = math.sqrt(max((uv[1] - uv[0]) / dy, 1.0e-30) * nu)
    y_plus_first = yv[0] * u_tau_guess / nu
    if y_plus_first < 5.0:
        return fit_wall_origin_viscous(y, u, nu)
    return fit_wall_origin_spalding(y, u, nu, **kwargs)


# ---------------------------------------------------------------------------
# 4. Laminar Poiseuille wall-origin calibration
# ---------------------------------------------------------------------------
def poiseuille_wall_origin(
    y: Sequence[float],
    u: Sequence[float],
    nu: float,
    body_force: float,
) -> dict:
    """Measure the effective wall position from a laminar plane-Poiseuille run.

    For a body-force driven plane channel the exact solution is a parabola

        u(y) = u_c - a (y - y_c)^2 / (2 nu),

    with ``a`` the body-force acceleration and ``y_c`` the symmetry line.  A
    quadratic least-squares fit of the *simulated* profile therefore yields the
    symmetry line, the centreline velocity, and - through the known ``a`` and
    ``nu`` - the effective half height ``h_eff`` and thus both effective wall
    positions.  The measured wall offset is the wall position relative to the
    first/last profile sample, i.e. exactly the quantity post-processing has to
    know (and previously assumed to be 0.5).

    This is the calibration that distinguishes an *assumed* ``y_w`` from a
    *measured* one; it is exact for the continuum parabola, so the residual
    reports the discrete (lattice) error directly.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    if body_force <= 0.0:
        raise ValueError("body_force must be positive")
    yv, uv = _as_profile(y, u)
    coefficients = np.polyfit(yv, uv, 2)  # [c2, c1, c0]
    c2, c1, c0 = (float(v) for v in coefficients)
    if c2 >= 0.0:
        raise ValueError("profile is not concave: not a Poiseuille parabola")
    y_c = -c1 / (2.0 * c2)
    u_c = c0 + c1 * y_c + c2 * y_c**2
    h_eff = 2.0 * math.sqrt(max(2.0 * nu * u_c / body_force, 0.0))
    y_wall_low = y_c - 0.5 * h_eff
    y_wall_high = y_c + 0.5 * h_eff
    fitted = np.polyval(coefficients, yv)
    return {
        "symmetry_line": y_c,
        "centreline_velocity": u_c,
        "effective_half_height": 0.5 * h_eff,
        "effective_height": h_eff,
        "wall_position_low": y_wall_low,
        "wall_position_high": y_wall_high,
        "wall_offset_from_first_sample": y_wall_low - float(yv[0]),
        "wall_offset_from_last_sample": y_wall_high - float(yv[-1]),
        "wall_offset_mean": 0.5
        * ((y_wall_low - float(yv[0])) + (y_wall_high - float(yv[-1]))),
        "parabola_coefficients": [c2, c1, c0],
        "residual_rms": float(np.sqrt(np.mean((uv - fitted) ** 2))),
        "n_points": int(yv.size),
    }


def wall_offset_bias(
    y: Sequence[float],
    u: Sequence[float],
    nu: float,
    u_tau: float,
    *,
    y_origin_assumed: float,
    y_origin_measured: float,
    y_plus_window: tuple[float, float] = (5.0, 100.0),
) -> dict:
    """Quantify the profile bias caused by assuming instead of measuring ``y_w``.

    Returns the wall-unit displacement ``(y_w,measured - y_w,assumed) u_tau/nu``
    and the induced ``u+`` deviations (RMS and maximum) over the requested
    wall-unit window.  This is the systematic, sign-known error the
    post-processing incurs *before* any comparison with reference data.
    """
    if nu <= 0.0 or u_tau <= 0.0:
        raise ValueError("nu and u_tau must be positive")
    yv, uv = _as_profile(y, u)
    y_plus_assumed = (yv - y_origin_assumed) * u_tau / nu
    y_plus_measured = (yv - y_origin_measured) * u_tau / nu
    lo, hi = y_plus_window
    mask = (y_plus_assumed >= lo) & (y_plus_assumed <= hi)
    if int(np.count_nonzero(mask)) < 2:
        mask = np.ones_like(yv, dtype=bool)
    u_plus = uv / u_tau
    order = np.argsort(y_plus_measured)
    yp_sorted = y_plus_measured[order]
    up_sorted = u_plus[order]
    shifted = np.interp(y_plus_assumed[mask], yp_sorted, up_sorted)
    delta = shifted - u_plus[mask]
    return {
        "y_origin_assumed": float(y_origin_assumed),
        "y_origin_measured": float(y_origin_measured),
        "y_origin_delta_cells": float(y_origin_measured - y_origin_assumed),
        "y_plus_shift": float((y_origin_measured - y_origin_assumed) * u_tau / nu),
        "u_plus_rms_bias": float(np.sqrt(np.mean(delta**2))),
        "u_plus_max_bias": float(np.max(np.abs(delta))),
        "window": [float(lo), float(hi)],
        "n_points": int(np.count_nonzero(mask)),
        "exceeds_0p1_cell_trigger": bool(
            abs(y_origin_measured - y_origin_assumed) > 0.1
        ),
    }


# ---------------------------------------------------------------------------
# 5. Three-way friction velocity closure
# ---------------------------------------------------------------------------
def u_tau_three_way(
    *,
    y: Sequence[float],
    u: Sequence[float],
    nu: float,
    body_force: float,
    half_height: float,
    rho: float = 1.0,
    wall_shear: float | None = None,
    kappa: float = KAPPA,
    b: float = B_LOG,
    y_plus_window: tuple[float, float] = (30.0, 120.0),
    y_origin: float | None = None,
    tolerance_pct: float = 3.0,
) -> dict:
    """Friction velocity from three independent routes, plus their closure.

    ``pressure_gradient``  ``u_tau = sqrt(a h)`` - global momentum balance of the
        periodic channel (exact for a fully developed channel, independent of
        the profile).
    ``wall_shear``  ``u_tau = sqrt(tau_w/rho)`` - ``tau_w`` supplied (momentum
        exchange / wall-stress diagnostic) or, when not supplied, recovered from
        the viscous-sublayer slope fit together with ``y_w``.
    ``log_law``  ``u_tau`` from the joint log-law regression using the measured
        wall origin.

    ``y_w`` is identified from the profile unless supplied; ``consistent`` is
    True iff all three pairwise relative deviations are within
    ``tolerance_pct``.
    """
    for name, value in (
        ("nu", nu),
        ("body_force", body_force),
        ("half_height", half_height),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if rho <= 0.0:
        raise ValueError("rho must be positive")

    u_tau_pressure = math.sqrt(body_force * half_height)

    viscous = fit_wall_origin_viscous(y, u, nu)
    if y_origin is not None:
        measured_origin = float(y_origin)
    elif wall_shear is not None:
        measured_origin = float(
            fit_wall_origin_wall_stress(y, u, nu, wall_shear, rho=rho).y_origin
        )
    else:
        measured_origin = float(viscous.y_origin)
    if wall_shear is None:
        u_tau_wall = float(viscous.u_tau)
        wall_source = "viscous_slope_fit"
    else:
        if wall_shear <= 0.0:
            raise ValueError("wall_shear must be positive")
        u_tau_wall = math.sqrt(wall_shear / rho)
        wall_source = "supplied_wall_stress"

    loglaw_slope = u_tau_from_log_law(
        y,
        u,
        nu,
        y_origin=measured_origin,
        kappa=kappa,
        b=b,
        y_plus_window=y_plus_window,
        mode="slope",
    )
    loglaw = u_tau_from_log_law(
        y,
        u,
        nu,
        y_origin=measured_origin,
        kappa=kappa,
        b=b,
        y_plus_window=y_plus_window,
    )

    routes = {
        "pressure_gradient": float(u_tau_pressure),
        "wall_shear": float(u_tau_wall),
        "log_law": float(loglaw["u_tau"]),
    }
    names = list(routes)
    pairs: dict[str, float] = {}
    for i, name_i in enumerate(names):
        for name_j in names[i + 1 :]:
            a, bb = routes[name_i], routes[name_j]
            scale = max(0.5 * (abs(a) + abs(bb)), 1.0e-30)
            pairs[f"{name_i}_vs_{name_j}"] = abs(a - bb) / scale * 100.0
    max_deviation = max(pairs.values())
    mean_route = sum(routes.values()) / len(routes)
    return {
        "u_tau": routes,
        "pairwise_deviation_pct": pairs,
        "max_pairwise_deviation_pct": float(max_deviation),
        "mean_u_tau": float(mean_route),
        "spread_pct_of_mean": float(
            (max(routes.values()) - min(routes.values())) / max(mean_route, 1.0e-30) * 100.0
        ),
        "tolerance_pct": float(tolerance_pct),
        "consistent": bool(max_deviation <= tolerance_pct),
        "wall_origin": {
            "measured": float(measured_origin),
            "source": (
                "supplied"
                if y_origin is not None
                else ("wall_stress_fit" if wall_shear is not None else "viscous_fit")
            ),
            "viscous_fit": viscous.as_dict(),
            "log_law_fit": loglaw,
        },
        "log_law_slope_variant": loglaw_slope,
        "log_law_model_form_band_pct": float(
            abs(loglaw_slope["u_tau"] - loglaw["u_tau"])
            / max(0.5 * (loglaw_slope["u_tau"] + loglaw["u_tau"]), 1.0e-30)
            * 100.0
        ),
        "wall_shear_source": wall_source,
        "log_law_points": int(loglaw["n_points"]),
        "log_law_window": list(loglaw["y_plus_window_used"]),
    }