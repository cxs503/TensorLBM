"""Mixing-length boundary-layer solver with a bubbly layer (IN62-N04).

This is the **numerical** route to ``Cf(alpha_w)`` -- independent of the
closed-form drag law in :mod:`tensorlbm.microbubble.loglaw`.  The mean momentum
equation of a zero-pressure-gradient turbulent boundary layer is integrated
with a Prandtl mixing length and van Driest damping, using the *density-weighted*
eddy viscosity that the void fraction produces:

    tau(y) = rho_l u_tau^2 (1 - y/delta) ,
    rho_l (nu + (1 - alpha(y)) l^2 |dU/dy|) dU/dy = tau(y) ,
    l(y) = kappa y (1 - exp(-y u_tau / (A^+ nu))) .

Solving the quadratic for ``dU/dy`` and integrating to ``y = delta`` gives
``U(delta)``; the friction velocity is the root of ``U(delta) = U_e``.  No
log-law is assumed anywhere -- the log law is an *output*, which is what makes
this an independent check of the analytic anchor.

Wall area coverage (mechanism M1 of the closure) is applied afterwards as the
area-weighted factor ``Cf_eff = (1 - alpha_w) Cf`` since the gas-covered part of
the plate transmits negligible shear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .loglaw import A_PLUS, B_INTERCEPT, KAPPA, cf_clean_correlation


@dataclass(frozen=True)
class BLParams:
    """Boundary-layer solver settings."""

    n: int = 2000
    stretch: float = 8.0          # grid clustering at the wall
    kappa: float = KAPPA
    a_plus: float = A_PLUS
    tol: float = 1e-10


def _grid(delta: float, n: int, stretch: float) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    y = delta * (np.exp(stretch * s) - 1.0) / (np.exp(stretch) - 1.0)
    y[0] = 0.0
    return y


def velocity_profile(u_tau: float, y: np.ndarray, delta: float, nu: float,
                     alpha: np.ndarray, p: BLParams = BLParams()) -> np.ndarray:
    """Mean velocity profile for a given friction velocity."""
    tau = u_tau ** 2 * np.maximum(1.0 - y / delta, 0.0)
    y_plus = y * u_tau / nu
    damping = 1.0 - np.exp(-y_plus / p.a_plus)
    ell = p.kappa * y * damping
    leff2 = (1.0 - np.clip(alpha, 0.0, 0.999)) * ell ** 2
    # quadratic (1-a) l^2 s^2 + nu s - tau = 0
    disc = nu ** 2 + 4.0 * leff2 * tau
    s = np.where(leff2 > 1e-300,
                 (-nu + np.sqrt(disc)) / np.maximum(2.0 * leff2, 1e-300),
                 tau / nu)
    u = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(y))])
    return u


def solve_u_tau(U_e: float, delta: float, nu: float, alpha_of_y,
                p: BLParams = BLParams()) -> dict:
    """Friction velocity that matches ``U(delta) = U_e`` (bisection)."""
    y = _grid(delta, p.n, p.stretch)
    alpha = np.asarray(alpha_of_y(y), dtype=float)
    if alpha.shape != y.shape:                      # scalar-returning callables
        alpha = np.asarray([float(alpha_of_y(float(yy))) for yy in y], dtype=float)
    lo, hi = 1e-8, 2.0 * U_e
    f_lo = velocity_profile(lo, y, delta, nu, alpha, p)[-1] - U_e
    f_hi = velocity_profile(hi, y, delta, nu, alpha, p)[-1] - U_e
    if f_lo > 0:                      # even a vanishing u_tau overshoots
        return {"u_tau": lo, "u": velocity_profile(lo, y, delta, nu, alpha, p),
                "y": y, "alpha": alpha, "converged": False}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = velocity_profile(mid, y, delta, nu, alpha, p)[-1] - U_e
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < p.tol * max(1.0, hi):
            break
    u_tau = 0.5 * (lo + hi)
    u = velocity_profile(u_tau, y, delta, nu, alpha, p)
    return {"u_tau": u_tau, "u": u, "y": y, "alpha": alpha, "converged": True}


def boundary_layer(U_e: float, delta: float, nu: float, alpha_of_y=None,
                   p: BLParams = BLParams()) -> dict:
    """Solve the layer and return the integral quantities."""
    if alpha_of_y is None:
        alpha_of_y = lambda _y: 0.0
    res = solve_u_tau(U_e, delta, nu, alpha_of_y, p)
    y, u, u_tau = res["y"], res["u"], res["u_tau"]
    ue = max(U_e, 1e-30)
    # momentum thickness and Re_theta
    integrand = (u / ue) * (1.0 - u / ue)
    theta = np.trapezoid(integrand, y)
    re_theta = ue * theta / nu
    cf = 2.0 * u_tau ** 2 / ue ** 2
    return {
        "u_tau": u_tau,
        "Cf": cf,
        "Re_theta": re_theta,
        "theta": theta,
        "y": y,
        "u": u,
        "u_plus": u / u_tau,
        "y_plus": y * u_tau / nu,
        "converged": res["converged"],
    }


def drag_reduction_re_theta(re_theta: float, alpha_w: float = 0.2,
                            alpha_of_y=None, p: BLParams = BLParams(),
                            u_e: float = 1.0, nu: float = 1e-5) -> dict:
    """DR at matched ``Re_theta`` (iterating the layer thickness).

    The clean layer is first tuned (its thickness ``delta``) to the requested
    ``Re_theta``; the bubbly layer is then solved with the *same* ``Re_theta``
    and the same void-fraction profile, which is the physically meaningful
    comparison (the plate is the same and the boundary layer has the same
    integral thickness).
    """
    delta = 0.02
    clean = boundary_layer(u_e, delta, nu, None, p)
    for _ in range(60):
        target = re_theta / max(clean["Re_theta"], 1e-30)
        if abs(target - 1.0) < 1e-6:
            break
        delta *= target ** 0.8
        clean = boundary_layer(u_e, delta, nu, None, p)
    bubbly = boundary_layer(u_e, delta, nu, alpha_of_y, p) if alpha_of_y else clean
    cf_ratio = (bubbly["u_tau"] / clean["u_tau"]) ** 2 if alpha_of_y else 1.0
    dr_mod = 1.0 - cf_ratio
    dr_total = 1.0 - (1.0 - alpha_w) * cf_ratio
    return {
        "re_theta_clean": clean["Re_theta"],
        "re_theta_bubbly": bubbly["Re_theta"],
        "cf_clean": clean["Cf"],
        "cf_bubbly": bubbly["Cf"],
        "cf_white": cf_clean_correlation(clean["Re_theta"]),
        "u_tau_ratio": bubbly["u_tau"] / clean["u_tau"],
        "DR_mixing_length": dr_mod,
        "DR_total": dr_total,
        "clean": clean,
        "bubbly": bubbly,
    }


def log_law_check(re_theta: float, p: BLParams = BLParams()) -> dict:
    """Verify the solver's own log region against the universal log law.

    The solver never uses the log law -- it integrates a mixing-length closure
    -- so reproducing ``u+ = (1/kappa) ln y+ + B`` in the overlap region
    ``30 < y+ < 0.15 delta+`` is a genuine check of the closure (and of the
    intercept ``B``, which is *not* an input of the solver).
    """
    res = drag_reduction_re_theta(re_theta, alpha_w=0.0, p=p)
    bl = res["clean"]
    yp, up = bl["y_plus"], bl["u_plus"]
    m = (yp > 30.0) & (yp < 0.15 * yp[-1])
    pred = np.log(yp[m]) / KAPPA + B_INTERCEPT
    err = up[m] - pred
    return {
        "re_theta": res["re_theta_clean"],
        "y_plus_range": [float(yp[m][0]), float(yp[m][-1])],
        "mean_error": float(err.mean()),
        "max_abs_error": float(np.abs(err).max()),
        "cf_solver": res["cf_clean"],
    }