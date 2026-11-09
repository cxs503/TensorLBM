"""Zero-pressure-gradient flat plate boundary layer driven by the wall law.

The wall model is exercised here exactly as it is inside a WMLES: the
law-of-the-wall (Spalding composite with ``kappa = 0.41``, ``B = 5.2``) supplies
the skin friction, and the outer layer is represented by Coles' law of the wake

    u+(eta) = u+_law-of-the-wall(eta*delta+) + (2 Pi / kappa) W(eta),
    W(eta)  = sin^2(pi eta / 2),        eta = y/delta,

with ``Pi = 0.45`` for a zero-pressure-gradient boundary layer.  The momentum
integral

    d(theta)/dx = c_f/2

is then marched in ``Re_theta``.  This is an *integral* boundary-layer model
(RANS grade, not LES): it is the honest way to obtain ``c_f(Re_x)`` over
``Re_theta = 1e4 ... 5e4`` when the LES itself cannot sustain resolved
turbulence at those Reynolds numbers on a coarse grid.  Every step of the
march is executed numerically here (numeric quadrature of the composite
profile, bisection on ``Re_theta``, finite-difference differentiation of the
ITTC-1957 line for its local form) -- nothing is tabulated.

Anchors
-------
``ITTC-1957``          ``C_F(Re) = 0.075 / (log10(Re) - 2)^2`` (mean line).
``ITTC local``         ``c_f(Re_x) = 2 [C_F + Re_x dC_F/dRe_x]``, obtained by
                       differentiating the line numerically.
``Prandtl-Schlichting````c_f = 0.455 / ln^2(0.06 Re_x)`` (independent check).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .laws import B_DEFAULT, KAPPA_DEFAULT

__all__ = [
    "itcc_1957_mean",
    "ittc_1957_mean",
    "ittc_1957_local",
    "prandtl_schlichting_local",
    "coles_profile",
    "FlatPlateResult",
    "run_flat_plate",
]

PI_WAKE = 0.45

_trapz = getattr(np, "trapezoid", None) or np.trapz


def ittc_1957_mean(reynolds) -> np.ndarray:
    """ITTC-1957 friction line ``C_F = 0.075 / (log10 Re - 2)^2``."""
    re = np.asarray(reynolds, dtype=float)
    return 0.075 / (np.log10(re) - 2.0) ** 2


itcc_1957_mean = ittc_1957_mean  # alias kept for readability of older scripts


def ittc_1957_local(reynolds) -> np.ndarray:
    """Local skin friction implied by the ITTC-1957 *mean* line.

    ``C_F(Re) = (1/Re) int_0^Re c_f dRe'``  =>  ``c_f = C_F + Re dC_F/dRe``
    for a plate of length ``L`` (``Re`` proportional to ``L``).  The derivative
    is taken by central differences on the analytic line, so this is a closed
    form, not a fitted correlation.
    """
    re = np.asarray(reynolds, dtype=float)
    h = 1.0e-4 * re
    d_cf_dre = (ittc_1957_mean(re + h) - ittc_1957_mean(re - h)) / (2.0 * h)
    return ittc_1957_mean(re) + re * d_cf_dre


def prandtl_schlichting_local(reynolds) -> np.ndarray:
    """``c_f = 0.455 / ln^2(0.06 Re_x)`` (Prandtl-Schlichting, 5e5 < Re < 1e7)."""
    re = np.asarray(reynolds, dtype=float)
    return 0.455 / np.log(0.06 * re) ** 2


def _spalding_u_plus_vec(y_plus: np.ndarray, kappa: float, b: float, iterations: int = 14) -> np.ndarray:
    """Vectorised Spalding law ``y+ -> u+`` (Newton, log-law start)."""
    a = math.exp(-kappa * b)
    y = np.asarray(y_plus, dtype=float)
    u = np.where(y > 11.0, np.log(np.clip(y, 1e-12, None)) / kappa + b, y)
    for _ in range(iterations):
        ku = kappa * u
        f = u + a * (np.exp(ku) - 1.0 - ku - 0.5 * ku * ku - ku**3 / 6.0) - y
        df = 1.0 + a * kappa * (np.exp(ku) - 1.0 - ku - 0.5 * ku * ku)
        step = np.where(df > 1e-12, f / np.maximum(df, 1e-12), 0.0)
        u = u - step
        u = np.where(u < 0.0, 0.5 * y, u)
    return u


def _spalding_y_plus_vec(u_plus: float, kappa: float, b: float, iterations: int = 14) -> float:
    """Scalar inverse of :func:`_spalding_u_plus_vec` by geometric bisection."""
    lo, hi = 1.0e-8, 1.0e9
    for _ in range(120):
        mid = math.sqrt(lo * hi)
        if float(_spalding_u_plus_vec(np.array([mid]), kappa, b, 20)[0]) < u_plus:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0e-12 * hi:
            break
    return math.sqrt(lo * hi)


def coles_profile(
    *,
    delta_plus: float,
    kappa: float = KAPPA_DEFAULT,
    b: float = B_DEFAULT,
    pi_wake: float = PI_WAKE,
    n_points: int = 801,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Coles log-wake profile in wall units for a boundary-layer thickness ``delta+``.

    Returns ``(y_plus, u_plus, ue_plus, c_f, theta_plus)``.  Parametrising the
    profile family by ``delta+`` (instead of by ``c_f``) makes the mapping
    ``delta+ -> Re_theta`` a single forward evaluation with no nested inversion:

        ``u+(delta+) = Spalding(delta+) + 2 Pi/kappa``,  ``c_f = 2/Ue+^2``,
        ``Re_theta = Ue+ * theta+``.
    """
    y_plus = np.linspace(0.0, delta_plus, n_points)
    eta = y_plus / delta_plus
    u_plus = _spalding_u_plus_vec(np.clip(y_plus, 1.0e-12, None), kappa, b)
    # Coles (1956) log-wake law: u+ = u+_law + (2 Pi/kappa) W(eta) with
    # W(eta) = sin^2(pi eta/2)  =>  W(1) = 1 and the edge wake increment is
    # exactly 2 Pi/kappa (= 2.20 for Pi = 0.45, kappa = 0.41).  An earlier
    # version multiplied the same W by 4 Pi/kappa (W(1) = 2), doubling the
    # wake and the boundary-layer edge velocity.
    wake = (2.0 * pi_wake / kappa) * np.sin(0.5 * math.pi * eta) ** 2
    u_plus = u_plus + wake
    ue_plus = float(u_plus[-1])
    c_f = 2.0 / ue_plus**2
    ratio = u_plus / ue_plus
    theta_plus = float(_trapz(ratio * (1.0 - ratio), y_plus))
    return y_plus, u_plus, ue_plus, c_f, theta_plus


@dataclass(frozen=True)
class FlatPlateResult:
    """Marched zero-pressure-gradient flat-plate boundary layer."""

    re_theta: np.ndarray
    re_x: np.ndarray
    c_f: np.ndarray
    ue_plus: np.ndarray
    delta_plus: np.ndarray
    shape_factor: np.ndarray
    pi_wake: float
    kappa: float
    b: float
    velocity: float = 1.0
    nu: float = 1.0e-6
    metadata: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        # Map Re_theta -> Re_x with the classical 1/7-power-law relation
        # Re_theta = 0.0365 Re_x^0.8 (documented closure of the comparison, not
        # part of the model): it only fixes the abscissa at which the ITTC-1957
        # and Prandtl-Schlichting lines are evaluated.
        re_x_ref = (self.re_theta / 0.0365) ** 1.25
        cf_ittc = ittc_1957_local(re_x_ref)
        cf_ps = prandtl_schlichting_local(re_x_ref)
        rel_ittc = np.abs(self.c_f - cf_ittc) / cf_ittc
        rel_ps = np.abs(self.c_f - cf_ps) / cf_ps
        return {
            "re_theta_min": float(self.re_theta.min()),
            "re_theta_max": float(self.re_theta.max()),
            "re_x_min": float(self.re_x.min()),
            "re_x_max": float(self.re_x.max()),
            "re_x_reference_min": float(re_x_ref.min()),
            "re_x_reference_max": float(re_x_ref.max()),
            "c_f_start": float(self.c_f[0]),
            "c_f_end": float(self.c_f[-1]),
            "shape_factor_start": float(self.shape_factor[0]),
            "shape_factor_end": float(self.shape_factor[-1]),
            "delta_plus_start": float(self.delta_plus[0]),
            "delta_plus_end": float(self.delta_plus[-1]),
            "c_f_ittc_local_start": float(cf_ittc[0]),
            "max_rel_error_vs_ittc_local": float(rel_ittc.max()),
            "rms_rel_error_vs_ittc_local": float(np.sqrt(np.mean(rel_ittc**2))),
            "max_rel_error_vs_prandtl_schlichting": float(rel_ps.max()),
            "rms_rel_error_vs_prandtl_schlichting": float(np.sqrt(np.mean(rel_ps**2))),
            "mean_c_f_model": float(np.mean(self.c_f)),
            "mean_c_f_ittc_local": float(np.mean(cf_ittc)),
            "mean_rel_error_vs_ittc_local": float(
                abs(np.mean(self.c_f) - np.mean(cf_ittc)) / np.mean(cf_ittc)
            ),
            "abscissa_mapping": "Re_theta = 0.0365 Re_x^0.8 (1/7-power law)",
        }


def run_flat_plate(
    *,
    re_theta_start: float = 1.0e4,
    re_theta_end: float = 5.0e4,
    n_steps: int = 200,
    kappa: float = KAPPA_DEFAULT,
    b: float = B_DEFAULT,
    pi_wake: float = PI_WAKE,
    velocity: float = 1.0,
    nu: float = 1.0e-6,
    n_profile_points: int = 801,
) -> FlatPlateResult:
    """March the momentum integral with the wall-law closure.

    At each station the boundary-layer thickness follows from the definition of
    ``Re_theta`` by bisection on ``delta+`` (monotone); the step in ``x`` is
    ``dx = dtheta / (c_f / 2)``.
    """
    re_theta = np.geomspace(re_theta_start, re_theta_end, n_steps)
    cache: dict[float, tuple[float, float, float, float]] = {}

    def state(delta_plus: float) -> tuple[float, float, float, float]:
        key = round(math.log(delta_plus), 8)
        if key not in cache:
            _, u_plus, ue_plus, c_f, theta_plus = coles_profile(
                delta_plus=delta_plus,
                kappa=kappa,
                b=b,
                pi_wake=pi_wake,
                n_points=n_profile_points,
            )
            ratio = u_plus / ue_plus
            delta_star_plus = float(_trapz(1.0 - ratio, np.linspace(0.0, delta_plus, n_profile_points)))
            cache[key] = (c_f, ue_plus, theta_plus, delta_star_plus / max(theta_plus, 1e-30))
        return cache[key]

    c_f = np.empty_like(re_theta)
    ue_plus = np.empty_like(re_theta)
    delta_plus = np.empty_like(re_theta)
    shape = np.empty_like(re_theta)
    x = np.zeros_like(re_theta)
    for i, re_t in enumerate(re_theta):
        lo, hi = 10.0, 1.0e6
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            _, ue_p, theta_p, _ = state(mid)
            if ue_p * theta_p < re_t:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1.0e-11 * hi:
                break
        d_plus = math.sqrt(lo * hi)
        cf, ue_p, theta_p, h_shape = state(d_plus)
        c_f[i] = cf
        ue_plus[i] = ue_p
        delta_plus[i] = d_plus
        shape[i] = h_shape
        if i > 0:
            dtheta = (re_theta[i] - re_theta[i - 1]) * nu / velocity
            cf_mean = 0.5 * (c_f[i] + c_f[i - 1])
            x[i] = x[i - 1] + dtheta / (0.5 * cf_mean)
    re_x = velocity * x / nu
    return FlatPlateResult(
        re_theta=re_theta,
        re_x=re_x,
        c_f=c_f,
        ue_plus=ue_plus,
        delta_plus=delta_plus,
        shape_factor=shape,
        pi_wake=pi_wake,
        kappa=kappa,
        b=b,
        velocity=velocity,
        nu=nu,
        metadata={
            "closure": "Spalding law of the wall + Coles wake (Pi=%.3f)" % pi_wake,
            "momentum_integral": "d(theta)/dx = c_f/2",
            "quadrature": f"trapezoid on {n_profile_points} wall-normal samples",
        },
    )
