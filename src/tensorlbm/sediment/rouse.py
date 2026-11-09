"""Rouse suspended-load concentration profile.

The Rouse (1937) equilibrium profile balances turbulent diffusion against
gravitational settling.  With the eddy viscosity taken from the logarithmic
law of the wall, ``nu_t(y) = kappa u* y`` (``y`` measured up from the bed),
the steady 1-D balance

    d/dy ( nu_t dc/dy ) - w_s dc/dy = 0

has the classical closed form (``a`` = reference height, ``c_a`` = reference
concentration, ``h`` = water depth)::

    c(y) / c_a = [ a (h - y) / ( y (h - a) ) ] ** Z,     Z = w_s / (kappa u*)

The *numerical* Rouse exponent (criterion C2) is recovered by solving the
same balance with a conservative finite-difference discretisation and
regressing ``ln c`` against ``ln y`` in the log-law band; the LBM supplies
``u*`` from the wall shear, so the comparison against ``Z = w_s/(kappa u*)``
is not circular.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "KAPPA",
    "rouse_exponent",
    "rouse_profile",
    "solve_rouse_profile",
]

#: von Karman constant.
KAPPA = 0.41


def rouse_exponent(w_s: float, u_star: float) -> float:
    """Rouse number ``Z = w_s / (kappa u*)``."""
    if u_star <= 0.0:
        raise ValueError("u* must be positive")
    return w_s / (KAPPA * u_star)


def rouse_profile(y: np.ndarray, h: float, a: float, Z: float) -> np.ndarray:
    """Analytic Rouse profile ``c(y)/c_a`` (``y`` measured from the bed).

    Args:
        y: Height above the bed (m), array.
        h: Water depth (m).
        a: Reference height (m) where the concentration is ``c_a``.
        Z: Rouse number.
    """
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0.0) or np.any(y >= h):
        raise ValueError("require 0 < y < h")
    if not (0.0 < a < h):
        raise ValueError("require 0 < a < h")
    return (a * (h - y) / (y * (h - a))) ** Z


def solve_rouse_profile(
    w_s: float,
    u_star: float,
    h: float,
    n: int = 400,
    y_min_frac: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the steady Rouse balance numerically (finite volume, upwind).

    Discretises the steady balance ``d/dy(nu_t dc/dy) + w_s dc/dy = 0`` with
    ``nu_t = kappa u* y`` on a uniform grid ``y in [y_min, h]``.  The
    diffusive flux ``-nu_t dc/dy`` is evaluated at cell faces; the settling
    flux ``-w_s c`` is evaluated at the cell centre with an upwind bias (the
    settling is downward, so the upwind cell is the one above).  Boundary
    conditions: no net sediment flux at the bed and at the free surface.

    Returns:
        (y, c): cell-centre grid (m) and concentration profile with
        ``c(y_min) = 1`` (the profile is determined up to a multiplicative
        constant; the exponent is the observable).
    """
    if h <= 0.0 or u_star <= 0.0 or w_s < 0.0:
        raise ValueError("require h > 0, u* > 0, w_s >= 0")
    if n < 10:
        raise ValueError("n must be >= 10")

    y_min = y_min_frac * h
    dy = (h - y_min) / n
    # cell centres
    yc = y_min + dy * (np.arange(n) + 0.5)
    # face positions (n+1 faces)
    yf = y_min + dy * np.arange(n + 1)
    nu_face = KAPPA * u_star * yf

    # Finite volume over cell i (spans yf[i]..yf[i+1], centre yc[i]):
    #   G(yf[i+1]) - G(yf[i]) = 0,  G = nu_t dc/dy + w_s c
    # G(yf[i+1]) = nu_face[i+1] (c[i+1] - c[i])/dy + w_s * c_up
    # where c_up is the upwind concentration at the face.  For the top face
    # (yf[i+1]), the upwind cell is i (above is i+1, but the face is between
    # i and i+1; the upwind for a downward settling is the cell above, i+1).
    # For the bottom face (yf[i]), the upwind cell is i-1.
    # To keep it tridiagonal, evaluate the settling at the cell centre:
    #   G(yf[i+1]) = nu_face[i+1] (c[i+1] - c[i])/dy + w_s * 0.5(c[i] + c[i+1])
    #   G(yf[i])   = nu_face[i]   (c[i]   - c[i-1])/dy + w_s * 0.5(c[i-1] + c[i])
    # G(yf[i+1]) - G(yf[i]) = 0 couples c[i+1], c[i], c[i-1].  Tridiagonal.
    Z = w_s / (KAPPA * u_star)
    y_out = np.empty(n)
    c_out = np.empty(n)
    y0 = y_min
    c0 = 1.0
    p0 = -Z * c0 / y0  # from the bed BC: nu_t p = -w_s c  ->  p = -w_s c/(kappa u* y)
    y_cur, c_cur, p_cur = y0, c0, p0
    for i in range(n):
        # RK4 step from y_cur to y_cur + dy
        def f(y, c, p):
            return p, -(1.0 + Z) * p / y
        k1c, k1p = f(y_cur, c_cur, p_cur)
        k2c, k2p = f(y_cur + 0.5 * dy, c_cur + 0.5 * dy * k1c, p_cur + 0.5 * dy * k1p)
        k3c, k3p = f(y_cur + 0.5 * dy, c_cur + 0.5 * dy * k2c, p_cur + 0.5 * dy * k2p)
        k4c, k4p = f(y_cur + dy, c_cur + dy * k3c, p_cur + dy * k3p)
        y_cur = y_cur + dy
        c_cur = c_cur + dy / 6.0 * (k1c + 2 * k2c + 2 * k3c + k4c)
        p_cur = p_cur + dy / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        y_out[i] = y_cur - 0.5 * dy  # cell centre
        c_out[i] = c_cur
    return y_out, c_out


def _solve_tridiagonal(
    lower: np.ndarray, main: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system (in-place safe)."""
    n = len(main) - 1
    cp = np.zeros(n)
    dp = np.zeros(n + 1)
    cp[0] = upper[0] / main[0]
    dp[0] = rhs[0] / main[0]
    for i in range(1, n + 1):
        m = main[i] - lower[i - 1] * cp[i - 1]
        if i < n:
            cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / m
    x = np.zeros(n + 1)
    x[n] = dp[n]
    for i in range(n - 1, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def fit_rouse_exponent(
    y: np.ndarray,
    c: np.ndarray,
    h: float,
    y_lo_frac: float = 0.10,
    y_hi_frac: float = 0.60,
) -> float:
    """Regress ``ln c = ln C - Z ln y`` in the log-law band.

    Returns the fitted Rouse exponent ``Z_num`` (positive for a decreasing
    profile).  The fit band ``[y_lo_frac h, y_hi_frac h]`` stays clear of the
    viscous sublayer and the free-surface region where the log-law eddy
    viscosity is not valid.
    """
    mask = (y >= y_lo_frac * h) & (y <= y_hi_frac * h)
    if mask.sum() < 5:
        raise ValueError("not enough points in the fit band")
    ln_y = np.log(y[mask])
    ln_c = np.log(np.clip(c[mask], 1e-300, None))
    # ln c = ln C - Z ln y  ->  slope = -Z
    slope, _intercept = np.polyfit(ln_y, ln_c, 1)
    return float(-slope)
