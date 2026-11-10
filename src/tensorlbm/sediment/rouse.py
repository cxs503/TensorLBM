"""Rouse suspended-sediment concentration profile.

Balance
-------
Steady, horizontally uniform suspended load balances gravitational settling
against turbulent diffusion.  With the settling velocity ``w_s`` downwards the
net upward sediment flux is ``F(y) = -D_s dc/dy - w_s c``; a steady state has
``dF/dy = 0`` and zero flux at both boundaries, hence ``F = 0`` everywhere::

    D_s(y) dc/dy + w_s c = 0            (0 < y < h)

With the log-law / parabolic eddy diffusivity of an open channel,
``D_s(y) = kappa u* y (1 - y/h)``, the exact solution is the classical Rouse
(1937) profile::

    c(y)/c_a = [ (h-y)/y * a/(h-a) ]^Z ,        Z = w_s / (kappa u*)

The profile is determined up to a multiplicative constant, so the *observable*
is the Rouse exponent ``Z = w_s/(kappa u*)``.

What is verified (criterion C2)
-------------------------------
1. A **passive-scalar D2Q9 lattice-Boltzmann solver**
   (:mod:`tensorlbm.sediment.passive_scalar`) transports the concentration
   field with the LBM velocity profile of
   :mod:`tensorlbm.sediment.channel` (one-way coupling), a downward settling
   drift ``-w_s`` and a position-dependent relaxation time that realises
   ``D_s(y)``.  The fitted ``Z`` must match ``w_s/(kappa u*)`` within 5 %.
2. The finite-volume solver below is an independent discretisation of the
   same balance (upwind settling, harmonic-mean face diffusivity, Dirichlet
   reference concentration at the bed); it is the cross-check that the LBM
   result is not an artefact of the lattice scheme.

Fit
---
``fit_rouse_exponent`` regresses ``ln c`` on ``[1, ln(h-y), ln y]``.  Because
the true profile contains the factor ``(h-y)^Z``, a one-parameter fit
``ln c vs ln y`` is *biased* whenever the fit band reaches an appreciable
fraction of ``h`` (bias ~9 % for ``y in [0.1h, 0.6h]``).  The two-predictor
fit is unbiased for the exact profile.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "KAPPA",
    "eddy_diffusivity",
    "fit_rouse_exponent",
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


def eddy_diffusivity(
    y: np.ndarray | float, h: float, u_star: float
) -> np.ndarray | float:
    """Open-channel sediment diffusivity ``D_s = kappa u* y (1 - y/h)``."""
    y = np.asarray(y, dtype=float)
    d = KAPPA * u_star * y * (1.0 - y / h)
    return float(d) if d.ndim == 0 else d


def rouse_profile(y: np.ndarray, h: float, a: float, Z: float) -> np.ndarray:
    """Analytic Rouse profile ``c(y)/c_a`` (``y`` measured up from the bed).

    ``c/c_a = [ a (h-y) / ( y (h-a) ) ]^Z``
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
    """Finite-volume solution of the steady Rouse balance (independent check).

    Flux-form discretisation of ``D_s dc/dy + w_s c = 0`` on a uniform grid
    ``y in [y_min, h]``: harmonic-mean face diffusivity, upwind (from above)
    settling flux, Dirichlet ``c = 1`` at ``y_min`` and zero flux at the free
    surface.

    Returns:
        ``(y, c)``: cell-centre heights (m) and concentration profile.
    """
    if h <= 0.0 or u_star <= 0.0 or w_s < 0.0:
        raise ValueError("require h > 0, u* > 0, w_s >= 0")
    if n < 10:
        raise ValueError("n must be >= 10")

    y_min = y_min_frac * h
    yf = np.linspace(y_min, h, n + 1)  # cell faces
    yc = 0.5 * (yf[:-1] + yf[1:])      # cell centres
    dy = yf[1] - yf[0]
    Dc = np.asarray(eddy_diffusivity(yc, h, u_star), dtype=float)
    Df = 2.0 * Dc[:-1] * Dc[1:] / (Dc[:-1] + Dc[1:])  # n-1 interior faces

    A = np.zeros((n, n))
    b = np.zeros(n)

    # interior faces i+1/2, i = 0..n-2 :  F = -Df[i]*(c_{i+1}-c_i)/dy - w_s c_{i+1}
    for i in range(n - 1):
        A[i, i + 1] += -Df[i] / dy - w_s
        A[i, i] += Df[i] / dy
        A[i + 1, i + 1] += Df[i] / dy + w_s
        A[i + 1, i] += -Df[i] / dy

    # bed face at y_min: F_bed = -Dc[0](c_0 - c_a)/(dy/2) - w_s c_0 with c_a = 1.
    # The cell-0 balance is  -F_bed + F_{1/2} = 0, hence the +2 Dc/dy terms.
    A[0, 0] += 2.0 * Dc[0] / dy + w_s
    b[0] += 2.0 * Dc[0] / dy
    # free surface: F_top = 0 is imposed by dropping the top face term.

    c = np.linalg.solve(A, b)
    return yc, c


def fit_rouse_exponent(
    y: np.ndarray,
    c: np.ndarray,
    h: float,
    y_lo_frac: float = 0.05,
    y_hi_frac: float = 0.60,
    return_asymmetry: bool = False,
):
    """Unbiased fit of the Rouse exponent from a concentration profile.

    Fits ``ln c = A + Z ln(h-y) - Z ln y`` (i.e. ``c = C[(h-y)/y]^Z``) over
    ``[y_lo_frac h, y_hi_frac h]`` by least squares on the two predictors
    ``ln(h-y)`` and ``ln y``; returns ``Z = -coef[ln y]``.

    With ``return_asymmetry=True`` the quantity
    ``|coef[ln(h-y)] + coef[ln y]|/Z`` is also returned; it is 0 for a pure
    Rouse profile and is used by the tests as an internal consistency check.
    """
    y = np.asarray(y, dtype=float)
    c = np.asarray(c, dtype=float)
    mask = (y >= y_lo_frac * h) & (y <= y_hi_frac * h) & (c > 0.0)
    if mask.sum() < 5:
        raise ValueError("not enough points in the fit band")
    X = np.column_stack(
        [np.ones(int(mask.sum())), np.log(h - y[mask]), np.log(y[mask])]
    )
    coef, *_ = np.linalg.lstsq(X, np.log(c[mask]), rcond=None)
    Z = float(-coef[2])
    if not return_asymmetry:
        return Z
    asym = abs(coef[1] + coef[2]) / max(abs(Z), 1.0e-12)
    return Z, float(asym)