"""Independent numerical solution of the Reynolds (lubrication) pressure equation.

Kept separate from :mod:`tensorlbm.squeeze_film` so that the closed-form anchors
there are certified by a *different* code path (grid + tridiagonal solve +
quadrature) rather than by differentiating the same formula.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["solve_tridiag", "reynolds_force_numeric", "reynolds_pressure_numeric"]


def solve_tridiag(diag: np.ndarray, lower: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system.

    Args:
        diag: main diagonal, length ``n``.
        lower: sub-diagonal, length ``n-1``.
        upper: super-diagonal, length ``n-1``.
        rhs: right-hand side, length ``n``.

    Returns:
        Solution vector, length ``n``.
    """
    n = diag.size
    if n == 1:
        return rhs / diag
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        den = diag[i] - lower[i - 1] * cp[i - 1]
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / den
        if i < n - 1:
            cp[i] = upper[i] / den
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def reynolds_pressure_numeric(
    kind: str,
    mu: float,
    a: float,
    h: float,
    V: float,
    n: int = 4001,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid solution of the Reynolds pressure equation on ``[0, a]``.

    Plane strain: ``d2p/dx2 = -12 mu V / h^3`` with ``dp/dx(0) = 0``, ``p(a) = 0``.
    Axisymmetric: ``(1/r) d/dr (r dp/dr) = -12 mu V / h^3`` with regularity at
    ``r = 0`` and ``p(a) = 0``.

    Args:
        kind: ``"plane"`` or ``"disk"``.
        mu, a, h, V: viscosity, half-width (radius), gap, approach speed (SI).
        n: number of nodes on ``[0, a]``.

    Returns:
        ``(s, p)`` with ``s`` the abscissa and ``p`` the gauge pressure [Pa].
    """
    if n < 9:
        raise ValueError("n must be >= 9")
    rhs = 12.0 * mu * V / h**3
    s = np.linspace(0.0, a, n)
    ds = s[1] - s[0]

    if kind == "plane":
        diag = np.full(n, -2.0)
        lower = np.ones(n - 1)
        upper = np.ones(n - 1)
        rhs_v = np.full(n, -rhs * ds * ds)
        # symmetry axis: ghost node p[-1] = p[1]  ->  2 p1 - 2 p0 = -rhs ds^2
        diag[0] = -2.0
        upper[0] = 2.0
        # p(a) = 0
        diag[-1] = 1.0
        lower[-1] = 0.0
        rhs_v[-1] = 0.0
        p = solve_tridiag(diag, lower, upper, rhs_v)
        return s, p

    if kind == "disk":
        r = s
        diag = np.zeros(n)
        lower = np.zeros(n - 1)
        upper = np.zeros(n - 1)
        rhs_v = np.zeros(n)
        for i in range(n - 1):
            rf_p = 0.5 * (r[i] + r[i + 1])
            rf_m = 0.5 * (r[i] + r[i - 1]) if i > 0 else 0.0
            vol = 0.5 * (rf_p**2 - rf_m**2)  # = int_{rf_m}^{rf_p} r dr
            coef_p = rf_p / ds
            coef_m = rf_m / ds
            diag[i] = coef_p + coef_m
            if i < n - 2:
                upper[i] = -coef_p
            if i > 0:
                lower[i - 1] = -coef_m
            rhs_v[i] = rhs * vol
        diag[-1] = 1.0
        lower[-2] = 0.0
        rhs_v[-1] = 0.0
        p = solve_tridiag(diag, lower, upper, rhs_v)
        return s, p

    raise ValueError(f"unknown kind {kind!r} (expected 'plane' or 'disk')")


def reynolds_force_numeric(
    kind: str,
    mu: float,
    a: float,
    h: float,
    V: float,
    length: float = 1.0,
    n: int = 4001,
) -> float:
    """Integrated film load from the numerical Reynolds solution.

    Args:
        kind: ``"plane"`` or ``"disk"``.
        mu, a, h, V: viscosity, half-width (radius), gap, approach speed (SI).
        length: out-of-plane extent (plane strain only) [m].
        n: number of grid nodes on ``[0, a]``.

    Returns:
        Film load [N].
    """
    s, p = reynolds_pressure_numeric(kind, mu, a, h, V, n=n)
    if kind == "plane":
        return float(2.0 * np.trapezoid(p, s) * length)
    return float(2.0 * math.pi * np.trapezoid(p * s, s))