"""Independent Graetz/Shah-London series reference (verification side).

This module is deliberately a *second implementation* of the classical series
solution, written for the audit of :mod:`tensorlbm.graetz`.  It shares nothing
with the library code except the published spectral constants:

* eigenvalues ``lambda_n`` of ``(rho R')' + 2 lambda^2 rho (1-rho^2) R = 0``,
  ``R'(0)=0``, ``R(1)=0`` -- Chandrasekhar (1961) / Shah & London (1978):
  ``2.70436, 6.67903, 10.67338, 14.67108, 18.66987, 22.66914, ...``
* developed Nusselt number ``Nu_inf = lambda_1^2/2 = 3.6568``.

Everything else (eigenfunctions by RK4, coefficients by the Sturm-Liouville
orthogonality weight ``rho(1-rho^2)``, cup-mixing temperature and wall flux) is
re-derived here so that the reference does not depend on the library's
eigen-solver or its post-processing.
"""

from __future__ import annotations

import numpy as np

#: published Graetz eigenvalues (Chandrasekhar 1961; Shah & London 1978)
LAMBDA = np.array(
    [2.70436, 6.67903, 10.67338, 14.67108, 18.66987, 22.66914,
     26.66850, 30.66790, 34.66740, 38.66700, 42.66660, 46.66630]
)
NU_INF = 3.6568  # = lambda_1^2 / 2


def _eigenfunction(lam: float, rho: np.ndarray, n_steps: int = 4000) -> np.ndarray:
    """RK4 integration of ``(rho R')' + 2 lam^2 rho(1-rho^2) R = 0`` from the axis."""
    beta = 2.0 * lam * lam
    eps = 1e-9
    rs = np.linspace(eps, 1.0, n_steps + 1)
    h = rs[1] - rs[0]
    R = 1.0 - beta * eps * eps / 8.0
    S = -beta * eps / 4.0
    out = np.empty(n_steps + 1)
    out[0] = R

    def f(r, Rv, Sv):
        return Sv, -0.5 * beta * (1.0 - r * r) * Rv - Sv / r

    for i in range(n_steps):
        r = rs[i]
        k1 = f(r, R, S)
        k2 = f(r + 0.5 * h, R + 0.5 * h * k1[0], S + 0.5 * h * k1[1])
        k3 = f(r + 0.5 * h, R + 0.5 * h * k2[0], S + 0.5 * h * k2[1])
        k4 = f(r + h, R + h * k3[0], S + h * k3[1])
        R += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        S += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        out[i + 1] = R
    return np.interp(rho, rs, out)


def series(x_star, n_modes: int = 10, n_r: int = 800) -> dict:
    """Nu(x*) of the constant-wall-temperature circular-tube Graetz problem."""
    rho = np.linspace(0.0, 1.0, n_r + 1)
    wgt = rho * (1.0 - rho * rho)          # Sturm-Liouville weight
    R = np.array([_eigenfunction(l, rho) for l in LAMBDA[:n_modes]])
    C = np.array([
        np.trapezoid(wgt * R[k], rho) / np.trapezoid(wgt * R[k] ** 2, rho)
        for k in range(n_modes)
    ])
    beta = 2.0 * LAMBDA[:n_modes] ** 2
    xs = np.atleast_1d(np.asarray(x_star, dtype=float))
    nu, theta_m, profiles = [], [], []
    drho = rho[1] - rho[0]
    for x in xs:
        th = (C[:, None] * np.exp(-beta[:, None] * x) * R).sum(axis=0)
        tm = 4.0 * np.trapezoid(rho * (1.0 - rho * rho) * th, rho)
        dTdr = (th[-1] - th[-2]) / drho      # dTheta/drho at rho=1 (negative)
        nu.append(float(-2.0 * dTdr / tm))
        theta_m.append(float(tm))
        profiles.append(th)
    return {
        "x_star": xs,
        "Nu": np.array(nu),
        "theta_m": np.array(theta_m),
        "profiles": profiles,
        "rho": rho,
        "C": C,
        "beta": beta,
        "Nu_inf": NU_INF,
        "source": "Chandrasekhar 1961 / Shah & London 1978 eigenvalues; "
                  "independent RK4 eigenfunctions + SL-orthogonality coefficients",
    }
