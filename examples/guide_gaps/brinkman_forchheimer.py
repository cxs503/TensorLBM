"""Darcy-Forchheimer / Brinkman solvers for the ``forchheimer`` guide gap.

Model (1D steady momentum balance of a Newtonian fluid in a rigid porous bed,
uniform porosity, Brinkman correction retained)::

    mu_eff d2u/dy2 - (mu/K) u - beta rho u|u| + G = 0 ,   G = -dp/dx

* ``K``      permeability [m^2]
* ``beta``   inertial (Forchheimer) coefficient [1/m], i.e. the quadratic
             pressure loss is ``beta * rho * u^2`` per unit length
* ``mu_eff`` effective (Brinkman) viscosity; ``mu_eff = mu`` everywhere in this
             pack, and stated as an assumption.

Three independent pieces of evidence are produced with these routines:

1. **bulk algebraic solution** (``bulk_velocity``) -- exact root of
   ``beta rho u^2 + (mu/K) u - G = 0``; reference for an unbounded (free-slip)
   bed.
2. **high-resolution channel reference** (``channel_reference``) -- Newton
   solve of the same ODE with no-slip walls on a fine uniform grid; its own
   grid error is quantified by Richardson extrapolation so it can serve as a
   reference for a coarse mesh.
3. **coarse-grid numerical channel** (``channel_picard``) -- Picard iteration on
   a deliberately coarse 2D (ny x nx) finite-volume mesh: the object under test.

Ergun (1952) correlations used as the independent cross-check of
``(K, beta)``::

    K    = d_p^2 phi^3 / (150 (1-phi)^2)
    beta = 1.75 (1-phi) / (phi^3 d_p)

References
----------
* Forchheimer, P. (1901) Wasserbewegung durch Boden. *Z. Ver. Dtsch. Ing.* 45, 1782.
* Ergun, S. (1952) Fluid flow through packed columns. *Chem. Eng. Prog.* 48, 89.
* Brinkman, H.C. (1947) A calculation of the viscous force exerted by a flowing
  fluid on a dense swarm of particles. *Appl. Sci. Res.* A1, 27.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bed:
    """A porous-bed configuration (SI units)."""

    name: str
    phi: float          # porosity
    d_p: float          # grain diameter [m]
    K: float            # permeability [m^2]
    beta: float         # inertial coefficient [1/m]
    mu: float           # dynamic viscosity [Pa s]
    rho: float          # density [kg/m^3]
    h: float            # channel half-width [m]

    @property
    def brinkman_ratio(self) -> float:
        """Wall-layer thickness over half-width (small => wall effects ~ 0)."""
        return math.sqrt(self.K) / self.h

    def replace(self, **kw) -> "Bed":
        d = dict(
            name=self.name, phi=self.phi, d_p=self.d_p, K=self.K, beta=self.beta,
            mu=self.mu, rho=self.rho, h=self.h,
        )
        d.update(kw)
        return Bed(**d)


def ergun(phi: float, d_p: float) -> tuple[float, float]:
    """Ergun (1952) permeability and inertial coefficient."""
    K = d_p**2 * phi**3 / (150.0 * (1.0 - phi) ** 2)
    beta = 1.75 * (1.0 - phi) / (phi**3 * d_p)
    return K, beta


def make_bed(name: str, phi: float, d_p: float, mu: float, rho: float, h: float) -> Bed:
    K, beta = ergun(phi, d_p)
    return Bed(name=name, phi=phi, d_p=d_p, K=K, beta=beta, mu=mu, rho=rho, h=h)


def bulk_velocity(G: float, K: float, beta: float, mu: float, rho: float) -> float:
    """Exact steady velocity of the unbounded Darcy-Forchheimer balance."""
    a = mu / K
    b = beta * rho
    if b == 0.0:
        return G / a
    return (-a + math.sqrt(a * a + 4.0 * b * G)) / (2.0 * b)


def bulk_forcing(u: float, K: float, beta: float, mu: float, rho: float) -> float:
    """Pressure gradient needed for velocity ``u`` in the unbounded bed."""
    return mu / K * u + beta * rho * u * u


# --------------------------------------------------------------------------- #
# tridiagonal solver (n systems stacked in the columns)
# --------------------------------------------------------------------------- #
def _thomas(
    sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve tridiagonal systems; ``diag``/``rhs`` may be (n,) or (n, m)."""
    from scipy.linalg import solve_banded

    rhs2 = rhs if rhs.ndim == 2 else rhs[:, None]
    n = rhs2.shape[0]
    m = rhs2.shape[1]
    diag2 = np.broadcast_to(np.asarray(diag).reshape(n, -1), (n, m))
    sub2 = np.broadcast_to(np.asarray(sub).reshape(n - 1, -1), (n - 1, m))
    sup2 = np.broadcast_to(np.asarray(sup).reshape(n - 1, -1), (n - 1, m))
    ab = np.zeros((3, n, m))
    ab[0, 1:, :] = sup2
    ab[1, :, :] = diag2
    ab[2, :-1, :] = sub2
    x = np.empty((n, m))
    for j in range(m):
        x[:, j] = solve_banded((1, 1), ab[:, :, j], rhs2[:, j])
    return x if rhs.ndim == 2 else x[:, 0]


# --------------------------------------------------------------------------- #
# reference: fine-grid Newton solve of the channel ODE
# --------------------------------------------------------------------------- #
def channel_reference(
    bed: Bed,
    Gs,
    n: int = 4001,
    mu_eff: float | None = None,
    tol: float = 1e-14,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """No-slip channel solution(s) on a uniform ``n``-point grid (Newton).

    ``Gs`` may be a scalar or a sequence; returns ``(y, U)`` with ``U`` of shape
    ``(n,)`` or ``(n, len(Gs))``.
    """
    mu_eff = bed.mu if mu_eff is None else mu_eff
    scalar = np.isscalar(Gs)
    Gl = np.atleast_1d(np.asarray(Gs, dtype=float))
    y = np.linspace(-bed.h, bed.h, n)
    dy = y[1] - y[0]
    k_lin = bed.mu / bed.K
    k_quad = bed.beta * bed.rho
    coef = mu_eff / dy**2
    sub = np.full(n - 1, coef)
    sup = np.full(n - 1, coef)
    sub_s = sub.copy()
    sup_s = sup.copy()
    sub_s[-1] = 0.0
    sup_s[0] = 0.0
    u = np.array(
        [[bulk_velocity(g, bed.K, bed.beta, bed.mu, bed.rho)] for g in Gl]
    ).T * np.ones((n, len(Gl)))
    u[0, :] = 0.0
    u[-1, :] = 0.0
    for _ in range(max_iter):
        diag = -2.0 * coef - k_lin - 2.0 * k_quad * np.abs(u)
        diag[0, :] = 1.0
        diag[-1, :] = 1.0
        res = np.zeros_like(u)
        res[1:-1, :] = (
            coef * (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :])
            - k_lin * u[1:-1, :]
            - k_quad * u[1:-1, :] * np.abs(u[1:-1, :])
            + Gl[None, :]
        )
        res[0, :] = u[0, :]
        res[-1, :] = u[-1, :]
        du = _thomas(sub_s, diag, sup_s, -res)
        u = u + du
        u[0, :] = 0.0
        u[-1, :] = 0.0
        if np.max(np.abs(du)) <= tol * max(float(np.max(np.abs(u))), 1e-300):
            break
    return y, (u[:, 0] if scalar else u)


def darcy_brinkman_exact(
    y: np.ndarray, G: float, K: float, mu: float, h: float, mu_eff: float | None = None
) -> np.ndarray:
    """Closed-form Brinkman (``beta = 0``) channel solution, ``mu_eff = mu``."""
    mu_eff = mu if mu_eff is None else mu_eff
    lam = math.sqrt(mu / (K * mu_eff)) if mu_eff != mu else 1.0 / math.sqrt(K)
    return (K * G / mu) * (1.0 - np.cosh(y * lam) / np.cosh(h * lam))


# --------------------------------------------------------------------------- #
# numerical: coarse 2D finite-volume Picard iteration
# --------------------------------------------------------------------------- #
def channel_coarse(
    bed: Bed,
    G: float,
    ny: int = 64,
    nx: int = 4,
    wall: str = "no-slip",
    mu_eff: float | None = None,
    tol: float = 1e-13,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Coarse 2D (ny x nx) finite-volume solve of the B-F channel, Newton.

    The mesh is independent of x (the problem is 1D in y); the x axis is kept so
    that the routine exercises a genuine 2D array layout -- every column must
    stay bit-identical, which is asserted through ``columns_identical``.  The
    Jacobian is tridiagonal (the drag term is local), so each Newton step is one
    banded solve per column.
    """
    mu_eff = bed.mu if mu_eff is None else mu_eff
    y = np.linspace(-bed.h, bed.h, ny)
    dy = y[1] - y[0]
    k_lin = bed.mu / bed.K
    k_quad = bed.beta * bed.rho
    coef = mu_eff / dy**2
    u0 = bulk_velocity(G, bed.K, bed.beta, bed.mu, bed.rho)
    u = np.full((ny, nx), u0)
    if wall == "no-slip":
        u[0, :] = 0.0
        u[-1, :] = 0.0

    def residual(field: np.ndarray) -> np.ndarray:
        res = np.zeros_like(field)
        res[1:-1, :] = (
            coef * (field[2:, :] - 2.0 * field[1:-1, :] + field[:-2, :])
            - k_lin * field[1:-1, :]
            - k_quad * field[1:-1, :] * np.abs(field[1:-1, :])
            + G
        )
        if wall == "no-slip":
            res[0, :] = field[0, :]
            res[-1, :] = field[-1, :]
        else:
            # free-slip: mirrored ghost node -> half-cell coupling at the walls
            res[0, :] = (
                2.0 * coef * (field[1, :] - field[0, :])
                - k_lin * field[0, :]
                - k_quad * field[0, :] * np.abs(field[0, :])
                + G
            )
            res[-1, :] = (
                2.0 * coef * (field[-2, :] - field[-1, :])
                - k_lin * field[-1, :]
                - k_quad * field[-1, :] * np.abs(field[-1, :])
                + G
            )
        return res

    sub = np.full(ny - 1, coef)
    sup = np.full(ny - 1, coef)
    sub_w = sub.copy()
    sup_w = sup.copy()
    if wall == "no-slip":
        sub_w[-1] = 0.0
        sup_w[0] = 0.0
    elif wall == "free-slip":
        sup_w[0] = 2.0 * coef
        sub_w[-1] = 2.0 * coef
    else:
        raise ValueError(f"unknown wall condition {wall!r}")

    n_iter = 0
    delta = math.inf
    for _ in range(max_iter):
        n_iter += 1
        res = residual(u)
        diag = -2.0 * coef - k_lin - 2.0 * k_quad * np.abs(u)
        if wall == "no-slip":
            diag[0, :] = 1.0
            diag[-1, :] = 1.0
        else:
            diag[0, :] = -2.0 * coef - k_lin - 2.0 * k_quad * np.abs(u[0, :])
            diag[-1, :] = -2.0 * coef - k_lin - 2.0 * k_quad * np.abs(u[-1, :])
        du = _thomas(sub_w, diag, sup_w, -res)
        u = u + du
        if wall == "no-slip":
            u[0, :] = 0.0
            u[-1, :] = 0.0
        delta = float(np.max(np.abs(du)))
        if delta <= tol * max(u0, 1e-300):
            break
    return y, u, {
        "iterations": n_iter,
        "last_delta": delta,
        "columns_identical": bool(np.allclose(u, u[:, [0]], rtol=0.0, atol=0.0)),
        "max_residual": float(np.max(np.abs(residual(u)))),
    }


def mean_velocity(u: np.ndarray) -> float:
    """Cross-section averaged (Darcy) velocity of a channel profile."""
    if u.ndim == 2:
        u = u[:, 0]
    return float(np.trapezoid(u, dx=1.0) / (u.size - 1))


def observed_order(e_coarse: float, e_fine: float, r: float = 2.0) -> float:
    """Observed convergence order from two grids with refinement ratio ``r``."""
    if e_fine <= 0.0 or e_coarse <= 0.0:
        return float("nan")
    return math.log(e_coarse / e_fine) / math.log(r)