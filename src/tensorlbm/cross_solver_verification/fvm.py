"""Cell-centred finite-volume reference solver: vorticity-streamfunction form.

Why this formulation
--------------------
``psi``-``omega`` makes the discrete mass constraint *structural*: the velocity
is obtained from a discrete stream function, hence the discrete divergence
vanishes identically. That gives the finite-volume leg of the cross-solver
benchmark an independent, exactly-solenoidal velocity field, which is exactly
what the mass-conservation ledger (C5) needs.

Discretisation
--------------
* uniform ``N x N`` cell-centred grid, ``h = 1/N``, cell centres ``(i+1/2)h``;
* Poisson problem ``lap(psi) = -omega`` solved *exactly* on the discrete 5-point
  stencil (DST-I eigendecomposition for Dirichlet walls, FFT eigenvalues for the
  periodic case).  Solving the discrete system exactly (rather than iteratively)
  keeps the observed convergence order that of the discretisation itself;
* second-order central fluxes for convection and diffusion;
* wall vorticity from the two-term Woods/Jensen formula, second order in ``h``::

      omega_wall = -4 (27 psi_1 - psi_2 - 12 h u_wall) / (9 h^2)

  derived from a Taylor expansion at distance ``h/2`` (cell centre) with the
  wall normal derivative ``dpsi/dn = u_t`` imposed exactly;
* pseudo-time marching with an adaptive step
  ``dt = CFL * min(h^2/(4 nu), h/max|u|)`` until the field stops changing, so
  the measured error is the *steady-state* discretisation error.

Known limitation (documented, not hidden): central convection is only
marginally stable when the cell Reynolds number ``|u| h / nu`` exceeds ~2, so the
Re = 1000 case is *not* part of the finite-volume leg of this benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dstn, idstn

__all__ = ["FVMResult", "solve_psi_omega"]


@dataclass
class FVMResult:
    u: np.ndarray
    v: np.ndarray
    psi: np.ndarray
    omega: np.ndarray
    omega_wall: dict
    n: int
    iterations: int
    converged: bool
    residual: float
    dt: float
    history: list[dict] = field(default_factory=list)

    @property
    def speed(self):
        return np.hypot(self.u, self.v)


# --------------------------------------------------------------------- helpers
def _poisson_dirichlet(omega: np.ndarray, h: float) -> np.ndarray:
    """Solve ``lap(psi) = -omega`` with ``psi = 0`` on all four walls (exact).

    The grid is cell-centred (centres at ``(j+1/2)h``) and the wall sits half a
    cell outside the first centre, so the boundary rows of the 5-point operator
    carry the odd-reflection ghost ``psi_ghost = -psi_1`` -> ``psi_2 - 3 psi_1``.
    That operator is diagonalised by the **DST-II** basis
    ``sin((j-1/2) m pi / n)`` with ``lam_m = (2 cos(m pi/n) - 2)/h^2``.

    (Using DST-I here -- i.e. pretending the wall is a full cell away -- is a
    real bug: it silently rescales the effective cavity and was caught by the
    Ghia comparison, which was 40 % off while the periodic MMS still passed.)
    """
    n = omega.shape[0]
    m = np.arange(1, n + 1)
    lam = (2.0 * np.cos(m * np.pi / n) - 2.0) / (h * h)
    denom = lam[:, None] + lam[None, :]
    rhs = dstn(-omega, type=2, norm="ortho")
    return idstn(rhs / denom, type=2, norm="ortho")


def _poisson_periodic(omega: np.ndarray, h: float) -> np.ndarray:
    """Solve the discrete periodic Poisson problem (zero-mean RHS)."""
    n = omega.shape[0]
    m = np.arange(n)
    lam = (2.0 * np.cos(2.0 * np.pi * m / n) - 2.0) / (h * h)
    denom = lam[:, None] + lam[None, :]
    denom = denom.copy()
    denom[0, 0] = 1.0
    rhs = np.fft.fft2(-omega)
    psi = np.real(np.fft.ifft2(rhs / denom))
    return psi - psi.mean()


def _pad_wrap(field: np.ndarray) -> np.ndarray:
    return np.pad(field, 1, mode="wrap")


def _pad_odd(field: np.ndarray) -> np.ndarray:
    """Ghost row/column for a quantity that is zero at (and antisymmetric about) the wall."""
    out = np.zeros((field.shape[0] + 2, field.shape[1] + 2))
    out[1:-1, 1:-1] = field
    out[0, 1:-1] = -field[0, :]
    out[-1, 1:-1] = -field[-1, :]
    out[1:-1, 0] = -field[:, 0]
    out[1:-1, -1] = -field[:, -1]
    return out


def _woods_wall_vorticity(psi: np.ndarray, h: float, u_lid: float):
    """Second-order wall vorticity on all four walls (see module docstring).

    Derived from a Taylor expansion in the inward normal direction ``d`` at
    distance ``a = h/2`` (cell centre) and ``3a`` (second cell), with the wall
    normal derivative ``g'(0) = dpsi/dn`` imposed exactly::

        g''(0) = [27 g(a) - g(3a) - 26 g(0) - 24 a g'(0)] / (9 a^2)

    In the cavity ``g(0) = psi_wall = 0`` on every wall, so the ``-26 g(0)``
    term drops out; ``g'(0) = u_wall`` on the lid and ``0`` elsewhere.
    Returns ``(omega_wall dict,)``; shapes ``(n,)`` per wall.
    """
    fac = 4.0 / (9.0 * h * h)

    def w(p1, p2, g_prime):
        return -fac * (27.0 * p1 - p2 - 12.0 * h * g_prime)

    return {
        "bottom": w(psi[0, :], psi[1, :], 0.0),
        "top": w(psi[-1, :], psi[-2, :], -u_lid),  # inward normal = -y
        "left": w(psi[:, 0], psi[:, 1], 0.0),
        "right": w(psi[:, -1], psi[:, -2], 0.0),
    }


def _pad_with_wall_ghost(omega: np.ndarray, ow: dict) -> np.ndarray:
    out = np.zeros((omega.shape[0] + 2, omega.shape[1] + 2))
    out[1:-1, 1:-1] = omega
    # ghost cell at h/2 outside: omega_ghost = 2 omega_wall - omega_1
    out[0, 1:-1] = 2.0 * ow["bottom"] - omega[0, :]
    out[-1, 1:-1] = 2.0 * ow["top"] - omega[-1, :]
    out[1:-1, 0] = 2.0 * ow["left"] - omega[:, 0]
    out[1:-1, -1] = 2.0 * ow["right"] - omega[:, -1]
    return out


def solve_psi_omega(
    n: int,
    nu: float,
    mode: str = "cavity",
    u_lid: float = 0.0,
    mms=None,
    max_iter: int = 400_000,
    tol: float = 1e-8,
    cfl: float = 0.4,
    check_every: int = 200,
    seed_velocity=None,
) -> FVMResult:
    """March the vorticity transport equation to a steady state.

    Parameters
    ----------
    mode:
        ``"cavity"`` (Dirichlet walls, optional moving lid) or ``"periodic"``
        (used with a manufactured solution ``mms``).
    """
    h = 1.0 / n
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")

    periodic = mode == "periodic"
    if periodic:
        if mms is None:
            raise ValueError("periodic mode requires a manufactured solution")
        omega = mms.vorticity(xx, yy).copy()
        src = mms.curl_source(xx, yy)
    else:
        omega = np.zeros((n, n))
        src = np.zeros((n, n))

    psi = (_poisson_periodic if periodic else _poisson_dirichlet)(omega, h)

    dt = cfl * h * h / (4.0 * nu)
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    history: list[dict] = []
    residual = np.inf
    converged = False
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        if periodic:
            psi_ext = _pad_wrap(psi)
            if mms is not None:
                wall_omega = None
            omega_ext = _pad_wrap(omega)
        else:
            psi_ext = _pad_odd(psi)
            wall_omega = _woods_wall_vorticity(psi, h, u_lid)
            omega_ext = _pad_with_wall_ghost(omega, wall_omega)
        u = (psi_ext[2:, 1:-1] - psi_ext[:-2, 1:-1]) / (2.0 * h)
        v = -(psi_ext[1:-1, 2:] - psi_ext[1:-1, :-2]) / (2.0 * h)
        lap = (
            omega_ext[2:, 1:-1] - 2.0 * omega + omega_ext[:-2, 1:-1]
            + omega_ext[1:-1, 2:] - 2.0 * omega + omega_ext[1:-1, :-2]
        ) / (h * h)
        conv = u * (omega_ext[1:-1, 2:] - omega_ext[1:-1, :-2]) / (2.0 * h) + v * (
            omega_ext[2:, 1:-1] - omega_ext[:-2, 1:-1]
        ) / (2.0 * h)
        umax = float(np.hypot(u, v).max()) + 1e-12
        dt = cfl * min(h * h / (4.0 * nu), h / umax)
        omega_new = omega + dt * (-conv + nu * lap + src)
        delta = float(np.abs(omega_new - omega).max()) / dt
        omega = omega_new
        if it % check_every == 0:
            psi_old = psi
            psi = (_poisson_periodic if periodic else _poisson_dirichlet)(omega, h)
            residual = float(np.abs(psi - psi_old).max() * (1.0 / h))
            history.append({"iter": it, "residual": residual, "omega_rate": delta, "dt": dt})
            if residual < tol:
                converged = True
                break
        else:
            psi = (_poisson_periodic if periodic else _poisson_dirichlet)(omega, h)
    psi = (_poisson_periodic if periodic else _poisson_dirichlet)(omega, h)
    if periodic:
        psi_ext = _pad_wrap(psi)
        wall_omega = None
    else:
        psi_ext = _pad_odd(psi)
        wall_omega = _woods_wall_vorticity(psi, h, u_lid)
    u = (psi_ext[2:, 1:-1] - psi_ext[:-2, 1:-1]) / (2.0 * h)
    v = -(psi_ext[1:-1, 2:] - psi_ext[1:-1, :-2]) / (2.0 * h)
    return FVMResult(
        u=u, v=v, psi=psi, omega=omega, omega_wall=wall_omega, n=n,
        iterations=iterations, converged=converged, residual=residual, dt=dt,
        history=history,
    )