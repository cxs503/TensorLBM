"""Minimal, dependency-light D2Q9 BGK kernels used by the unit anchors.

The unit-bootstrap gate needs *real* lattice kinetic simulations whose
discretisation error is either analytically zero (Couette, Poiseuille) or
small and controllable (Taylor-Green), so that the residual error reported by
an anchor is attributable to the dimensionalisation chain rather than to the
solver.

Kernels
-------
``channel_poiseuille``   plane channel, half-way bounce-back walls, Guo body
                         force; steady solution is the exact parabola with
                         walls at ``y = -0.5`` and ``y = Ny - 0.5``
``channel_couette``      plane channel, moving top wall (half-way bounce-back
                         with wall velocity); steady solution is exactly
                         linear for BGK
``taylor_green``         fully periodic 2D Taylor-Green vortex; measures the
                         exponential decay rate ``2 nu k^2``

All arrays are ``(9, Nx, Ny)`` with unit lattice density.  Geometry is
Cartesian and the lattice never rotates; nothing here depends on physical
units -- dimensionalisation is the bridge's job
(:mod:`tensorlbm.units.bridge`).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "C",
    "W",
    "OPP",
    "equilibrium",
    "channel_poiseuille",
    "channel_couette",
    "taylor_green",
]

#: D2Q9 discrete velocities
C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
#: D2Q9 weights
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
#: opposite direction indices
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

_CX = C[:, 0].reshape(9, 1, 1).astype(float)
_CY = C[:, 1].reshape(9, 1, 1).astype(float)
_W3 = W.reshape(9, 1, 1)

#: incoming-from-below directions (cy = +1) and their outgoing partners
_BOTTOM_IN = (2, 5, 6)
_BOTTOM_OUT = (4, 7, 8)
#: incoming-from-above directions (cy = -1) and their outgoing partners
_TOP_IN = (4, 7, 8)
_TOP_OUT = (2, 5, 6)


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order Maxwell-Boltzmann equilibrium, shape ``(9, Nx, Ny)``."""
    cu = _CX * ux[None] + _CY * uy[None]
    usq = ux**2 + uy**2
    return _W3 * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq[None])


def _stream(f: np.ndarray) -> np.ndarray:
    """Periodic streaming in both directions (boundary rows fixed afterwards)."""
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(f[i], (int(C[i, 0]), int(C[i, 1])), axis=(0, 1))
    return out


def _collide_bgk(f: np.ndarray, tau: float) -> np.ndarray:
    rho = f.sum(axis=0)
    ux = (f * _CX).sum(axis=0) / rho
    uy = (f * _CY).sum(axis=0) / rho
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau


def _wall_reflect(
    f_new: np.ndarray,
    f_post: np.ndarray,
    rho_wall: float,
    u_wall_x: np.ndarray | None,
    side: str,
) -> None:
    """Half-way bounce-back on the last/first row (in place).

    ``f_new`` is post-streaming, ``f_post`` post-collision.  With wall
    velocity ``u_w`` the standard correction
    ``f_i = f_opp + 6 w_i rho (c_i . u_w)`` is applied (``2/cs^2 = 6``).
    """
    y = 0 if side == "bottom" else f_new.shape[2] - 1
    ins = _BOTTOM_IN if side == "bottom" else _TOP_IN
    outs = _BOTTOM_OUT if side == "bottom" else _TOP_OUT
    for i_in, i_out in zip(ins, outs):
        f_new[i_in, :, y] = f_post[i_out, :, y]
        if u_wall_x is not None:
            f_new[i_in, :, y] += 6.0 * W[i_in] * rho_wall * C[i_in, 0] * u_wall_x


def _body_force_source(force_x: float) -> np.ndarray:
    """Classic body-force term ``3 w_i c_ix F`` added after collision.

    With this term, ``u = sum_i c_i f_i`` (no half-force shift) and the steady
    channel solution is the exact parabola; the residual deviation measured by
    the gate is the ``O(Ma^2)`` compressibility bias (identical for every grid
    because the anchor pins the lattice velocity), i.e. it cancels in the
    grid-invariance criterion and stays far below the absolute tolerance.
    """
    return _W3 * 3.0 * _CX * force_x


def channel_poiseuille(
    ny: int,
    nx: int,
    tau: float,
    force_lb: float,
    *,
    max_steps: int = 200_000,
    tol: float = 1e-11,
    check_every: int = 200,
) -> dict:
    """Steady plane-Poiseuille channel driven by a body force.

    Walls are half-way bounce-back planes at ``y = -0.5`` and ``y = Ny-0.5``,
    so the channel height is exactly ``Ny`` lattice units and the steady
    solution is ``u(y) = G/(2 nu) * s * (Ny - s)`` with ``s = y + 0.5``.

    Returns
    -------
    dict with ``ux`` (steady profile per row), ``u_max``, ``u_bulk``,
    ``wall_stress`` (``G * Ny / 2``, exact by momentum balance), ``steps``
    and ``converged``.
    """
    if ny < 4 or nx < 1:
        raise ValueError("channel needs ny >= 4 and nx >= 1")
    nu = (tau - 0.5) / 3.0
    rho = np.ones((nx, ny))
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    f = equilibrium(rho, ux, uy)

    converged = False
    u_prev = np.zeros(ny)
    steps = 0
    for step in range(1, max_steps + 1):
        f = _collide_bgk(f, tau) + _body_force_source(force_lb)
        f_post = f
        f = _stream(f_post)
        _wall_reflect(f, f_post, 1.0, None, "bottom")
        _wall_reflect(f, f_post, 1.0, None, "top")
        steps = step
        if step % check_every == 0:
            rho_now = f.sum(axis=0)
            ux_now = (f * _CX).sum(axis=0) / rho_now
            prof = ux_now.mean(axis=0)
            if np.max(np.abs(prof - u_prev)) < tol * max(1.0, np.max(np.abs(prof))):
                converged = True
                break
            u_prev = prof

    rho_end = f.sum(axis=0)
    ux_end = (f * _CX).sum(axis=0) / rho_end
    prof = ux_end.mean(axis=0)
    return {
        "ux": prof,
        "u_max": float(prof.max()),
        "u_bulk": float(prof.mean()),
        "wall_stress": float(force_lb * ny / 2.0),
        "steps": steps,
        "converged": converged,
        "nu_lb": nu,
    }


def channel_couette(
    ny: int,
    nx: int,
    tau: float,
    u_wall_lb: float,
    *,
    max_steps: int = 200_000,
    tol: float = 1e-11,
    check_every: int = 200,
) -> dict:
    """Steady Couette flow with a moving top wall (half-way bounce-back).

    Returns
    -------
    dict with the steady ``ux`` profile, the least-squares slope
    ``slope`` (``d u / dy`` in lattice units), the maximum relative deviation
    from the linear fit ``linearity_error`` and run diagnostics.
    """
    nu = (tau - 0.5) / 3.0
    rho = np.ones((nx, ny))
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    f = equilibrium(rho, ux, uy)
    u_top = np.full(nx, u_wall_lb)

    converged = False
    u_prev = np.zeros(ny)
    steps = 0
    for step in range(1, max_steps + 1):
        f_post = _collide_bgk(f, tau)
        f = _stream(f_post)
        _wall_reflect(f, f_post, 1.0, None, "bottom")
        _wall_reflect(f, f_post, 1.0, u_top, "top")
        steps = step
        if step % check_every == 0:
            rho_now = f.sum(axis=0)
            prof = ((f * _CX).sum(axis=0) / rho_now).mean(axis=0)
            if np.max(np.abs(prof - u_prev)) < tol * max(1.0, np.max(np.abs(prof))):
                converged = True
                u_prev = prof
                break
            u_prev = prof

    rho_end = f.sum(axis=0)
    prof = ((f * _CX).sum(axis=0) / rho_end).mean(axis=0)
    s = np.arange(ny) + 0.5
    slope, intercept = np.polyfit(s, prof, 1)
    fit = slope * s + intercept
    lin_err = float(np.max(np.abs(prof - fit)) / max(abs(u_wall_lb), 1e-30))
    return {
        "ux": prof,
        "slope": float(slope),
        "intercept": float(intercept),
        "linearity_error": lin_err,
        "steps": steps,
        "converged": converged,
        "nu_lb": nu,
    }


def taylor_green(
    n: int,
    tau: float,
    u0_lb: float,
    *,
    steps: int = 400,
    fit_from: int = 100,
) -> dict:
    """2D Taylor-Green vortex; measures the exponential decay rate.

    ``u_x = u0 sin(k x) cos(k y)``, ``u_y = -u0 cos(k x) sin(k y)`` on a
    periodic ``n x n`` grid with ``k = 2 pi / n``.  The exact solution decays
    as ``exp(-2 nu k^2 t)``; the rate is obtained by a least-squares fit of
    ``ln A(t)`` over ``steps >= fit_from``.

    Returns
    -------
    dict with ``rate`` (per lattice step), ``k`` (lattice wave number),
    ``amplitude`` history and run diagnostics.
    """
    k = 2.0 * np.pi / n
    x = (np.arange(n) + 0.5)
    X, Y = np.meshgrid(x, x, indexing="ij")
    ux = u0_lb * np.sin(k * X) * np.cos(k * Y)
    uy = -u0_lb * np.cos(k * X) * np.sin(k * Y)
    rho = np.ones((n, n))
    f = equilibrium(rho, ux, uy)

    amp_hist: list[float] = []
    for step in range(steps + 1):
        rho_now = f.sum(axis=0)
        ux_now = (f * _CX).sum(axis=0) / rho_now
        uy_now = (f * _CY).sum(axis=0) / rho_now
        amp = float(np.sqrt(np.mean(ux_now**2 + uy_now**2)))
        amp_hist.append(amp)
        if step == steps:
            break
        f = _stream(_collide_bgk(f, tau))

    t = np.arange(steps + 1, dtype=float)
    sel = t >= fit_from
    slope = np.polyfit(t[sel], np.log(np.array(amp_hist)[sel]), 1)[0]
    return {
        "rate": float(-slope),
        "k": float(k),
        "amplitude": amp_hist,
        "nu_lb": (tau - 0.5) / 3.0,
        "steps": steps,
    }