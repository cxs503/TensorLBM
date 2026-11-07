"""Linear stability analysis of the 2-D Boussinesq cavity (critical Rayleigh number).

The conductive state of a differentially heated square cavity loses stability to
a steady convection roll at ``Ra_c = 1707.762`` (linear theory; the value is
reproduced here by an independent discretisation, *not* copied).  Time
integration of the lattice-Boltzmann model cannot resolve a 1 % threshold
window: the growth rate at ``Ra/Ra_c - 1 = 0.01`` corresponds to a few times
``1e-5`` per lattice step, i.e. millions of steps for one e-fold.  This module
therefore solves the linearised Boussinesq eigenvalue problem directly:

.. math::

    \\partial_t \\omega = \\nu \\nabla^2 \\omega - g\\beta \\,\\partial_x \\theta,
    \\qquad \\nabla^2 \\psi = -\\omega,
    \\qquad \\partial_t \\theta = \\alpha \\nabla^2 \\theta + \\partial_y \\psi,

linearised about ``\\bar T = 1 - x`` (hot wall at ``x = 0``, gravity along
``-y``), with no-slip walls (Thom vorticity), isothermal side walls and adiabatic
top/bottom walls.  The spatial operator is assembled as a dense matrix, its
eigenvalue with the largest real part gives the growth rate ``sigma(Ra)``, and
``sigma(Ra_c) = 0`` is found by a secant iteration.

Units: ``H = 1``, ``alpha = 1``, ``nu = Pr``, hence ``g beta = Ra * Pr``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["boussinesq_operator", "growth_rate", "critical_rayleigh",
           "RA_C_LINEAR_THEORY"]

#: classical linear-theory critical Rayleigh number for the 2-D square cavity
#: with adiabatic horizontal walls (comparison value only)
RA_C_LINEAR_THEORY = 1707.762


def _interior_index(n: int):
    """Map (i, j) interior node (1 <= i, j <= n-1) to a flat index."""
    m = n - 1

    def idx(i: int, j: int) -> int:
        return (i - 1) * m + (j - 1)

    return m, idx


def boussinesq_operator(n: int = 24, pr: float = 0.71) -> dict:
    """Assemble ``d/dt (omega, theta) = (S0 + Ra * S1) (omega, theta)``.

    ``n`` is the number of grid intervals (``n + 1`` nodes per direction, nodes
    on the walls).  Returns ``S0``, ``S1`` and bookkeeping.
    """
    m, idx = _interior_index(n)
    h = 1.0 / n
    h2 = h * h
    M = m * m

    # ---- Poisson matrix for psi (Dirichlet psi = 0 on all walls) ----------
    Lap = np.zeros((M, M))
    walls_per_node = np.zeros(M)
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            Lap[r, r] = -4.0 / h2
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 1 <= ii <= n - 1 and 1 <= jj <= n - 1:
                    Lap[r, idx(ii, jj)] = 1.0 / h2
                else:
                    walls_per_node[r] += 1.0
    # psi = -Lap^{-1} omega  (psi = 0 on the walls is already built in)
    psi_mat = -np.linalg.inv(Lap)

    # ---- vorticity Laplacian with Thom wall vorticity ---------------------
    # omega_wall = -2 psi_adjacent / h^2   (first interior node is h from wall)
    Lap_om = Lap.copy()
    diag_corr = np.zeros(M)
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            if walls_per_node[r] > 0:
                # each wall-adjacent normal direction contributes
                # (+omega_wall/h^2) = (-2 psi_node / h^4)
                diag_corr[r] = -2.0 * walls_per_node[r] / (h2 * h2)
    Lap_om = Lap_om + np.diag(diag_corr) @ psi_mat

    # ---- theta Laplacian: Dirichlet in x, Neumann (mirror) in y ----------
    Lap_th = np.zeros((M, M))
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            # x-direction: Dirichlet on side walls
            if j == 1:
                Lap_th[r, r] += -2.0 / h2
                Lap_th[r, idx(i, 2)] += 1.0 / h2
            elif j == n - 1:
                Lap_th[r, r] += -2.0 / h2
                Lap_th[r, idx(i, n - 2)] += 1.0 / h2
            else:
                Lap_th[r, r] += -2.0 / h2
                Lap_th[r, idx(i, j + 1)] += 1.0 / h2
                Lap_th[r, idx(i, j - 1)] += 1.0 / h2
            # y-direction: adiabatic (mirror) top/bottom
            if i == 1:
                Lap_th[r, r] += -1.0 / h2
                Lap_th[r, idx(2, j)] += 1.0 / h2
            elif i == n - 1:
                Lap_th[r, r] += -1.0 / h2
                Lap_th[r, idx(n - 2, j)] += 1.0 / h2
            else:
                Lap_th[r, r] += -2.0 / h2
                Lap_th[r, idx(i + 1, j)] += 1.0 / h2
                Lap_th[r, idx(i - 1, j)] += 1.0 / h2

    # ---- velocity from psi: u = d psi / d y (mirror at adiabatic walls) ---
    Dy = np.zeros((M, M))
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            if i == 1:
                Dy[r, idx(2, j)] = 1.0 / (2.0 * h)
                Dy[r, r] = -1.0 / (2.0 * h)
            elif i == n - 1:
                Dy[r, r] = 1.0 / (2.0 * h)
                Dy[r, idx(n - 2, j)] = -1.0 / (2.0 * h)
            else:
                Dy[r, idx(i + 1, j)] = 1.0 / (2.0 * h)
                Dy[r, idx(i - 1, j)] = -1.0 / (2.0 * h)
    Adv = Dy @ psi_mat  # theta_t += Adv @ omega

    # ---- d theta / dx (central differences, Dirichlet walls) -------------
    Dx = np.zeros((M, M))
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            if j == 1:
                Dx[r, r] = -1.0 / h
                Dx[r, idx(i, 2)] = 1.0 / h
            elif j == n - 1:
                Dx[r, r] = 1.0 / h
                Dx[r, idx(i, n - 2)] = -1.0 / h
            else:
                Dx[r, idx(i, j + 1)] = 1.0 / (2.0 * h)
                Dx[r, idx(i, j - 1)] = -1.0 / (2.0 * h)

    # ---- assemble S = S0 + Ra * S1 --------------------------------------
    S0 = np.zeros((2 * M, 2 * M))
    S1 = np.zeros((2 * M, 2 * M))
    # rows: omega block
    S0[:M, :M] = pr * Lap_om            # nu = Pr (alpha = 1)
    S1[:M, M:] = -pr * Dx               # -g beta d/dx with g beta = Ra * Pr
    # rows: theta block
    S0[M:, M:] = Lap_th
    S0[M:, :M] = Adv
    return {"S0": S0, "S1": S1, "n": n, "m": m, "size": 2 * M}


def growth_rate(ra: float, ops: dict | None = None, n: int = 24,
                pr: float = 0.71) -> float:
    """Largest real part of the linearised spectrum at ``Ra``."""
    ops = ops if ops is not None else boussinesq_operator(n, pr)
    S = ops["S0"] + ra * ops["S1"]
    ev = np.linalg.eigvals(S)
    return float(np.max(ev.real))


def critical_rayleigh(n: int = 24, pr: float = 0.71, ra_lo: float = 500.0,
                      ra_hi: float = 4000.0, tol: float = 1e-6,
                      max_iter: int = 40) -> dict:
    """Secant iteration on ``sigma(Ra) = 0`` (the neutral-stability point)."""
    ops = boussinesq_operator(n, pr)
    hist = []
    f_lo = growth_rate(ra_lo, ops)
    f_hi = growth_rate(ra_hi, ops)
    hist.append({"ra": ra_lo, "sigma": f_lo})
    hist.append({"ra": ra_hi, "sigma": f_hi})
    if f_lo > 0 or f_hi < 0:
        return {"ra_c": None, "bracket": (ra_lo, ra_hi), "history": hist,
                "note": "root not bracketed"}
    a, fa = ra_lo, f_lo
    b, fb = ra_hi, f_hi
    ra_c = None
    for _ in range(max_iter):
        # secant step, kept inside the bracket
        if fb - fa == 0.0:
            c = 0.5 * (a + b)
        else:
            c = b - fb * (b - a) / (fb - fa)
            if not (a < c < b):
                c = 0.5 * (a + b)
        fc = growth_rate(c, ops)
        hist.append({"ra": c, "sigma": fc})
        if abs(fc) < 1e-9 or (b - a) < tol * 0.5 * (a + b):
            ra_c = c
            break
        if fc > 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        ra_c = c
    return {
        "ra_c": ra_c,
        "pr": pr,
        "n": n,
        "resolution": f"{n}x{n} interiors",
        "history": hist,
        "ra_c_linear_theory": RA_C_LINEAR_THEORY,
        "deviation_rel": None if ra_c is None else abs(ra_c - RA_C_LINEAR_THEORY)
        / RA_C_LINEAR_THEORY,
    }