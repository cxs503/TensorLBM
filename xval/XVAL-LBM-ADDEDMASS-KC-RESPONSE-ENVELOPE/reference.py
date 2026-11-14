#!/usr/bin/env python3
"""Independent references for the added-mass ("mass analogue") coefficient C_a.

Nothing here reuses the module under test; the repo's own potential solve
(``fsi_added_mass.potential_added_mass``) appears only in :func:`repo_potential`
so that the two references can be compared *with each other*.

R1  unbounded circular cylinder (potential flow)          C_a = 1
R2  cylinder inside a concentric circular container       C_a = (Ro^2+R^2)/(Ro^2-R^2)
    (Lamb 1932 §71; Landau & Lifshitz, Fluid Mechanics §2 problem) plus an
    independent radial-FD / energy-integral check of the same BVP.
R3  repo same-mask periodic potential solve, with a refinement study that
    isolates its discretisation bias.
"""
from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- #
# R1
# --------------------------------------------------------------------------- #
def c_a_unbounded() -> float:
    """``m_a = rho*pi*R^2`` per unit length => ``C_a = m_a/(rho*pi*R^2) = 1``."""
    return 1.0


# --------------------------------------------------------------------------- #
# R2  concentric circular container
# --------------------------------------------------------------------------- #
def c_a_annulus_closed(radius: float, r_out: float) -> float:
    """Exact potential added mass of a cylinder in a concentric container.

    phi = (A r + B/r) cos(theta) with ``dphi/dr = U cos(theta)`` on r = radius and
    ``dphi/dr = 0`` on r = r_out; the kinetic energy gives

        m_a = rho*pi*R^2 * (Ro^2 + R^2)/(Ro^2 - R^2)
    """
    if not 0.0 < radius < r_out:
        raise ValueError("need 0 < radius < r_out")
    r2, ro2 = radius * radius, r_out * r_out
    return (ro2 + r2) / (ro2 - r2)


def c_a_annulus_numeric(radius: float, r_out: float, n: int = 4000) -> dict:
    """Independent check of :func:`c_a_annulus_closed` by radial FD + energy.

    Solves ``g'' + g'/r - g/r^2 = 0`` with ``g'(R) = 1``, ``g'(Ro) = 0`` and takes

        C_a = (1/(pi R^2)) * pi * int_R^Ro [ (g')^2 + (g/r)^2 ] r dr

    i.e. the energy path rather than the boundary-flux path used by the closed
    form, so an algebra slip in either shows up as a mismatch.
    """
    rr = np.linspace(radius, r_out, n + 1)
    h = rr[1] - rr[0]
    g = np.zeros(n + 1)
    # unknowns g[0..n]; ghost-free elimination of the two Neumann BCs
    m = n + 1
    amat = np.zeros((m, m))
    rhs = np.zeros(m)
    for i in range(m):
        r = rr[i]
        if i == 0:                      # g'(R) = 1  ->  g_0 - g_1 = -h
            amat[i, i], amat[i, i + 1] = 1.0, -1.0
            rhs[i] = -h
            continue
        if i == m - 1:                  # g'(Ro) = 0 ->  g_n - g_{n-1} = 0
            amat[i, i], amat[i, i - 1] = 1.0, -1.0
            continue
        amat[i, i] = -2.0 / h**2 - 1.0 / r**2
        amat[i, i - 1] = 1.0 / h**2 - 0.5 / (h * r)
        amat[i, i + 1] = 1.0 / h**2 + 0.5 / (h * r)
    g = np.linalg.solve(amat, rhs)
    gp = np.gradient(g, h)
    integrand = (gp**2 + (g / rr) ** 2) * rr
    energy = math.pi * float(np.trapezoid(integrand, rr))
    return {
        "c_a_numeric": energy / (math.pi * radius**2),
        "c_a_closed": c_a_annulus_closed(radius, r_out),
        "g_R": float(g[0]),
    }


# --------------------------------------------------------------------------- #
# spin-up criterion
# --------------------------------------------------------------------------- #
def spinup_periods(beta: float, box_over_D: float) -> float:
    """Periods of spin-up needed before the periodic state is reached.

    The slowest viscous mode of the periodic box (wavenumber ``2*pi/L``) decays as
    ``exp(-nu (2*pi/L)^2 t)``; in units of the forcing period ``T`` and with
    ``nu*T = D^2/beta`` this is

        N_spin ~ beta * (L/(2*pi*D))^2 = beta * (box_over_D/(2*pi))^2 .
    """
    return float(beta) * (box_over_D / (2.0 * math.pi)) ** 2


# --------------------------------------------------------------------------- #
# R3  repo same-mask periodic potential solve
# --------------------------------------------------------------------------- #
def repo_potential(nx: int, ny: int, radius: float) -> dict:
    """The repo reference used by ``oscillatory_cylinder_inertia`` itself."""
    from tensorlbm.fsi_added_mass import potential_added_mass

    out = potential_added_mass(nx, ny, radius=radius)
    out["c_a_unbounded"] = c_a_unbounded()
    return out


def reference_table(radii=(8.0, 12.0, 16.0, 24.0, 32.0), boxes=(4.0, 6.0, 8.0, 12.0, 16.0),
                    area_equiv: bool = True) -> list[dict]:
    """C_a references for each (radius, box/D) pair.

    The concentric-container anchors use two radii for the square periodic cell:
    the inscribed circle ``Ro = L/2`` (most confining) and the equal-area circle
    ``Ro = L/sqrt(pi)``.  A square cell has corners that the inscribed circle does
    not, so the periodic-box value is expected to sit at or below the ``L/2``
    anchor - this is a bracket, not a theorem.
    """
    rows = []
    for R in radii:
        for box in boxes:
            L = box * 2.0 * R
            row = {
                "R": R,
                "box_over_D": box,
                "L_over_R": L / R,
                "blockage": 2.0 * R / L,
                "c_a_unbounded": 1.0,
                "c_a_annulus_inscribed": c_a_annulus_closed(R, L / 2.0),
                "c_a_annulus_areaeq": c_a_annulus_closed(R, L / math.sqrt(math.pi)),
                "solid_fraction": math.pi * R**2 / L**2,
            }
            rows.append(row)
    return rows