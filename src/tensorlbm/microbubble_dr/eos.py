"""Shan-Chen EOS utilities for the IN67-N01 microbubble channel.

The micro-bubble runs need the coexistence densities of the pseudopotential
model actually used in the solver, because the void-fraction field is defined as

    alpha = (rho_l - rho) / (rho_l - rho_g)   clamped to [0, 1]

and a hard-coded pair of densities would silently mis-scale alpha.  Here the
densities are obtained from the model's own mechanical pressure

    p(rho) = c_s^2 [ rho - (G/2) psi(rho)^2 ]

with the Maxwell equal-area construction.  This is the lattice-EOS coexistence
(maxwell construction on the *lattice* pressure, not on a continuum EOS), which
is what the Shan-Chen solver actually phase-separates to at leading order.

References
----------
Shan & Chen (1993) Phys. Rev. E 47, 1815; (1994) 49, 2941.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

CS2 = 1.0 / 3.0


def psi_exp_np(rho: np.ndarray | float, rho0: float = 1.0):
    """Exponential pseudopotential ``psi = rho0 (1 - exp(-rho/rho0))``."""
    return rho0 * (1.0 - np.exp(-np.asarray(rho, dtype=float) / rho0))


def mechanical_pressure(rho, G: float = 5.0, rho0: float = 1.0):
    """Shan-Chen mechanical pressure ``p = cs^2 (rho - G psi^2 / 2)``."""
    psi = psi_exp_np(rho, rho0)
    return CS2 * (np.asarray(rho, dtype=float) - 0.5 * G * psi * psi)


def spinodal_window(G: float = 5.0, rho0: float = 1.0, rho_max: float = 20.0):
    """Return ``(rho_spinodal_a, rho_spinodal_b)`` where ``dp/drho < 0``.

    The window exists only above the critical coupling ``G_c`` (classic
    ``G_c = 4`` in this library's sign convention); below it the model cannot
    phase separate and a two-phase reference run is meaningless.
    """
    grid = np.linspace(1e-5, rho_max, 400001)
    p = mechanical_pressure(grid, G, rho0)
    neg = np.where(np.diff(p) < 0.0)[0]
    if neg.size == 0:
        raise ValueError(
            f"G={G} is below the critical coupling: p(rho) is monotone, no "
            "liquid-gas coexistence is possible"
        )
    return float(grid[neg[0]]), float(grid[neg[-1] + 1])


def maxwell_coexistence(G: float = 5.0, rho0: float = 1.0,
                        rho_max: float = 20.0, xtol: float = 2e-10):
    """Maxwell equal-area coexistence ``(rho_gas, rho_liquid)`` of the lattice EOS.

    Solved as a one-dimensional root problem in ``rho_g``: for a trial gas
    density the liquid density is the (unique) larger root of
    ``p(rho_l) = p(rho_g)``, and the residual is the Maxwell area
    ``int (p0 - p(rho))/rho^2 drho`` over ``[rho_g, rho_l]``.  The area changes
    sign across the vapor spinodal, so a bracketed Brent solve is enough.
    """
    s_lo, s_hi = spinodal_window(G, rho0, rho_max)

    def rho_liquid_of(rho_g: float) -> float:
        target = float(mechanical_pressure(rho_g, G, rho0))
        f = lambda r: float(mechanical_pressure(r, G, rho0)) - target
        return brentq(f, s_hi, rho_max, xtol=xtol)

    def area(rho_g: float) -> float:
        rho_l = rho_liquid_of(rho_g)
        p0 = float(mechanical_pressure(rho_g, G, rho0))
        val, _ = quad(
            lambda r: (p0 - float(mechanical_pressure(r, G, rho0))) / (r * r),
            rho_g, rho_l, limit=400,
        )
        return val

    lo, hi = 1e-6, s_lo
    f_lo, f_hi = area(lo), area(hi)
    if f_lo * f_hi > 0.0:
        raise RuntimeError(
            f"Maxwell construction not bracketed (G={G}): area({lo})={f_lo}, "
            f"area({hi})={f_hi}"
        )
    rho_g = brentq(area, lo, hi, xtol=xtol, maxiter=200)
    return rho_g, rho_liquid_of(rho_g)


def void_fraction(rho, rho_liquid: float, rho_gas: float):
    """``alpha = (rho_l - rho)/(rho_l - rho_g)`` clamped to ``[0, 1]``."""
    if rho_liquid <= rho_gas:
        raise ValueError("rho_liquid must exceed rho_gas")
    a = (rho_liquid - np.asarray(rho, dtype=float)) / (rho_liquid - rho_gas)
    return np.clip(a, 0.0, 1.0)


def kinematic_viscosity(tau: float) -> float:
    """BGK kinematic viscosity of this library (``nu = (tau - 1/2)/3``)."""
    return (tau - 0.5) / 3.0