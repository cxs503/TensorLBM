"""Maxwell equal-area construction on the Shan-Chen *lattice* equation of state.

Independent analytic anchor for the SCMP (single-component pseudopotential)
verification.  Everything here is derived from the model definition, no code
from ``tensorlbm`` is imported for the thermodynamics, so the result is a
genuine reference and not a transcription of the implementation under test.

Lattice EOS (library convention of ``tensorlbm.multiphase``, ``G > 0``
attractive, backward-gather force ``F = -G cs^2 psi grad(psi)``)::

    p(rho) = rho/3 - (G/6) psi(rho)^2                (cs^2 = 1/3)

Proof of the sign: with ``G_eff = -G_lib`` (see ``multiphase_edm`` module
docstring) the mechanical pressure is
``p = rho cs^2 + (G_eff/2) cs^2 psi^2 = rho/3 - (G/6) psi^2``.

Maxwell construction in the ``(rho, p)`` plane is done with the specific
volume ``v = 1/rho``::

    p(rho_g) = p(rho_l) = p_sat                (mechanical equilibrium)
    \int_{rho_g}^{rho_l} [p(rho) - p_sat] / rho^2  d(rho) = 0   (equal area)

The second condition is ``\int p dv = p_sat (v_l - v_g)`` rewritten with
``dv = -d(rho)/rho^2``.

Reference for the construction: Shan & Chen (1993) PRE 47, 1815;
Yuan & Schaefer (2006) Phys. Fluids 18, 042101.

All routines are pure numpy; ``psi`` is supplied as a callable on ndarrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Pseudopotentials (same formulas as the library, re-typed here on purpose:
# the Maxwell anchor must not depend on the code under test)
# ---------------------------------------------------------------------------


def psi_exp(rho, rho0: float = 1.0):
    """Shan-Chen 1993 pseudopotential psi = rho0 (1 - exp(-rho/rho0))."""
    return rho0 * (1.0 - np.exp(-rho / rho0))


def psi_cs(rho, T: float, a: float = 1.0, b: float = 4.0, R: float = 1.0):
    """CS-EOS pseudopotential psi = sqrt(6 (rho/3 - p_CS))  (TCLB d2q9_scmp form).

    TCLB writes ``psi = sqrt(-p_CS(rho) + rho/3)``; the factor 6 is a constant
    rescaling of psi, i.e. an equivalent rescaling of the coupling (``G``), so
    with ``G = 1`` the lattice EOS collapses onto the Carnahan-Starling EOS
    exactly:
        p = rho/3 - (G/6) psi^2 = rho/3 - (rho/3 - p_CS) = p_CS .
    The argument is clamped at 0 (library behaviour) which makes the EOS
    piecewise above the crossing density ``p_CS = rho/3``; coexistence points
    used here are kept below that crossing.
    """
    eta = b * rho / 4.0
    eta1 = np.clip(1.0 - eta, 1e-8, None)
    p = rho * R * T * (1.0 + eta + eta**2 - eta**3) / eta1**3 - a * rho**2
    return np.sqrt(np.clip(6.0 * (rho * CS2 - p), 0.0, None))


def cs_crossing_density(T: float, a: float = 1.0, b: float = 4.0, R: float = 1.0) -> float:
    """Smallest rho > rho_c where p_CS(rho) = rho/3 (psi clamp onset)."""
    grid = np.linspace(1e-3, 4.0 / b - 1e-3, 200001)
    eta = b * grid / 4.0
    eta1 = np.clip(1.0 - eta, 1e-8, None)
    p = grid * R * T * (1.0 + eta + eta**2 - eta**3) / eta1**3 - a * grid**2
    d = grid * CS2 - p
    idx = np.where(d < 0.0)[0]
    if idx.size == 0:
        return 4.0 / b
    return float(grid[idx[0]])


# ---------------------------------------------------------------------------
# EOS helpers
# ---------------------------------------------------------------------------


def p_of_rho(rho, G: float, psi):
    """Lattice mechanical pressure ``p = rho/3 - (G/6) psi^2``."""
    return rho * CS2 - 0.5 * G * CS2 * np.asarray(psi(rho)) ** 2


def dp_drho(rho, G: float, psi, h: float = 1e-6):
    return (p_of_rho(rho + h, G, psi) - p_of_rho(rho - h, G, psi)) / (2.0 * h)


@dataclass
class Coexistence:
    rho_g: float
    rho_l: float
    p_sat: float
    area_residual: float
    pressure_residual: float
    converged: bool
    critical: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "rho_g": self.rho_g,
            "rho_l": self.rho_l,
            "p_sat": self.p_sat,
            "area_residual": self.area_residual,
            "pressure_residual": self.pressure_residual,
            "converged": self.converged,
            "critical": self.critical,
            "note": self.note,
        }


def spinodal(G: float, psi, rho_lo: float = 1e-4, rho_hi: float = None, n: int = 4001):
    """Return ``(rho_s1, rho_s2)`` where ``dp/drho = 0`` (None if G <= G_c)."""
    if rho_hi is None:
        rho_hi = 6.0
    grid = np.linspace(rho_lo, rho_hi, n)
    d = dp_drho(grid, G, psi)
    sign = np.sign(d)
    idx = np.where(np.diff(sign) != 0)[0]
    if idx.size < 2:
        return None
    roots = []
    for i in idx:
        lo, hi = grid[i], grid[i + 1]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.sign(dp_drho(mid, G, psi)) == np.sign(dp_drho(lo, G, psi)):
                lo = mid
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))
    return (roots[0], roots[-1]) if len(roots) >= 2 else None


def critical_coupling(psi, rho_lo: float = 1e-4, rho_hi: float = 6.0, n: int = 400001):
    """``G_c`` = the coupling at which ``min_rho dp/drho`` first touches 0.

    ``dp/drho = 1/3 - (G/3) psi psi'``  ->  ``G_c = 1 / max(psi psi')``.
    For ``psi_exp`` this returns exactly 4 at ``rho = ln 2`` (analytic check).
    """
    grid = np.linspace(rho_lo, rho_hi, n)
    h = 1e-6
    pp = np.asarray(psi(grid)) * (np.asarray(psi(grid + h)) - np.asarray(psi(grid - h))) / (2 * h)
    i = int(np.argmax(pp))
    return 1.0 / pp[i], float(grid[i])


def maxwell(
    G: float,
    psi,
    rho_lo: float = 1e-5,
    rho_hi: float = None,
    n_solve: int = 4000,
    tol: float = 1e-13,
    max_iter: int = 300,
) -> Coexistence:
    """Solve the Maxwell construction for the lattice EOS ``p = rho/3 - (G/6) psi^2``.

    Two-stage approach (fully independent of any library usage):
      1. locate the spinodal densities (``dp/drho = 0``);
      2. bisect on ``p_sat`` inside ``(p(rho_s2), p(rho_s1))``; for each
         ``p_sat`` invert the monotone branches to get ``rho_g``, ``rho_l``
         and evaluate the equal-area integral.
    """
    if rho_hi is None:
        rho_hi = 6.0
    sp = spinodal(G, psi, rho_lo, rho_hi)
    if sp is None:
        return Coexistence(0.0, 0.0, 0.0, 0.0, 0.0, True, True,
                           note="no spinodal (G <= G_c): single phase")
    rho_s1, rho_s2 = sp
    p_hi = float(p_of_rho(rho_s1, G, psi))   # local maximum of p
    p_lo = float(p_of_rho(rho_s2, G, psi))   # local minimum of p

    def branch_root(p_target: float, lo: float, hi: float) -> float:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            v = float(p_of_rho(mid, G, psi))
            if (v - p_target) * (float(p_of_rho(lo, G, psi)) - p_target) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def area(p_sat: float) -> tuple[float, float, float]:
        rg = branch_root(p_sat, rho_lo, rho_s1)
        rl = branch_root(p_sat, 4.0 / 3.0 * rho_s2 if rho_s2 < 0 else rho_s2, rho_hi)
        # if the right branch root collapses onto the spinodal, widen the search
        if abs(rl - rho_s2) < 1e-12:
            rl = branch_root(p_sat, rho_s2, rho_hi)
        grid = np.linspace(rg, rl, n_solve)
        integrand = (p_of_rho(grid, G, psi) - p_sat) / grid**2
        trap = getattr(np, "trapezoid", None) or np.trapz
        return float(trap(integrand, grid)), rg, rl

    lo_p, hi_p = p_lo, p_hi
    a_lo = area(lo_p + 1e-14 * abs(lo_p) + 1e-16)[0]
    a_hi = area(hi_p - 1e-14 * abs(hi_p) - 1e-16)[0]
    converged = False
    p_sat = 0.5 * (lo_p + hi_p)
    for _ in range(max_iter):
        p_sat = 0.5 * (lo_p + hi_p)
        a_mid, rg, rl = area(p_sat)
        if abs(a_mid) < tol or (hi_p - lo_p) < 1e-15:
            converged = True
            break
        if a_mid * a_lo > 0:
            lo_p, a_lo = p_sat, a_mid
        else:
            hi_p = p_sat
    a_mid, rg, rl = area(p_sat)
    p_res = max(abs(float(p_of_rho(rg, G, psi)) - p_sat), abs(float(p_of_rho(rl, G, psi)) - p_sat))
    return Coexistence(
        rho_g=rg,
        rho_l=rl,
        p_sat=p_sat,
        area_residual=a_mid,
        pressure_residual=p_res,
        converged=converged,
        critical=False,
    )


# ---------------------------------------------------------------------------
# Known analytic limits, used as self-tests (see tests/test_maxwell_sc_xval.py)
# ---------------------------------------------------------------------------


def analytic_psi_exp():
    """Analytic critical point of the psi_exp model.

    ``dp/drho = 1/3 - (G/3) psi psi'``, ``psi psi' = e^{-rho}(1-e^{-rho})``
    with maximum 1/4 at ``rho = ln 2`` -> ``G_c = 4``, ``rho_c = ln 2``,
    ``p_c = ln2/3 - (4/6)(1/2)^2``.
    """
    return {
        "G_c": 4.0,
        "rho_c": math.log(2.0),
        "p_c": math.log(2.0) / 3.0 - (4.0 / 6.0) * 0.25,
    }