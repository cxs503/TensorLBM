"""Independent (from-scratch) D2Q9 single-component Shan-Chen reference solver.

Purpose
-------
This module is the *reference practice* half of the cross-validation
``XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION``: a minimal, self-contained
implementation of the SCMP (Shan-Chen single component, pseudopotential)
Laplace-pressure experiment written from the textbook recipe
(Shan & Chen 1993/1994; Krueger et al., *The Lattice Boltzmann Method:
Principles and Practice*, Ch. 9 "Multiphase and multicomponent flows", and the
companion ``lbm-principles-practice/code`` ``shan-chen`` examples).

It deliberately imports **nothing** from :mod:`tensorlbm`.  The only thing it
shares with the library is the D2Q9 lattice velocity ordering, which is the
standard textbook ordering ``[(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),
(-1,-1),(1,-1)]`` and is kept identical solely so that distribution functions
can be compared cell-by-cell against ``tensorlbm`` (see
``tests/test_xval_sc_laplace.py``).

Model (all quantities in lattice units)
---------------------------------------
* D2Q9 BGK, ``tau`` = 1.0, ``cs2 = 1/3``.
* Pseudopotential ``psi(rho) = 1 - exp(-rho)``  (Shan-Chen 1994 form, rho0 = 1).
* Interaction force, backward-gather form
      ``F(x) = -G psi(x) sum_i w_i c_i psi(x - c_i)``
  with the standard continuum limit ``F = -(G/2) grad(psi^2)``; the *sign*
  convention is ``G > 0`` = attraction (phase separation) as in TensorLBM
  (physical standard-SC coupling ``G_eff = -G``).
* Forcing: velocity shift (the scheme used by the ``laplace_droplet``
  benchmark), ``u_eq = u + tau F / rho``.
* Periodic streaming.

The equilibrium distribution and the stress tensor of the model are the plain
textbook ones, so this file can be re-derived without reading TensorLBM.
"""

from __future__ import annotations

import numpy as np

# --- D2Q9 lattice -----------------------------------------------------------
C = np.array(
    [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
    dtype=np.int64,
)
W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float64,
)
CX = C[:, 0].astype(np.float64)
CY = C[:, 1].astype(np.float64)
CS2 = 1.0 / 3.0


def psi_exp(rho: np.ndarray, rho0: float = 1.0) -> np.ndarray:
    """Shan-Chen pseudopotential ``psi(rho) = rho0 (1 - exp(-rho/rho0))``."""
    return rho0 * (1.0 - np.exp(-rho / rho0))


def sc_force_standard(rho: np.ndarray, G: float, psi_fn=psi_exp) -> tuple[np.ndarray, np.ndarray]:
    """``F = -G psi(x) sum_i w_i c_i psi(x - c_i)`` (backward gather, periodic).

    ``G > 0`` is attractive in this convention.
    """
    psi = psi_fn(rho)
    fx = np.zeros_like(psi)
    fy = np.zeros_like(psi)
    for i in range(9):
        # psi(x - c_i): roll by +c_i
        psi_m = np.roll(psi, shift=(C[i, 1], C[i, 0]), axis=(0, 1))
        fx += W[i] * CX[i] * psi_m
        fy += W[i] * CY[i] * psi_m
    return -G * psi * fx, -G * psi * fy


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium, shape ``(9, ny, nx)``."""
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    usq = ux * ux + uy * uy
    return W[:, None, None] * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    rho_s = np.maximum(rho, 1e-12)
    ux = (f * CX[:, None, None]).sum(axis=0) / rho_s
    uy = (f * CY[:, None, None]).sum(axis=0) / rho_s
    return rho, ux, uy


def stream(f: np.ndarray) -> np.ndarray:
    """Periodic streaming: ``f_new_i(x) = f_i(x - c_i)``."""
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(f[i], shift=(C[i, 1], C[i, 0]), axis=(0, 1))
    return out


def collide_sc(
    f: np.ndarray,
    G: float = 5.0,
    tau: float = 1.0,
    psi_fn=psi_exp,
    forcing: str = "velocity_shift",
) -> np.ndarray:
    """SCMP BGK collision with the Shan-Chen interaction force."""
    rho, ux, uy = macroscopic(f)
    Fx, Fy = sc_force_standard(rho, G, psi_fn)
    rho_s = np.maximum(rho, 1e-12)
    if forcing == "velocity_shift":
        feq = equilibrium(rho, ux + tau * Fx / rho_s, uy + tau * Fy / rho_s)
    elif forcing == "edm":
        feq_u = equilibrium(rho, ux, uy)
        feq = equilibrium(rho, ux + Fx / rho_s, uy + Fy / rho_s)
        return f - (f - feq_u) / tau + (feq - feq_u)
    elif forcing == "guo":
        # Guo (2002) source term with the half-force corrected physical velocity
        uxp = ux + 0.5 * Fx / rho_s
        uyp = uy + 0.5 * Fy / rho_s
        feq = equilibrium(rho, uxp, uyp)
        cu = CX[:, None, None] * uxp[None] + CY[:, None, None] * uyp[None]
        cf = CX[:, None, None] * Fx[None] + CY[:, None, None] * Fy[None]
        uf = (uxp * Fx + uyp * Fy)[None]
        S = W[:, None, None] * (1.0 - 1.0 / (2.0 * tau)) * ((cf - uf) / CS2 + cu * cf / CS2**2)
        return f - (f - feq) / tau + S
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown forcing {forcing!r}")
    return f - (f - feq) / tau


def eos_pressure(rho: np.ndarray, G_eff: float = -5.0) -> np.ndarray:
    """SC equation of state in the standard (Shan-Chen) sign convention.

    ``p = cs^2 rho + (G_eff cs^2 / 2) psi^2`` — the Maxwell-consistent bulk
    pressure of the pseudopotential model (``G_eff < 0`` = attraction).
    """
    psi = psi_exp(rho)
    return CS2 * rho + 0.5 * G_eff * CS2 * psi * psi


def maxwell_coexistence(G_eff: float = -5.0, tol: float = 1e-12):
    """Maxwell equal-area construction on the SC equation of state.

    Independent *analytic* reference for the coexistence state of the
    pseudopotential model (no lattice solver involved):

        p(rho) = cs^2 rho + (G_eff cs^2 / 2) psi(rho)^2

    find ``p_c`` such that the two outer roots ``rho_v < rho_l`` of
    ``p(rho) = p_c`` satisfy the equal-chemical-potential (equal-area)
    condition

        mu(rho_l) - mu(rho_v) = int_{rho_v}^{rho_l} (dp/drho) / rho drho = 0 .

    A 1-D bisection on ``p_c`` between the two spinodal pressures is used,
    which cannot collapse to the trivial ``rho_v = rho_l`` solution.
    Returns ``(rho_v, rho_l, p_c)``.
    """
    grid = np.linspace(1e-5, 8.0, 400001)
    pgrid = eos_pressure(grid, G_eff)

    def roots_at(pc: float) -> np.ndarray:
        f = pgrid - pc
        idx = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        out = []
        for i in idx:
            lo, hi = grid[i], grid[i + 1]
            flo = f[i]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fm = eos_pressure(np.asarray(mid), G_eff) - pc
                if (fm < 0) == (flo < 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            out.append(0.5 * (lo + hi))
        return np.array(out)

    def dpdrho(rho):
        return CS2 + 0.5 * G_eff * CS2 * 2.0 * psi_exp(rho) * np.exp(-rho)

    def equal_area_residual(pc: float) -> float:
        r = roots_at(pc)
        if r.size < 3:
            return np.nan
        rv, rl = r[0], r[-1]
        g = np.linspace(rv, rl, 8001)
        return float(np.trapezoid(dpdrho(g) / g, g))

    # Spinodal pressures bracket the two-phase window.  dp/drho = 0 gives
    #   cs2 + G_eff cs2 (1-u) u = 0  ->  (1-u)u = -1/G_eff,  u = exp(-rho).
    target = -1.0 / G_eff
    u = np.roots([1.0, -1.0, target])
    rho_sp = sorted(-np.log(u))
    pc_lo, pc_hi = sorted([float(eos_pressure(np.asarray(rho_sp[0]), G_eff)),
                           float(eos_pressure(np.asarray(rho_sp[1]), G_eff))])
    lo, hi = pc_lo + 1e-9, pc_hi - 1e-9
    f_lo = equal_area_residual(lo)
    f_hi = equal_area_residual(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        # fall back to a coarse scan for a sign change
        lo_s = hi_s = None
        prev_pc, prev_f = None, None
        for pc in np.linspace(pc_lo + 1e-6, pc_hi - 1e-6, 400):
            fv = equal_area_residual(pc)
            if prev_f is not None and np.isfinite(fv) and np.isfinite(prev_f) and fv * prev_f < 0:
                lo_s, hi_s = prev_pc, pc
                break
            prev_pc, prev_f = pc, fv
        if lo_s is None:
            return float("nan"), float("nan"), float("nan")
        lo, hi = lo_s, hi_s
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        f_mid = equal_area_residual(mid)
        if not np.isfinite(f_mid):
            break
        if (f_mid < 0) == (equal_area_residual(lo) < 0):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    pc = 0.5 * (lo + hi)
    r = roots_at(pc)
    return float(r[0]), float(r[-1]), float(pc)
