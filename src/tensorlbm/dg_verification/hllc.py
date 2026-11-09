"""HLLC interface flux + Zhang-Shu positivity-preserving limiter (IN65-N04).

Why this module exists
----------------------
The DG verification face shipped by IN70-N01 used a *local Lax--Friedrichs*
(Rusanov) interface flux for the 1-D Euler equations and reported two open
issues: the ideal-gas Sod tube was pinned at a 25 % "shock position error" and
the isothermal P2 run degraded at ``t = 400``.  Rusanov smears a contact
discontinuity over ``O(|S_R - S_L| / (S_M-ish))`` cells -- the contact is the
dominant density jump of the Sod problem, so a smeared contact also corrupts the
*measurement* of the shock position (see :mod:`...frontier`, which locates the
shock on the pressure, the field that is continuous across a contact).

This module adds the two missing ingredients of a shock-capturing Euler DG:

* :func:`hllc_flux` -- Toro's HLLC approximate Riemann solver (Harten-Lax-van
  Leer with the **contact** restored).  Three waves: ``S_L < S_M < S_R``; the
  middle state ``U*_K`` carries the contact.  Conservative by construction (it
  is a single-valued numerical flux), so the discrete mass/momentum/energy
  ledger is exact up to round-off.  Both the ideal-gas closure
  (``p = (gamma-1)(E - m^2/2rho)``) and the isothermal closure of the D2Q9
  lattice (``p = cs^2 rho``, ``cs^2 = 1/3``) are supported -- the latter is what
  makes the DG problem comparable to the verified LBM Sod tube.
* :func:`zhang_shu_limit` -- the positivity-preserving limiter of Zhang & Shu
  (2010): a troubled cell is scaled *towards its own cell average*,
  ``U <- Ubar + theta (U - Ubar)``, with ``theta`` the largest admissible value
  in ``[0, 1]``.  Because the average is untouched, the limiter is exactly
  conservative (this is the difference to the cruder "replace the cell by its
  mean" repair that the earlier solver used, which throws away the whole
  element, not just the offending part).

Admissibility theory (why the bisection below is valid)
------------------------------------------------------
For the ideal gas, ``p(theta)`` is the *concave* quadratic

    p(theta) = (gamma-1) [ E_theta - m_theta^2 / (2 rho_theta) ] ,
    E_theta = Ebar + theta dE,  m_theta = mbar + theta dm,  rho_theta = rhobar + theta drho

(concavity of ``p`` in ``U`` is the standard fact behind the admissible-set
argument of Zhang-Shu).  The admissible set ``{theta : rho_theta > 0, p_theta > 0}
intersected with ``[0,1]`` is therefore an interval containing ``0`` (``0`` is
admissible because the cell average of the *previous* stage is admissible), so a
bisection on ``theta`` converges to the largest admissible scaling.  For the
isothermal closure ``p`` is proportional to ``rho`` and the condition collapses
to ``rho_theta > 0``.

References
----------
* E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  3rd ed., Springer 2009, §10.4 (HLLC) and §9.5.2 (pressure-based wave speeds).
* X. Zhang, C.-W. Shu, "Positivity-preserving high order discontinuous Galerkin
  schemes for compressible Euler equations with source terms",
  J. Comput. Phys. 230 (2011) 1238-1248.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "AdmissibleError",
    "hllc_flux",
    "admissible_nodes",
    "zhang_shu_limit",
    "rusanov_flux",
]


class AdmissibleError(RuntimeError):
    """Raised when a cell average itself violates positivity (should not happen)."""


# ---------------------------------------------------------------------------
# primitive / flux helpers
# ---------------------------------------------------------------------------


def _primitive(U: np.ndarray, gamma: float | None, cs2: float) -> tuple:
    rho = U[0]
    m = U[1]
    u = m / np.maximum(rho, 1e-300)
    if gamma is None:  # isothermal closure p = cs2 rho
        p = cs2 * rho
    else:
        p = (gamma - 1.0) * (U[2] - 0.5 * m * u)
    return rho, u, p


def _physical_flux(U: np.ndarray, gamma: float | None, cs2: float) -> np.ndarray:
    rho, u, p = _primitive(U, gamma, cs2)
    F = np.empty_like(U)
    F[0] = U[1]
    F[1] = U[1] * u + p
    if gamma is not None:
        F[2] = u * (U[2] + p)
    return F


def _sound_speed(rho, p, gamma, cs2):
    rho = np.maximum(rho, 1e-300)
    if gamma is None:
        return np.full_like(rho, np.sqrt(cs2))
    return np.sqrt(gamma * np.maximum(p, 0.0) / rho)


def rusanov_flux(UL, UR, gamma=None, cs2=1.0 / 3.0):
    """Local Lax-Friedrichs flux (kept for A/B comparison against HLLC)."""
    FL = _physical_flux(UL, gamma, cs2)
    FR = _physical_flux(UR, gamma, cs2)
    rhoL, uL, pL = _primitive(UL, gamma, cs2)
    rhoR, uR, pR = _primitive(UR, gamma, cs2)
    aL = np.abs(uL) + _sound_speed(rhoL, pL, gamma, cs2)
    aR = np.abs(uR) + _sound_speed(rhoR, pR, gamma, cs2)
    a = np.maximum(aL, aR)
    return 0.5 * (FL + FR) - 0.5 * a * (UR - UL)


def hllc_flux(
    UL: np.ndarray,
    UR: np.ndarray,
    gamma: float | None = 1.4,
    cs2: float = 1.0 / 3.0,
    wave_speed: str = "toro",
) -> np.ndarray:
    """HLLC numerical flux for the 1-D Euler equations (Toro 2009, §10.4).

    Args:
        UL, UR: conserved states on both sides of a face, shape ``(n_comp, ...)``
            with ``n_comp = 3`` (ideal gas, ``gamma`` given) or ``2``
            (isothermal, ``gamma=None``, pressure ``cs2*rho``).
        gamma: ratio of specific heats, or ``None`` for the isothermal closure.
        cs2: isothermal sound speed squared (used only when ``gamma is None``).
        wave_speed: ``"toro"`` -- pressure-based (PVRS + shock-aware) estimates,
            which *see* the shock and are the ones used for the Sod runs;
            ``"davis"`` -- min/max acoustic estimates
            ``S_L = min(u_L - c_L, u_R - c_R)``, ``S_R = max(u_L + c_L, u_R + c_R)``;
            ``"einfeldt"`` -- Roe-averaged acoustic estimates.

    Returns:
        Array of shape ``(n_comp, ...)`` -- the interface flux.
    """
    UL = np.asarray(UL, dtype=np.float64)
    UR = np.asarray(UR, dtype=np.float64)
    rhoL, uL, pL = _primitive(UL, gamma, cs2)
    rhoR, uR, pR = _primitive(UR, gamma, cs2)
    cL = _sound_speed(rhoL, pL, gamma, cs2)
    cR = _sound_speed(rhoR, pR, gamma, cs2)
    FL = _physical_flux(UL, gamma, cs2)
    FR = _physical_flux(UR, gamma, cs2)

    if wave_speed == "davis":
        SL = np.minimum(uL - cL, uR - cR)
        SR = np.maximum(uL + cL, uR + cR)
    elif wave_speed == "einfeldt":
        wr = np.sqrt(np.maximum(rhoL, 0.0))
        wl = np.sqrt(np.maximum(rhoR, 0.0))
        denom = np.maximum(wr + wl, 1e-300)
        u_t = (wr * uL + wl * uR) / denom
        c_t = 0.5 * (cL + cR)
        SL = np.minimum(uL - cL, u_t - c_t)
        SR = np.maximum(uR + cR, u_t + c_t)
    elif wave_speed == "toro":
        # Toro 2009, §9.5.2: PVRS plus the two-rarefaction estimate, take the
        # smaller (PVRS over-estimates strong shocks, the two-rarefaction
        # estimate is the accurate one there; for the Sod jump PVRS gives
        # p* = 0.55 while two-rarefaction gives 0.307 against the exact
        # p* = 0.3031).
        rho_bar = 0.5 * (rhoL + rhoR)
        c_bar = 0.5 * (cL + cR)
        p_pvrs = 0.5 * (pL + pR) - 0.5 * (uR - uL) * rho_bar * c_bar
        if gamma is None:
            # isothermal: the two-rarefaction form degenerates (z = 0); the
            # linearised estimate is used (documented limitation, see docs).
            p_star = np.maximum(p_pvrs, 0.0)
        else:
            z = (gamma - 1.0) / (2.0 * gamma)
            p_tr = (
                (cL + cR - 0.5 * (gamma - 1.0) * (uR - uL))
                / (cL / np.maximum(pL, 1e-300) ** z + cR / np.maximum(pR, 1e-300) ** z)
            ) ** (1.0 / z)
            p_star = np.maximum(np.minimum(p_pvrs, p_tr), 0.0)
        gam = 1.0 if gamma is None else float(gamma)  # isothermal ~ isentropic limit
        # q_K = 1 for a rarefaction, sqrt(1 + (g+1)/(2g)(p*/p_K - 1)) for a shock
        def _q(pK):
            pK = np.maximum(pK, 1e-300)
            q_shock = np.sqrt(np.maximum(1.0 + (gam + 1.0) / (2.0 * gam) * (p_star / pK - 1.0), 1.0))
            return np.where(p_star > pK, q_shock, 1.0)

        SL = uL - cL * _q(pL)
        SR = uR + cR * _q(pR)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown wave_speed {wave_speed!r}")

    # HLL middle wave speed (contact)
    denom = rhoL * (SL - uL) - rhoR * (SR - uR)
    # denom < 0 strictly for any admissible pair of estimates; guard for round-off
    bad = np.abs(denom) < 1e-300
    denom = np.where(bad, -1e-300, denom)
    SM = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / denom

    # star states
    dL = SL - uL
    dR = SR - uR
    dL = np.where(np.abs(dL) < 1e-300, -1e-300, dL)
    dR = np.where(np.abs(dR) < 1e-300, 1e-300, dR)
    facL = rhoL * dL / (SL - SM)
    facR = rhoR * dR / (SR - SM)
    UstL = np.empty_like(UL)
    UstR = np.empty_like(UR)
    UstL[0] = facL
    UstL[1] = facL * SM
    UstR[0] = facR
    UstR[1] = facR * SM
    if gamma is not None:
        UstL[2] = facL * (UL[2] / rhoL + (SM - uL) * (SM + pL / (rhoL * dL)))
        UstR[2] = facR * (UR[2] / rhoR + (SM - uR) * (SM + pR / (rhoR * dR)))

    F = np.where(
        (SL >= 0.0)[None, ...],
        FL,
        np.where(
            (SM >= 0.0)[None, ...],
            FL + SL[None, ...] * (UstL - UL),
            np.where((SR >= 0.0)[None, ...], FR + SR[None, ...] * (UstR - UR), FR),
        ),
    )
    if np.any(bad):
        # degenerate acoustic state: fall back to Rusanov on those faces
        Fr = rusanov_flux(UL, UR, gamma, cs2)
        F = np.where(bad[None, ...], Fr, F)
    return F


# ---------------------------------------------------------------------------
# positivity-preserving limiter
# ---------------------------------------------------------------------------


def admissible_nodes(U: np.ndarray, gamma=None, cs2=1.0 / 3.0, eps: float = 1e-13):
    """Boolean array: ``rho > eps`` and (ideal gas) ``p > eps`` at every node."""
    rho, u, p = _primitive(U, gamma, cs2)
    ok = rho > eps
    if gamma is not None:
        ok = ok & (p > eps)
    return ok


def zhang_shu_limit(
    U: np.ndarray,
    gamma: float | None = 1.4,
    cs2: float = 1.0 / 3.0,
    eps: float = 1e-13,
    n_bisect: int = 40,
):
    """Zhang-Shu positivity-preserving scaling limiter (conservation exact).

    Args:
        U: nodal conserved state ``(n_comp, n_cells, n_node)``.
        gamma: ratio of specific heats, ``None`` for the isothermal closure.
        cs2: isothermal sound speed squared.
        eps: positivity floor.
        n_bisect: bisection iterations on ``theta``.

    Returns:
        ``(U_limited, theta)`` where ``theta[c]`` is the scaling used in cell
        ``c`` (``1`` = untouched).  The cell mean of every component is exactly
        preserved: ``mean(Ubar + theta (U - Ubar)) == Ubar``.

    Raises:
        AdmissibleError: if a cell *average* is not admissible -- that would mean
            the update itself was not positivity preserving (CFL too large).
    """
    U = np.asarray(U, dtype=np.float64)
    Ubar = U.mean(axis=-1, keepdims=True)
    if not np.all(admissible_nodes(Ubar, gamma, cs2, eps)):
        raise AdmissibleError(
            "non-admissible cell average: the scheme lost positivity "
            "(reduce the CFL number)"
        )
    d = U - Ubar
    if np.all(admissible_nodes(U, gamma, cs2, eps)):
        return U, np.ones(U.shape[1])

    n_cells = U.shape[1]
    lo = np.zeros(n_cells)
    hi = np.ones(n_cells)
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo + hi)
        trial = Ubar + mid[None, :, None] * d
        ok = np.all(admissible_nodes(trial, gamma, cs2, eps), axis=-1)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    theta = lo
    out = Ubar + theta[None, :, None] * d
    # a hair of margin so the next stage cannot sit exactly on the boundary
    out = np.where(np.all(admissible_nodes(out, gamma, cs2, eps), axis=-1)[None, :, None], out,
                   Ubar * np.ones_like(U))
    return out, theta