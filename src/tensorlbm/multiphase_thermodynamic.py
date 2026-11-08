"""Thermodynamic consistency verification for pseudopotential multiphase LBM.

Provides:
1. EOS implementations (van der Waals, Carnahan-Starling) with analytical derivatives
2. Maxwell equal-area construction via Brent's method
3. Mechanical pressure from the SCMP lattice scheme
4. Surface tension estimation from the mechanical EOS

References
----------
Shan & Chen (1993) Phys. Rev. E 47 1815
Yuan & Schaefer (2006) Phys. Fluids 18 042101
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import brentq


# ============================================================================
# Equation of State implementations
# ============================================================================

class VDWEOS:
    """van der Waals equation of state.

    p(rho, T) = rho*R*T/(1 - b*rho) - a*rho^2

    Critical point:
        T_c = 8a/(27bR),  rho_c = 1/(3b),  p_c = a/(27b^2)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0 / 3.0, R: float = 1.0):
        self.a = a
        self.b = b
        self.R = R

    @property
    def T_c(self) -> float:
        return 8.0 * self.a / (27.0 * self.b * self.R)

    @property
    def rho_c(self) -> float:
        return 1.0 / (3.0 * self.b)

    @property
    def p_c(self) -> float:
        return self.a / (27.0 * self.b ** 2)

    def pressure(self, rho, T: float):
        """p(rho, T) = rho*R*T/(1-b*rho) - a*rho^2."""
        rho = np.asarray(rho, dtype=np.float64)
        denom = np.clip(1.0 - self.b * rho, 1e-15, None)
        return self.R * T * rho / denom - self.a * rho ** 2

    def dp_drho(self, rho, T: float):
        """dp/drho at constant T."""
        rho = np.asarray(rho, dtype=np.float64)
        denom = np.clip(1.0 - self.b * rho, 1e-15, None)
        return self.R * T / (denom ** 2) - 2.0 * self.a * rho


class CSEOS:
    """Carnahan-Starling equation of state (Yuan & Schaefer 2006).

    p(rho, T) = rho*R*T*(1 + eta + eta^2 - eta^3)/(1 - eta)^3 - a*rho^2
    where eta = b*rho/4.

    Default: a=1, b=4, R=1 -> T_c ~ 0.09433, rho_c ~ 0.13044
    """

    def __init__(self, a: float = 1.0, b: float = 4.0, R: float = 1.0):
        self.a = a
        self.b = b
        self.R = R
        self._T_c: Optional[float] = None
        self._rho_c: Optional[float] = None

    @property
    def T_c(self) -> float:
        if self._T_c is None:
            self._compute_critical()
        return self._T_c

    @property
    def rho_c(self) -> float:
        if self._rho_c is None:
            self._compute_critical()
        return self._rho_c

    def _compute_critical(self):
        """Find critical point from dp/drho = d2p/drho2 = 0."""
        from scipy.optimize import fsolve

        def equations(params):
            rho_c, T_c = params
            if rho_c <= 0 or T_c <= 0:
                return [1e10, 1e10]
            eta_c = self.b * rho_c / 4.0
            if eta_c >= 1.0 or eta_c <= 0:
                return [1e10, 1e10]
            drho = 1e-7
            p0 = float(self.pressure(rho_c, T_c))
            p1 = float(self.pressure(rho_c + drho, T_c))
            p2 = float(self.pressure(rho_c + 2 * drho, T_c))
            dp = (p1 - p0) / drho
            d2p = (p2 - 2 * p1 + p0) / drho ** 2
            return [dp, d2p]

        sol, info, ier, msg = fsolve(equations, [0.13, 0.094], full_output=True)
        if ier == 1 and sol[0] > 0 and sol[1] > 0:
            self._rho_c = float(sol[0])
            self._T_c = float(sol[1])
        else:
            self._rho_c = 0.13044
            self._T_c = 0.09433

    def pressure(self, rho, T: float):
        """CS pressure."""
        rho = np.asarray(rho, dtype=np.float64)
        eta = self.b * rho / 4.0
        eta = np.clip(eta, 0.0, 1.0 - 1e-12)
        one_m_eta = 1.0 - eta
        p_rep = self.R * T * rho * (1.0 + eta + eta ** 2 - eta ** 3) / (one_m_eta ** 3)
        return p_rep - self.a * rho ** 2

    def dp_drho(self, rho, T: float):
        """Analytical dp/drho."""
        rho = np.asarray(rho, dtype=np.float64)
        eta = self.b * rho / 4.0
        eta = np.clip(eta, 1e-15, 1.0 - 1e-12)
        one_m_eta = 1.0 - eta
        f = 1.0 + eta + eta ** 2 - eta ** 3
        fp = 1.0 + 2.0 * eta - 3.0 * eta ** 2
        b4 = self.b / 4.0
        term1 = f / one_m_eta ** 3
        term2 = rho * fp * b4 / one_m_eta ** 3
        term3 = rho * f * 3.0 * b4 / one_m_eta ** 4
        return self.R * T * (term1 + term2 + term3) - 2.0 * self.a * rho


# ============================================================================
# Maxwell equal-area construction
# ============================================================================

@dataclass
class MaxwellCoexistence:
    """Result of Maxwell equal-area construction.

    Attributes
    ----------
    T : float        Absolute temperature.
    T_r : float      Reduced temperature T/T_c.
    rho_v : float    Vapor coexistence density.
    rho_l : float    Liquid coexistence density.
    p_sat : float    Saturation pressure.
    area_residual : float   |integral(p - p_sat) dv| (should be ~0).
    """
    T: float
    T_r: float
    rho_v: float
    rho_l: float
    p_sat: float
    area_residual: float


def maxwell_construction(
    eos,
    T: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> MaxwellCoexistence:
    """Maxwell equal-area construction for a given EOS and temperature.

    Finds (rho_v, rho_l, p_sat) such that:
      1. p(rho_v, T) = p(rho_l, T) = p_sat
      2. integral_{v_v}^{v_l} [p(1/v, T) - p_sat] dv = 0

    Uses Brent's method with tolerance ``tol``.
    """
    T_c = eos.T_c
    rho_c = eos.rho_c
    T_r = T / T_c

    if T_r >= 1.0:
        raise ValueError(f"T_r = {T_r:.6f} >= 1.0: no phase separation above T_c")
    if T_r < 0.3:
        raise ValueError(f"T_r = {T_r:.6f} < 0.3: too far below T_c for reliable construction")

    # Find spinodal densities where dp/drho = 0
    drho = 1e-8

    def dp_func(rho_val):
        if hasattr(eos, 'dp_drho'):
            return float(eos.dp_drho(rho_val, T))
        return (float(eos.pressure(rho_val + drho, T)) -
                float(eos.pressure(rho_val - drho, T))) / (2 * drho)

    # Vapor spinodal: search below rho_c
    rho_sv = None
    for rho_test in np.linspace(0.01 * rho_c, 0.98 * rho_c, 500):
        if dp_func(rho_test) < 0:
            rho_lo = max(1e-8, rho_test - 0.005 * rho_c)
            rho_hi = min(rho_test + 0.005 * rho_c, 0.99 * rho_c)
            if dp_func(rho_lo) * dp_func(rho_hi) < 0:
                rho_sv = brentq(dp_func, rho_lo, rho_hi, xtol=tol, maxiter=max_iter)
            else:
                rho_sv = rho_test
            break

    # Liquid spinodal: search above rho_c
    rho_sl = None
    rho_max = 0.95 / eos.b if hasattr(eos, 'b') else 3.0 * rho_c
    for rho_test in np.linspace(1.02 * rho_c, rho_max, 500):
        if dp_func(rho_test) < 0:
            rho_lo = max(rho_test - 0.005 * rho_c, 1.01 * rho_c)
            rho_hi = min(rho_test + 0.005 * rho_c, rho_max)
            if dp_func(rho_lo) * dp_func(rho_hi) < 0:
                rho_sl = brentq(dp_func, rho_lo, rho_hi, xtol=tol, maxiter=max_iter)
            else:
                rho_sl = rho_test
            break

    if rho_sv is None or rho_sl is None:
        raise ValueError(f"Could not find spinodals at T_r={T_r:.4f}")

    # For a trial p_sat, find coexistence densities on each branch
    def coexistence_densities(p_sat: float):
        """Find rho_v < rho_sv and rho_l > rho_sl where p(rho,T) = p_sat."""
        # Vapor branch
        def f_v(rho_val):
            return float(eos.pressure(rho_val, T)) - p_sat

        rho_v = None
        # p(rho->0) -> 0, p(rho_sv) = p_sv
        p_sv = float(eos.pressure(rho_sv, T))
        if 0 < p_sat < p_sv:
            try:
                rho_v = brentq(f_v, 1e-10, rho_sv * 0.9999, xtol=tol, maxiter=max_iter)
            except ValueError:
                pass

        # Liquid branch
        def f_l(rho_val):
            return float(eos.pressure(rho_val, T)) - p_sat

        rho_l = None
        p_sl = float(eos.pressure(rho_sl, T))
        if p_sat > min(p_sv, p_sl) and p_sat < float(eos.pressure(rho_max, T)):
            try:
                rho_l = brentq(f_l, rho_sl * 1.0001, rho_max, xtol=tol, maxiter=max_iter)
            except ValueError:
                pass

        return rho_v, rho_l

    def equal_area_integral(p_sat: float) -> float:
        """Compute integral_{v_l}^{v_v} (p - p_sat) dv."""
        rho_v, rho_l = coexistence_densities(p_sat)
        if rho_v is None or rho_l is None:
            return float('nan')
        v_v = 1.0 / rho_v
        v_l = 1.0 / rho_l
        n_quad = 500
        v_arr = np.linspace(v_l, v_v, n_quad)
        rho_arr = 1.0 / v_arr
        p_arr = np.array([float(eos.pressure(r, T)) for r in rho_arr])
        integrand = p_arr - p_sat
        return float(np.trapezoid(integrand, v_arr))

    # Bracket p_sat
    p_sv = float(eos.pressure(rho_sv, T))
    p_sl = float(eos.pressure(rho_sl, T))
    p_lo = min(p_sv, p_sl) * 0.5
    p_hi = max(p_sv, p_sl) * 1.2

    # Scan for bracket
    p_scan = np.linspace(p_lo, p_hi, 200)
    res_scan = []
    for p in p_scan:
        r = equal_area_integral(p)
        res_scan.append(r)
    res_scan = np.array(res_scan)

    valid = ~np.isnan(res_scan)
    if not np.any(valid):
        raise ValueError("Cannot evaluate equal-area integral in search range")

    p_valid = p_scan[valid]
    r_valid = res_scan[valid]

    sign_changes = np.where(np.diff(np.sign(r_valid)))[0]
    if len(sign_changes) == 0:
        raise ValueError("Cannot bracket equal-area root")

    idx = sign_changes[0]
    p_bracket_lo = p_valid[idx]
    p_bracket_hi = p_valid[idx + 1]

    p_sat = brentq(equal_area_integral, p_bracket_lo, p_bracket_hi,
                   xtol=tol, maxiter=max_iter)

    rho_v, rho_l = coexistence_densities(p_sat)
    if rho_v is None or rho_l is None:
        raise ValueError(f"Maxwell construction failed at T_r={T_r:.4f}")

    area_res = abs(equal_area_integral(p_sat))

    return MaxwellCoexistence(
        T=T, T_r=T_r,
        rho_v=float(rho_v), rho_l=float(rho_l),
        p_sat=float(p_sat), area_residual=float(area_res),
    )


def maxwell_vdw(T_r: float, a: float = 1.0, b: float = 1.0 / 3.0,
                R: float = 1.0, tol: float = 1e-12) -> MaxwellCoexistence:
    """Maxwell construction for vdW EOS at reduced temperature T_r."""
    eos = VDWEOS(a=a, b=b, R=R)
    T = T_r * eos.T_c
    return maxwell_construction(eos, T, tol=tol)


def maxwell_cs(T_r: float, a: float = 1.0, b: float = 4.0,
               R: float = 1.0, tol: float = 1e-12) -> MaxwellCoexistence:
    """Maxwell construction for CS EOS at reduced temperature T_r."""
    eos = CSEOS(a=a, b=b, R=R)
    T = T_r * eos.T_c
    return maxwell_construction(eos, T, tol=tol)


# ============================================================================
# SCMP mechanical pressure
# ============================================================================

def scmp_mechanical_pressure(
    rho,
    G: float,
    psi_fn: Callable,
    cs2: float = 1.0 / 3.0,
):
    """Mechanical pressure of the SCMP lattice scheme.

    p_mech(rho) = rho*cs^2 - (G*cs^2/2)*psi^2(rho)

    With corrected-sign psi and G=1, this reproduces the target EOS.
    """
    import torch
    if not isinstance(rho, torch.Tensor):
        rho_t = torch.tensor(rho, dtype=torch.float64)
    else:
        rho_t = rho.double()
    psi_val = psi_fn(rho_t)
    p = rho_t * cs2 - 0.5 * G * cs2 * psi_val ** 2
    if isinstance(rho, (float, int)):
        return float(p.item())
    if isinstance(rho, np.ndarray):
        return p.numpy()
    return p
