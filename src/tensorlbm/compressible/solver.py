"""Compressible flow solver for shock wave verification.

Implements:
- 1D Godunov solver with exact Riemann solver for normal shock verification
- Rankine-Hugoniot exact solutions
- Verification utilities

References
----------
- Toro, E.F., "Riemann Solvers and Numerical Methods for Fluid Dynamics", 3rd ed.
- Anderson, J.D., "Modern Compressible Flow", 3rd ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================================
# Rankine-Hugoniot exact solutions
# ============================================================================

@dataclass(frozen=True)
class NormalShockResult:
    M2: float
    rho_ratio: float
    p_ratio: float
    T_ratio: float
    p0_ratio: float


def normal_shock_relations(M1: float, gamma: float = 1.4) -> NormalShockResult:
    """Compute normal shock relations."""
    if M1 <= 1.0:
        raise ValueError(f"M1 must be > 1, got {M1}")
    M1_sq = M1 * M1
    M2_sq = (1.0 + 0.5 * (gamma - 1.0) * M1_sq) / (gamma * M1_sq - 0.5 * (gamma - 1.0))
    M2 = math.sqrt(M2_sq)
    rho_ratio = ((gamma + 1.0) * M1_sq) / (2.0 + (gamma - 1.0) * M1_sq)
    p_ratio = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (M1_sq - 1.0)
    T_ratio = p_ratio / rho_ratio
    term1 = ((gamma + 1.0) * M1_sq) / (2.0 + (gamma - 1.0) * M1_sq)
    term2 = (gamma + 1.0) / (2.0 * gamma * M1_sq - (gamma - 1.0))
    p0_ratio = (term1 ** (gamma / (gamma - 1.0))) * (term2 ** (1.0 / (gamma - 1.0)))
    return NormalShockResult(M2=M2, rho_ratio=rho_ratio, p_ratio=p_ratio, T_ratio=T_ratio, p0_ratio=p0_ratio)


def oblique_shock_relations(M1: float, theta: float, gamma: float = 1.4):
    """Compute oblique shock relations (weak shock solution)."""
    if M1 <= 1.0:
        raise ValueError(f"M1 must be > 1, got {M1}")
    theta_rad = theta if abs(theta) < 0.1 else math.radians(theta) if theta > 1 else theta
    beta = math.asin(1.0 / M1) + 0.1
    for _ in range(100):
        sin_b = math.sin(beta)
        cos_b = math.cos(beta)
        M1n_sq = M1 * M1 * sin_b * sin_b
        num = M1n_sq - 1.0
        den = M1 * M1 * (gamma + math.cos(2 * beta)) + 2.0
        if den <= 0:
            beta += 0.01
            continue
        tan_theta_calc = 2.0 * cos_b * num / den
        f = math.tan(theta_rad) - tan_theta_calc
        if abs(f) < 1e-12:
            break
        db = 1e-8
        sin_bd = math.sin(beta + db)
        cos_bd = math.cos(beta + db)
        cos_2bd = math.cos(2 * (beta + db))
        M1n_sq_d = M1 * M1 * sin_bd * sin_bd
        num_d = M1n_sq_d - 1.0
        den_d = M1 * M1 * (gamma + cos_2bd) + 2.0
        tan_theta_d = 2.0 * cos_bd * num_d / den_d if den_d > 0 else tan_theta_calc
        df = -(tan_theta_d - tan_theta_calc) / db
        if abs(df) < 1e-15:
            beta += 0.01
            continue
        beta -= f / df
        beta = max(math.asin(1.0 / M1) + 0.001, min(math.pi / 2 - 0.001, beta))

    M1n = M1 * math.sin(beta)
    ns = normal_shock_relations(M1n, gamma)
    M2n = ns.M2
    M2 = M2n / math.sin(beta - theta_rad)
    return type('ObliqueShockResult', (), {
        'beta': beta, 'theta': theta_rad, 'M1n': M1n, 'M2n': M2n, 'M2': M2,
        'rho_ratio': ns.rho_ratio, 'p_ratio': ns.p_ratio, 'T_ratio': ns.T_ratio, 'p0_ratio': ns.p0_ratio
    })()


# ============================================================================
# 1D Godunov solver with exact Riemann solver
# ============================================================================

def _p_star(p_L, p_R, rho_L, rho_R, u_L, u_R, gamma):
    """Compute pressure at star region using Newton iteration."""
    def f(p):
        A_L = 2.0 / ((gamma + 1.0) * rho_L)
        B_L = (gamma - 1.0) / (gamma + 1.0) * p_L
        A_R = 2.0 / ((gamma + 1.0) * rho_R)
        B_R = (gamma - 1.0) / (gamma + 1.0) * p_R

        if p > p_L:
            f_L = (p - p_L) / math.sqrt((p + B_L) * A_L)
        else:
            c_L = math.sqrt(gamma * p_L / rho_L)
            f_L = 2.0 * c_L / (gamma - 1.0) * ((p / p_L) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)

        if p > p_R:
            f_R = (p - p_R) / math.sqrt((p + B_R) * A_R)
        else:
            c_R = math.sqrt(gamma * p_R / rho_R)
            f_R = 2.0 * c_R / (gamma - 1.0) * ((p / p_R) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)

        return f_L + f_R + u_R - u_L

    p = 0.5 * (p_L + p_R)
    for _ in range(50):
        fp = f(p)
        if abs(fp) < 1e-10:
            break
        dp = 1e-6 * p
        fp_plus = f(p + dp)
        dfdp = (fp_plus - fp) / dp
        if abs(dfdp) < 1e-15:
            p *= 1.1
            continue
        p -= fp / dfdp
        p = max(p, 1e-10)
    return p


def _riemann_solve_1d(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma=1.4):
    """Exact Riemann solver for 1D Euler equations.

    Returns flux at the interface.
    """
    p_star = _p_star(p_L, p_R, rho_L, rho_R, u_L, u_R, gamma)

    A_L = 2.0 / ((gamma + 1.0) * rho_L)
    B_L = (gamma - 1.0) / (gamma + 1.0) * p_L
    A_R = 2.0 / ((gamma + 1.0) * rho_R)
    B_R = (gamma - 1.0) / (gamma + 1.0) * p_R

    if p_star > p_L:
        u_star = u_L - (p_star - p_L) / math.sqrt((p_star + B_L) * A_L)
        rho_star_L = rho_L * (p_star / p_L + B_L / p_L) / (B_L / p_L * p_star / p_L + 1.0)
    else:
        c_L = math.sqrt(gamma * p_L / rho_L)
        u_star = u_L - 2.0 * c_L / (gamma - 1.0) * ((p_star / p_L) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)

    if p_star > p_R:
        rho_star_R = rho_R * (p_star / p_R + B_R / p_R) / (B_R / p_R * p_star / p_R + 1.0)
    else:
        rho_star_R = rho_R * (p_star / p_R) ** (1.0 / gamma)

    # Determine wave structure and sample solution at x/t = 0
    S_L = u_L - math.sqrt(gamma * p_L / rho_L) * math.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / p_L + (gamma - 1.0) / (2.0 * gamma)) if p_star > p_L else u_L - math.sqrt(gamma * p_L / rho_L) * (p_star / p_L) ** ((gamma - 1.0) / (2.0 * gamma))
    S_R = u_R + math.sqrt(gamma * p_R / rho_R) * math.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / p_R + (gamma - 1.0) / (2.0 * gamma)) if p_star > p_R else u_R + math.sqrt(gamma * p_R / rho_R) * (p_star / p_R) ** ((gamma - 1.0) / (2.0 * gamma))
    S_star = u_star

    if S_L >= 0:
        rho_s, u_s, p_s = rho_L, u_L, p_L
    elif S_star >= 0:
        rho_s, u_s, p_s = rho_star_L, u_star, p_star
    elif S_R >= 0:
        rho_s, u_s, p_s = rho_star_R, u_star, p_star
    else:
        rho_s, u_s, p_s = rho_R, u_R, p_R

    E_s = p_s / (gamma - 1.0) + 0.5 * rho_s * u_s * u_s
    return rho_s * u_s, rho_s * u_s * u_s + p_s, u_s * (E_s + p_s)


class GodunovSolver1D:
    """1D Godunov solver for compressible Euler equations."""

    def __init__(self, nx: int, gamma: float = 1.4, dx: float = 1.0, cfl: float = 0.4):
        self.nx = nx
        self.gamma = gamma
        self.dx = dx
        self.cfl = cfl
        self.rho = np.ones(nx, dtype=np.float64)
        self.u = np.zeros(nx, dtype=np.float64)
        self.p = np.ones(nx, dtype=np.float64)

    def _compute_dt(self) -> float:
        c = np.sqrt(np.maximum(self.gamma * self.p / np.maximum(self.rho, 1e-12), 1e-12))
        max_speed = np.max(np.abs(self.u) + c)
        return self.cfl * self.dx / max_speed if max_speed > 0 else self.dx

    def initialize_normal_shock(self, M1: float, gamma: float = 1.4, shock_pos: float = 0.5):
        rh = normal_shock_relations(M1, gamma)
        rho1, p1 = 1.0, 1.0
        cs1 = math.sqrt(gamma * p1 / rho1)
        ux1 = M1 * cs1
        rho2 = rho1 * rh.rho_ratio
        p2 = p1 * rh.p_ratio
        cs2 = math.sqrt(gamma * p2 / rho2)
        ux2 = rh.M2 * cs2
        nx_mid = int(shock_pos * self.nx)
        self.rho[:nx_mid] = rho1
        self.rho[nx_mid:] = rho2
        self.u[:nx_mid] = ux1
        self.u[nx_mid:] = ux2
        self.p[:nx_mid] = p1
        self.p[nx_mid:] = p2

    def step(self):
        dt = self._compute_dt()
        flux = np.zeros((3, self.nx + 1))

        for i in range(self.nx + 1):
            if i == 0:
                rho_L, u_L, p_L = self.rho[0], self.u[0], self.p[0]
            else:
                rho_L, u_L, p_L = self.rho[i - 1], self.u[i - 1], self.p[i - 1]

            if i == self.nx:
                rho_R, u_R, p_R = self.rho[-1], self.u[-1], self.p[-1]
            else:
                rho_R, u_R, p_R = self.rho[i], self.u[i], self.p[i]

            f_rho, f_mom, f_E = _riemann_solve_1d(rho_L, u_L, p_L, rho_R, u_R, p_R, self.gamma)
            flux[0, i] = f_rho
            flux[1, i] = f_mom
            flux[2, i] = f_E

        dU = np.zeros((3, self.nx))
        dU[0] = -dt / self.dx * (flux[0, 1:] - flux[0, :-1])
        dU[1] = -dt / self.dx * (flux[1, 1:] - flux[1, :-1])
        dU[2] = -dt / self.dx * (flux[2, 1:] - flux[2, :-1])

        # Update conservative variables
        U = np.zeros((3, self.nx))
        U[0] = self.rho
        U[1] = self.rho * self.u
        U[2] = self.p / (self.gamma - 1.0) + 0.5 * self.rho * self.u ** 2
        U += dU

        # Convert back to primitive
        self.rho = np.maximum(U[0], 1e-12)
        self.u = U[1] / self.rho
        self.p = np.maximum((self.gamma - 1.0) * (U[2] - 0.5 * self.rho * self.u ** 2), 1e-6)

    def run(self, n_steps: int, report_interval: int = 100):
        history = []
        for s in range(n_steps):
            self.step()
            if (s + 1) % report_interval == 0 or s == 0:
                history.append((s + 1, float(self.rho.max()), float(np.abs(self.u).max()), float(self.p.max())))
        return history


# ============================================================================
# D2Q16 lattice (for LBM initialization reference)
# ============================================================================

@dataclass
class D2Q16Lattice:
    Q: int = 16
    cx: list[int] = None
    cy: list[int] = None
    w: list[float] = None

    def __post_init__(self):
        if self.cx is None:
            self.cx = [0, 1, -1, 0, 0, 1, -1, -1, 1, 2, -2, 0, 0, 2, -2, -2]
        if self.cy is None:
            self.cy = [0, 0, 0, 1, -1, 1, 1, -1, -1, 0, 0, 2, -2, 1, 1, -1]
        if self.w is None:
            self.w = [0.266667, 0.133333, 0.133333, 0.133333, 0.133333,
                      0.033333, 0.033333, 0.033333, 0.033333,
                      0.016667, 0.016667, 0.016667, 0.016667,
                      0.008333, 0.008333, 0.008333]
            w_sum = sum(self.w)
            self.w = [wi / w_sum for wi in self.w]
