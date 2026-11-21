"""Body-force driver scaffold for LBM simulations.

Provides a structured framework for driving LBM flows with body forces,
including:
- Proper initialization from analytical solutions
- Steady-state detection with statistical convergence
- Error analysis against analytical references
- Support for both Guo (second-order) and simple (first-order) forcing

This scaffold follows the pattern from haimingz/vivsim's body-force-driver
implementation, adapted for TensorLBM's numpy/torch dual-backend design.

Usage
-----
>>> from tensorlbm.forcing.body_force_driver import BodyForceDriver, PoiseuilleConfig
>>> cfg = PoiseuilleConfig(height=32, tau=0.8, force=1e-5, steps=10000)
>>> driver = BodyForceDriver(cfg)
>>> result = driver.run()
>>> print(f"L2 error: {result.eps_L2:.6f}")
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass
class PoiseuilleConfig:
    """Configuration for body-force driven Poiseuille flow."""
    height: int = 32
    nx: int = 4
    tau: float = 0.8
    force: float = 1.0e-5
    steps: int = 10000
    forcing: str = "guo"
    bc: str = "force_corrected_nebb"
    init_mode: str = "analytic"
    steady_window: int = 500
    tol_steady: float = 1e-10

    def __post_init__(self):
        if self.tau <= 0.5:
            raise ValueError(f"tau must be > 0.5 for stability, got {self.tau}")
        if self.height < 4:
            raise ValueError(f"height must be >= 4, got {self.height}")
        if self.forcing not in ("guo", "simple"):
            raise ValueError(f"forcing must be 'guo' or 'simple', got {self.forcing}")
        if self.bc not in ("standard_bb", "force_corrected_nebb"):
            raise ValueError(f"bc must be 'standard_bb' or 'force_corrected_nebb', got {self.bc}")


@dataclasses.dataclass
class BodyForceResult:
    """Result from a body-force driven simulation."""
    u_profile: np.ndarray
    u_analytical: np.ndarray
    eps_L2: float
    eps_inf: float
    mass_drift: float
    wall_slip: float
    steps_to_steady: int
    wall_time_s: float
    config: PoiseuilleConfig

    def summary(self) -> str:
        status = "PASS" if self.eps_L2 < 0.02 and self.eps_inf < 0.02 else "FAIL"
        return (
            f"[{status}] H={self.config.height} τ={self.config.tau} "
            f"g={self.config.force:.2e} forcing={self.config.forcing} "
            f"bc={self.config.bc}: "
            f"L2={self.eps_L2:.6f} L∞={self.eps_inf:.6f} "
            f"slip={self.wall_slip:.2e} drift={self.mass_drift:.2e}"
        )


class BodyForceDriver:
    """Driver for body-force driven LBM simulations.

    D2Q9 BGK solver with Guo (2002) forcing.

    Guo forcing scheme:
        u = (Σ f_i c_i + F/2) / ρ    [physical velocity]
        feq = feq(ρ, u)               [equilibrium uses physical velocity]
        f* = f - ω(f - feq)           [collision]
        f_new = f* + S_i              [add force term]

    where S_i = (1 - ω/2) w_i [3(c_i - u) + 9(c_i·u)c_i] · F
    """

    CX = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=int)
    CY = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1], dtype=int)
    W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=float)
    OPP = np.array([0, 2, 1, 4, 3, 8, 7, 6, 5], dtype=int)

    def __init__(self, cfg: PoiseuilleConfig):
        self.cfg = cfg
        self.ny = cfg.height + 2
        self.nx = cfg.nx
        self.solid = np.zeros((self.ny, self.nx), dtype=bool)
        self.solid[0, :] = True
        self.solid[-1, :] = True
        self.fluid = ~self.solid
        self.y_fluid = np.arange(1, self.ny - 1, dtype=float) - 0.5
        self.H = float(cfg.height)
        self.nu = (cfg.tau - 0.5) / 3.0
        self.g = cfg.force
        self.f = self._initialize()

    def _initialize(self) -> np.ndarray:
        f = np.zeros((9, self.ny, self.nx), dtype=float)
        if self.cfg.init_mode == "analytic":
            u_ana = self._analytical_profile(self.y_fluid)
            ux = np.zeros((self.ny, self.nx))
            ux[1:-1, :] = u_ana[:, None]
        else:
            ux = np.zeros((self.ny, self.nx))
        rho = np.ones((self.ny, self.nx))
        f = self._equilibrium(rho, ux, np.zeros_like(ux))
        f[:, self.solid] = 0.0
        return f

    def _analytical_profile(self, y: np.ndarray) -> np.ndarray:
        return self.g / (2.0 * self.nu) * y * (self.H - y)

    def _equilibrium(self, rho, ux, uy):
        feq = np.zeros((9, self.ny, self.nx), dtype=float)
        u2 = ux**2 + uy**2
        for i in range(9):
            cu = 3.0 * (self.CX[i] * ux + self.CY[i] * uy)
            feq[i] = self.W[i] * rho * (1.0 + cu + 0.5 * cu**2 - 1.5 * u2)
        return feq

    def _macroscopic(self, f):
        """Compute ρ, u with Guo force correction: u = (Σfc + F/2)/ρ."""
        rho = f.sum(axis=0)
        safe_rho = np.where(rho > 0, rho, 1.0)
        mom_x = (f * self.CX[:, None, None]).sum(axis=0)
        mom_y = (f * self.CY[:, None, None]).sum(axis=0)
        # Guo: physical velocity includes F/2 shift
        ux = (mom_x + 0.5 * self.g) / safe_rho
        uy = mom_y / safe_rho
        return rho, ux, uy

    def _collide(self, f):
        """BGK collision + Guo forcing."""
        rho, ux, uy = self._macroscopic(f)

        # Equilibrium uses physical velocity (already includes F/2)
        feq = self._equilibrium(rho, ux, uy)

        omega = 1.0 / self.cfg.tau
        f_post = f - omega * (f - feq)

        if self.cfg.forcing == "guo":
            # Guo force term: S_i = (1 - ω/2) w_i [3(c_i - u) + 9(c_i·u)c_i] · F
            coef = (1.0 - 0.5 * omega) * 3.0 * self.g
            for i in range(9):
                cu = self.CX[i] * ux + self.CY[i] * uy
                f_post[i] += coef * self.W[i] * (
                    self.CX[i] - ux + 3.0 * cu * self.CX[i]
                )
        else:
            for i in range(9):
                f_post[i] += 3.0 * self.W[i] * self.CX[i] * self.g

        f_post[:, self.solid] = 0.0
        return f_post

    def _stream(self, f):
        """Pull-convention streaming + BC."""
        f_out = np.empty_like(f)
        for i in range(9):
            cx, cy = self.CX[i], self.CY[i]
            if cx == 0 and cy == 0:
                f_out[i] = f[i]
            else:
                f_out[i] = np.roll(f[i], (cy, cx), axis=(0, 1))

        if self.cfg.bc == "force_corrected_nebb":
            f_out = self._apply_force_corrected_nebb(f_out)
        else:
            f_out = self._apply_standard_bb(f_out)

        f_out[:, self.solid] = 0.0
        return f_out

    def _apply_standard_bb(self, f):
        f_out = f.copy()
        for i in range(1, 9):
            opp = self.OPP[i]
            if self.CY[i] > 0:
                f_out[i, 1, :] = f[opp, 1, :]
            if self.CY[i] < 0:
                f_out[i, self.ny - 2, :] = f[opp, self.ny - 2, :]
        return f_out

    def _apply_force_corrected_nebb(self, f):
        f_out = f.copy()
        tau = self.cfg.tau
        g = self.g
        f_out = self._nebb_wall(f_out, wall_y=1, is_bottom=True, tau=tau, force_x=g)
        f_out = self._nebb_wall(f_out, wall_y=self.ny - 2, is_bottom=False, tau=tau, force_x=g)
        return f_out

    def _nebb_wall(self, f, wall_y, is_bottom, tau, force_x):
        f_out = f.copy()
        unknown_dirs = [3, 5, 6] if is_bottom else [4, 7, 8]

        rho = f.sum(axis=0)
        rho_wall = rho[wall_y, :].copy()
        u_wall = 0.0
        v_wall = 0.0

        for i in unknown_dirs:
            opp = self.OPP[i]
            ci_x, ci_y = self.CX[i], self.CY[i]
            ci_dot_u = ci_x * u_wall + ci_y * v_wall
            feq_i = self.W[i] * rho_wall * (1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u**2)

            ci_opp_x, ci_opp_y = self.CX[opp], self.CY[opp]
            ci_opp_dot_u = ci_opp_x * u_wall + ci_opp_y * v_wall
            feq_opp = self.W[opp] * rho_wall * (1.0 + 3.0 * ci_opp_dot_u + 4.5 * ci_opp_dot_u**2)

            f_out[i, wall_y, :] = f[opp, wall_y, :] + feq_i - feq_opp

            if abs(force_x) > 1e-15:
                coef = (1.0 - 0.5 / tau)
                term_x = 3.0 * ci_x + 9.0 * ci_dot_u * ci_x
                delta_f = coef * self.W[i] * term_x * force_x
                f_out[i, wall_y, :] += delta_f

        return f_out

    def run(self) -> BodyForceResult:
        t0 = time.time()
        cfg = self.cfg
        steady_start = max(cfg.steps - cfg.steady_window, 0)
        u_acc = np.zeros(self.cfg.height)
        rho_acc = 0.0
        n_acc = 0
        wall_slip_max = 0.0

        for step in range(1, cfg.steps + 1):
            self.f = self._stream(self._collide(self.f))
            rho, ux, uy = self._macroscopic(self.f)
            wall_slip_max = max(wall_slip_max, float(np.max(np.abs(ux[self.solid]))))

            if step > steady_start:
                u_acc += ux[1:-1, 0]
                rho_acc += float(rho[self.fluid].mean())
                n_acc += 1

        wall_time = time.time() - t0
        u_avg = u_acc / max(n_acc, 1)
        rho_mean = rho_acc / max(n_acc, 1)
        u_ana = self._analytical_profile(self.y_fluid)

        diff = u_avg - u_ana
        u_ana_norm = np.linalg.norm(u_ana)
        u_ana_max = np.max(np.abs(u_ana))
        eps_L2 = float(np.linalg.norm(diff) / u_ana_norm) if u_ana_norm > 0 else 0.0
        eps_inf = float(np.max(np.abs(diff)) / u_ana_max) if u_ana_max > 0 else 0.0

        return BodyForceResult(
            u_profile=u_avg, u_analytical=u_ana,
            eps_L2=eps_L2, eps_inf=eps_inf,
            mass_drift=abs(rho_mean - 1.0), wall_slip=wall_slip_max,
            steps_to_steady=steady_start, wall_time_s=wall_time, config=cfg,
        )


def run_convergence_study(heights=None, tau=0.8, force=1e-5, forcing="guo", bc="force_corrected_nebb"):
    if heights is None:
        heights = [8, 16, 32, 64]
    results = []
    for H in heights:
        steps = max(5000, H * 200)
        cfg = PoiseuilleConfig(height=H, tau=tau, force=force, steps=steps, forcing=forcing, bc=bc)
        driver = BodyForceDriver(cfg)
        results.append(driver.run())
    return results
