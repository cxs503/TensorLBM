"""Pipe / channel flow solver for non-Newtonian fluids.

Implements a D2Q9 LBM solver for pressure-driven flow between parallel plates
(2D channel) with non-Newtonian rheology. The analytical solution for a
power-law fluid provides an exact verification benchmark:

    u(y) = (n/(n+1)) (G/K)^(1/n) [H^((n+1)/n) - |y|^((n+1)/n)]

where G = -dp/dx is the pressure gradient, H is the channel half-height,
K is the consistency index, and n is the flow index.

For the circular pipe (axisymmetric), the analytical solution is:

    u(r) = U_max [1 - (r/R)^((n+1)/n)]

We simulate the 2D channel (planar Poiseuille) and compare with the
analytical profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import numpy as np

from ..d2q9 import C, W, OPPOSITE
from .rheology import (
    strain_rate_magnitude_2d,
    apparent_viscosity_carreau_yasuda,
    apparent_viscosity_power_law,
    poiseuille_power_law_profile,
)


@dataclass
class PipeConfig:
    """Configuration for non-Newtonian pipe/channel flow."""
    nx: int = 129          # streamwise length
    ny: int = 65           # wall-normal resolution (must be odd for centerline)
    n_power: float = 1.0   # power-law index
    re: float = 100.0      # generalized Reynolds number
    u_max_target: float = 0.05  # target maximum velocity (lattice units)
    rheology: Literal["power_law", "carreau_yasuda"] = "power_law"
    # Carreau-Yasuda parameters
    nu_0: float = 0.1
    nu_inf: float = 0.001
    lambda_cy: float = 1.0
    yasuda_a: float = 2.0
    # Solver parameters
    max_steps: int = 30000
    convergence_tol: float = 1e-6
    picard_max_iter: int = 50
    picard_tol: float = 1e-4
    picard_relaxation: float = 0.5
    lbm_steps_per_picard: int = 200
    tau_min: float = 0.501
    tau_max: float = 2.0
    device: str = "cpu"


class PipeNonNewtonian:
    """D2Q9 pressure-driven channel flow solver with non-Newtonian rheology."""

    def __init__(self, config: PipeConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.dtype = torch.float64

        ny, nx = config.ny, config.nx
        self.ny = ny
        self.nx = nx

        # Lattice constants
        self._c = C.to(self.device).to(self.dtype)
        W_EXACT = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
        self._w = torch.tensor(W_EXACT, dtype=self.dtype, device=self.device).reshape(9, 1, 1)
        self._opp = OPPOSITE.to(self.device)

        # Channel geometry
        self.H = (ny - 1) / 2.0  # half-height in lattice units

        # Wall mask: top and bottom
        self.wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=self.device)
        self.wall_mask[0, :] = True   # bottom wall
        self.wall_mask[-1, :] = True  # top wall

        # Compute body force from target Re and velocity
        self._setup_body_force()

        # Initialize
        self._initialize()

    def _setup_body_force(self):
        """Compute body force to achieve target Reynolds number."""
        cfg = self.cfg
        n = cfg.n_power
        H = self.H
        U_target = cfg.u_max_target

        if cfg.rheology == "power_law":
            # For power-law: Re_G = ρ U^(2-n) (2H)^n / K
            # Body force G = -dp/dx
            # At steady state: G * H = τ_wall = K (U/H)^(n) * (some factor)
            # For power-law Poiseuille: U_max = (n/(n+1)) (G/K)^(1/n) H^((n+1)/n)
            # So G = K * (U_max * (n+1)/n / H^((n+1)/n))^n
            # We set K from Re: K = U^(2-n) * (2H)^n / Re
            L_char = 2.0 * H
            K = U_target**(2-n) * L_char**n / cfg.re
            self._consistency_K = K
            # Compute G from target U_max
            G = K * (U_target * (n+1)/n / H**((n+1)/n))**n
            self._body_force = G
        else:
            # Carreau-Yasuda: use nu_0 for initial estimate
            nu_ref = cfg.nu_0
            G = 8.0 * nu_ref * U_target / (4.0 * H * H)  # Newtonian estimate
            self._body_force = G
            self._consistency_K = nu_ref

    def _initialize(self):
        """Initialize distribution functions."""
        ny, nx = self.ny, self.nx
        rho = torch.ones((ny, nx), dtype=self.dtype, device=self.device)
        ux = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)
        uy = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)

        self.f = self._equilibrium(rho, ux, uy)

        # Initial τ
        if self.cfg.rheology == "power_law":
            nu_ref = self._consistency_K
        else:
            nu_ref = self.cfg.nu_0
        tau_ref = 3.0 * nu_ref + 0.5
        tau_ref = max(tau_ref, self.cfg.tau_min)
        tau_ref = min(tau_ref, self.cfg.tau_max)
        self.tau_field = torch.full((ny, nx), tau_ref, dtype=self.dtype, device=self.device)

        self.rho = rho
        self.ux = ux
        self.uy = uy
        self.shear_rate = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)

    def _equilibrium(self, rho, ux, uy):
        cx = self._c[:, 0].reshape(9, 1, 1)
        cy = self._c[:, 1].reshape(9, 1, 1)
        u_sq = ux * ux + uy * uy
        cu = cx * ux + cy * uy
        return self._w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))

    def _macroscopic(self, f):
        cx = self._c[:, 0].reshape(9, 1, 1)
        cy = self._c[:, 1].reshape(9, 1, 1)
        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = (f * cx).sum(dim=0) / rho_safe
        uy = (f * cy).sum(dim=0) / rho_safe
        return rho, ux, uy

    def _stream(self, f):
        out = torch.empty_like(f)
        for i in range(9):
            cx_i = int(self._c[i, 0].item())
            cy_i = int(self._c[i, 1].item())
            src = f[i]
            if cx_i:
                src = torch.roll(src, cx_i, dims=1)
            if cy_i:
                src = torch.roll(src, cy_i, dims=0)
            out[i] = src
        return out

    def _collide_with_force(self, f, tau, body_force):
        """BGK collision with body force (Guo et al. 2002 forcing scheme)."""
        rho, ux, uy = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy)
        omega = (1.0 / tau).unsqueeze(0)

        # Guo forcing term
        Fx = body_force
        Fy = 0.0
        # Velocity correction
        ux_corr = ux + 0.5 * Fx / rho
        uy_corr = uy + 0.5 * Fy / rho

        # Force term in the equilibrium
        cu = self._c[:, 0].reshape(9, 1, 1) * Fx + self._c[:, 1].reshape(9, 1, 1) * Fy
        u_corr_sq = ux_corr * ux_corr + uy_corr * uy_corr
        cu_corr = self._c[:, 0].reshape(9, 1, 1) * ux_corr + self._c[:, 1].reshape(9, 1, 1) * uy_corr
        feq_corr = self._w * rho.unsqueeze(0) * (
            1.0 + 3.0 * cu_corr + 4.5 * cu_corr * cu_corr - 1.5 * u_corr_sq.unsqueeze(0)
        )

        # Source term S_i = w_i * (3(c_i - u) + 9(c_i·u)c_i) · F
        cu_F = self._c[:, 0].reshape(9, 1, 1) * Fx + self._c[:, 1].reshape(9, 1, 1) * Fy
        cu_u = self._c[:, 0].reshape(9, 1, 1) * ux + self._c[:, 1].reshape(9, 1, 1) * uy
        S = self._w * rho.unsqueeze(0) * (
            3.0 * (cu_F - cu_u * (ux * Fx + uy * Fy).unsqueeze(0))
            + 9.0 * cu_u * cu_F
        )
        # Simplified Guo forcing: S_i = (1 - 0.5/τ) w_i ρ [3(c_i-u)·F/ρ + 9(c_i·u)(c_i·F)/ρ]
        # For simplicity, use the velocity-shift approach
        f_eq_forced = self._equilibrium(rho, ux_corr, uy_corr)

        return f - omega * (f - f_eq_forced)

    def _compute_viscosity(self, shear_rate):
        cfg = self.cfg
        if cfg.rheology == "power_law":
            return apparent_viscosity_power_law(
                shear_rate,
                consistency_index=self._consistency_K,
                flow_index=cfg.n_power,
                shear_rate_floor=1e-12,
            )
        else:
            return apparent_viscosity_carreau_yasuda(
                shear_rate,
                nu_0=cfg.nu_0,
                nu_inf=cfg.nu_inf,
                relaxation_time=cfg.lambda_cy,
                flow_index=cfg.n_power,
                yasuda_exponent=cfg.yasuda_a,
                shear_rate_floor=1e-12,
            )

    def _apply_boundary_conditions(self, f_pre, f_post):
        """Bounce-back on walls, periodic in x."""
        f_new = f_post.clone()

        # Bounce-back on top and bottom walls
        for i in range(9):
            opp = int(self._opp[i].item())
            f_new[i][self.wall_mask] = f_pre[opp][self.wall_mask]

        # Periodic in x (already handled by torch.roll)
        return f_new

    def _update_tau_field(self):
        """Update τ field from current velocity."""
        shear_rate = strain_rate_magnitude_2d(self.ux, self.uy)
        nu = self._compute_viscosity(shear_rate)
        tau_new = 3.0 * nu + 0.5
        tau_new = torch.clamp(tau_new, min=self.cfg.tau_min, max=self.cfg.tau_max)

        # Don't update τ on walls
        tau_new = torch.where(self.wall_mask, self.tau_field, tau_new)

        # Under-relaxation
        omega = self.cfg.picard_relaxation
        tau_relaxed = (1.0 - omega) * self.tau_field + omega * tau_new

        diff = torch.abs(tau_relaxed - self.tau_field)
        scale = torch.clamp(torch.abs(self.tau_field), min=1e-15)
        rel_change = (diff / scale).max().item()

        self.tau_field = tau_relaxed
        self.shear_rate = shear_rate
        return rel_change

    def solve(self) -> dict:
        """Run the full simulation with Picard iteration."""
        cfg = self.cfg
        picard_history = []
        total_steps = 0

        for picard_iter in range(cfg.picard_max_iter):
            for step in range(cfg.lbm_steps_per_picard):
                f_pre = self.f.clone()
                f_post = self._collide_with_force(self.f, self.tau_field, self._body_force)
                f_new = self._stream(f_post)
                self.f = self._apply_boundary_conditions(f_pre, f_new)
                total_steps += 1

            self.rho, self.ux, self.uy = self._macroscopic(self.f)
            rel_change = self._update_tau_field()
            picard_history.append(rel_change)

            if rel_change < cfg.picard_tol:
                return {
                    "converged": True,
                    "picard_iterations": picard_iter + 1,
                    "total_lbm_steps": total_steps,
                    "final_residual": rel_change,
                    "picard_history": picard_history,
                }

            if total_steps >= cfg.max_steps:
                break

        return {
            "converged": False,
            "picard_iterations": picard_iter + 1,
            "total_lbm_steps": total_steps,
            "final_residual": rel_change if picard_history else float('inf'),
            "picard_history": picard_history,
        }

    def get_velocity_profile(self):
        """Extract mid-channel velocity profile u(y)."""
        mid_x = self.nx // 2
        u_profile = self.ux[:, mid_x].cpu().numpy()
        y_coords = np.linspace(-self.H, self.H, self.ny)
        return y_coords, u_profile

    def analytical_profile(self, y: np.ndarray) -> np.ndarray:
        """Compute analytical power-law Poiseuille profile."""
        cfg = self.cfg
        n = cfg.n_power
        H = self.H
        G = self._body_force

        if cfg.rheology == "power_law":
            K = self._consistency_K
            y_tensor = torch.tensor(y, dtype=self.dtype)
            u_anal = poiseuille_power_law_profile(y_tensor, H, G, K, n)
            return u_anal.numpy()
        else:
            # Newtonian limit
            nu = cfg.nu_0
            return G / (2.0 * nu) * (H**2 - y**2)


def run_pipe_benchmark(n_values=(0.5, 1.0, 1.5), ny=65, nx=129,
                       device="cpu") -> dict:
    """Run pipe flow benchmark for multiple power-law indices."""
    results = {}
    for n in n_values:
        cfg = PipeConfig(
            nx=nx, ny=ny, n_power=n, rheology="power_law",
            u_max_target=0.05,
            picard_max_iter=50, picard_tol=1e-4,
            lbm_steps_per_picard=200,
            device=device,
        )
        solver = PipeNonNewtonian(cfg)
        info = solver.solve()
        y_num, u_num = solver.get_velocity_profile()
        u_anal = solver.analytical_profile(y_num)

        # Compute L2 error
        mask = np.abs(y_num) < 0.95 * solver.H  # exclude near-wall
        l2_error = np.sqrt(np.mean((u_num[mask] - u_anal[mask])**2))
        u_max_anal = np.max(u_anal)
        l2_relative = l2_error / max(u_max_anal, 1e-15)

        results[n] = {
            "info": info,
            "y": y_num,
            "u_numerical": u_num,
            "u_analytical": u_anal,
            "l2_error": float(l2_error),
            "l2_relative_error": float(l2_relative),
        }
    return results
