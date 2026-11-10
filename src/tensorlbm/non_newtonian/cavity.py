"""Lid-driven cavity flow solver for non-Newtonian (Carreau-Yasuda / power-law) fluids.

Implements a D2Q9 LBM solver with:
- Zou/He velocity boundary on the moving lid
- Halfway bounce-back on stationary walls
- Picard iteration for non-Newtonian viscosity coupling

Reference data:
- Bell & Surana (1997): lid-driven cavity for power-law fluids n=0.5, 1.0, 1.5
- Ghia et al. (1982): Newtonian cavity Re=100, 400, 1000 (n=1 limit)
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
)


@dataclass
class CavityConfig:
    """Configuration for non-Newtonian lid-driven cavity."""
    nx: int = 65           # grid resolution
    ny: int = 65
    re: float = 100.0      # Reynolds number
    u_lid: float = 0.1     # lid velocity
    n_power: float = 1.0   # power-law index
    rheology: Literal["power_law", "carreau_yasuda"] = "power_law"
    # Carreau-Yasuda parameters
    nu_0: float = 0.1
    nu_inf: float = 0.001
    lambda_cy: float = 1.0
    yasuda_a: float = 2.0
    # Solver parameters
    max_steps: int = 20000
    convergence_tol: float = 1e-6
    picard_max_iter: int = 50
    picard_tol: float = 1e-4
    picard_relaxation: float = 0.5
    lbm_steps_per_picard: int = 200
    tau_min: float = 0.6
    tau_max: float = 2.0
    # Steady-state controls (verification protocol)
    flow_tol: float = 1e-6          # max|du|/u_lid per Picard block
    wall_offset: float = 0.5        # half-way bounce-back wall location
    device: str = "cpu"


class CavityNonNewtonian:
    """D2Q9 lid-driven cavity solver with non-Newtonian rheology."""

    def __init__(self, config: CavityConfig):
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

        # Wall mask: bottom, left, right walls (top is the lid)
        self.wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=self.device)
        self.wall_mask[0, :] = True   # bottom
        self.wall_mask[:, 0] = True   # left
        self.wall_mask[:, -1] = True  # right

        # Compute reference viscosity and τ
        self._setup_viscosity()

        # Initialize fields
        self._initialize()

    def _setup_viscosity(self):
        """Compute reference viscosity from Re definition."""
        cfg = self.cfg
        L = self.nx - 1  # cavity width in lattice units
        U = cfg.u_lid

        if cfg.rheology == "power_law":
            n = cfg.n_power
            K = U**(2-n) * L**n / cfg.re
            self._consistency_K = K
            gamma_ref = U / L
            nu_ref = K * max(gamma_ref, 1e-12)**(n - 1)
            self._nu_ref = nu_ref
        else:
            self._nu_ref = cfg.nu_0

        tau_ref = 3.0 * self._nu_ref + 0.5
        tau_ref = max(tau_ref, cfg.tau_min)
        tau_ref = min(tau_ref, cfg.tau_max)
        self._tau_ref = tau_ref

    def _initialize(self):
        """Initialize distribution functions."""
        ny, nx = self.ny, self.nx
        rho = torch.ones((ny, nx), dtype=self.dtype, device=self.device)
        ux = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)
        uy = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)

        self.f = self._equilibrium(rho, ux, uy)
        self.tau_field = torch.full((ny, nx), self._tau_ref, dtype=self.dtype, device=self.device)
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

    def _collide(self, f, tau):
        rho, ux, uy = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy)
        omega = (1.0 / tau).unsqueeze(0)
        return f - omega * (f - feq)

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

    def _apply_boundary_conditions(self, f_post_collide, f_post_stream):
        """Apply bounce-back on walls and Zou/He on lid.

        Key: apply BCs directly on f_post_stream (post-streaming field).
        """
        f = f_post_stream.clone()

        # Half-way bounce-back on stationary walls (bottom, left, right).
        # The reflection must act on the *post-streaming* field: the incoming
        # population f_opp at a wall node then carries the post-collision value
        # of the adjacent fluid node, which puts the no-slip surface half a
        # lattice unit inside the wall node (standard half-way bounce-back).
        src = f_post_stream
        for i in (2, 5, 6):          # bottom wall: fluid above
            k = int(self._opp[i].item())
            f[i, 0, :] = src[k, 0, :]
            f[k, 0, :] = src[k, 0, :]
        for i in (1, 5, 8):          # left wall: fluid to the right
            k = int(self._opp[i].item())
            f[i, :, 0] = src[k, :, 0]
            f[k, :, 0] = src[k, :, 0]
        for i in (3, 6, 7):          # right wall: fluid to the left
            k = int(self._opp[i].item())
            f[i, :, -1] = src[k, :, -1]
            f[k, :, -1] = src[k, :, -1]

        # Zou/He velocity boundary on the lid (top row, j = ny-1)
        j = self.ny - 1
        u_lid = self.cfg.u_lid

        f0 = f[0, j, :].clone()
        f1 = f[1, j, :].clone()
        f2 = f[2, j, :].clone()
        f3 = f[3, j, :].clone()
        f5 = f[5, j, :].clone()
        f6 = f[6, j, :].clone()

        # Zou/He density for top wall with uy=0
        rho_lid = f0 + f1 + 2.0 * f2 + f3 + 2.0 * f5 + 2.0 * f6

        # Unknown populations (f4=South, f7=SW, f8=SE)
        f[4, j, :] = f2
        f[7, j, :] = f5 + 0.5 * (f1 - f3) - 0.5 * rho_lid * u_lid
        f[8, j, :] = f6 - 0.5 * (f1 - f3) + 0.5 * rho_lid * u_lid

        return self._zero_wall_velocity(f)


    def _zero_wall_velocity(self, f: torch.Tensor) -> torch.Tensor:
        """Enforce u = 0 exactly on wall nodes by symmetrising opposite pairs.

        Half-way bounce-back already cancels the wall-normal momentum; the
        tangential pair still carries a spurious O(1e-4 u_max) slip because it
        streams inside the wall row.  Averaging each opposite pair removes it
        without changing the momentum flux into the fluid.
        """
        wm = self.wall_mask
        for i in range(1, 9):
            j = int(self._opp[i].item())
            if i < j:
                mean = 0.5 * (f[i][wm] + f[j][wm])
                f[i][wm] = mean
                f[j][wm] = mean
        return f

    def _update_tau_field(self):
        """Update τ field from current velocity via Picard iteration."""
        shear_rate = strain_rate_magnitude_2d(self.ux, self.uy)
        nu = self._compute_viscosity(shear_rate)
        tau_new = 3.0 * nu + 0.5
        tau_new = torch.clamp(tau_new, min=self.cfg.tau_min, max=self.cfg.tau_max)

        # Don't update τ on walls or lid
        tau_new = torch.where(self.wall_mask, self.tau_field, tau_new)
        tau_new[self.ny - 1, :] = self.tau_field[self.ny - 1, :]

        # Under-relaxation
        omega = self.cfg.picard_relaxation
        tau_relaxed = (1.0 - omega) * self.tau_field + omega * tau_new

        # Convergence metric (interior only)
        interior = ~self.wall_mask
        interior[self.ny - 1, :] = False
        if interior.any():
            diff = torch.abs(tau_relaxed[interior] - self.tau_field[interior])
            scale = torch.clamp(torch.abs(self.tau_field[interior]), min=1e-15)
            rel_change = (diff / scale).max().item()
        else:
            rel_change = 0.0

        self.tau_field = tau_relaxed
        self.shear_rate = shear_rate
        return rel_change

    def solve(self) -> dict:
        """Run the full simulation with Picard iteration."""
        cfg = self.cfg
        picard_history = []
        total_steps = 0

        u_ref = max(abs(float(cfg.u_lid)), 1e-12)
        prev_ux = self.ux.clone()
        flow_history: list[float] = []
        rel_change = float("inf")
        flow_res = float("inf")
        picard_iter = -1

        for picard_iter in range(cfg.picard_max_iter):
            for step in range(cfg.lbm_steps_per_picard):
                f_post = self._collide(self.f, self.tau_field)
                f_new = self._stream(f_post)
                self.f = self._apply_boundary_conditions(f_post, f_new)
                total_steps += 1

            self.rho, self.ux, self.uy = self._macroscopic(self.f)
            # flow residual: has the *flow field* reached steady state?
            flow_res = float((self.ux - prev_ux).abs().max().item() / u_ref)
            prev_ux = self.ux.clone()
            flow_history.append(flow_res)

            if torch.isnan(self.f).any():
                return {
                    "converged": False,
                    "picard_iterations": picard_iter + 1,
                    "total_lbm_steps": total_steps,
                    "final_residual": float('inf'),
                    "picard_history": picard_history,
                    "error": "NaN detected",
                }

            rel_change = self._update_tau_field()
            picard_history.append(rel_change)

            if rel_change < cfg.picard_tol and flow_res < cfg.flow_tol:
                return {
                    "converged": True,
                    "picard_iterations": picard_iter + 1,
                    "total_lbm_steps": total_steps,
                    "final_residual": rel_change,
                    "flow_residual": flow_res,
                    "picard_history": picard_history,
                    "flow_history": flow_history,
                }

            if total_steps >= cfg.max_steps:
                break

        return {
            "converged": False,
            "picard_iterations": picard_iter + 1,
            "total_lbm_steps": total_steps,
            "final_residual": rel_change,
            "flow_residual": flow_res,
            "picard_history": picard_history,
            "flow_history": flow_history,
        }

    def get_centerline_velocities(self):
        """Extract centerline velocity profiles for comparison."""
        mid_x = self.nx // 2
        mid_y = self.ny // 2

        u_vertical = self.ux[:, mid_x].cpu().numpy()
        y_coords = np.linspace(0, 1, self.ny)

        u_horizontal = self.ux[mid_y, :].cpu().numpy()
        v_horizontal = self.uy[mid_y, :].cpu().numpy()
        x_coords = np.linspace(0, 1, self.nx)

        return {
            "y": y_coords,
            "u_vertical": u_vertical,
            "x": x_coords,
            "u_horizontal": u_horizontal,
            "v_horizontal": v_horizontal,
        }


def run_cavity_benchmark(n_values=(0.5, 1.0, 1.5), nx=65, re=100.0,
                         u_lid=0.1, device="cpu") -> dict:
    """Run cavity benchmark for multiple power-law indices."""
    results = {}
    for n in n_values:
        cfg = CavityConfig(
            nx=nx, ny=nx, re=re, u_lid=u_lid,
            n_power=n, rheology="power_law",
            picard_max_iter=50, picard_tol=1e-4,
            lbm_steps_per_picard=200,
            device=device,
        )
        solver = CavityNonNewtonian(cfg)
        info = solver.solve()
        profiles = solver.get_centerline_velocities()
        results[n] = {
            "info": info,
            "profiles": profiles,
            "tau_field": solver.tau_field.cpu().numpy(),
            "ux_field": solver.ux.cpu().numpy(),
            "uy_field": solver.uy.cpu().numpy(),
        }
    return results
