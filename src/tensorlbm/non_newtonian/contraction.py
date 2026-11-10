"""4:1 planar contraction flow solver for non-Newtonian fluids.

Geometry:
    Inlet channel height = 4H, outlet channel height = H
    Contraction ratio = 4:1
    The contraction is sudden (step change in height)

Reference:
    Xue, S.-C., Phan-Thien, N. & Tanner, R.I. (2004).
    "Three dimensional numerical study of viscoelastic flow in expansion",
    (and related contraction flow benchmarks).

    Key quantities for comparison:
    - Vortex length in the corner upstream of the contraction
    - Centerline velocity recovery downstream
    - Pressure drop across the contraction
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
class ContractionConfig:
    """Configuration for 4:1 planar contraction flow."""
    # Geometry: inlet height = 4*H_out, outlet height = H_out
    H_out: int = 16          # outlet half-height in lattice units
    L_inlet: int = 128       # inlet length (upstream of contraction)
    L_outlet: int = 128      # outlet length (downstream of contraction)
    n_power: float = 1.0     # power-law index
    re: float = 100.0        # Reynolds number based on outlet height and velocity
    u_inlet: float = 0.02    # inlet average velocity
    rheology: Literal["power_law", "carreau_yasuda"] = "power_law"
    # Carreau-Yasuda parameters
    nu_0: float = 0.1
    nu_inf: float = 0.001
    lambda_cy: float = 1.0
    yasuda_a: float = 2.0
    # Solver parameters
    max_steps: int = 50000
    convergence_tol: float = 1e-6
    picard_max_iter: int = 50
    picard_tol: float = 1e-4
    picard_relaxation: float = 0.5
    lbm_steps_per_picard: int = 300
    tau_min: float = 0.501
    tau_max: float = 2.0
    device: str = "cpu"


class Contraction4to1:
    """D2Q9 4:1 planar contraction flow solver with non-Newtonian rheology."""

    def __init__(self, config: ContractionConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.dtype = torch.float64

        # Domain dimensions
        H_in = 4 * config.H_out  # inlet full height
        H_out = config.H_out     # outlet full height
        self.H_in = H_in
        self.H_out = H_out
        self.ny = H_in + 2  # +2 for walls
        self.nx = config.L_inlet + config.L_outlet + 1

        # Contraction location
        self.x_contract = config.L_inlet  # x-index of contraction

        # Lattice constants
        self._c = C.to(self.device).to(self.dtype)
        W_EXACT = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
        self._w = torch.tensor(W_EXACT, dtype=self.dtype, device=self.device).reshape(9, 1, 1)
        self._opp = OPPOSITE.to(self.device)

        # Build geometry mask (True = solid wall)
        self._build_geometry()

        # Compute body force
        self._setup_body_force()

        # Initialize
        self._initialize()

    def _build_geometry(self):
        """Build the wall mask for the 4:1 contraction geometry."""
        ny, nx = self.ny, self.nx
        H_in = self.H_in
        H_out = self.H_out
        x_c = self.x_contract

        self.wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=self.device)

        # Top and bottom walls for the entire domain
        self.wall_mask[0, :] = True   # bottom wall
        self.wall_mask[-1, :] = True  # top wall

        # Step: above the outlet channel, there's a corner
        # Outlet channel: from y = (H_in - H_out)/2 to y = (H_in + H_out)/2
        # (centered vertically)
        y_low_out = (H_in - H_out) // 2 + 1  # bottom of outlet channel
        y_high_out = (H_in + H_out) // 2     # top of outlet channel

        # Corner walls: downstream of contraction
        # Bottom corner: y < y_low_out for x > x_contract
        # Top corner: y > y_high_out for x > x_contract
        for j in range(ny):
            for i in range(x_c + 1, nx):
                if j < y_low_out or j > y_high_out:
                    self.wall_mask[j, i] = True

        self.y_low_out = y_low_out
        self.y_high_out = y_high_out

    def _setup_body_force(self):
        """Compute body force for target Reynolds number."""
        cfg = self.cfg
        n = cfg.n_power
        H_out = self.H_out
        U_out = cfg.u_inlet * 4.0  # continuity: U_out = 4 * U_inlet

        if cfg.rheology == "power_law":
            L_char = 2.0 * H_out
            K = U_out**(2-n) * L_char**n / cfg.re
            self._consistency_K = K
            # Estimate body force from fully-developed relation
            G = K * (U_out * (n+1)/n / H_out**((n+1)/n))**n
            self._body_force = G * 0.5  # scale down for stability
        else:
            nu_ref = cfg.nu_0
            G = 8.0 * nu_ref * U_out / (4.0 * H_out * H_out)
            self._body_force = G * 0.5
            self._consistency_K = nu_ref

    def _initialize(self):
        """Initialize distribution functions."""
        ny, nx = self.ny, self.nx
        rho = torch.ones((ny, nx), dtype=self.dtype, device=self.device)
        ux = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)
        uy = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)

        # Set initial parabolic profile in inlet
        H_in_half = self.H_in / 2.0
        for j in range(ny):
            y = j - H_in_half
            if abs(y) <= H_in_half:
                u_profile = 1.5 * self.cfg.u_inlet * (1.0 - (y / H_in_half)**2)
                ux[j, :] = u_profile

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
        """BGK collision with body force."""
        rho, ux, uy = self._macroscopic(f)
        # Velocity correction for forcing
        ux_corr = ux + 0.5 * body_force / torch.clamp(rho, min=1e-12)
        uy_corr = uy
        feq = self._equilibrium(rho, ux_corr, uy_corr)
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

    def _apply_boundary_conditions(self, f_pre, f_post):
        """Bounce-back on walls, convective outlet."""
        f_new = f_post.clone()

        # Bounce-back on walls
        for i in range(9):
            opp = int(self._opp[i].item())
            f_new[i][self.wall_mask] = f_pre[opp][self.wall_mask]

        # Inlet (left boundary): prescribed velocity profile
        i_in = 0
        H_in_half = self.H_in / 2.0
        for j in range(self.ny):
            y = j - H_in_half
            if not self.wall_mask[j, i_in]:
                u_target = 1.5 * self.cfg.u_inlet * (1.0 - (y / H_in_half)**2)
                # Zou/He inlet
                f0 = f_new[0, j, i_in]
                f2 = f_new[2, j, i_in]
                f4 = f_new[4, j, i_in]
                f5 = f_new[5, j, i_in]
                f6 = f_new[6, j, i_in]
                # Actually use simple extrapolation for inlet
                f_new[1, j, i_in] = f_new[1, j, i_in + 1] if i_in + 1 < self.nx else f_new[1, j, i_in]
                f_new[5, j, i_in] = f_new[5, j, i_in + 1] if i_in + 1 < self.nx else f_new[5, j, i_in]
                f_new[8, j, i_in] = f_new[8, j, i_in + 1] if i_in + 1 < self.nx else f_new[8, j, i_in]

        # Outlet (right boundary): convective (zero gradient)
        i_out = self.nx - 1
        for j in range(self.ny):
            if not self.wall_mask[j, i_out]:
                for d in [1, 5, 8]:  # outgoing directions
                    f_new[d, j, i_out] = f_new[d, j, i_out - 1]

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

    def compute_vortex_length(self):
        """Compute the recirculation vortex length upstream of contraction.

        The vortex forms in the corner upstream of the contraction step.
        We detect it by finding the extent of negative u-velocity along
        the top wall (above the outlet channel).
        """
        # Look along the top wall of the inlet, just before contraction
        # The corner is at y = y_high_out + 1, x = x_contract
        y_corner = self.y_high_out + 1
        if y_corner >= self.ny:
            y_corner = self.ny - 2

        # Find extent of recirculation along the wall
        ux_wall = self.ux[y_corner, :self.x_contract].cpu().numpy()
        # Vortex starts from the contraction and extends upstream
        vortex_length = 0
        for i in range(self.x_contract - 1, -1, -1):
            if ux_wall[i] < 0:
                vortex_length = self.x_contract - i
            else:
                break

        return float(vortex_length)

    def get_centerline_velocity(self):
        """Extract centerline velocity downstream of contraction."""
        y_center = (self.y_low_out + self.y_high_out) // 2
        u_centerline = self.ux[y_center, self.x_contract:].cpu().numpy()
        x_downstream = np.arange(len(u_centerline))
        return x_downstream, u_centerline


def run_contraction_benchmark(n_values=(0.5, 1.0, 1.5), H_out=16,
                              device="cpu") -> dict:
    """Run 4:1 contraction benchmark for multiple power-law indices."""
    results = {}
    for n in n_values:
        cfg = ContractionConfig(
            H_out=H_out, n_power=n, rheology="power_law",
            u_inlet=0.02,
            picard_max_iter=50, picard_tol=1e-4,
            lbm_steps_per_picard=300,
            device=device,
        )
        solver = Contraction4to1(cfg)
        info = solver.solve()
        vortex_length = solver.compute_vortex_length()
        x_cl, u_cl = solver.get_centerline_velocity()

        results[n] = {
            "info": info,
            "vortex_length": vortex_length,
            "x_centerline": x_cl,
            "u_centerline": u_cl,
            "ux_field": solver.ux.cpu().numpy(),
            "uy_field": solver.uy.cpu().numpy(),
        }
    return results
