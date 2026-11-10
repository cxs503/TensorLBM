"""Picard iteration framework for non-Newtonian LBM.

The local relaxation time τ(x,t) depends on the strain-rate magnitude |γ̇|,
which in turn depends on the velocity field obtained from the LBM solve.
This creates a nonlinear coupling that is resolved by Picard (fixed-point)
iteration within each time step or at steady state.

Algorithm:
    1. Initialise τ = τ₀ (from ν₀ or a reference viscosity)
    2. Loop:
       a. Run LBM collision-streaming with current τ field
       b. Compute velocity field (ρ, u) from distributions
       c. Compute strain-rate magnitude |γ̇| from velocity gradients
       d. Update viscosity ν(|γ̇|) from the rheology model
       e. Update τ = 3ν + 0.5
       f. Check convergence: max|τ_new - τ_old| / τ_old < tol
    3. Return converged state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch

from ..d2q9 import equilibrium, macroscopic, C, W, OPPOSITE
from .rheology import (
    strain_rate_magnitude_2d,
    apparent_viscosity_carreau_yasuda,
    apparent_viscosity_power_law,
)


@dataclass
class PicardConfig:
    """Configuration for Picard iteration."""
    max_iterations: int = 50
    tolerance: float = 1e-4
    relaxation_factor: float = 0.5  # under-relaxation for stability
    rheology_model: Literal["carreau_yasuda", "power_law", "newtonian"] = "carreau_yasuda"
    # Carreau-Yasuda parameters
    nu_0: float = 0.1       # zero-shear viscosity
    nu_inf: float = 0.001   # infinite-shear viscosity
    lambda_time: float = 1.0  # relaxation time λ
    flow_index: float = 1.0   # power-law index n
    yasuda_exponent: float = 2.0  # Yasuda exponent a
    # Power-law parameters
    consistency_index: float = 0.05
    # Newtonian
    newtonian_viscosity: float = 0.01
    # Clamping
    tau_min: float = 0.501
    tau_max: float = 2.0
    shear_rate_floor: float = 1e-12


class PicardNonNewtonianSolver:
    """Picard iteration solver for non-Newtonian LBM.

    This solver wraps the standard D2Q9 LBM loop and adds Picard iteration
    to resolve the nonlinear coupling between strain rate and viscosity.
    """

    def __init__(self, ny: int, nx: int, config: PicardConfig,
                 device: torch.device | None = None):
        self.ny = ny
        self.nx = nx
        self.config = config
        self.device = device or torch.device("cpu")
        self.dtype = torch.float64  # double precision for accuracy

        # D2Q9 lattice constants on device
        self._c = C.to(self.device).to(torch.float64)
        self._w = W.to(self.device).to(torch.float64)
        self._opp = OPPOSITE.to(self.device)

        # Build exact weights for float64
        W_EXACT = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
        self._w_exact = torch.tensor(W_EXACT, dtype=self.dtype, device=self.device)

        # State
        self.f = None  # distribution functions (9, ny, nx)
        self.tau_field = None  # local relaxation time (ny, nx)
        self.rho = None
        self.ux = None
        self.uy = None
        self.shear_rate = None

        # Convergence history
        self.convergence_history: list[float] = []
        self.iteration_count = 0

    def _equilibrium(self, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
        """D2Q9 equilibrium distribution (float64)."""
        cx = self._c[:, 0].reshape(9, 1, 1)
        cy = self._c[:, 1].reshape(9, 1, 1)
        w = self._w_exact.reshape(9, 1, 1)

        u_sq = ux * ux + uy * uy
        cu = cx * ux + cy * uy
        feq = w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
        return feq

    def _macroscopic(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recover macroscopic variables from distributions."""
        cx = self._c[:, 0].reshape(9, 1, 1)
        cy = self._c[:, 1].reshape(9, 1, 1)

        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = (f * cx).sum(dim=0) / rho_safe
        uy = (f * cy).sum(dim=0) / rho_safe
        return rho, ux, uy

    def _stream(self, f: torch.Tensor) -> torch.Tensor:
        """Streaming step with periodic or bounce-back handled externally."""
        out = torch.empty_like(f)
        for i in range(9):
            cx = int(self._c[i, 0].item())
            cy = int(self._c[i, 1].item())
            src = f[i]
            if cx:
                src = torch.roll(src, cx, dims=1)
            if cy:
                src = torch.roll(src, cy, dims=0)
            out[i] = src
        return out

    def _collide_with_tau_field(self, f: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """BGK collision with spatially-varying relaxation time."""
        rho, ux, uy = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy)
        omega = 1.0 / tau.unsqueeze(0)  # (1, ny, nx)
        return f - omega * (f - feq)

    def compute_viscosity(self, shear_rate: torch.Tensor) -> torch.Tensor:
        """Compute apparent viscosity from the configured rheology model."""
        cfg = self.config
        if cfg.rheology_model == "carreau_yasuda":
            nu = apparent_viscosity_carreau_yasuda(
                shear_rate,
                nu_0=cfg.nu_0,
                nu_inf=cfg.nu_inf,
                relaxation_time=cfg.lambda_time,
                flow_index=cfg.flow_index,
                yasuda_exponent=cfg.yasuda_exponent,
                shear_rate_floor=cfg.shear_rate_floor,
            )
        elif cfg.rheology_model == "power_law":
            nu = apparent_viscosity_power_law(
                shear_rate,
                consistency_index=cfg.consistency_index,
                flow_index=cfg.flow_index,
                shear_rate_floor=cfg.shear_rate_floor,
            )
        elif cfg.rheology_model == "newtonian":
            nu = torch.full_like(shear_rate, cfg.newtonian_viscosity)
        else:
            raise ValueError(f"Unknown rheology model: {cfg.rheology_model}")
        return nu

    def compute_tau_field(self, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
        """Compute local τ field from velocity field."""
        shear_rate = strain_rate_magnitude_2d(ux, uy)
        nu = self.compute_viscosity(shear_rate)
        tau = 3.0 * nu + 0.5
        tau = torch.clamp(tau, min=self.config.tau_min, max=self.config.tau_max)
        return tau, shear_rate

    def initialize(self, rho0: float = 1.0, ux0: float = 0.0, uy0: float = 0.0):
        """Initialize distribution functions and τ field."""
        rho = torch.full((self.ny, self.nx), rho0, dtype=self.dtype, device=self.device)
        ux = torch.full((self.ny, self.nx), ux0, dtype=self.dtype, device=self.device)
        uy = torch.full((self.ny, self.nx), uy0, dtype=self.dtype, device=self.device)
        self.f = self._equilibrium(rho, ux, uy)

        # Initial τ from reference viscosity
        if self.config.rheology_model == "carreau_yasuda":
            nu_ref = self.config.nu_0  # zero-shear (max) viscosity
        elif self.config.rheology_model == "power_law":
            nu_ref = self.config.consistency_index
        else:
            nu_ref = self.config.newtonian_viscosity
        tau_ref = 3.0 * nu_ref + 0.5
        tau_ref = max(tau_ref, self.config.tau_min)
        tau_ref = min(tau_ref, self.config.tau_max)
        self.tau_field = torch.full((self.ny, self.nx), tau_ref, dtype=self.dtype, device=self.device)

        self.rho = rho
        self.ux = ux
        self.uy = uy
        self.shear_rate = torch.zeros((self.ny, self.nx), dtype=self.dtype, device=self.device)

    def picard_step(self, wall_mask: torch.Tensor | None = None,
                    boundary_fn: Callable | None = None) -> float:
        """One Picard iteration: collide → stream → apply BCs → update τ.

        Returns the relative change in τ (convergence metric).
        """
        tau_old = self.tau_field.clone()

        # Collision with current τ field
        f_post = self._collide_with_tau_field(self.f, self.tau_field)

        # Streaming
        f_new = self._stream(f_post)

        # Apply boundary conditions
        if wall_mask is not None:
            f_new = self._apply_bounce_back(f_post, f_new, wall_mask)
        if boundary_fn is not None:
            f_new = boundary_fn(f_new, self.f)

        self.f = f_new

        # Update macroscopic variables
        self.rho, self.ux, self.uy = self._macroscopic(self.f)

        # Update τ field from new velocity
        tau_new, self.shear_rate = self.compute_tau_field(self.ux, self.uy)

        # Apply wall mask to τ (walls keep reference τ)
        if wall_mask is not None:
            tau_new = torch.where(wall_mask, tau_old, tau_new)

        # Under-relaxation
        omega = self.config.relaxation_factor
        tau_relaxed = (1.0 - omega) * tau_old + omega * tau_new

        # Convergence metric
        diff = torch.abs(tau_relaxed - tau_old)
        tau_scale = torch.clamp(torch.abs(tau_old), min=1e-15)
        rel_change = (diff / tau_scale).max().item()

        self.tau_field = tau_relaxed
        self.convergence_history.append(rel_change)
        self.iteration_count += 1

        return rel_change

    def _apply_bounce_back(self, f_pre: torch.Tensor, f_post: torch.Tensor,
                           wall_mask: torch.Tensor) -> torch.Tensor:
        """Halfway bounce-back on wall nodes."""
        f_new = f_post.clone()
        for i in range(9):
            opp = int(self._opp[i].item())
            f_new[i][wall_mask] = f_pre[opp][wall_mask]
        return f_new

    def run_picard(self, num_lbm_steps: int = 1,
                   wall_mask: torch.Tensor | None = None,
                   boundary_fn: Callable | None = None) -> dict:
        """Run Picard iteration to convergence.

        For steady-state problems, this runs multiple LBM steps per Picard
        iteration to improve the velocity field before updating τ.

        Returns convergence info dict.
        """
        self.convergence_history = []
        self.iteration_count = 0

        for picard_iter in range(self.config.max_iterations):
            # Run LBM steps to advance the flow
            for _ in range(num_lbm_steps):
                self.picard_step(wall_mask=wall_mask, boundary_fn=boundary_fn)

            # Check convergence
            if self.convergence_history and self.convergence_history[-1] < self.config.tolerance:
                return {
                    "converged": True,
                    "iterations": self.iteration_count,
                    "final_residual": self.convergence_history[-1],
                    "history": self.convergence_history,
                }

        return {
            "converged": False,
            "iterations": self.iteration_count,
            "final_residual": self.convergence_history[-1] if self.convergence_history else float('inf'),
            "history": self.convergence_history,
        }
