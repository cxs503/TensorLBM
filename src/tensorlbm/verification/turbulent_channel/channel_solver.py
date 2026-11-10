"""D3Q19 MRT channel flow solver with Smagorinsky SGS and van Driest damping.

Implements a periodic plane channel flow driven by a constant body force,
with no-slip walls at y=0 and y=H (half-height h = H/2).

Geometry: Lx = 2*pi*h, Ly = 2h, Lz = pi*h  (streamwise, wall-normal, spanwise)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------

def log_law_u_plus(y_plus: torch.Tensor, kappa: float = 0.41, B: float = 5.2) -> torch.Tensor:
    """von Karman log law: U+ = (1/kappa) * ln(y+) + B."""
    y_plus = torch.as_tensor(y_plus, dtype=torch.float64)
    return torch.log(y_plus.clamp(min=1.0)) / kappa + B


def reichardt_u_plus(y_plus: torch.Tensor, kappa: float = 0.41) -> torch.Tensor:
    """Reichardt (1951) composite law of the wall."""
    y_plus = torch.as_tensor(y_plus, dtype=torch.float64)
    return (torch.log1p(kappa * y_plus) / kappa
            + 7.8 * (1.0 - torch.exp(-y_plus / 11.0)
                     - (y_plus / 11.0) * torch.exp(-y_plus / 3.0)))


def van_driest_damping(y_plus: torch.Tensor, A_plus: float = 25.0) -> torch.Tensor:
    """van Driest (1956) damping function: f_v = 1 - exp(-y+/A+)."""
    y_plus = torch.as_tensor(y_plus, dtype=torch.float64)
    return 1.0 - torch.exp(-y_plus.clamp(min=0.0) / A_plus)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelConfig:
    """Configuration for a turbulent channel flow simulation."""
    nx: int = 128
    ny: int = 65
    nz: int = 64
    re_tau: float = 180.0
    u_tau: float = 0.05
    smagorinsky_cs: float = 0.1
    van_driest_a_plus: float = 25.0
    use_van_driest: bool = True
    steps: int = 50000
    warmup_steps: int = 15000
    sample_interval: int = 5
    perturbation_amplitude: float = 0.3
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    label: str = "channel"

    @property
    def half_height(self) -> float:
        return 0.5 * (self.ny - 2)

    @property
    def nu(self) -> float:
        return self.u_tau * self.half_height / self.re_tau

    @property
    def tau_plus(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def tau_minus(self) -> float:
        return 0.5 + (3.0 / 16.0) / (self.tau_plus - 0.5)

    @property
    def body_force(self) -> float:
        return 2.0 * self.u_tau ** 2 / (self.ny - 2)

    @property
    def dy_plus(self) -> float:
        return self.u_tau / self.nu

    @property
    def dx_plus(self) -> float:
        Lx = 2.0 * math.pi * self.half_height
        return Lx / self.nx * self.u_tau / self.nu

    @property
    def dz_plus(self) -> float:
        Lz = math.pi * self.half_height
        return Lz / self.nz * self.u_tau / self.nu

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.nx < 16 or self.ny < 10 or self.nz < 8:
            raise ValueError("Grid too small")
        if self.re_tau <= 0 or self.u_tau <= 0:
            raise ValueError("re_tau and u_tau must be positive")
        if self.tau_plus - 0.5 < 1e-4:
            raise ValueError(f"tau+ too close to 0.5: {self.tau_plus:.6f}")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")


# ---------------------------------------------------------------------------
# Solver - vectorized implementation
# ---------------------------------------------------------------------------

class ChannelSolver:
    """D3Q19 TRT channel flow solver with Smagorinsky SGS.

    Populations stored as (19, nz, ny, nx).
    Periodic in x and z, no-slip walls at y=0 and y=ny-1.
    Uses vectorized streaming via roll operations.
    """

    def __init__(self, cfg: ChannelConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dt = torch.float32 if cfg.dtype == "float32" else torch.float64

        from tensorlbm.d3q19 import C, W, OPPOSITE
        from tensorlbm import collide_smagorinsky_mrt3d, stream3d, equilibrium3d, macroscopic3d

        self.C = C.to(self.device, dtype=self.dt)
        self.W = W.to(self.device, dtype=self.dt)
        self.OPPOSITE = OPPOSITE.to(self.device)
        self.collide_smag = collide_smagorinsky_mrt3d
        self.stream = stream3d
        self.equilibrium = equilibrium3d
        self.macroscopic = macroscopic3d

        self.nx, self.ny, self.nz = cfg.nx, cfg.ny, cfg.nz
        self.nu = cfg.nu
        self.tau_plus = cfg.tau_plus

        # Pre-compute wall mask indices for bounce-back
        cy = self.C[:, 1]
        self.up_indices = torch.where(cy > 0)[0]   # directions going up
        self.down_indices = torch.where(cy < 0)[0]  # directions going down

        self._init_populations()

    def _init_populations(self) -> None:
        """Initialize with parabolic profile + perturbation."""
        cfg = self.cfg
        h = cfg.half_height
        a = cfg.body_force

        # Wall-normal coordinate
        y = torch.arange(self.ny, device=self.device, dtype=self.dt).view(1, self.ny, 1)

        # Parabolic profile
        u_mean = a * y * (2 * h - y) / (2.0 * self.nu)

        # Perturbation
        torch.manual_seed(cfg.seed)
        pert = cfg.perturbation_amplitude * cfg.u_tau * torch.randn(
            self.nz, self.ny, self.nx, device=self.device, dtype=self.dt
        )
        pert[:, 0, :] = 0.0
        pert[:, -1, :] = 0.0

        ux = u_mean.expand(self.nz, self.ny, self.nx) + pert
        uy = torch.zeros(self.nz, self.ny, self.nx, device=self.device, dtype=self.dt)
        uz = 0.1 * cfg.perturbation_amplitude * cfg.u_tau * torch.randn(
            self.nz, self.ny, self.nx, device=self.device, dtype=self.dt
        )
        uz[:, 0, :] = 0.0
        uz[:, -1, :] = 0.0

        rho = torch.ones(self.nz, self.ny, self.nx, device=self.device, dtype=self.dt)
        self.f = self.equilibrium(rho, ux, uy, uz)

    def _apply_walls(self) -> None:
        """Vectorized bounce-back at walls."""
        opp = self.OPPOSITE

        # Bottom wall: for directions going up (cy > 0), reflect from opposite
        for i in self.up_indices:
            j = opp[i]
            self.f[i, :, 0, :] = self.f[j, :, 0, :]

        # Top wall: for directions going down (cy < 0), reflect from opposite
        for i in self.down_indices:
            j = opp[i]
            self.f[i, :, self.ny - 1, :] = self.f[j, :, self.ny - 1, :]

    def _apply_body_force(self) -> None:
        """Apply body force using Guo forcing scheme."""
        fx = self.cfg.body_force
        cx = self.C[:, 0]
        factor = 1.0 - 0.5 / self.tau_plus

        # Vectorized force application
        force_term = self.W * cx * fx * factor  # (19,)
        for i in range(19):
            self.f[i] += force_term[i]

    def step(self) -> None:
        """One time step: collide -> stream -> walls -> force."""
        tau = self.tau_plus
        cs = self.cfg.smagorinsky_cs if self.cfg.smagorinsky_cs > 0 else 0.0

        if cs > 0:
            self.f = self.collide_smag(self.f, tau=tau, C_s=cs)
        else:
            from tensorlbm.solver import collide_mrt
            self.f = collide_mrt(self.f, tau=tau)

        self.f = self.stream(self.f)
        self._apply_walls()
        self._apply_body_force()

    def get_macroscopic(self):
        """Compute macroscopic quantities."""
        return self.macroscopic(self.f)

    def run(self, n_steps: int, callback=None) -> dict:
        """Run simulation for n_steps."""
        import time
        t0 = time.time()

        for step in range(1, n_steps + 1):
            self.step()
            if callback and step % 1000 == 0:
                callback(step, self)

        elapsed = time.time() - t0
        return {
            "steps": n_steps,
            "elapsed_s": elapsed,
            "steps_per_sec": n_steps / elapsed if elapsed > 0 else 0,
        }


def run_channel(cfg: ChannelConfig) -> dict:
    """Convenience function to run a channel flow simulation."""
    solver = ChannelSolver(cfg)
    info = solver.run(cfg.steps)
    return {"solver": solver, "config": cfg, "run_info": info}
