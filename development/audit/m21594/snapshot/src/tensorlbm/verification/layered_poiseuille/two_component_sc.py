"""Two-component Shan-Chen LBM with EDM forcing for layered Poiseuille validation.

This module implements a binary (two-component) Shan-Chen multiphase model
where each component has its own distribution function and relaxation time.
The interaction force between components creates phase separation, and when
combined with different viscosities, naturally forms layered flows.

References
----------
- Shan & Chen (1993) Phys. Rev. E 47, 1815 - original SC model
- Kupershtokh (2004) - Exact Difference Method (EDM) forcing
- Martys & Chen (1996) Phys. Rev. E 53, 743 - two-component SC
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# D2Q9 lattice constants
CS2 = 1.0 / 3.0
W = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=torch.float32)
C = torch.tensor([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
], dtype=torch.float32)
OPPOSITE = [0, 3, 4, 1, 2, 7, 8, 5, 6]


def psi_exp(rho: torch.Tensor, rho0: float = 1.0) -> torch.Tensor:
    """Exponential pseudopotential: psi = rho0 * (1 - exp(-rho/rho0))."""
    return rho0 * (1.0 - torch.exp(-rho / rho0))


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium distribution."""
    feq = torch.zeros((9,) + rho.shape, dtype=rho.dtype, device=rho.device)
    u2 = ux**2 + uy**2
    for i in range(9):
        cu = 3.0 * (C[i, 0] * ux + C[i, 1] * uy)
        feq[i] = W[i] * rho * (1.0 + cu + 0.5 * cu**2 - 1.5 * u2)
    return feq


def macroscopic(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute macroscopic density and velocity from distribution function."""
    rho = f.sum(dim=0)
    safe = torch.clamp(rho, min=1e-12)
    ux = (f * C[:, 0].view(9, 1, 1)).sum(dim=0) / safe
    uy = (f * C[:, 1].view(9, 1, 1)).sum(dim=0) / safe
    return rho, ux, uy


def stream_periodic(f: torch.Tensor) -> torch.Tensor:
    """Periodic streaming."""
    f_new = torch.empty_like(f)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        f_new[i] = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
    return f_new


def stream_bounce_back_y(f: torch.Tensor, solid_top: torch.Tensor, solid_bot: torch.Tensor) -> torch.Tensor:
    """Streaming with bounce-back at top/bottom walls, periodic in x."""
    f_new = torch.empty_like(f)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        # Periodic in x
        streamed = torch.roll(f[i], shifts=(0, -cx), dims=(0, 1))
        # Bounce-back at walls
        if cy > 0:  # Moving upward
            streamed[1, :] = f[OPPOSITE[i], 1, :]
        if cy < 0:  # Moving downward
            streamed[-2, :] = f[OPPOSITE[i], -2, :]
        f_new[i] = streamed
    # Solid cells keep their value
    f_new[:, solid_top] = f[:, solid_top]
    f_new[:, solid_bot] = f[:, solid_bot]
    return f_new


@dataclass
class TwoComponentSCConfig:
    """Configuration for two-component SC simulation."""
    ny: int
    nx: int
    tau_a: float  # Relaxation time for component A
    tau_b: float  # Relaxation time for component B
    G: float  # SC coupling constant (>0 = repulsion between components)
    g_x: float  # Body force in x (drives flow)
    g_y: float  # Body force in y
    steps: int
    init_rho_a: float = 2.0  # Initial density of component A
    init_rho_b: float = 2.0  # Initial density of component B
    init_layer_frac: float = 0.5  # Fraction of domain for component A (from bottom)
    rho0_psi: float = 1.0  # Reference density for pseudopotential
    forcing: str = "edm"  # "edm", "guo", or "velocity_shift"
    sample_every: int = 100
    device: str = "cpu"


class TwoComponentSCSolver:
    """Two-component Shan-Chen solver with pluggable forcing schemes."""
    
    def __init__(self, cfg: TwoComponentSCConfig):
        self.cfg = cfg
        self.ny = cfg.ny
        self.nx = cfg.nx
        self.device = cfg.device
        
        # Initialize distribution functions for both components
        self.f_a = self._initialize_component(cfg.init_rho_a, cfg.init_layer_frac, is_a=True)
        self.f_b = self._initialize_component(cfg.init_rho_b, cfg.init_layer_frac, is_a=False)
        
        # Solid masks (top and bottom walls)
        self.solid_top = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=self.device)
        self.solid_top[0, :] = True
        self.solid_bot = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=self.device)
        self.solid_bot[-1, :] = True
        
        self.step_count = 0
        
    def _initialize_component(self, rho0: float, layer_frac: float, is_a: bool) -> torch.Tensor:
        """Initialize a component with layered configuration."""
        rho = torch.zeros(self.ny, self.nx, dtype=torch.float32, device=self.device)
        interface_y = int(self.ny * layer_frac)
        
        if is_a:
            # Component A in bottom layer
            rho[interface_y:-1, :] = rho0
        else:
            # Component B in top layer
            rho[1:interface_y, :] = rho0
        
        # Small perturbation to break symmetry
        rho += 0.01 * torch.randn_like(rho)
        rho = torch.clamp(rho, min=0.1)
        
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        return equilibrium(rho, ux, uy)
    
    def _compute_interaction_force(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SC interaction forces between components.
        
        For two-component SC:
        F^A = -G * psi^A * sum_i w_i * psi^B(x+c_i) * c_i
        F^B = -G * psi^B * sum_i w_i * psi^A(x+c_i) * c_i
        
        Returns (fx_a, fy_a, fx_b, fy_b)
        """
        rho_a, _, _ = macroscopic(self.f_a)
        rho_b, _, _ = macroscopic(self.f_b)
        
        psi_a = psi_exp(rho_a, self.cfg.rho0_psi)
        psi_b = psi_exp(rho_b, self.cfg.rho0_psi)
        
        # Zero out solid cells
        psi_a = psi_a.masked_fill(self.solid_top | self.solid_bot, 0.0)
        psi_b = psi_b.masked_fill(self.solid_top | self.solid_bot, 0.0)
        
        # Neighbor sum for component A force (uses psi_b)
        fx_a = torch.zeros_like(rho_a)
        fy_a = torch.zeros_like(rho_a)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            if cx == 0 and cy == 0:
                continue
            psi_b_shift = torch.roll(psi_b, shifts=(-cy, -cx), dims=(0, 1))
            psi_b_shift = psi_b_shift.masked_fill(self.solid_top | self.solid_bot, 0.0)
            fx_a += W[i] * cx * psi_b_shift
            fy_a += W[i] * cy * psi_b_shift
        
        fx_a = -self.cfg.G * psi_a * fx_a
        fy_a = -self.cfg.G * psi_a * fy_a
        
        # Neighbor sum for component B force (uses psi_a)
        fx_b = torch.zeros_like(rho_b)
        fy_b = torch.zeros_like(rho_b)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            if cx == 0 and cy == 0:
                continue
            psi_a_shift = torch.roll(psi_a, shifts=(-cy, -cx), dims=(0, 1))
            psi_a_shift = psi_a_shift.masked_fill(self.solid_top | self.solid_bot, 0.0)
            fx_b += W[i] * cx * psi_a_shift
            fy_b += W[i] * cy * psi_a_shift
        
        fx_b = -self.cfg.G * psi_b * fx_b
        fy_b = -self.cfg.G * psi_b * fy_b
        
        # Add external body force
        if self.cfg.g_x != 0.0:
            fx_a += rho_a * self.cfg.g_x
            fx_b += rho_b * self.cfg.g_x
        if self.cfg.g_y != 0.0:
            fy_a += rho_a * self.cfg.g_y
            fy_b += rho_b * self.cfg.g_y
        
        return fx_a, fy_a, fx_b, fy_b
    
    def _collide_edm(self, f: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, tau: float) -> torch.Tensor:
        """EDM forcing: f_out = f - (f - feq_u)/tau + (feq_d - feq_u)."""
        rho, ux, uy = macroscopic(f)
        rho_safe = torch.clamp(rho, min=1e-12)
        
        feq_u = equilibrium(rho, ux, uy)
        uxd = ux + fx / rho_safe
        uyd = uy + fy / rho_safe
        feq_d = equilibrium(rho, uxd, uyd)
        
        return f - (f - feq_u) / tau + (feq_d - feq_u)
    
    def _collide_guo(self, f: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, tau: float) -> torch.Tensor:
        """Guo forcing scheme."""
        rho, ux, uy = macroscopic(f)
        rho_safe = torch.clamp(rho, min=1e-12)
        
        uxp = ux + fx / (2.0 * rho_safe)
        uyp = uy + fy / (2.0 * rho_safe)
        feq = equilibrium(rho, uxp, uyp)
        
        cf = C[:, 0].view(9, 1, 1) * fx.unsqueeze(0) + C[:, 1].view(9, 1, 1) * fy.unsqueeze(0)
        cu = C[:, 0].view(9, 1, 1) * uxp.unsqueeze(0) + C[:, 1].view(9, 1, 1) * uyp.unsqueeze(0)
        uf = uxp * fx + uyp * fy
        
        src = W.view(9, 1, 1) * (1.0 - 1.0 / (2.0 * tau)) * ((cf - uf.unsqueeze(0)) / CS2 + cu * cf / CS2**2)
        return f - (f - feq) / tau + src
    
    def _collide_velocity_shift(self, f: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, tau: float) -> torch.Tensor:
        """Velocity-shift forcing."""
        rho, ux, uy = macroscopic(f)
        rho_safe = torch.clamp(rho, min=1e-12)
        
        uxeq = ux + tau * fx / rho_safe
        uyeq = uy + tau * fy / rho_safe
        feq = equilibrium(rho, uxeq, uyeq)
        
        return f - (f - feq) / tau
    
    def step(self) -> None:
        """One time step: compute forces, collide, stream."""
        fx_a, fy_a, fx_b, fy_b = self._compute_interaction_force()
        
        # Collide
        if self.cfg.forcing == "edm":
            collide_fn = self._collide_edm
        elif self.cfg.forcing == "guo":
            collide_fn = self._collide_guo
        elif self.cfg.forcing == "velocity_shift":
            collide_fn = self._collide_velocity_shift
        else:
            raise ValueError(f"Unknown forcing: {self.cfg.forcing}")
        
        f_a_post = collide_fn(self.f_a, fx_a, fy_a, self.cfg.tau_a)
        f_b_post = collide_fn(self.f_b, fx_b, fy_b, self.cfg.tau_b)
        
        # Stream with bounce-back
        self.f_a = stream_bounce_back_y(f_a_post, self.solid_top, self.solid_bot)
        self.f_b = stream_bounce_back_y(f_b_post, self.solid_top, self.solid_bot)
        
        self.step_count += 1
    
    def get_macroscopic(self) -> Dict[str, torch.Tensor]:
        """Get current macroscopic fields."""
        rho_a, ux_a, uy_a = macroscopic(self.f_a)
        rho_b, ux_b, uy_b = macroscopic(self.f_b)
        
        rho_total = rho_a + rho_b
        ux_total = (rho_a * ux_a + rho_b * ux_b) / torch.clamp(rho_total, min=1e-12)
        uy_total = (rho_a * uy_a + rho_b * uy_b) / torch.clamp(rho_total, min=1e-12)
        
        return {
            "rho_a": rho_a,
            "rho_b": rho_b,
            "rho_total": rho_total,
            "ux_a": ux_a,
            "ux_b": ux_b,
            "ux_total": ux_total,
            "uy_a": uy_a,
            "uy_b": uy_b,
            "uy_total": uy_total,
        }
    
    def get_x_averaged_profile(self) -> Dict[str, np.ndarray]:
        """Get x-averaged velocity and density profiles."""
        macro = self.get_macroscopic()
        
        # Average over x, exclude walls
        y_slice = slice(1, self.ny - 1)
        
        return {
            "y": np.arange(1, self.ny - 1, dtype=float) - 0.5,
            "rho_a": macro["rho_a"][y_slice, :].mean(dim=1).cpu().numpy(),
            "rho_b": macro["rho_b"][y_slice, :].mean(dim=1).cpu().numpy(),
            "rho_total": macro["rho_total"][y_slice, :].mean(dim=1).cpu().numpy(),
            "ux_a": macro["ux_a"][y_slice, :].mean(dim=1).cpu().numpy(),
            "ux_b": macro["ux_b"][y_slice, :].mean(dim=1).cpu().numpy(),
            "ux_total": macro["ux_total"][y_slice, :].mean(dim=1).cpu().numpy(),
        }
    
    def run(self, monitor_every: Optional[int] = None) -> Dict[str, Any]:
        """Run simulation to completion."""
        history = []
        
        for step in range(1, self.cfg.steps + 1):
            self.step()
            
            if monitor_every and step % monitor_every == 0:
                macro = self.get_macroscopic()
                rho_a_mean = macro["rho_a"].mean().item()
                rho_b_mean = macro["rho_b"].mean().item()
                ux_max = macro["ux_total"].max().item()
                history.append({
                    "step": step,
                    "rho_a_mean": rho_a_mean,
                    "rho_b_mean": rho_b_mean,
                    "ux_max": ux_max,
                })
            
            if step % self.cfg.sample_every == 0:
                # Could add convergence check here
                pass
        
        return {
            "steps": self.step_count,
            "history": history,
            "final_profile": self.get_x_averaged_profile(),
        }
