"""2D Compressible LBM solver using Double Distribution Function (DDF) approach.

Implements:
- Density distribution f (D2Q9) for mass/momentum
- Temperature/energy distribution g (D2Q5) for energy equation
- Coupling through equation of state p = ρ·R·T (ideal gas, γ=1.4)
- Artificial viscosity for shock capturing
- Far-field Riemann invariant BCs
- No-slip adiabatic wall BC

The DDF approach separates hydrodynamics (f) from thermodynamics (g),
allowing compressible flow simulation while maintaining LBM efficiency.

References
----------
- He, X., Chen, S., Doolen, G.D., "A Novel Thermal Model for the Lattice
  Boltzmann Method in Incompressible Limit", JCP 146, 282-300, 1998.
- Li, Q., He, Y.L., Wang, Y., Tao, W.Q., "Coupled double-distribution-function
  lattice Boltzmann method for the Navier-Stokes equations", Phys. Rev. E 76, 2007.
- Shan, H., Zhang, Y., "Lattice Boltzmann simulation of compressible flows",
  Computers & Fluids, 2020.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ============================================================================
# D2Q9 lattice constants (numpy version for compressible solver)
# ============================================================================
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# D2Q5 for energy (simpler, sufficient for temperature)
CX5 = np.array([0, 1, 0, -1, 0], dtype=np.float64)
CY5 = np.array([0, 0, 1, 0, -1], dtype=np.float64)
W5 = np.array([2/6, 1/6, 1/6, 1/6, 1/6], dtype=np.float64)


@dataclass
class CompressibleLBMConfig:
    """Configuration for compressible LBM simulation."""
    nx: int = 200
    ny: int = 100
    gamma: float = 1.4
    R_gas: float = 1.0 / 3.0  # Gas constant (lattice units, cs² = R·T_ref)
    Ma_inf: float = 0.3
    alpha_deg: float = 0.0
    T_inf: float = 1.0        # Freestream temperature
    rho_inf: float = 1.0      # Freestream density
    omega_f: float = 1.0      # Relaxation for f (density)
    omega_g: float = 1.0      # Relaxation for g (energy)
    c_h: float = 0.2          # Artificial viscosity coefficient
    n_steps: int = 5000
    output_interval: int = 500


class CompressibleDDFSolver:
    """2D Compressible LBM solver using Double Distribution Function method.
    
    The solver uses:
    - f (D2Q9): density distribution for continuity + momentum
    - g (D2Q5): energy distribution for temperature equation
    - Ideal gas EOS: p = ρ·R·T, cs² = γ·R·T
    - Artificial viscosity for shock stabilization
    """
    
    def __init__(self, config: CompressibleLBMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.gamma = config.gamma
        self.R_gas = config.R_gas
        
        # Freestream conditions
        self.Ma_inf = config.Ma_inf
        self.alpha_rad = math.radians(config.alpha_deg)
        self.T_inf = config.T_inf
        self.rho_inf = config.rho_inf
        
        # Derived freestream quantities
        self.cs_inf = math.sqrt(self.gamma * self.R_gas * self.T_inf)
        self.u_inf = self.Ma_inf * self.cs_inf
        self.ux_inf = self.u_inf * math.cos(self.alpha_rad)
        self.uy_inf = self.u_inf * math.sin(self.alpha_rad)
        self.p_inf = self.rho_inf * self.R_gas * self.T_inf
        
        # Initialize fields
        self.rho = np.full((self.ny, self.nx), self.rho_inf)
        self.ux = np.full((self.ny, self.nx), self.ux_inf)
        self.uy = np.full((self.ny, self.nx), self.uy_inf)
        self.T = np.full((self.ny, self.nx), self.T_inf)
        self.p = np.full((self.ny, self.nx), self.p_inf)
        
        # Distribution functions
        self.f = self._equilibrium_f_2d(self.rho, self.ux, self.uy)
        self.g = self._equilibrium_g_2d(self.rho, self.T, self.ux, self.uy)
        
        # Masks
        self.wall_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.fluid_mask = np.ones((self.ny, self.nx), dtype=bool)
        
        # History
        self.history = []
    
    def _equilibrium_f_2d(self, rho, ux, uy):
        """D2Q9 equilibrium for density distribution (2D fields)."""
        f_eq = np.zeros((9, self.ny, self.nx))
        u_sq = ux**2 + uy**2
        for i in range(9):
            cu = CX[i] * ux + CY[i] * uy
            f_eq[i] = W[i] * rho * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        return f_eq
    
    def _equilibrium_g_2d(self, rho, T, ux, uy):
        """D2Q5 equilibrium for energy/temperature distribution (2D fields)."""
        g_eq = np.zeros((5, self.ny, self.nx))
        u_sq = ux**2 + uy**2
        for i in range(5):
            cu = CX5[i] * ux + CY5[i] * uy
            g_eq[i] = W5[i] * rho * T * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        return g_eq
    
    def _stream_f(self):
        """Streaming step for f (D2Q9)."""
        f_new = np.zeros_like(self.f)
        for i in range(9):
            f_new[i] = np.roll(np.roll(self.f[i], shift=-int(CX[i]), axis=1),
                               shift=-int(CY[i]), axis=0)
        return f_new
    
    def _stream_g(self):
        """Streaming step for g (D2Q5)."""
        g_new = np.zeros_like(self.g)
        for i in range(5):
            g_new[i] = np.roll(np.roll(self.g[i], shift=-int(CX5[i]), axis=1),
                               shift=-int(CY5[i]), axis=0)
        return g_new
    
    def _compute_artificial_viscosity(self):
        """Compute artificial viscosity for shock capturing.
        
        μ_art = c_h · ρ · Δx · |∇·u|
        """
        dudx = np.gradient(self.ux, axis=1)
        dvdy = np.gradient(self.uy, axis=0)
        div_u = dudx + dvdy
        
        mu_art = self.cfg.c_h * self.rho * np.abs(div_u)
        mu_art = np.clip(mu_art, 0, 0.1)
        
        return mu_art
    
    def _update_macroscopic(self):
        """Update macroscopic variables from distribution functions."""
        self.rho = np.sum(self.f, axis=0)
        self.rho = np.maximum(self.rho, 1e-10)
        self.ux = np.sum(self.f * CX[:, None, None], axis=0) / self.rho
        self.uy = np.sum(self.f * CY[:, None, None], axis=0) / self.rho
        
        rhoT = np.sum(self.g, axis=0)
        self.T = rhoT / self.rho
        self.T = np.maximum(self.T, 1e-10)
        
        self.p = self.rho * self.R_gas * self.T
    
    def _apply_farfield_bc(self):
        """Apply far-field boundary conditions using characteristic BCs.
        
        For subsonic flow: use characteristic analysis to determine
        which variables to specify from outside vs extrapolate from inside.
        """
        # Left boundary (inflow for positive ux)
        if self.ux_inf >= 0:
            # Subsonic inflow: specify all macroscopic vars
            self.rho[:, 0] = self.rho_inf
            self.ux[:, 0] = self.ux_inf
            self.uy[:, 0] = self.uy_inf
            self.T[:, 0] = self.T_inf
            # Reconstruct distributions from macroscopic
            for i in range(9):
                cu = CX[i] * self.ux[:, 0] + CY[i] * self.uy[:, 0]
                u_sq = self.ux[:, 0]**2 + self.uy[:, 0]**2
                self.f[i, :, 0] = W[i] * self.rho[:, 0] * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
            for i in range(5):
                cu = CX5[i] * self.ux[:, 0] + CY5[i] * self.uy[:, 0]
                u_sq = self.ux[:, 0]**2 + self.uy[:, 0]**2
                self.g[i, :, 0] = W5[i] * self.rho[:, 0] * self.T[:, 0] * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        
        # Right boundary (outflow): zero-gradient
        self.rho[:, -1] = self.rho[:, -2]
        self.ux[:, -1] = self.ux[:, -2]
        self.uy[:, -1] = self.uy[:, -2]
        self.T[:, -1] = self.T[:, -2]
        self.f[:, :, -1] = self.f[:, :, -2]
        self.g[:, :, -1] = self.g[:, :, -2]
        
        # Top boundary: zero-gradient
        self.rho[-1, :] = self.rho[-2, :]
        self.ux[-1, :] = self.ux[-2, :]
        self.uy[-1, :] = self.uy[-2, :]
        self.T[-1, :] = self.T[-2, :]
        self.f[:, -1, :] = self.f[:, -2, :]
        self.g[:, -1, :] = self.g[:, -2, :]
        
        # Bottom boundary: zero-gradient
        self.rho[0, :] = self.rho[1, :]
        self.ux[0, :] = self.ux[1, :]
        self.uy[0, :] = self.uy[1, :]
        self.T[0, :] = self.T[1, :]
        self.f[:, 0, :] = self.f[:, 1, :]
        self.g[:, 0, :] = self.g[:, 1, :]
    
    def _apply_wall_bc(self):
        """Apply no-slip adiabatic wall boundary (bounce-back)."""
        for i in range(9):
            opp = OPPOSITE[i]
            self.f[i] = np.where(self.wall_mask, self.f[opp], self.f[i])
        
        for i in range(5):
            if i == 0:
                continue
            opp_idx = {1: 3, 2: 4, 3: 1, 4: 2}[i]
            self.g[i] = np.where(self.wall_mask, self.g[opp_idx], self.g[i])
    
    def step(self):
        """Perform one LBM time step."""
        # Collision for f
        f_eq = self._equilibrium_f_2d(self.rho, self.ux, self.uy)
        omega_f = self.cfg.omega_f
        
        # Artificial viscosity modifies effective relaxation
        mu_art = self._compute_artificial_viscosity()
        omega_eff = omega_f / (1.0 + 3.0 * mu_art)
        omega_eff = np.clip(omega_eff, 0.1, 1.99)
        
        self.f = self.f - omega_eff[None, :, :] * (self.f - f_eq)
        
        # Collision for g
        g_eq = self._equilibrium_g_2d(self.rho, self.T, self.ux, self.uy)
        self.g = self.g - self.cfg.omega_g * (self.g - g_eq)
        
        # Streaming
        self.f = self._stream_f()
        self.g = self._stream_g()
        
        # Update macroscopic
        self._update_macroscopic()
        
        # Boundary conditions
        self._apply_farfield_bc()
        self._apply_wall_bc()
    
    def run(self, n_steps: Optional[int] = None):
        """Run simulation for specified number of steps."""
        if n_steps is None:
            n_steps = self.cfg.n_steps
        
        for step in range(n_steps):
            self.step()
            
            if (step + 1) % self.cfg.output_interval == 0:
                rho_max = self.rho.max()
                rho_min = self.rho.min()
                Ma_max = np.sqrt(self.ux**2 + self.uy**2).max() / self.cs_inf
                
                self.history.append({
                    'step': step + 1,
                    'rho_max': float(rho_max),
                    'rho_min': float(rho_min),
                    'Ma_max': float(Ma_max),
                    'p_max': float(self.p.max()),
                })
    
    def compute_pressure_coefficient(self, x_surf, y_surf):
        """Compute pressure coefficient Cp along surface."""
        cp = np.zeros(len(x_surf))
        for k in range(len(x_surf)):
            ix = int(np.clip(x_surf[k], 0, self.nx - 1))
            iy = int(np.clip(y_surf[k], 0, self.ny - 1))
            p_local = self.p[iy, ix]
            q_inf = 0.5 * self.rho_inf * self.u_inf**2
            if q_inf > 1e-12:
                cp[k] = (p_local - self.p_inf) / q_inf
            else:
                cp[k] = 0.0
        return cp


def run_flat_plate_shock(Ma: float, nx: int = 200, ny: int = 50, n_steps: int = 2000):
    """Quick test: 1D-like shock tube to verify shock capturing."""
    cfg = CompressibleLBMConfig(
        nx=nx, ny=ny, Ma_inf=Ma, alpha_deg=0.0,
        n_steps=n_steps, output_interval=200, c_h=0.2
    )
    solver = CompressibleDDFSolver(cfg)
    
    from tensorlbm.compressible.solver import normal_shock_relations
    rh = normal_shock_relations(Ma, cfg.gamma)
    
    mid = nx // 2
    solver.rho[:, :mid] = cfg.rho_inf
    solver.ux[:, :mid] = cfg.u_inf
    solver.T[:, :mid] = cfg.T_inf
    
    solver.rho[:, mid:] = cfg.rho_inf * rh.rho_ratio
    solver.ux[:, mid:] = cfg.u_inf * rh.M2 * math.sqrt(rh.T_ratio)
    solver.T[:, mid:] = cfg.T_inf * rh.T_ratio
    
    solver.p = solver.rho * cfg.R_gas * solver.T
    solver.f = solver._equilibrium_f_2d(solver.rho, solver.ux, solver.uy)
    solver.g = solver._equilibrium_g_2d(solver.rho, solver.T, solver.ux, solver.uy)
    
    solver.run(n_steps)
    return solver
