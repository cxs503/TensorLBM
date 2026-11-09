"""NACA0012 airfoil simulation with compressible DDF LBM.

Implements 2D compressible LBM simulation of NACA0012 airfoil.

Note: Standard D2Q9 LBM is limited to Ma < 0.3 for stability.
For higher Mach numbers, use the Godunov solver or kinetic flux solver.

Features:
- Airfoil geometry embedded in Cartesian grid via voxelization
- Double Distribution Function (DDF): f (D2Q9) + g (D2Q5)
- Ideal gas EOS: p = ρRT, γ = 1.4
- Artificial viscosity for shock capturing
- Entropy monitoring for shock detection
- Total enthalpy conservation tracking
- Pressure coefficient (Cp) extraction along airfoil surface

References
----------
- He, X., Chen, S., Doolen, G.D., JCP 146, 282-300, 1998
- Li, Q., He, Y.L., Wang, Y., Tao, W.Q., PRE 76, 2007
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict

from .airfoil import naca0012_surface_points
from .solver import normal_shock_relations

# Lattice constants (numpy, 2D)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

CX5 = np.array([0, 1, 0, -1, 0], dtype=np.float64)
CY5 = np.array([0, 0, 1, 0, -1], dtype=np.float64)
W5 = np.array([2/6, 1/6, 1/6, 1/6, 1/6], dtype=np.float64)


@dataclass
class NACA0012LBMConfig:
    nx: int = 400
    ny: int = 200
    Ma_inf: float = 0.15  # Keep low for D2Q9 stability
    alpha_deg: float = 2.0
    gamma: float = 1.4
    R_gas: float = 1.0 / 3.0
    T_inf: float = 1.0
    rho_inf: float = 1.0
    omega_f: float = 0.8  # Lower omega for stability
    omega_g: float = 0.8
    c_h: float = 0.05
    n_steps: int = 3000
    output_interval: int = 500
    chord_pixels: int = 80
    chord_offset_x: float = 0.25
    chord_offset_y: float = 0.50


class NACA0012LBMSolver:
    """Compressible DDF LBM solver for NACA0012 airfoil.
    
    Note: Limited to Ma < 0.3 for D2Q9 stability.
    For transonic/supersonic flows, use Godunov or kinetic flux solver.
    """
    
    def __init__(self, config: NACA0012LBMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.gamma = config.gamma
        self.R_gas = config.R_gas
        
        self.Ma_inf = config.Ma_inf
        self.alpha_rad = math.radians(config.alpha_deg)
        self.T_inf = config.T_inf
        self.rho_inf = config.rho_inf
        
        # Sound speed in lattice units: cs² = γ·R·T
        self.cs_inf = math.sqrt(self.gamma * self.R_gas * self.T_inf)
        self.u_inf = self.Ma_inf * self.cs_inf
        self.ux_inf = self.u_inf * math.cos(self.alpha_rad)
        self.uy_inf = self.u_inf * math.sin(self.alpha_rad)
        self.p_inf = self.rho_inf * self.R_gas * self.T_inf
        self.q_inf = 0.5 * self.rho_inf * self.u_inf**2
        
        self._init_fields()
        self._init_airfoil_mask()
        self._init_distributions()
        
        self.history: List[Dict] = []
        self.entropy_field: np.ndarray = np.zeros((self.ny, self.nx))
        self.total_enthalpy_field: np.ndarray = np.zeros((self.ny, self.nx))
        self.cp_surface: Optional[np.ndarray] = None
        self.x_surface: Optional[np.ndarray] = None
        self.y_surface: Optional[np.ndarray] = None
    
    def _init_fields(self):
        self.rho = np.full((self.ny, self.nx), self.rho_inf, dtype=np.float64)
        self.ux = np.full((self.ny, self.nx), self.ux_inf, dtype=np.float64)
        self.uy = np.full((self.ny, self.nx), self.uy_inf, dtype=np.float64)
        self.T = np.full((self.ny, self.nx), self.T_inf, dtype=np.float64)
        self.p = np.full((self.ny, self.nx), self.p_inf, dtype=np.float64)
    
    def _init_airfoil_mask(self):
        self.wall_mask = np.zeros((self.ny, self.nx), dtype=bool)
        
        n_surf = 200
        x_surf_all, y_surf_all = naca0012_surface_points(n_points=n_surf // 2)
        
        cp = self.cfg.chord_pixels
        ox = int(self.cfg.chord_offset_x * self.nx)
        oy = int(self.cfg.chord_offset_y * self.ny)
        
        x_all = x_surf_all * cp + ox
        y_all = y_surf_all * cp + oy
        
        for j in range(self.ny):
            intersections = []
            for k in range(len(x_all) - 1):
                y1, y2 = y_all[k], y_all[k + 1]
                x1, x2 = x_all[k], x_all[k + 1]
                if (y1 <= j < y2) or (y2 <= j < y1):
                    if abs(y2 - y1) > 1e-12:
                        x_int = x1 + (j - y1) * (x2 - x1) / (y2 - y1)
                        intersections.append(x_int)
            
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                x_start = int(max(0, min(self.nx - 1, int(intersections[k]))))
                x_end = int(max(0, min(self.nx - 1, int(intersections[k + 1]))))
                self.wall_mask[j, x_start:x_end + 1] = True
        
        try:
            from scipy.ndimage import binary_dilation
            struct = np.ones((3, 3), dtype=bool)
            self.wall_mask = binary_dilation(self.wall_mask, structure=struct)
        except ImportError:
            pass
        
        self.fluid_mask = ~self.wall_mask
        self._surface_x = x_all
        self._surface_y = y_all
    
    def _init_distributions(self):
        self.f = self._equilibrium_f(self.rho, self.ux, self.uy)
        self.g = self._equilibrium_g(self.rho, self.T, self.ux, self.uy)
    
    def _equilibrium_f(self, rho, ux, uy):
        f_eq = np.zeros((9, self.ny, self.nx), dtype=np.float64)
        u_sq = ux**2 + uy**2
        for i in range(9):
            cu = CX[i] * ux + CY[i] * uy
            f_eq[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)
        return f_eq
    
    def _equilibrium_g(self, rho, T, ux, uy):
        g_eq = np.zeros((5, self.ny, self.nx), dtype=np.float64)
        u_sq = ux**2 + uy**2
        for i in range(5):
            cu = CX5[i] * ux + CY5[i] * uy
            g_eq[i] = W5[i] * rho * T * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)
        return g_eq
    
    def _stream_f(self):
        f_new = np.zeros_like(self.f)
        for i in range(9):
            f_new[i] = np.roll(
                np.roll(self.f[i], shift=-int(CX[i]), axis=1),
                shift=-int(CY[i]), axis=0
            )
        return f_new
    
    def _stream_g(self):
        g_new = np.zeros_like(self.g)
        for i in range(5):
            g_new[i] = np.roll(
                np.roll(self.g[i], shift=-int(CX5[i]), axis=1),
                shift=-int(CY5[i]), axis=0
            )
        return g_new
    
    def _compute_entropy(self):
        rho_safe = np.maximum(self.rho, 1e-12)
        p_safe = np.maximum(self.p, 1e-12)
        self.entropy_field = np.log(p_safe / (rho_safe ** self.gamma))
        return self.entropy_field
    
    def _compute_total_enthalpy(self):
        cp = self.gamma * self.R_gas / (self.gamma - 1.0)
        u_sq = self.ux**2 + self.uy**2
        self.total_enthalpy_field = cp * self.T + 0.5 * u_sq
        return self.total_enthalpy_field
    
    def _compute_artificial_viscosity(self):
        dudx = np.gradient(self.ux, axis=1)
        dvdy = np.gradient(self.uy, axis=0)
        div_u = dudx + dvdy
        mu_art = self.cfg.c_h * self.rho * np.maximum(-div_u, 0.0)
        mu_art = np.clip(mu_art, 0, 0.1)
        return mu_art
    
    def _update_macroscopic(self):
        self.rho = np.sum(self.f, axis=0)
        self.rho = np.maximum(self.rho, 1e-10)
        self.ux = np.sum(self.f * CX[:, None, None], axis=0) / self.rho
        self.uy = np.sum(self.f * CY[:, None, None], axis=0) / self.rho
        self.ux[self.wall_mask] = 0.0
        self.uy[self.wall_mask] = 0.0
        rhoT = np.sum(self.g, axis=0)
        self.T = rhoT / self.rho
        self.T = np.maximum(self.T, 1e-10)
        self.p = self.rho * self.R_gas * self.T
    
    def _apply_boundary_conditions(self):
        # Left: freestream
        self.rho[:, 0] = self.rho_inf
        self.ux[:, 0] = self.ux_inf
        self.uy[:, 0] = self.uy_inf
        self.T[:, 0] = self.T_inf
        for i in range(9):
            cu = CX[i] * self.ux[:, 0] + CY[i] * self.uy[:, 0]
            u_sq = self.ux[:, 0]**2 + self.uy[:, 0]**2
            self.f[i, :, 0] = W[i] * self.rho[:, 0] * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        for i in range(5):
            cu = CX5[i] * self.ux[:, 0] + CY5[i] * self.uy[:, 0]
            u_sq = self.ux[:, 0]**2 + self.uy[:, 0]**2
            self.g[i, :, 0] = W5[i] * self.rho[:, 0] * self.T[:, 0] * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        
        # Right: zero-gradient
        self.rho[:, -1] = self.rho[:, -2]
        self.ux[:, -1] = self.ux[:, -2]
        self.uy[:, -1] = self.uy[:, -2]
        self.T[:, -1] = self.T[:, -2]
        self.f[:, :, -1] = self.f[:, :, -2]
        self.g[:, :, -1] = self.g[:, :, -2]
        
        # Top: freestream
        self.rho[-1, :] = self.rho_inf
        self.ux[-1, :] = self.ux_inf
        self.uy[-1, :] = self.uy_inf
        self.T[-1, :] = self.T_inf
        for i in range(9):
            cu = CX[i] * self.ux[-1, :] + CY[i] * self.uy[-1, :]
            u_sq = self.ux[-1, :]**2 + self.uy[-1, :]**2
            self.f[i, -1, :] = W[i] * self.rho[-1, :] * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        for i in range(5):
            cu = CX5[i] * self.ux[-1, :] + CY5[i] * self.uy[-1, :]
            u_sq = self.ux[-1, :]**2 + self.uy[-1, :]**2
            self.g[i, -1, :] = W5[i] * self.rho[-1, :] * self.T[-1, :] * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        
        # Bottom: freestream
        self.rho[0, :] = self.rho_inf
        self.ux[0, :] = self.ux_inf
        self.uy[0, :] = self.uy_inf
        self.T[0, :] = self.T_inf
        for i in range(9):
            cu = CX[i] * self.ux[0, :] + CY[i] * self.uy[0, :]
            u_sq = self.ux[0, :]**2 + self.uy[0, :]**2
            self.f[i, 0, :] = W[i] * self.rho[0, :] * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        for i in range(5):
            cu = CX5[i] * self.ux[0, :] + CY5[i] * self.uy[0, :]
            u_sq = self.ux[0, :]**2 + self.uy[0, :]**2
            self.g[i, 0, :] = W5[i] * self.rho[0, :] * self.T[0, :] * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        
        # Wall: bounce-back
        for i in range(9):
            opp = OPPOSITE[i]
            self.f[i] = np.where(self.wall_mask, self.f[opp], self.f[i])
        for i in range(5):
            if i == 0:
                continue
            opp_idx = {1: 3, 2: 4, 3: 1, 4: 2}[i]
            self.g[i] = np.where(self.wall_mask, self.g[opp_idx], self.g[i])
    
    def step(self):
        f_eq = self._equilibrium_f(self.rho, self.ux, self.uy)
        mu_art = self._compute_artificial_viscosity()
        omega_eff = self.cfg.omega_f / (1.0 + 3.0 * mu_art)
        omega_eff = np.clip(omega_eff, 0.1, 1.8)  # Lower max for stability
        self.f = self.f - omega_eff[None, :, :] * (self.f - f_eq)
        
        g_eq = self._equilibrium_g(self.rho, self.T, self.ux, self.uy)
        self.g = self.g - self.cfg.omega_g * (self.g - g_eq)
        
        self.f = self._stream_f()
        self.g = self._stream_g()
        
        self._update_macroscopic()
        self._apply_boundary_conditions()
    
    def run(self, n_steps: Optional[int] = None):
        if n_steps is None:
            n_steps = self.cfg.n_steps
        
        for step in range(n_steps):
            self.step()
            
            if (step + 1) % self.cfg.output_interval == 0:
                self._compute_entropy()
                self._compute_total_enthalpy()
                self._extract_surface_cp()
                
                u_mag = np.sqrt(self.ux**2 + self.uy**2)
                Ma_local = u_mag / self.cs_inf
                
                s_ref = np.log(self.p_inf / (self.rho_inf ** self.gamma))
                ds_max = float(np.max(self.entropy_field - s_ref))
                
                h0_ref = self.gamma * self.R_gas / (self.gamma - 1.0) * self.T_inf + 0.5 * self.u_inf**2
                h0_var = float(np.max(np.abs(self.total_enthalpy_field - h0_ref)) / h0_ref * 100)
                
                self.history.append({
                    'step': step + 1,
                    'rho_max': float(self.rho.max()),
                    'rho_min': float(self.rho.min()),
                    'Ma_max': float(Ma_local.max()),
                    'p_max': float(self.p.max()),
                    'ds_max': ds_max,
                    'h0_variation_pct': h0_var,
                })
    
    def _extract_surface_cp(self):
        if self._surface_x is None:
            return
        
        cp = np.zeros(len(self._surface_x))
        x_surf = np.zeros(len(self._surface_x))
        y_surf = np.zeros(len(self._surface_x))
        
        for k in range(len(self._surface_x)):
            ix = int(np.clip(self._surface_x[k], 0, self.nx - 1))
            iy = int(np.clip(self._surface_y[k], 0, self.ny - 1))
            
            if self.wall_mask[iy, ix]:
                found = False
                for r in range(1, 8):
                    for di in range(-r, r + 1):
                        for dj in range(-r, r + 1):
                            ni, nj = iy + dj, ix + di
                            if 0 <= ni < self.ny and 0 <= nj < self.nx:
                                if not self.wall_mask[ni, nj]:
                                    iy, ix = ni, nj
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            
            p_local = self.p[iy, ix]
            if self.q_inf > 1e-12:
                cp[k] = (p_local - self.p_inf) / self.q_inf
            else:
                cp[k] = 0.0
            
            ox = int(self.cfg.chord_offset_x * self.nx)
            cp_val = self.cfg.chord_pixels
            x_surf[k] = (self._surface_x[k] - ox) / cp_val
            y_surf[k] = self._surface_y[k]
        
        self.cp_surface = cp
        self.x_surface = x_surf
        self.y_surface = y_surf
    
    def get_diagnostics(self) -> Dict:
        self._compute_entropy()
        self._compute_total_enthalpy()
        
        s_ref = np.log(self.p_inf / (self.rho_inf ** self.gamma))
        ds_field = self.entropy_field - s_ref
        
        h0_ref = self.gamma * self.R_gas / (self.gamma - 1.0) * self.T_inf + 0.5 * self.u_inf**2
        h0_var = np.abs(self.total_enthalpy_field - h0_ref) / h0_ref * 100
        
        u_mag = np.sqrt(self.ux**2 + self.uy**2)
        Ma_local = u_mag / self.cs_inf
        
        return {
            'entropy_max': float(ds_field.max()),
            'entropy_min': float(ds_field.min()),
            'entropy_in_shock_region': float(ds_field.max()),
            'h0_max_variation_pct': float(h0_var.max()),
            'Ma_max': float(Ma_local.max()),
            'Ma_min': float(Ma_local.min()),
            'rho_max': float(self.rho.max()),
            'rho_min': float(self.rho.min()),
            'p_max': float(self.p.max()),
            'p_min': float(self.p.min()),
        }
