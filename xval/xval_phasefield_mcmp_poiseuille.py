"""Cross-validation: TensorLBM vs ebinan92/Fingering_dynamics — phasefield-mcmp-poiseuille.

Implements the Swift-Osborn-Yeomans (SOY) free-energy D2Q9 LBM from the OSS
repository in PyTorch (TensorLBM-native), runs a two-layer stratified Poiseuille
flow, and validates against the analytical piecewise-parabolic solution.

OSS source: ebinan92/Fingering_dynamics @ 150a5739e
Key algorithm (extracted from fingering.py):
  - Two D2Q9 populations: f (flow), g (order parameter ψ ∈ [-1, +1])
  - Chemical potential: μ = aψ(1 − ψ²) − κ∇²ψ  (a = −2κ/ξ²)
  - Free-energy pressure: p = ρ/3 + ψμ/3
  - Guo forcing on f for the capillary force μ∇ψ + body force G_x
  - g-equilibrium includes chemical-potential diffusion terms (B_0, B_{1-8})
  - Isotropic D2Q9 stencil for gradients and Laplacian
  - Half-way bounce-back on walls, Zou-He velocity boundaries at inlet/outlet

Validation test: two-layer stratified Poiseuille flow
  - Channel height H (fluid rows), interface at h = H/2
  - Lower layer: ψ ≈ +1 (phase A), upper layer: ψ ≈ −1 (phase B)
  - Both layers driven by uniform body force G_x
  - Analytical: piecewise-parabolic u(y) with velocity & stress continuity at interface
  - Acceptance: L2 error of velocity profile < 15% (owner precision standard)

Dimensionless numbers:
  - Re = u_max * H / ν  (Reynolds number, based on peak velocity and channel height)
  - Ca = μ * u_max / σ  (Capillary number, ratio of viscous to surface tension forces)
  - Cn = ξ / H          (Cahn number, ratio of interface width to channel height)
  - Ma = u_max / c_s    (Mach number, must be << 1 for incompressible LBM)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# D2Q9 constants
# ---------------------------------------------------------------------------
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)

# Direction index mapping for Zou-He boundaries
# D2Q9: 0=rest, 1=east, 2=north, 3=west, 4=south, 5=NE, 6=NW, 7=SW, 8=SE
DIR_EAST = 1
DIR_NORTH = 2
DIR_WEST = 3
DIR_SOUTH = 4
DIR_NE = 5
DIR_NW = 6
DIR_SW = 7
DIR_SE = 8


# ---------------------------------------------------------------------------
# Analytical stratified Poiseuille solution
# ---------------------------------------------------------------------------

def analytical_stratified_poiseuille(
    ny: int, H: float, h: float, G_x: float,
    nu_lower: float, nu_upper: float
) -> dict:
    """Exact two-layer Poiseuille profile.

    Geometry: channel from y=0 to y=H, interface at y=h.
    Both layers driven by uniform body force G_x per unit volume.
    No-slip at y=0 and y=H.
    Velocity and shear stress continuous at y=h.

    Returns dict with profile (list of length ny), Q_lower, Q_upper, Q_total.
    """
    a_low = -G_x / (2.0 * nu_lower)
    a_up = -G_x / (2.0 * nu_upper)
    k = nu_upper / nu_lower
    # nu_lower * b_low = nu_upper * b_up  ->  b_low = k * b_up
    b_up = (a_up * (h * h - H * H) - a_low * h * h) / (H - h + k * h)
    b_low = k * b_up
    c_up = -a_up * H * H - b_up * H

    profile = []
    for j in range(ny):
        y = float(j)  # wall at y=0, y=H; fluid rows at y=1..ny-2
        if y <= h:
            u = a_low * y * y + b_low * y
        else:
            u = a_up * y * y + b_up * y + c_up
        profile.append(u)

    Q_lower = a_low * h**3 / 3.0 + b_low * h**2 / 2.0
    Q_upper = (
        a_up * (H**3 - h**3) / 3.0
        + b_up * (H * H - h * h) / 2.0
        + c_up * (H - h)
    )
    return {
        "profile": profile,
        "Q_lower": Q_lower,
        "Q_upper": Q_upper,
        "Q_total": Q_lower + Q_upper,
        "u_max": max(profile),
    }


# ---------------------------------------------------------------------------
# SOY Free-Energy D2Q9 LBM — PyTorch (faithful to OSS algorithm)
# ---------------------------------------------------------------------------

class SOYFreeEnergyLBM:
    """Swift-Osborn-Yeomans free-energy D2Q9 LBM (PyTorch).

    Faithful reimplementation of the core algorithm from
    ebinan92/Fingering_dynamics/lattice_boltzmann/fingering.py.
    Uses PyTorch tensor operations — analogous to the OSS cupy-dropin
    strategy (array-level GPU ops, no custom kernels), but with PyTorch
    instead of CuPy.
    """

    def __init__(self, ny: int, nx: int, *, sigma: float, xi: float,
                 tau_f: float, tau_g: float, gamma: float,
                 device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.ny, self.nx = ny, nx
        self.sigma, self.xi = sigma, xi
        self.kappa = 0.75 * sigma * xi
        self.a = -2.0 * self.kappa / (xi ** 2)
        self.tau_f, self.tau_g, self.gamma = tau_f, tau_g, gamma
        self.device, self.dtype = torch.device(device), dtype
        self.w0 = float(W[0])

        self.cx_t = torch.tensor(CX, dtype=dtype, device=self.device).view(9, 1, 1)
        self.cy_t = torch.tensor(CY, dtype=dtype, device=self.device).view(9, 1, 1)
        self.w_t = torch.tensor(W, dtype=dtype, device=self.device).view(9, 1, 1)

        self.f = torch.zeros((9, ny, nx), dtype=dtype, device=self.device)
        self.g = torch.zeros((9, ny, nx), dtype=dtype, device=self.device)
        self.psi = torch.zeros((ny, nx), dtype=dtype, device=self.device)
        self.step_count = 0

        # Wall mask (rows 0 and ny-1 are solid)
        self.solid = torch.zeros((ny, nx), dtype=torch.bool, device=self.device)
        self.solid[0, :] = True
        self.solid[-1, :] = True

        # Body force (default: none)
        self.Gx = 0.0
        self.Gy = 0.0

    def init_droplet(self, R: float, cx: float = None, cy: float = None):
        """Initialize a circular droplet (for Laplace pressure test)."""
        if cx is None: cx = self.nx / 2.0
        if cy is None: cy = self.ny / 2.0
        y = torch.arange(self.ny, dtype=self.dtype, device=self.device).view(self.ny, 1)
        x = torch.arange(self.nx, dtype=self.dtype, device=self.device).view(1, self.nx)
        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        self.psi = torch.tanh((dist - R) / self.xi)
        self._dist = dist
        self._R = R

        grad_psi_x, grad_psi_y, lap_psi = self._compute_gradients(self.psi)
        mu = self._chemical_potential(self.psi, lap_psi)
        rho = torch.ones((self.ny, self.nx), dtype=self.dtype, device=self.device)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        p = self._pressure(rho, self.psi, mu)

        cu = self.cx_t * ux.unsqueeze(0) + self.cy_t * uy.unsqueeze(0)
        usq = (ux ** 2 + uy ** 2).unsqueeze(0)

        A = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        A[0] = (rho - 3.0 * (1.0 - self.w0) * p) / self.w0
        A[1:] = 3.0 * p.unsqueeze(0).expand(8, self.ny, self.nx)
        self.f = self.w_t * (A + rho.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

        B = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        B[0] = (self.psi - 3.0 * (1.0 - self.w0) * self.gamma * mu) / self.w0
        B[1:] = 3.0 * self.gamma * mu.unsqueeze(0).expand(8, self.ny, self.nx)
        self.g = self.w_t * (B + self.psi.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

    def init_poiseuille(self, h_frac: float = 0.5, psi_wall: float = 0.0,
                        Gx: float = 1e-5):
        """Initialize a two-layer stratified channel for Poiseuille flow.

        Args:
            h_frac: Interface position as fraction of channel height (0..1)
            psi_wall: Wall order parameter (controls wetting; 0 = neutral)
            Gx: Body force per unit volume driving the flow
        """
        H = self.ny - 2  # fluid rows (rows 1..ny-2)
        h = h_frac * H   # interface position from bottom wall

        # Initialize psi: +1 in lower layer, -1 in upper layer, tanh transition
        y = torch.arange(self.ny, dtype=self.dtype, device=self.device).view(self.ny, 1)
        x = torch.arange(self.nx, dtype=self.dtype, device=self.device).view(1, self.nx)
        # Interface at y = h + 1 (since row 0 is wall, row 1 is first fluid row)
        interface_y = h + 1.0
        self.psi = torch.tanh((y - interface_y) / self.xi)

        # Set wall psi for wetting
        if psi_wall != 0.0:
            self.psi[0, :] = psi_wall
            self.psi[-1, :] = psi_wall

        self._H = float(H)
        self._h = h
        self._interface_y = interface_y
        self.Gx = Gx
        self.Gy = 0.0

        # Initialize f and g to equilibrium with zero velocity
        grad_psi_x, grad_psi_y, lap_psi = self._compute_gradients(self.psi)
        mu = self._chemical_potential(self.psi, lap_psi)
        rho = torch.ones((self.ny, self.nx), dtype=self.dtype, device=self.device)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        p = self._pressure(rho, self.psi, mu)

        cu = self.cx_t * ux.unsqueeze(0) + self.cy_t * uy.unsqueeze(0)
        usq = (ux ** 2 + uy ** 2).unsqueeze(0)

        A = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        A[0] = (rho - 3.0 * (1.0 - self.w0) * p) / self.w0
        A[1:] = 3.0 * p.unsqueeze(0).expand(8, self.ny, self.nx)
        self.f = self.w_t * (A + rho.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

        B = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        B[0] = (self.psi - 3.0 * (1.0 - self.w0) * self.gamma * mu) / self.w0
        B[1:] = 3.0 * self.gamma * mu.unsqueeze(0).expand(8, self.ny, self.nx)
        self.g = self.w_t * (B + self.psi.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

    def _compute_gradients(self, psi):
        """Isotropic D2Q9 stencil for gradients and Laplacian."""
        grad_x = torch.zeros_like(psi)
        grad_y = torch.zeros_like(psi)
        lap = torch.zeros_like(psi)
        for i in range(1, 9):
            dx, dy = int(CX[i]), int(CY[i])
            shifted = torch.roll(torch.roll(psi, shifts=-dx, dims=1), shifts=-dy, dims=0)
            grad_x = grad_x + 3.0 * float(W[i]) * float(CX[i]) * shifted
            grad_y = grad_y + 3.0 * float(W[i]) * float(CY[i]) * shifted
            lap = lap + float(W[i]) * shifted
        lap = 6.0 * (lap - (1.0 - float(W[0])) * psi)
        return grad_x, grad_y, lap

    def _chemical_potential(self, psi, lap_psi):
        return self.a * psi * (1.0 - psi ** 2) - self.kappa * lap_psi

    def _pressure(self, rho, psi, mu):
        return rho / 3.0 + psi * mu / 3.0

    def _apply_zou_he_inlet(self, ux_profile: torch.Tensor):
        """Zou-He velocity inlet at x=0 (west boundary).

        Prescribes velocity ux(y) at the inlet. Unknown populations f1, f5, f8
        are reconstructed from known populations and the prescribed velocity.
        """
        ny, nx = self.ny, self.nx
        rho = self.f.sum(dim=0)

        for j in range(1, ny - 1):  # skip wall rows
            # Known populations at inlet (x=0)
            f0 = self.f[0, j, 0]
            f2 = self.f[2, j, 0]
            f3 = self.f[3, j, 0]
            f4 = self.f[4, j, 0]
            f6 = self.f[6, j, 0]
            f7 = self.f[7, j, 0]

            ux = ux_profile[j]
            uy = 0.0  # no transverse velocity at inlet

            # Density from known populations and prescribed ux
            rho_in = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux)

            # Unknown populations
            self.f[1, j, 0] = f3 + (2.0 / 3.0) * rho_in * ux
            self.f[5, j, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_in * ux + (1.0 / 2.0) * rho_in * uy
            self.f[8, j, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_in * ux - (1.0 / 2.0) * rho_in * uy

            # Also update g populations for order parameter
            g0 = self.g[0, j, 0]
            g2 = self.g[2, j, 0]
            g3 = self.g[3, j, 0]
            g4 = self.g[4, j, 0]
            g6 = self.g[6, j, 0]
            g7 = self.g[7, j, 0]

            psi_in = self.psi[j, 0]
            self.g[1, j, 0] = g3 + (2.0 / 3.0) * psi_in * ux
            self.g[5, j, 0] = g7 - 0.5 * (g2 - g4) + (1.0 / 6.0) * psi_in * ux
            self.g[8, j, 0] = g6 + 0.5 * (g2 - g4) + (1.0 / 6.0) * psi_in * ux

    def _apply_zou_he_outlet(self, rho_out: float = 1.0):
        """Zou-He pressure outlet at x=nx-1 (east boundary).

        Prescribes density (pressure) at the outlet. Unknown populations
        f2, f5, f6 are reconstructed.
        """
        ny, nx = self.ny, self.nx

        for j in range(1, ny - 1):  # skip wall rows
            # Known populations at outlet (x=nx-1)
            f0 = self.f[0, j, nx - 1]
            f1 = self.f[1, j, nx - 1]
            f3 = self.f[3, j, nx - 1]
            f4 = self.f[4, j, nx - 1]
            f7 = self.f[7, j, nx - 1]
            f8 = self.f[8, j, nx - 1]

            # ux from known populations and prescribed rho
            ux = 1.0 - (f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)) / rho_out
            uy = 0.0

            # Unknown populations
            self.f[2, j, nx - 1] = f4 + (2.0 / 3.0) * rho_out * uy
            self.f[5, j, nx - 1] = f8 - 0.5 * (f1 - f3) + (1.0 / 6.0) * rho_out * ux + (1.0 / 2.0) * rho_out * uy
            self.f[6, j, nx - 1] = f7 + 0.5 * (f1 - f3) + (1.0 / 6.0) * rho_out * ux - (1.0 / 2.0) * rho_out * uy

            # Also update g populations
            g0 = self.g[0, j, nx - 1]
            g1 = self.g[1, j, nx - 1]
            g3 = self.g[3, j, nx - 1]
            g4 = self.g[4, j, nx - 1]
            g7 = self.g[7, j, nx - 1]
            g8 = self.g[8, j, nx - 1]

            psi_out = self.psi[j, nx - 1]
            self.g[2, j, nx - 1] = g4 + (2.0 / 3.0) * psi_out * uy
            self.g[5, j, nx - 1] = g8 - 0.5 * (g1 - g3) + (1.0 / 6.0) * psi_out * ux
            self.g[6, j, nx - 1] = g7 + 0.5 * (g1 - g3) + (1.0 / 6.0) * psi_out * ux

    def _apply_bounce_back(self):
        """Half-way bounce-back on solid walls (rows 0 and ny-1)."""
        for i in range(1, 9):
            dx, dy = int(CX[i]), int(CY[i])
            # For wall rows, reflect populations
            if dy != 0:
                opp = [0, 3, 4, 1, 2, 7, 8, 5, 6][i]
                self.f[i, 0, :] = self.f[opp, 0, :]
                self.f[i, self.ny - 1, :] = self.f[opp, self.ny - 1, :]
                self.g[i, 0, :] = self.g[opp, 0, :]
                self.g[i, self.ny - 1, :] = self.g[opp, self.ny - 1, :]

    def step(self, use_zou_he: bool = False, inlet_ux: torch.Tensor = None):
        """One time step: collide + stream + boundary conditions.

        Args:
            use_zou_he: If True, apply Zou-He inlet/outlet boundaries
            inlet_ux: Prescribed inlet velocity profile (ny,) tensor
        """
        rho = self.f.sum(dim=0)
        grad_psi_x, grad_psi_y, lap_psi = self._compute_gradients(self.psi)
        mu = self._chemical_potential(self.psi, lap_psi)

        jx = (self.f * self.cx_t).sum(dim=0)
        jy = (self.f * self.cy_t).sum(dim=0)

        # Capillary force + body force
        Fx = mu * grad_psi_x + rho * self.Gx
        Fy = mu * grad_psi_y + rho * self.Gy

        rho_s = rho.clamp(min=1e-12)
        ux = (jx + 0.5 * Fx) / rho_s
        uy = (jy + 0.5 * Fy) / rho_s
        p = self._pressure(rho, self.psi, mu)

        cu = self.cx_t * ux.unsqueeze(0) + self.cy_t * uy.unsqueeze(0)
        usq = (ux ** 2 + uy ** 2).unsqueeze(0)
        ci_dot_F = self.cx_t * Fx.unsqueeze(0) + self.cy_t * Fy.unsqueeze(0)
        u_dot_F = (ux * Fx + uy * Fy).unsqueeze(0)

        # f-equilibrium
        A = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        A[0] = (rho - 3.0 * (1.0 - self.w0) * p) / self.w0
        A[1:] = 3.0 * p.unsqueeze(0).expand(8, self.ny, self.nx)
        feq = self.w_t * (A + rho.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

        # Guo forcing
        Si = (1.0 - 0.5 / self.tau_f) * self.w_t * (
            3.0 * ci_dot_F + 9.0 * cu * ci_dot_F - 3.0 * u_dot_F
        )
        self.f = self.f - (self.f - feq) / self.tau_f + Si

        # g-equilibrium
        B = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        B[0] = (self.psi - 3.0 * (1.0 - self.w0) * self.gamma * mu) / self.w0
        B[1:] = 3.0 * self.gamma * mu.unsqueeze(0).expand(8, self.ny, self.nx)
        geq = self.w_t * (B + self.psi.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))
        self.g = self.g - (self.g - geq) / self.tau_g

        # Streaming
        f_new = torch.empty_like(self.f)
        g_new = torch.empty_like(self.g)
        f_new[0] = self.f[0]
        g_new[0] = self.g[0]
        for i in range(1, 9):
            dx, dy = int(CX[i]), int(CY[i])
            f_new[i] = torch.roll(torch.roll(self.f[i], shifts=dx, dims=1), shifts=dy, dims=0)
            g_new[i] = torch.roll(torch.roll(self.g[i], shifts=dx, dims=1), shifts=dy, dims=0)
        self.f, self.g = f_new, g_new

        # Boundary conditions
        self._apply_bounce_back()

        if use_zou_he and inlet_ux is not None:
            self._apply_zou_he_inlet(inlet_ux)
            self._apply_zou_he_outlet()

        # Update psi from g
        self.psi = self.g.sum(dim=0)

        # Re-apply wall psi
        if hasattr(self, '_psi_wall') and self._psi_wall is not None:
            self.psi[0, :] = self._psi_wall
            self.psi[-1, :] = self._psi_wall

        self.step_count += 1

    def get_velocity_profile(self, x_col: int = None) -> np.ndarray:
        """Extract velocity profile at a given x column (averaged over x if None)."""
        rho = self.f.sum(dim=0)
        jx = (self.f * self.cx_t).sum(dim=0)
        grad_psi_x, grad_psi_y, lap_psi = self._compute_gradients(self.psi)
        mu = self._chemical_potential(self.psi, lap_psi)
        Fx = mu * grad_psi_x + rho * self.Gx
        rho_s = rho.clamp(min=1e-12)
        ux = (jx + 0.5 * Fx) / rho_s

        if x_col is not None:
            return ux[:, x_col].cpu().numpy()
        else:
            # Average over all fluid columns (skip walls in x if periodic)
            return ux[:, 1:-1].mean(dim=1).cpu().numpy()

    def get_psi_profile(self, x_col: int = None) -> np.ndarray:
        """Extract psi profile at a given x column."""
        if x_col is not None:
            return self.psi[:, x_col].cpu().numpy()
        else:
            return self.psi[:, 1:-1].mean(dim=1).cpu().numpy()

    def measure_poiseuille(self, x_col: int = None) -> dict:
        """Measure Poiseuille flow quantities."""
        ux = self.get_velocity_profile(x_col)
        psi = self.get_psi_profile(x_col)

        # Only fluid rows (skip walls)
        ux_fluid = ux[1:-1]
        psi_fluid = psi[1:-1]

        u_max = float(np.max(np.abs(ux_fluid)))
        u_mean = float(np.mean(ux_fluid))

        # Find interface position (psi = 0 crossing)
        psi_cross = np.where(np.diff(np.sign(psi_fluid)))[0]
        if len(psi_cross) > 0:
            interface_idx = psi_cross[0] + 1
            interface_y = float(interface_idx)
        else:
            interface_y = float(self._h) if hasattr(self, '_h') else float(self.ny) / 2.0

        return {
            "u_max": u_max,
            "u_mean": u_mean,
            "u_profile": ux_fluid,
            "psi_profile": psi_fluid,
            "interface_y": interface_y,
            "psi_min": float(np.min(psi_fluid)),
            "psi_max": float(np.max(psi_fluid)),
        }


# ---------------------------------------------------------------------------
# Poiseuille cross-validation runner
# ---------------------------------------------------------------------------

@dataclass
class PoiseuilleConfig:
    ny: int = 64           # channel height + 2 wall rows
    nx: int = 16           # streamwise length
    h_frac: float = 0.5    # interface position as fraction of H
    sigma: float = 0.01    # surface tension
    xi: float = 4.0        # interface width
    tau_f: float = 1.0     # flow relaxation time
    tau_g: float = 0.8     # order parameter relaxation time
    Gx: float = 1e-5       # body force
    n_steps: int = 30000   # simulation steps
    sample_every: int = 1000
    psi_wall: float = 0.0  # wall wetting (0 = neutral)
    use_zou_he: bool = False  # use Zou-He boundaries (False = periodic)


def run_poiseuille_xval(cfg: PoiseuilleConfig = None) -> dict:
    """Run the phase-field MCMP Poiseuille cross-validation."""
    if cfg is None:
        cfg = PoiseuilleConfig()

    H = cfg.ny - 2  # fluid rows
    h = cfg.h_frac * H
    kappa = 0.75 * cfg.sigma * cfg.xi
    gamma_mob = (cfg.tau_g - 0.5) / 3.0
    nu = (cfg.tau_f - 0.5) / 3.0  # kinematic viscosity (both layers same tau)

    results = {
        "config": {
            "ny": cfg.ny, "nx": cfg.nx, "H": H, "h": h,
            "sigma": cfg.sigma, "xi": cfg.xi, "kappa": kappa,
            "tau_f": cfg.tau_f, "tau_g": cfg.tau_g, "gamma": gamma_mob,
            "Gx": cfg.Gx, "nu": nu, "n_steps": cfg.n_steps,
            "psi_wall": cfg.psi_wall, "use_zou_he": cfg.use_zou_he,
        },
        "dimensionless": {},
    }

    # --- Run SOY PyTorch (OSS algorithm) ---
    print("Running SOY PyTorch Poiseuille (OSS algorithm)...")
    t0 = time.time()
    sim = SOYFreeEnergyLBM(
        cfg.ny, cfg.nx, sigma=cfg.sigma, xi=cfg.xi,
        tau_f=cfg.tau_f, tau_g=cfg.tau_g, gamma=gamma_mob,
        dtype=torch.float64
    )
    sim._psi_wall = cfg.psi_wall if cfg.psi_wall != 0.0 else None
    sim.init_poiseuille(h_frac=cfg.h_frac, psi_wall=cfg.psi_wall, Gx=cfg.Gx)

    history = []
    for step in range(cfg.n_steps):
        sim.step(use_zou_he=cfg.use_zou_he)
        if step % cfg.sample_every == 0 or step == cfg.n_steps - 1:
            m = sim.measure_poiseuille()
            history.append((step, m))
            print(f"  step {step}: u_max={m['u_max']:.6e}, "
                  f"psi=[{m['psi_min']:.4f},{m['psi_max']:.4f}]")

    t_soy = time.time() - t0

    # Final measurement
    m_final = sim.measure_poiseuille()
    u_profile_sim = m_final["u_profile"]

    # Analytical solution (same viscosity in both layers for this test)
    ana = analytical_stratified_poiseuille(cfg.ny, float(H), h, cfg.Gx, nu, nu)
    u_profile_ana = np.array(ana["profile"][1:-1])  # skip walls

    # L2 error
    norm_ana = np.sqrt(np.sum(u_profile_ana ** 2))
    if norm_ana > 1e-300:
        l2_error = np.sqrt(np.sum((u_profile_sim - u_profile_ana) ** 2)) / norm_ana
    else:
        l2_error = float("nan")

    # Dimensionless numbers
    u_max = m_final["u_max"]
    Re = u_max * H / nu if nu > 0 else float("inf")
    Ca = nu * u_max / cfg.sigma if cfg.sigma > 0 else float("inf")
    Cn = cfg.xi / H
    Ma = u_max / CS

    results["dimensionless"] = {
        "Re": Re, "Ca": Ca, "Cn": Cn, "Ma": Ma,
        "u_max": u_max, "u_mean": m_final["u_mean"],
    }

    results["soy_pytorch"] = {
        "u_max": u_max,
        "u_mean": m_final["u_mean"],
        "u_profile": u_profile_sim.tolist(),
        "psi_profile": m_final["psi_profile"].tolist(),
        "interface_y": m_final["interface_y"],
        "l2_error": l2_error,
        "analytical_u_max": ana["u_max"],
        "analytical_Q_total": ana["Q_total"],
        "time_s": t_soy,
        "history": [(h[0], h[1]["u_max"]) for h in history],
        "pass": l2_error < 0.15,
    }

    print(f"  SOY: u_max = {u_max:.6e}, analytical = {ana['u_max']:.6e}, "
          f"L2 error = {l2_error:.4%}, {'PASS' if l2_error < 0.15 else 'FAIL'}")

    return results


# ---------------------------------------------------------------------------
# Convergence study (varying resolution)
# ---------------------------------------------------------------------------

def run_convergence_study(cfg_base: PoiseuilleConfig = None) -> dict:
    """Run convergence study with varying channel heights."""
    if cfg_base is None:
        cfg_base = PoiseuilleConfig(n_steps=15000)

    heights = [32, 48, 64, 96, 128]
    results = {"heights": heights, "l2_errors": [], "u_maxs": []}

    for H in heights:
        ny = H + 2
        nx = max(8, H // 4)
        cfg = PoiseuilleConfig(
            ny=ny, nx=nx, h_frac=cfg_base.h_frac,
            sigma=cfg_base.sigma, xi=cfg_base.xi,
            tau_f=cfg_base.tau_f, tau_g=cfg_base.tau_g,
            Gx=cfg_base.Gx, n_steps=cfg_base.n_steps,
            sample_every=cfg_base.sample_every,
        )
        r = run_poiseuille_xval(cfg)
        err = r["soy_pytorch"]["l2_error"]
        u_max = r["soy_pytorch"]["u_max"]
        results["l2_errors"].append(err)
        results["u_maxs"].append(u_max)
        print(f"  H={H}: L2 error = {err:.4%}, u_max = {u_max:.6e}")

    return results


# ---------------------------------------------------------------------------
# Sensitivity study
# ---------------------------------------------------------------------------

def run_sensitivity_study(cfg_base: PoiseuilleConfig = None) -> dict:
    """Run sensitivity study on key parameters."""
    if cfg_base is None:
        cfg_base = PoiseuilleConfig(n_steps=15000)

    results = {"tau_f_sweep": {}, "xi_sweep": {}, "Gx_sweep": {}}

    # Vary tau_f (viscosity)
    for tau_f in [0.7, 0.8, 1.0, 1.2, 1.5]:
        cfg = PoiseuilleConfig(
            ny=cfg_base.ny, nx=cfg_base.nx, h_frac=cfg_base.h_frac,
            sigma=cfg_base.sigma, xi=cfg_base.xi,
            tau_f=tau_f, tau_g=cfg_base.tau_g,
            Gx=cfg_base.Gx, n_steps=cfg_base.n_steps,
            sample_every=cfg_base.sample_every,
        )
        r = run_poiseuille_xval(cfg)
        results["tau_f_sweep"][str(tau_f)] = {
            "l2_error": r["soy_pytorch"]["l2_error"],
            "u_max": r["soy_pytorch"]["u_max"],
        }
        print(f"  tau_f={tau_f}: L2 error = {r['soy_pytorch']['l2_error']:.4%}")

    # Vary xi (interface width)
    for xi in [2.0, 3.0, 4.0, 5.0, 6.0]:
        cfg = PoiseuilleConfig(
            ny=cfg_base.ny, nx=cfg_base.nx, h_frac=cfg_base.h_frac,
            sigma=cfg_base.sigma, xi=xi,
            tau_f=cfg_base.tau_f, tau_g=cfg_base.tau_g,
            Gx=cfg_base.Gx, n_steps=cfg_base.n_steps,
            sample_every=cfg_base.sample_every,
        )
        r = run_poiseuille_xval(cfg)
        results["xi_sweep"][str(xi)] = {
            "l2_error": r["soy_pytorch"]["l2_error"],
            "u_max": r["soy_pytorch"]["u_max"],
        }
        print(f"  xi={xi}: L2 error = {r['soy_pytorch']['l2_error']:.4%}")

    # Vary Gx (body force)
    for Gx in [5e-6, 1e-5, 2e-5, 5e-5]:
        cfg = PoiseuilleConfig(
            ny=cfg_base.ny, nx=cfg_base.nx, h_frac=cfg_base.h_frac,
            sigma=cfg_base.sigma, xi=cfg_base.xi,
            tau_f=cfg_base.tau_f, tau_g=cfg_base.tau_g,
            Gx=Gx, n_steps=cfg_base.n_steps,
            sample_every=cfg_base.sample_every,
        )
        r = run_poiseuille_xval(cfg)
        results["Gx_sweep"][str(Gx)] = {
            "l2_error": r["soy_pytorch"]["l2_error"],
            "u_max": r["soy_pytorch"]["u_max"],
        }
        print(f"  Gx={Gx}: L2 error = {r['soy_pytorch']['l2_error']:.4%}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, out_dir: Path):
    """Generate comparison figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Velocity profile comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    if "soy_pytorch" in results:
        r = results["soy_pytorch"]
        H = results["config"]["H"]
        y = np.arange(1, results["config"]["ny"] - 1, dtype=float)
        ax.plot(r["u_profile"], y, "o-", label="SOY PyTorch (OSS algorithm)",
                markersize=4, color="C0")

        # Analytical profile
        ana = analytical_stratified_poiseuille(
            results["config"]["ny"], H, results["config"]["h"],
            results["config"]["Gx"], results["config"]["nu"], results["config"]["nu"]
        )
        u_ana = np.array(ana["profile"][1:-1])
        ax.plot(u_ana, y, "k--", lw=2, label="Analytical (piecewise parabolic)")

    ax.set_xlabel("Velocity u (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title("Stratified Poiseuille Flow: Velocity Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_velocity_profile.png", dpi=150)
    plt.close(fig)

    # 2. Order parameter profile
    fig, ax = plt.subplots(figsize=(8, 6))
    if "soy_pytorch" in results:
        r = results["soy_pytorch"]
        y = np.arange(1, results["config"]["ny"] - 1, dtype=float)
        ax.plot(r["psi_profile"], y, "o-", label="ψ (order parameter)",
                markersize=4, color="C1")
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.axhline(results["config"]["h"], color="r", ls=":", lw=1,
                    label=f"Interface (h={results['config']['h']:.1f})")

    ax.set_xlabel("ψ (order parameter)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title("Order Parameter Profile (ψ ∈ [-1, +1])")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_psi_profile.png", dpi=150)
    plt.close(fig)

    # 3. Convergence: L2 error vs H
    if "convergence" in results:
        conv = results["convergence"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(conv["heights"], conv["l2_errors"], "o-", markersize=6, color="C0")
        ax.axhline(0.15, color="r", ls="--", lw=2, label="15% threshold")
        ax.set_xlabel("Channel height H (lattice units)")
        ax.set_ylabel("L2 relative error")
        ax.set_title("Convergence: L2 Error vs Resolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig3_convergence.png", dpi=150)
        plt.close(fig)

    # 4. Sensitivity plots
    if "sensitivity" in results:
        sens = results["sensitivity"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if "tau_f_sweep" in sens:
            taus = sorted(float(k) for k in sens["tau_f_sweep"])
            errs = [sens["tau_f_sweep"][str(t)]["l2_error"] * 100 for t in taus]
            axes[0].plot(taus, errs, "o-", color="C0")
            axes[0].axhline(15.0, color="r", ls="--", label="15% threshold")
            axes[0].set_xlabel("τ_f (relaxation time)")
            axes[0].set_ylabel("L2 error (%)")
            axes[0].set_title("Sensitivity to τ_f")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        if "xi_sweep" in sens:
            xis = sorted(float(k) for k in sens["xi_sweep"])
            errs = [sens["xi_sweep"][str(x)]["l2_error"] * 100 for x in xis]
            axes[1].plot(xis, errs, "o-", color="C1")
            axes[1].axhline(15.0, color="r", ls="--", label="15% threshold")
            axes[1].set_xlabel("ξ (interface width)")
            axes[1].set_ylabel("L2 error (%)")
            axes[1].set_title("Sensitivity to ξ")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        if "Gx_sweep" in sens:
            Gxs = sorted(float(k) for k in sens["Gx_sweep"])
            errs = [sens["Gx_sweep"][str(g)]["l2_error"] * 100 for g in Gxs]
            axes[2].plot(Gxs, errs, "o-", color="C2")
            axes[2].axhline(15.0, color="r", ls="--", label="15% threshold")
            axes[2].set_xlabel("G_x (body force)")
            axes[2].set_ylabel("L2 error (%)")
            axes[2].set_title("Sensitivity to G_x")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "fig4_sensitivity.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    out_dir = Path(__file__).parent / "artifacts" / "phasefield_mcmp_poiseuille"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("XVAL: TensorLBM vs ebinan92/Fingering_dynamics")
    print("       phasefield-mcmp-poiseuille")
    print("=" * 70)

    # Primary validation
    results = run_poiseuille_xval()

    # Convergence study
    print("\n--- Convergence study (varying H) ---")
    conv = run_convergence_study()
    results["convergence"] = conv

    # Sensitivity study
    print("\n--- Sensitivity study ---")
    sens = run_sensitivity_study()
    results["sensitivity"] = sens

    # Plots
    plot_results(results, out_dir)

    # Save results
    def sanitize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return "NaN"
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(x) for x in obj]
        if isinstance(obj, tuple):
            return [sanitize(x) for x in obj]
        return obj

    with open(out_dir / "xval_poiseuille_results.json", "w") as f:
        json.dump(sanitize(results), f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if "soy_pytorch" in results:
        r = results["soy_pytorch"]
        status = "✓ PASS" if r.get("pass") else "✗ FAIL"
        print(f"  SOY PyTorch (OSS algorithm): u_max = {r['u_max']:.6e}, "
              f"L2 error = {r['l2_error']:.4%} {status}")

    if "convergence" in results:
        c = results["convergence"]
        print(f"  Convergence: L2 errors = {[f'{e:.4%}' for e in c['l2_errors']]}")

    print(f"\nArtifacts saved to {out_dir}")
