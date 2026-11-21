"""Circular Poiseuille flow solver using D2Q9 LBM.

Implements 2D cross-section of pipe flow with:
- Body force driving (equivalent to pressure gradient)
- Standard half-way bounce-back on curved boundary (staircase approximation)
- Inamuro wet-node boundary condition (interpolated bounce-back)

The solver compares both methods against the analytical parabolic profile.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple, Optional, Dict

import torch
import numpy as np

from tensorlbm.boundary.inamuro_wetnode import (
    C2Q9,
    OPPOSITE_2Q9,
    compute_boundary_distance_from_levelset,
    apply_inamuro_all_directions,
    create_curved_boundary_masks,
)


@dataclasses.dataclass
class CircularPoiseuilleConfig:
    """Configuration for circular Poiseuille flow simulation."""
    n: int = 65  # Grid size (n x n)
    radius: float = 25.0  # Pipe radius in lattice units
    body_force: float = 1e-5  # Body force (equivalent to -dp/dx / rho)
    tau: float = 1.0  # Relaxation time
    max_steps: int = 30000
    use_inamuro: bool = True  # Use Inamuro wet-node vs standard bounce-back
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self):
        if self.n < 33:
            raise ValueError(f"Grid too small: n={self.n}, need >= 33")
        if self.radius > self.n / 2 - 2:
            raise ValueError(f"Radius {self.radius} too large for grid {self.n}")
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau} <= 0.5, unstable")

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def nu(self) -> float:
        """Kinematic viscosity from tau."""
        return (self.tau - 0.5) / 3.0


class CircularPoiseuilleSolver:
    """D2Q9 circular Poiseuille flow solver.

    Simulates 2D cross-section of pipe flow driven by a body force.
    The circular boundary is handled either by:
    - Standard half-way bounce-back (staircase approximation)
    - Inamuro wet-node (interpolated bounce-back, second-order accurate)
    """

    C = torch.tensor([
        [ 0,  0], [ 1,  0], [-1,  0], [ 0,  1], [ 0, -1],
        [ 1,  1], [-1,  1], [ 1, -1], [-1, -1],
    ], dtype=torch.int64)
    W = torch.tensor([
        4./9., 1./9., 1./9., 1./9., 1./9.,
        1./36., 1./36., 1./36., 1./36.,
    ], dtype=torch.float64)
    OPPOSITE = torch.tensor([0, 2, 1, 4, 3, 8, 7, 6, 5], dtype=torch.int64)

    def __init__(self, config: CircularPoiseuilleConfig):
        self.cfg = config
        n = config.n
        self.n = n
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # Create circular boundary
        center = (n - 1) / 2.0
        self.fluid_mask, self.solid_mask, self.levelset = create_curved_boundary_masks(
            n, n, center, center, config.radius, interior_solid=True
        )
        self.fluid_mask = self.fluid_mask.to(self.device)
        self.solid_mask = self.solid_mask.to(self.device)
        self.levelset = self.levelset.to(self.device)

        # Pre-compute boundary information for Inamuro
        self.q_maps = torch.zeros((8, n, n), dtype=torch.float64, device=self.device)
        self.boundary_masks = torch.zeros((8, n, n), dtype=torch.bool, device=self.device)

        if config.use_inamuro:
            for i in range(1, 9):
                q, mask = compute_boundary_distance_from_levelset(
                    self.fluid_mask, self.levelset, i
                )
                self.q_maps[i-1] = q.to(self.device)
                self.boundary_masks[i-1] = mask.to(self.device)

        # Initialize at equilibrium (fluid at rest)
        rho = torch.ones((n, n), dtype=config.dtype, device=self.device)
        ux = torch.zeros((n, n), dtype=config.dtype, device=self.device)
        uy = torch.zeros((n, n), dtype=config.dtype, device=self.device)
        self.f = self._equilibrium(rho, ux, uy)

        # Streaming shifts
        self._shifts = []
        for i in range(9):
            cx, cy = int(self.C[i, 0]), int(self.C[i, 1])
            self._shifts.append((-cy, -cx))

    def _equilibrium(self, rho, ux, uy):
        c = self.C.to(device=self.device, dtype=rho.dtype)
        w = self.W.to(device=self.device, dtype=rho.dtype)
        uc = c[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(-1, 1, 1) * uy.unsqueeze(0)
        u2 = ux**2 + uy**2
        return w.view(-1, 1, 1) * rho.unsqueeze(0) * (
            1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
        )

    def _stream(self, f):
        f_out = torch.empty_like(f)
        for i in range(9):
            sy, sx = self._shifts[i]
            if sy == 0 and sx == 0:
                f_out[i] = f[i]
            else:
                f_out[i] = torch.roll(f[i], shifts=(sy, sx), dims=(0, 1))
        return f_out

    def _macroscopic(self, f):
        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = (f * self.C[:, 0].view(-1, 1, 1).to(f.dtype)).sum(dim=0) / rho_safe
        uy = (f * self.C[:, 1].view(-1, 1, 1).to(f.dtype)).sum(dim=0) / rho_safe
        return rho, ux, uy

    def _collide_bgk(self, f):
        """BGK collision with body force on fluid nodes only."""
        rho, ux, uy = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy)

        # Body force term (Guo forcing scheme)
        # F_i = w_i * (1 - 1/(2*tau)) * (c_i - u) . F / cs2
        cs2 = 1.0 / 3.0
        c = self.C.to(device=self.device, dtype=f.dtype)
        w = self.W.to(device=self.device, dtype=f.dtype)

        # Body force in x-direction
        Fx = self.cfg.body_force
        Fy = 0.0

        force_term = w.view(-1, 1, 1) * (1.0 - 0.5 * self.cfg.omega) / cs2 * (
            (c[:, 0].view(-1, 1, 1) - ux.unsqueeze(0)) * Fx +
            (c[:, 1].view(-1, 1, 1) - uy.unsqueeze(0)) * Fy
        )

        f_out = f.clone()
        f_out[:, self.fluid_mask] = (
            f[:, self.fluid_mask] +
            self.cfg.omega * (feq[:, self.fluid_mask] - f[:, self.fluid_mask]) +
            force_term[:, self.fluid_mask]
        )
        return f_out

    def _apply_bounce_back(self, f, f_pre_stream):
        """Standard half-way bounce-back on curved boundary (staircase)."""
        f_out = f.clone()
        for i in range(1, 9):
            opp = self.OPPOSITE[i]
            # Find fluid nodes that have solid neighbor in direction i
            cx, cy = int(self.C[i, 0]), int(self.C[i, 1])

            # Create shifted solid mask
            solid_shifted = torch.zeros_like(self.solid_mask)
            src_y_start = max(0, -cy)
            src_y_end = min(self.n, self.n - cy)
            src_x_start = max(0, -cx)
            src_x_end = min(self.n, self.n - cx)

            dst_y_start = max(0, cy)
            dst_y_end = min(self.n, self.n + cy)
            dst_x_start = max(0, cx)
            dst_x_end = min(self.n, self.n + cx)

            if src_y_end > src_y_start and src_x_end > src_x_start:
                solid_shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    self.solid_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            else:
                solid_shifted = torch.ones_like(self.solid_mask)

            # Boundary nodes: fluid nodes with solid neighbor in direction i
            boundary = self.fluid_mask & solid_shifted
            f_out[i, boundary] = f_pre_stream[opp, boundary]

        return f_out

    def _apply_inamuro(self, f, f_pre_stream):
        """Apply Inamuro wet-node boundary condition."""
        rho, _, _ = self._macroscopic(f)
        u_wall = torch.zeros_like(rho)
        v_wall = torch.zeros_like(rho)

        return apply_inamuro_all_directions(
            f, f_pre_stream, self.fluid_mask,
            self.q_maps, self.boundary_masks,
            rho, u_wall, v_wall
        )

    def step(self):
        """One LBM step: collide -> stream -> boundary conditions."""
        # Collision (fluid nodes only, with body force)
        f_collided = self._collide_bgk(self.f)

        # Streaming
        f_streamed = self._stream(f_collided)

        # Boundary conditions
        if self.cfg.use_inamuro:
            self.f = self._apply_inamuro(f_streamed, f_collided)
        else:
            self.f = self._apply_bounce_back(f_streamed, f_collided)

    def get_radial_profile(self):
        """Extract velocity profile along horizontal centerline.

        Returns
        -------
        r : np.ndarray
            Radial positions from center
        u : np.ndarray
            x-velocity at each position
        """
        rho, ux, uy = self._macroscopic(self.f)
        mid = self.n // 2
        center = (self.n - 1) / 2.0

        # Horizontal centerline
        u_center = ux[mid, :].cpu().numpy()
        x = np.arange(self.n)
        r = np.abs(x - center)

        return r, u_center

    def get_full_field(self):
        """Get full velocity field for visualization."""
        rho, ux, uy = self._macroscopic(self.f)
        return (
            rho.cpu().numpy(),
            ux.cpu().numpy(),
            uy.cpu().numpy(),
        )

    def compute_flow_rate(self):
        """Compute volumetric flow rate by integrating velocity over cross-section."""
        rho, ux, uy = self._macroscopic(self.f)
        # Q = sum(u * dA) where dA = 1 in lattice units
        Q = ux[self.fluid_mask].sum().item()
        return Q

    def compute_max_velocity(self):
        """Compute maximum (centerline) velocity."""
        rho, ux, uy = self._macroscopic(self.f)
        return ux[self.fluid_mask].max().item()

    def get_residual(self, ux_prev, uy_prev):
        """Compute convergence residual."""
        rho, ux, uy = self._macroscopic(self.f)
        du = float(torch.max(
            torch.abs(ux[self.fluid_mask] - ux_prev[self.fluid_mask]).max(),
            torch.abs(uy[self.fluid_mask] - uy_prev[self.fluid_mask]).max(),
        ))
        return du

    def run(self, callback=None, resid_tol=1e-7, monitor_interval=2000):
        """Run simulation to steady state."""
        rho, ux_prev, uy_prev = self._macroscopic(self.f)

        for step in range(self.cfg.max_steps):
            self.step()
            if step % monitor_interval == 0 and step > 0:
                resid = self.get_residual(ux_prev, uy_prev)
                rho, ux_prev, uy_prev = self._macroscopic(self.f)
                if callback:
                    callback(step, self, resid)
                if resid < resid_tol:
                    break

        return self


def run_circular_poiseuille(config: CircularPoiseuilleConfig) -> CircularPoiseuilleSolver:
    """Run circular Poiseuille flow simulation."""
    solver = CircularPoiseuilleSolver(config)
    solver.run()
    return solver


def run_comparison(
    n: int = 65,
    radius: float = 25.0,
    body_force: float = 1e-5,
    tau: float = 1.0,
    max_steps: int = 30000,
) -> Dict:
    """Run both standard bounce-back and Inamuro wet-node for comparison.

    Returns
    -------
    results : dict
        Dictionary with 'bb' and 'inamuro' solver results
    """
    cfg_bb = CircularPoiseuilleConfig(
        n=n, radius=radius, body_force=body_force, tau=tau,
        max_steps=max_steps, use_inamuro=False
    )
    cfg_inamuro = CircularPoiseuilleConfig(
        n=n, radius=radius, body_force=body_force, tau=tau,
        max_steps=max_steps, use_inamuro=True
    )

    solver_bb = CircularPoiseuilleSolver(cfg_bb)
    solver_bb.run()

    solver_inamuro = CircularPoiseuilleSolver(cfg_inamuro)
    solver_inamuro.run()

    return {
        'bb': solver_bb,
        'inamuro': solver_inamuro,
        'config': cfg_bb,
    }
