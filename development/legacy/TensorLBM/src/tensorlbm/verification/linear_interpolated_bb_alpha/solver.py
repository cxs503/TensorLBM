"""Solver for linear-interpolated bounce-back (alpha scheme) validation.

Implements circular Poiseuille flow with the Bouzidi/Inamuro interpolated
bounce-back scheme for curved boundaries, enabling systematic validation
against the analytical parabolic profile.

The alpha scheme (linear interpolated bounce-back) uses the fractional
distance q = |x_b - x_f| / |x_s - x_f| to interpolate unknown distributions:

  For q >= 0.5:
    f_i(x_f) = (1/(2q)) * f_opp(x_f) + ((2q-1)/(2q)) * f_opp(x_f - c_i)
               + (2/(2q)) * rho * w_i * (c_i . u_w) / cs2

  For q < 0.5:
    f_i(x_f) = 2q * f_opp(x_f) + (1 - 2q) * f_i(x_f)
               + (2/(2q)) * rho * w_i * (c_i . u_w) / cs2

This achieves second-order accuracy for curved boundaries, compared to
first-order accuracy of standard half-way bounce-back.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple, Optional, List, Dict

import torch
import numpy as np

from tensorlbm.boundary.inamuro_wetnode import (
    compute_boundary_distance_from_levelset,
    apply_inamuro_all_directions,
    create_curved_boundary_masks,
)


@dataclasses.dataclass
class LinearInterpolatedBBAlphaConfig:
    """Configuration for linear-interpolated BB alpha scheme validation.

    Parameters
    ----------
    n : int
        Grid size (n x n). Must be >= 33 for meaningful results.
    radius : float
        Pipe radius in lattice units.
    body_force : float
        Body force driving the flow (equivalent to -dp/dx / rho).
    tau : float
        BGK relaxation time. Must be > 0.5 for stability.
    max_steps : int
        Maximum simulation steps.
    resid_tol : float
        Convergence tolerance for steady-state residual.
    monitor_interval : int
        Steps between convergence checks.
    dtype : torch.dtype
        Floating point precision.
    device : str
        Computation device.
    """
    n: int = 65
    radius: float = 25.0
    body_force: float = 1e-5
    tau: float = 1.0
    max_steps: int = 30000
    resid_tol: float = 1e-7
    monitor_interval: int = 2000
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
        """Kinematic viscosity: nu = (tau - 0.5) / 3."""
        return (self.tau - 0.5) / 3.0

    @property
    def reynolds(self) -> float:
        """Reynolds number based on max velocity and diameter."""
        u_max = self.body_force * self.radius**2 / (4.0 * self.nu)
        return u_max * (2 * self.radius) / self.nu


class LinearInterpolatedBBAlphaSolver:
    """D2Q9 circular Poiseuille solver with linear-interpolated BB alpha scheme.

    Validates the interpolated bounce-back boundary condition against
    the analytical parabolic velocity profile for pipe flow.

    Parameters
    ----------
    config : LinearInterpolatedBBAlphaConfig
        Simulation configuration.
    """

    # D2Q9 lattice vectors
    C = torch.tensor([
        [ 0,  0], [ 1,  0], [-1,  0], [ 0,  1], [ 0, -1],
        [ 1,  1], [-1,  1], [ 1, -1], [-1, -1],
    ], dtype=torch.int64)
    W = torch.tensor([
        4./9., 1./9., 1./9., 1./9., 1./9.,
        1./36., 1./36., 1./36., 1./36.,
    ], dtype=torch.float64)
    OPPOSITE = torch.tensor([0, 2, 1, 4, 3, 8, 7, 6, 5], dtype=torch.int64)

    def __init__(self, config: LinearInterpolatedBBAlphaConfig):
        self.cfg = config
        n = config.n
        self.n = n
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # Create circular boundary (annular domain: solid cylinder in fluid)
        self.fluid_mask, self.solid_mask, self.levelset = create_curved_boundary_masks(
            n, n, (n - 1) / 2.0, (n - 1) / 2.0, config.radius, interior_solid=True
        )
        self.fluid_mask = self.fluid_mask.to(self.device)
        self.solid_mask = self.solid_mask.to(self.device)
        self.levelset = self.levelset.to(self.device)

        # Pre-compute boundary distance maps for all 8 directions
        self.q_maps = torch.zeros((8, n, n), dtype=torch.float64, device=self.device)
        self.boundary_masks = torch.zeros((8, n, n), dtype=torch.bool, device=self.device)

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

        # Pre-compute streaming shifts
        self._shifts = [(-int(self.C[i, 1]), -int(self.C[i, 0])) for i in range(9)]

    def _equilibrium(self, rho, ux, uy):
        """Compute D2Q9 equilibrium distribution."""
        c = self.C.to(device=self.device, dtype=rho.dtype)
        w = self.W.to(device=self.device, dtype=rho.dtype)
        uc = (c[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) +
              c[:, 1].view(-1, 1, 1) * uy.unsqueeze(0))
        u2 = ux**2 + uy**2
        return w.view(-1, 1, 1) * rho.unsqueeze(0) * (
            1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
        )

    def _stream(self, f):
        """Stream distributions with periodic boundary conditions."""
        f_out = torch.empty_like(f)
        for i in range(9):
            sy, sx = self._shifts[i]
            if sy == 0 and sx == 0:
                f_out[i] = f[i]
            else:
                f_out[i] = torch.roll(f[i], shifts=(sy, sx), dims=(0, 1))
        return f_out

    def _macroscopic(self, f):
        """Compute macroscopic variables from distributions."""
        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = (f * self.C[:, 0].view(-1, 1, 1).to(f.dtype)).sum(dim=0) / rho_safe
        uy = (f * self.C[:, 1].view(-1, 1, 1).to(f.dtype)).sum(dim=0) / rho_safe
        return rho, ux, uy

    def _collide_bgk(self, f):
        """BGK collision with Guo body force on fluid nodes."""
        rho, ux, uy = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy)

        # Guo forcing scheme
        cs2 = 1.0 / 3.0
        c = self.C.to(device=self.device, dtype=f.dtype)
        w = self.W.to(device=self.device, dtype=f.dtype)

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
        """Standard half-way bounce-back (staircase approximation)."""
        f_out = f.clone()
        for i in range(1, 9):
            opp = self.OPPOSITE[i]
            cx, cy = int(self.C[i, 0]), int(self.C[i, 1])

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

            boundary = self.fluid_mask & solid_shifted
            f_out[i, boundary] = f_pre_stream[opp, boundary]

        return f_out

    def _apply_interpolated_bb_alpha(self, f, f_pre_stream):
        """Apply linear-interpolated bounce-back (alpha/Bouzidi scheme).

        This delegates to the Inamuro wet-node implementation which
        implements the same alpha-scheme interpolation.
        """
        rho, _, _ = self._macroscopic(f)
        u_wall = torch.zeros_like(rho)
        v_wall = torch.zeros_like(rho)

        return apply_inamuro_all_directions(
            f, f_pre_stream, self.fluid_mask,
            self.q_maps, self.boundary_masks,
            rho, u_wall, v_wall
        )

    def step(self, use_interpolated: bool = True):
        """One LBM step: collide -> stream -> boundary conditions.

        Parameters
        ----------
        use_interpolated : bool
            If True, use linear-interpolated BB alpha scheme.
            If False, use standard half-way bounce-back.
        """
        f_collided = self._collide_bgk(self.f)
        f_streamed = self._stream(f_collided)

        if use_interpolated:
            self.f = self._apply_interpolated_bb_alpha(f_streamed, f_collided)
        else:
            self.f = self._apply_bounce_back(f_streamed, f_collided)

    def get_radial_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract velocity profile along horizontal centerline.

        Returns
        -------
        r : np.ndarray
            Radial positions from center.
        u : np.ndarray
            x-velocity at each position.
        """
        _, ux, _ = self._macroscopic(self.f)
        mid = self.n // 2
        center = (self.n - 1) / 2.0

        u_center = ux[mid, :].cpu().numpy()
        x = np.arange(self.n)
        r = np.abs(x - center)

        return r, u_center

    def get_full_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get full velocity field."""
        rho, ux, uy = self._macroscopic(self.f)
        return rho.cpu().numpy(), ux.cpu().numpy(), uy.cpu().numpy()

    def compute_flow_rate(self) -> float:
        """Compute volumetric flow rate."""
        _, ux, _ = self._macroscopic(self.f)
        return float(ux[self.fluid_mask].sum().item())

    def compute_max_velocity(self) -> float:
        """Compute maximum (centerline) velocity."""
        _, ux, _ = self._macroscopic(self.f)
        return float(ux[self.fluid_mask].max().item())

    def get_residual(self, ux_prev: torch.Tensor, uy_prev: torch.Tensor) -> float:
        """Compute convergence residual."""
        _, ux, uy = self._macroscopic(self.f)
        du = float(torch.max(
            torch.abs(ux[self.fluid_mask] - ux_prev[self.fluid_mask]).max(),
            torch.abs(uy[self.fluid_mask] - uy_prev[self.fluid_mask]).max(),
        ))
        return du

    def run(self, use_interpolated: bool = True, callback=None) -> int:
        """Run simulation to steady state.

        Parameters
        ----------
        use_interpolated : bool
            Which boundary condition to use.
        callback : callable, optional
            Called with (step, solver, residual) at each monitor interval.

        Returns
        -------
        steps : int
            Number of steps taken.
        """
        _, ux_prev, uy_prev = self._macroscopic(self.f)

        for step in range(self.cfg.max_steps):
            self.step(use_interpolated=use_interpolated)
            if step % self.cfg.monitor_interval == 0 and step > 0:
                resid = self.get_residual(ux_prev, uy_prev)
                _, ux_prev, uy_prev = self._macroscopic(self.f)
                if callback:
                    callback(step, self, resid)
                if resid < self.cfg.resid_tol:
                    return step + 1

        return self.cfg.max_steps


def analytical_circular_poiseuille(
    r: np.ndarray,
    radius: float,
    body_force: float,
    nu: float,
) -> np.ndarray:
    """Analytical velocity profile for circular Poiseuille flow.

    u(r) = (F / (4 * nu)) * (R^2 - r^2)

    where F is the body force, nu is kinematic viscosity, R is radius.

    Parameters
    ----------
    r : np.ndarray
        Radial positions.
    radius : float
        Pipe radius.
    body_force : float
        Body force magnitude.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    u : np.ndarray
        Analytical velocity at each radial position.
    """
    u = np.zeros_like(r, dtype=np.float64)
    mask = r <= radius
    u[mask] = (body_force / (4.0 * nu)) * (radius**2 - r[mask]**2)
    return u


def run_single_case(
    n: int = 65,
    radius: float = 25.0,
    body_force: float = 1e-5,
    tau: float = 1.0,
    use_interpolated: bool = True,
) -> Dict:
    """Run a single validation case.

    Returns
    -------
    result : dict
        Dictionary with simulation results and error metrics.
    """
    cfg = LinearInterpolatedBBAlphaConfig(
        n=n, radius=radius, body_force=body_force, tau=tau,
        max_steps=30000, resid_tol=1e-7, monitor_interval=2000
    )
    solver = LinearInterpolatedBBAlphaSolver(cfg)
    steps = solver.run(use_interpolated=use_interpolated)

    r_sim, u_sim = solver.get_radial_profile()
    u_max_sim = solver.compute_max_velocity()
    Q_sim = solver.compute_flow_rate()

    # Analytical values
    nu = cfg.nu
    u_analytical = analytical_circular_poiseuille(r_sim, radius, body_force, nu)
    u_max_analytical = body_force * radius**2 / (4.0 * nu)
    Q_analytical = np.pi * radius**2 * u_max_analytical / 2.0

    # Error metrics
    mask = r_sim <= radius
    if mask.any():
        l2_error = float(np.sqrt(np.mean((u_sim[mask] - u_analytical[mask])**2)) / u_max_analytical)
        linf_error = float(np.max(np.abs(u_sim[mask] - u_analytical[mask])) / u_max_analytical)
    else:
        l2_error = float('nan')
        linf_error = float('nan')

    max_vel_error = abs(u_max_sim - u_max_analytical) / u_max_analytical
    flow_rate_error = abs(Q_sim - Q_analytical) / Q_analytical

    return {
        'config': cfg,
        'steps': steps,
        'r_sim': r_sim,
        'u_sim': u_sim,
        'u_analytical': u_analytical,
        'u_max_sim': u_max_sim,
        'u_max_analytical': u_max_analytical,
        'Q_sim': Q_sim,
        'Q_analytical': Q_analytical,
        'l2_error': l2_error,
        'linf_error': linf_error,
        'max_vel_error': max_vel_error,
        'flow_rate_error': flow_rate_error,
        'use_interpolated': use_interpolated,
    }


def run_convergence_study(
    grid_sizes: List[int] = None,
    body_force: float = 1e-5,
    tau: float = 1.0,
) -> List[Dict]:
    """Run convergence study for the linear-interpolated BB alpha scheme.

    Parameters
    ----------
    grid_sizes : list of int
        Grid sizes to test.
    body_force : float
        Body force magnitude.
    tau : float
        Relaxation time.

    Returns
    -------
    results : list of dict
        Results for each grid size, for both BB and interpolated BB.
    """
    if grid_sizes is None:
        grid_sizes = [33, 49, 65, 81]

    all_results = []

    for n in grid_sizes:
        radius = n * 0.4  # Keep radius proportional to grid

        for use_interp in [False, True]:
            result = run_single_case(
                n=n, radius=radius, body_force=body_force, tau=tau,
                use_interpolated=use_interp
            )
            result['grid_size'] = n
            result['radius'] = radius
            all_results.append(result)

    return all_results
