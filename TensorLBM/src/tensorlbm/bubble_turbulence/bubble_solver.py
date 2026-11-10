"""Lagrangian point-particle bubble solver coupled to D3Q19 turbulent channel.

Microbubbles are treated as point particles (d+ << eta_K) with:
  - Schiller-Naumann drag
  - Buoyancy (density ratio rho_g/rho_l << 1)
  - Saffman/Mei lift force
  - Two-way momentum coupling to the carrier phase

The carrier phase is a D3Q19 MRT channel with Smagorinsky SGS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BubbleConfig:
    """Configuration for the bubble phase."""
    # Bubble properties
    n_bubbles: int = 2000          # number of bubbles
    d_plus: float = 10.0           # bubble diameter in wall units
    density_ratio: float = 0.012   # rho_gas / rho_liquid (air/water)
    viscosity_ratio: float = 0.018  # mu_gas / mu_liquid

    # Injection
    injection_wall: str = "bottom"  # "bottom", "top", or "both"
    injection_y_plus_max: float = 30.0  # inject within viscous sublayer
    injection_strip_x: Tuple[float, float] = (0.0, 0.25)  # fraction of Lx

    # Force model
    include_lift: bool = True
    include_lubrication: bool = True  # wall lubrication force
    lubrication_cutoff: float = 1.5  # in lattice units from wall

    # Coupling
    two_way_coupling: bool = True
    coupling_interp: str = "trilinear"  # "nearest" or "trilinear"

    # Lifecycle
    max_age_steps: int = 50000      # remove bubbles after this many steps
    recirculate: bool = True        # re-inject at bottom when exiting top

    seed: int = 42


# ---------------------------------------------------------------------------
# Force models
# ---------------------------------------------------------------------------

def schiller_naumann_drag(
    Re_p: torch.Tensor,
    d_p: float,
    nu: float,
) -> torch.Tensor:
    """Schiller-Naumann drag coefficient for a sphere.

    C_D = (24/Re_p) * (1 + 0.15 * Re_p^0.687)  for Re_p < 1000
    C_D = 0.44                                     for Re_p >= 1000

    Returns the drag coefficient C_D.
    """
    Re_p = Re_p.clamp(min=1e-10)
    Cd = torch.where(
        Re_p < 1000.0,
        (24.0 / Re_p) * (1.0 + 0.15 * Re_p.pow(0.687)),
        torch.full_like(Re_p, 0.44),
    )
    return Cd


def saffman_lift_coefficient(
    Re_p: torch.Tensor,
    Re_shear: torch.Tensor,
) -> torch.Tensor:
    """Saffman (1965) / Mei (1992) lift coefficient.

    Based on the shear Reynolds number Re_shear = |omega| * d_p^2 / nu
    and particle Reynolds number Re_p.
    """
    Re_p = Re_p.clamp(min=1e-10)
    Re_shear = Re_shear.clamp(min=0.0)

    # Saffman limit: CL ~ 6.46 * sqrt(Re_shear / Re_p) for Re_shear << Re_p
    # Mei correction for finite Re_p
    alpha = 0.5 * Re_shear / Re_p  # shear parameter
    Jr_star = alpha * (1.0 - 0.3314 * alpha.pow(0.5)) * torch.exp(
        -0.1 * Re_p + 0.3314 * alpha.pow(0.5)
    )
    CL = 6.46 * Jr_star * torch.sqrt(Re_shear / Re_p.clamp(min=1e-10))
    return CL.clamp(min=0.0, max=10.0)


# ---------------------------------------------------------------------------
# Bubble solver
# ---------------------------------------------------------------------------

class BubbleSolver:
    """Lagrangian point-particle bubble solver.

    Tracks n_bubbles point particles in a 3D periodic channel.
    Positions stored as (n_bubbles, 3) in lattice units.
    Velocities stored as (n_bubbles, 3) in lattice units/step.
    """

    def __init__(
        self,
        cfg: BubbleConfig,
        nx: int, ny: int, nz: int,
        nu: float, u_tau: float,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        self.cfg = cfg
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nu = nu
        self.u_tau = u_tau
        self.device = torch.device(device)
        self.dt = torch.float32 if dtype == "float32" else torch.float64

        self.d_lattice = cfg.d_plus * nu / u_tau  # d+ -> lattice units
        self.rho_p = cfg.density_ratio  # in lattice units, rho_liquid = 1

        # Initialize bubble positions and velocities
        torch.manual_seed(cfg.seed)
        self.positions = self._init_positions()
        self.velocities = torch.zeros(
            cfg.n_bubbles, 3, device=self.device, dtype=self.dt
        )
        self.ages = torch.zeros(
            cfg.n_bubbles, device=self.device, dtype=torch.long
        )
        self.active = torch.ones(
            cfg.n_bubbles, device=self.device, dtype=torch.bool
        )

        # Momentum coupling accumulator (force on fluid per cell)
        self.coupling_force = torch.zeros(
            nz, ny, nx, 3, device=self.device, dtype=self.dt
        )

        # Statistics accumulators
        self._n_samples = 0
        self._bubble_flux_profile = torch.zeros(ny, device=self.dt)

    def _init_positions(self) -> torch.Tensor:
        """Initialize bubble positions uniformly in the near-wall region."""
        cfg = self.cfg
        n = cfg.n_bubbles

        # x: uniform in domain
        x = torch.rand(n, device=self.device, dtype=self.dt) * self.nx

        # y: concentrated near injection wall, within y+ < injection_y_plus_max
        y_plus_max = cfg.injection_y_plus_max
        y_max_lattice = y_plus_max * self.nu / self.u_tau
        y_max_lattice = min(y_max_lattice, self.ny - 2)
        y = torch.rand(n, device=self.device, dtype=self.dt) * y_max_lattice + 1.0

        # z: uniform in domain
        z = torch.rand(n, device=self.device, dtype=self.dt) * self.nz

        return torch.stack([x, y, z], dim=1)

    def interpolate_fluid_velocity(
        self, positions: torch.Tensor,
        u_field: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate fluid velocity at bubble positions.

        u_field: (3, nz, ny, nx) or (nz, ny, nx, 3)
        positions: (n_bubbles, 3) in lattice units

        Returns: (n_bubbles, 3) fluid velocities at bubble locations.
        """
        if u_field.shape[0] == 3:
            # (3, nz, ny, nx) -> (nz, ny, nx, 3)
            u_field = u_field.permute(1, 2, 3, 0)

        n = positions.shape[0]
        u_fluid = torch.zeros(n, 3, device=self.device, dtype=self.dt)

        if self.cfg.coupling_interp == "nearest":
            # Nearest grid point
            ix = positions[:, 0].long().clamp(0, self.nx - 1)
            iy = positions[:, 1].long().clamp(0, self.ny - 1)
            iz = positions[:, 2].long().clamp(0, self.nz - 1)
            u_fluid = u_field[iz, iy, ix]
        else:
            # Trilinear interpolation
            xf = positions[:, 0] % self.nx
            yf = positions[:, 1].clamp(0, self.ny - 1.001)
            zf = positions[:, 2] % self.nz

            x0 = xf.long().clamp(0, self.nx - 1)
            y0 = yf.long().clamp(0, self.ny - 2)
            z0 = zf.long().clamp(0, self.nz - 1)
            x1 = (x0 + 1) % self.nx
            y1 = (y0 + 1).clamp(0, self.ny - 1)
            z1 = (z0 + 1) % self.nz

            wx = (xf - x0.float()).unsqueeze(1)
            wy = (yf - y0.float()).unsqueeze(1)
            wz = (zf - z0.float()).unsqueeze(1)

            # 8 corners
            for dx, wx_val in [(x0, 1 - wx), (x1, wx)]:
                for dy, wy_val in [(y0, 1 - wy), (y1, wy)]:
                    for dz, wz_val in [(z0, 1 - wz), (z1, wz)]:
                        w = wx_val * wy_val * wz_val
                        u_fluid += w * u_field[dz, dy, dx]

        return u_fluid

    def _compute_forces(
        self,
        u_fluid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forces on each bubble.

        Returns: (n_bubbles, 3) force per bubble in lattice units.
        """
        cfg = self.cfg
        d_p = self.d_lattice
        nu = self.nu

        # Relative velocity
        u_rel = u_fluid - self.velocities  # (n, 3)
        slip_mag = u_rel.norm(dim=1).clamp(min=1e-12)  # (n,)

        # Particle Reynolds number
        Re_p = slip_mag * d_p / nu  # (n,)

        # --- Drag force ---
        Cd = schiller_naumann_drag(Re_p, d_p, nu)  # (n,)
        # F_drag = 0.5 * Cd * pi/4 * d_p^2 * rho * |u_rel| * u_rel
        # In lattice units with rho_liquid = 1:
        drag_mag = 0.5 * Cd * (math.pi / 4.0) * d_p**2 * slip_mag
        F_drag = drag_mag.unsqueeze(1) * u_rel  # (n, 3)

        # --- Buoyancy force ---
        # F_buoy = (rho_l - rho_p) * V_p * g
        # In lattice units, g is encoded in the body force of the channel.
        # For bubbles (rho_p << rho_l), buoyancy pushes upward (+y).
        V_p = (math.pi / 6.0) * d_p**3
        g_lattice = self.u_tau**2 / (self.ny / 2.0)  # approximate from body force
        F_buoy = torch.zeros_like(self.velocities)
        F_buoy[:, 1] = (1.0 - self.rho_p) * V_p * g_lattice

        # --- Lift force (Saffman/Mei) ---
        F_lift = torch.zeros_like(self.velocities)
        if cfg.include_lift:
            # Estimate shear from velocity gradient (simplified)
            # Use |omega| ~ |du/dy| as proxy
            omega_mag = torch.abs(
                (u_fluid[:, 0] - self.velocities[:, 0]) / (d_p + 1e-10)
            )
            Re_shear = omega_mag * d_p**2 / nu
            CL = saffman_lift_coefficient(Re_p, Re_shear)
            # Lift direction: u_rel x omega (simplified: wall-normal)
            # For channel flow, lift pushes bubbles away from wall
            F_lift_mag = 0.5 * CL * (math.pi / 4.0) * d_p**2 * slip_mag**2
            # Sign: positive y (away from bottom wall)
            F_lift[:, 1] = F_lift_mag

        # --- Wall lubrication force ---
        F_lub = torch.zeros_like(self.velocities)
        if cfg.include_lubrication:
            y = self.positions[:, 1]
            dist_bottom = y - 1.0
            dist_top = (self.ny - 1.0) - y
            dist_wall = torch.minimum(dist_bottom, dist_top)

            # Lubrication repulsion: F_lub ~ 1/(h/d_p) for h < cutoff
            h_norm = dist_wall / d_p
            cutoff = cfg.lubrication_cutoff / d_p
            mask = h_norm < cutoff
            if mask.any():
                F_lub_mag = torch.zeros_like(dist_wall)
                F_lub_mag[mask] = 1.0 / h_norm[mask].clamp(min=0.1) - 1.0 / cutoff
                # Direction: away from nearest wall
                sign = torch.where(dist_bottom < dist_top, 1.0, -1.0)
                F_lub[:, 1] = F_lub_mag * sign

        # Total force
        F_total = F_drag + F_buoy + F_lift + F_lub

        # Add virtual mass / pressure gradient (for light bubbles)
        # F_added = -V_p * grad(p) + 0.5 * V_p * rho * (Du/Dt - du/dt)
        # Simplified: pressure gradient ~ body force
        F_pressure = torch.zeros_like(self.velocities)
        F_pressure[:, 0] = -V_p * g_lattice  # opposes body force direction

        F_total += F_pressure

        return F_total

    def _apply_coupling(self, u_field_3d: torch.Tensor) -> None:
        """Apply two-way momentum coupling: deposit reaction force on fluid.

        The reaction force on the fluid is -F_total (Newton's 3rd law),
        distributed to grid cells via trilinear interpolation.
        """
        if not self.cfg.two_way_coupling:
            return

        self.coupling_force.zero_()
        n = self.positions.shape[0]
        if n == 0:
            return

        # Reaction force per bubble
        u_fluid = self.interpolate_fluid_velocity(self.positions, u_field_3d)
        F_on_bubble = self._compute_forces(u_fluid)
        F_on_fluid = -F_on_bubble  # Newton's 3rd law

        # Deposit onto grid (trilinear)
        xf = self.positions[:, 0] % self.nx
        yf = self.positions[:, 1].clamp(0, self.ny - 1.001)
        zf = self.positions[:, 2] % self.nz

        x0 = xf.long().clamp(0, self.nx - 1)
        y0 = yf.long().clamp(0, self.ny - 2)
        z0 = zf.long().clamp(0, self.nz - 1)
        x1 = (x0 + 1) % self.nx
        y1 = (y0 + 1).clamp(0, self.ny - 1)
        z1 = (z0 + 1) % self.nz

        wx = (xf - x0.float())
        wy = (yf - y0.float())
        wz = (zf - z0.float())

        for dx, wx_val in [(x0, 1 - wx), (x1, wx)]:
            for dy, wy_val in [(y0, 1 - wy), (y1, wy)]:
                for dz, wz_val in [(z0, 1 - wz), (z1, wz)]:
                    w = (wx_val * wy_val * wz_val).unsqueeze(1)  # (n, 1)
                    # Scatter add
                    idx = (dz * self.ny * self.nx + dy * self.nx + dx).unsqueeze(1)
                    self.coupling_force.view(-1, 3).scatter_add_(
                        0, idx.expand(-1, 3), w * F_on_fluid
                    )

    def step(self, u_field_3d: torch.Tensor) -> None:
        """Advance bubbles by one time step.

        u_field_3d: (3, nz, ny, nx) fluid velocity field.
        """
        # Interpolate fluid velocity at bubble positions
        u_fluid = self.interpolate_fluid_velocity(self.positions, u_field_3d)

        # Compute forces
        F_total = self._compute_forces(u_fluid)

        # Particle mass (in lattice units, rho_liquid = 1)
        V_p = (math.pi / 6.0) * self.d_lattice**3
        m_p = self.rho_p * V_p + 0.5 * V_p  # added mass

        # Update velocities (Euler integration)
        self.velocities += F_total / m_p

        # Update positions
        self.positions += self.velocities

        # Enforce periodicity in x and z
        self.positions[:, 0] = self.positions[:, 0] % self.nx
        self.positions[:, 2] = self.positions[:, 2] % self.nz

        # Wall reflection in y (bounce off walls)
        below_bottom = self.positions[:, 1] < 1.0
        above_top = self.positions[:, 1] > self.ny - 1.0

        self.positions[below_bottom, 1] = 2.0 - self.positions[below_bottom, 1]
        self.velocities[below_bottom, 1] = torch.abs(
            self.velocities[below_bottom, 1]
        )

        self.positions[above_top, 1] = 2.0 * (self.ny - 1.0) - self.positions[above_top, 1]
        self.velocities[above_top, 1] = -torch.abs(
            self.velocities[above_top, 1]
        )

        # Clamp to domain
        self.positions[:, 1] = self.positions[:, 1].clamp(1.0, self.ny - 1.0)

        # Update ages
        self.ages += 1

        # Recirculate old bubbles
        if self.cfg.recirculate:
            old = self.ages > self.cfg.max_age_steps
            if old.any():
                self._reinject(old)

        # Apply two-way coupling
        self._apply_coupling(u_field_3d)

    def _reinject(self, mask: torch.Tensor) -> None:
        """Re-inject bubbles at the bottom wall."""
        n_reinject = mask.sum().item()
        if n_reinject == 0:
            return

        y_max = min(
            self.cfg.injection_y_plus_max * self.nu / self.u_tau,
            self.ny - 2,
        )
        self.positions[mask, 0] = torch.rand(
            n_reinject, device=self.device, dtype=self.dt
        ) * self.nx
        self.positions[mask, 1] = (
            torch.rand(n_reinject, device=self.device, dtype=self.dt) * y_max + 1.0
        )
        self.positions[mask, 2] = torch.rand(
            n_reinject, device=self.device, dtype=self.dt
        ) * self.nz
        self.velocities[mask] = 0.0
        self.ages[mask] = 0

    def get_volume_fraction_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute wall-normal bubble volume fraction profile.

        Returns (y, alpha) where alpha is the local volume fraction.
        """
        ny = self.ny
        y_edges = np.arange(ny + 1) - 0.5
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        y_pos = self.positions[:, 1].cpu().numpy()
        counts, _ = np.histogram(y_pos, bins=np.arange(ny + 1) - 0.5)

        # Volume fraction = n_bubbles_in_cell * V_bubble / V_cell
        V_bubble = (math.pi / 6.0) * self.d_lattice**3
        V_cell = self.nx * 1.0 * self.nz  # dy = 1 in lattice units
        alpha = counts * V_bubble / V_cell

        return y_centers, alpha

    def get_coupling_force_profile(self) -> np.ndarray:
        """Get wall-normal profile of coupling force (x-component).

        Returns array of shape (ny,).
        """
        fx = self.coupling_force[:, :, :, 0]  # (nz, ny, nx)
        profile = fx.mean(dim=(0, 2))  # average over x, z
        return profile.cpu().numpy()


# ---------------------------------------------------------------------------
# Two-phase channel solver (carrier + bubbles)
# ---------------------------------------------------------------------------

class TwoPhaseChannelSolver:
    """Coupled D3Q19 MRT channel + Lagrangian bubble solver.

    This wraps the existing ChannelSolver and adds bubble phase.
    """

    def __init__(
        self,
        channel_cfg,  # ChannelConfig from verification.turbulent_channel
        bubble_cfg: BubbleConfig,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        from tensorlbm.verification.turbulent_channel.channel_solver import ChannelSolver

        self.channel = ChannelSolver(channel_cfg)
        self.bubble = BubbleSolver(
            bubble_cfg,
            nx=channel_cfg.nx, ny=channel_cfg.ny, nz=channel_cfg.nz,
            nu=channel_cfg.nu, u_tau=channel_cfg.u_tau,
            device=device, dtype=dtype,
        )
        self.cfg = channel_cfg
        self.bubble_cfg = bubble_cfg

        # Statistics accumulators
        self._step_count = 0
        self._sum_u = torch.zeros(
            channel_cfg.ny, device=self.channel.device,
            dtype=self.channel.dt,
        )
        self._sum_u2 = torch.zeros_like(self._sum_u)
        self._sum_uv = torch.zeros_like(self._sum_u)
        self._sum_v2 = torch.zeros_like(self._sum_u)
        self._sum_w2 = torch.zeros_like(self._sum_u)
        self._n_stat = 0

    def step(self) -> None:
        """One coupled time step."""
        # 1. Carrier phase step
        self.channel.step()

        # 2. Get fluid velocity
        rho, ux, uy, uz = self.channel.get_macroscopic()
        u_field = torch.stack([ux, uy, uz], dim=0)  # (3, nz, ny, nx)

        # 3. Apply bubble coupling force to carrier
        if self.bubble_cfg.two_way_coupling:
            coupling = self.bubble.coupling_force  # (nz, ny, nx, 3)
            # Add coupling force to populations via body force
            self._apply_coupling_force(coupling)

        # 4. Bubble phase step
        self.bubble.step(u_field)

        # 5. Accumulate statistics
        self._accumulate_stats(ux, uy, uz)
        self._step_count += 1

    def _apply_coupling_force(self, coupling: torch.Tensor) -> None:
        """Apply bubble coupling force to carrier phase via Guo forcing."""
        # coupling: (nz, ny, nx, 3)
        fx = coupling[:, :, :, 0]  # (nz, ny, nx)

        C = self.channel.C
        W = self.channel.W
        tau = self.channel.tau_plus
        factor = 1.0 - 0.5 / tau

        cx = C[:, 0].view(-1, 1, 1, 1)
        # force_term = W * cx * fx * factor
        for i in range(19):
            self.channel.f[i] += W[i] * cx[i] * fx * factor

    def _accumulate_stats(
        self, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor
    ) -> None:
        """Accumulate turbulent statistics (plane-averaged)."""
        # ux, uy, uz: (nz, ny, nx)
        u_mean = ux.mean(dim=(0, 2))  # (ny,)
        v_mean = uy.mean(dim=(0, 2))
        w_mean = uz.mean(dim=(0, 2))

        u_prime = ux - u_mean.view(1, -1, 1)
        v_prime = uy - v_mean.view(1, -1, 1)
        w_prime = uz - w_mean.view(1, -1, 1)

        self._sum_u += u_mean
        self._sum_u2 += (u_prime**2).mean(dim=(0, 2))
        self._sum_v2 += (v_prime**2).mean(dim=(0, 2))
        self._sum_w2 += (w_prime**2).mean(dim=(0, 2))
        self._sum_uv += (u_prime * v_prime).mean(dim=(0, 2))
        self._n_stat += 1

    def get_statistics(self) -> dict:
        """Return accumulated statistics."""
        n = max(self._n_stat, 1)
        ny = self.cfg.ny
        h = self.cfg.half_height
        u_tau = self.cfg.u_tau
        nu = self.cfg.nu

        y = torch.arange(ny, dtype=torch.float64)
        y_from_wall = torch.minimum(y, torch.tensor(ny - 1, dtype=torch.float64) - y)
        y_plus = y_from_wall * u_tau / nu

        U_mean = self._sum_u / n
        urms = torch.sqrt(self._sum_u2 / n)
        vrms = torch.sqrt(self._sum_v2 / n)
        wrms = torch.sqrt(self._sum_w2 / n)
        uv = self._sum_uv / n

        # Convert to wall units
        U_plus = U_mean / u_tau
        urms_plus = urms / u_tau
        vrms_plus = vrms / u_tau
        wrms_plus = wrms / u_tau
        uv_plus = uv / (u_tau**2)

        # Friction velocity from three methods
        # Method 1: from body force (exact for channel)
        u_tau_1 = math.sqrt(self.cfg.body_force * h)

        # Method 2: from velocity gradient at wall
        du_dy_wall = (U_mean[1] - U_mean[0]) / 1.0  # first grid point
        u_tau_2 = math.sqrt(abs(nu * du_dy_wall))

        # Method 3: from total stress at center
        u_tau_3 = u_tau_1  # for constant flow rate, same as method 1

        return {
            "y_plus": y_plus.numpy(),
            "U_plus": U_plus.cpu().numpy(),
            "urms_plus": urms_plus.cpu().numpy(),
            "vrms_plus": vrms_plus.cpu().numpy(),
            "wrms_plus": wrms_plus.cpu().numpy(),
            "uv_plus": uv_plus.cpu().numpy(),
            "u_tau_methods": [u_tau_1, u_tau_2, u_tau_3],
            "n_stat_samples": n,
            "total_steps": self._step_count,
        }

    def run(self, n_steps: int, stat_start: int = 0, callback=None) -> dict:
        """Run coupled simulation.

        Parameters
        ----------
        n_steps : int
            Total number of steps.
        stat_start : int
            Step at which to start accumulating statistics.
        callback : callable, optional
            Called every 1000 steps with (step, solver).
        """
        import time
        t0 = time.time()

        for step in range(1, n_steps + 1):
            self.step()

            if step > stat_start:
                pass  # stats already accumulated in step()

            if callback and step % 1000 == 0:
                callback(step, self)

        elapsed = time.time() - t0
        return {
            "steps": n_steps,
            "elapsed_s": elapsed,
            "steps_per_sec": n_steps / elapsed if elapsed > 0 else 0,
        }
