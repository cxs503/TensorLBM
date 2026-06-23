"""Lagrangian particle tracker for TensorLBM.

Tracks massless or massive particles through a frozen (or time-averaged)
velocity field using 4th-order Runge-Kutta time integration.

Applications:
- Surface soiling / contamination (automotive, aerospace)
- Spray cooling trajectory mapping
- Particle erosion probability maps
- Passive scalar transport visualisation

The tracker operates in *lattice units* internally and optionally converts
results to physical units when a scale factor is supplied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ParticleState:
    """State of a single tracked particle."""
    pid: int
    x: float          # current x position (lattice units)
    y: float          # current y position
    z: float = 0.0    # z (for 3-D fields)
    vx: float = 0.0   # particle velocity x
    vy: float = 0.0   # particle velocity y
    vz: float = 0.0
    age: float = 0.0  # time steps alive
    status: Literal["active", "deposited", "escaped"] = "active"
    deposit_x: float | None = None
    deposit_y: float | None = None


@dataclass
class ParticleTrackResult:
    """Result for a single particle."""
    pid: int
    trajectory_x: list[float] = field(default_factory=list)
    trajectory_y: list[float] = field(default_factory=list)
    status: Literal["active", "deposited", "escaped"] = "active"
    deposit_x: float | None = None
    deposit_y: float | None = None
    age_steps: int = 0


# ---------------------------------------------------------------------------
# Bilinear interpolation of velocity at a point
# ---------------------------------------------------------------------------

def _interp_velocity_2d(
    ux: torch.Tensor,   # (ny, nx)
    uy: torch.Tensor,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Bilinear interpolation of (ux, uy) at real-valued grid coordinates."""
    ny, nx = ux.shape
    # Clamp to interior
    x = max(0.0, min(x, nx - 1.001))
    y = max(0.0, min(y, ny - 1.001))
    ix, iy = int(x), int(y)
    fx, fy = x - ix, y - iy
    ix1 = min(ix + 1, nx - 1)
    iy1 = min(iy + 1, ny - 1)

    def _lerp(f: torch.Tensor) -> float:
        return float(
            (1 - fx) * (1 - fy) * f[iy, ix]
            + fx * (1 - fy) * f[iy, ix1]
            + (1 - fx) * fy * f[iy1, ix]
            + fx * fy * f[iy1, ix1]
        )

    return _lerp(ux), _lerp(uy)


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

def track_particles(
    ux: torch.Tensor,           # (ny, nx) time-averaged x-velocity field
    uy: torch.Tensor,           # (ny, nx) time-averaged y-velocity field
    obstacle_mask: torch.Tensor,  # (ny, nx) bool – True = solid
    injection_x: list[float],   # injection x coordinates (lattice)
    injection_y: list[float],   # injection y coordinates (lattice)
    n_steps: int = 2000,
    dt: float = 0.5,            # time step in lattice units
    stokes_number: float = 0.0, # St = 0 → massless (follows flow exactly)
    record_every: int = 10,
    dx_phys: float = 1.0,       # m per lattice cell (for output conversion)
) -> list[ParticleTrackResult]:
    """Track particles through the 2-D velocity field.

    Parameters
    ----------
    ux, uy : Tensor
        2-D velocity fields (lattice units, shape (ny, nx)).
    obstacle_mask : Tensor
        Boolean mask; True inside solid bodies.
    injection_x, injection_y : list[float]
        Starting positions for each particle.
    n_steps : int
        Maximum number of time steps per particle.
    dt : float
        Integration time step (lattice units).
    stokes_number : float
        Particle inertia parameter (0 = massless tracer).
    record_every : int
        Record trajectory position every N steps.
    dx_phys : float
        Physical grid spacing (m) for coordinate output.

    Returns
    -------
    list[ParticleTrackResult]
    """
    ny, nx = ux.shape
    n_particles = len(injection_x)
    results: list[ParticleTrackResult] = []

    for pid in range(n_particles):
        state = ParticleState(
            pid=pid,
            x=injection_x[pid],
            y=injection_y[pid],
        )
        # Initialise particle velocity to local fluid velocity
        state.vx, state.vy = _interp_velocity_2d(ux, uy, state.x, state.y)

        res = ParticleTrackResult(pid=pid)
        res.trajectory_x.append(state.x * dx_phys)
        res.trajectory_y.append(state.y * dx_phys)

        for step in range(n_steps):
            if state.status != "active":
                break

            x0, y0 = state.x, state.y
            vx0, vy0 = state.vx, state.vy

            # Fluid velocity at current position
            uf0, vf0 = _interp_velocity_2d(ux, uy, x0, y0)

            if stokes_number < 1e-6:
                # Massless: RK4 on position using fluid velocity directly
                # k1
                k1x, k1y = uf0, vf0
                # k2
                k2x, k2y = _interp_velocity_2d(ux, uy, x0 + 0.5 * dt * k1x, y0 + 0.5 * dt * k1y)
                # k3
                k3x, k3y = _interp_velocity_2d(ux, uy, x0 + 0.5 * dt * k2x, y0 + 0.5 * dt * k2y)
                # k4
                k4x, k4y = _interp_velocity_2d(ux, uy, x0 + dt * k3x, y0 + dt * k3y)

                dx = dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
                dy = dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)

                state.x += dx
                state.y += dy
                state.vx = k4x
                state.vy = k4y
            else:
                # Massive: integrate Maxey-Riley (drag only) ẍ = (u_f - v_p)/St
                # k1
                ax0 = (uf0 - vx0) / stokes_number
                ay0 = (vf0 - vy0) / stokes_number
                # midpoint velocity
                vx_m = vx0 + 0.5 * dt * ax0
                vy_m = vy0 + 0.5 * dt * ay0
                xm = x0 + 0.5 * dt * vx0
                ym = y0 + 0.5 * dt * vy0
                ufm, vfm = _interp_velocity_2d(ux, uy, xm, ym)
                axm = (ufm - vx_m) / stokes_number
                aym = (vfm - vy_m) / stokes_number
                # endpoint
                vx_e = vx0 + dt * axm
                vy_e = vy0 + dt * aym
                xe = x0 + dt * vx_m
                ye = y0 + dt * vy_m
                ufe, vfe = _interp_velocity_2d(ux, uy, xe, ye)
                axe = (ufe - vx_e) / stokes_number
                aye = (vfe - vy_e) / stokes_number
                # RK4 combine
                state.vx += dt / 6.0 * (ax0 + 2 * axm + 2 * axm + axe)
                state.vy += dt / 6.0 * (ay0 + 2 * aym + 2 * aym + aye)
                state.x += dt / 6.0 * (vx0 + 2 * vx_m + 2 * vx_m + vx_e)
                state.y += dt / 6.0 * (vy0 + 2 * vy_m + 2 * vy_m + vy_e)

            state.age += 1

            # Boundary checks
            ix, iy = int(state.x), int(state.y)
            if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
                state.status = "escaped"
                break

            # Deposition check: hit solid wall
            if obstacle_mask[min(max(iy, 0), ny - 1), min(max(ix, 0), nx - 1)]:
                state.status = "deposited"
                state.deposit_x = state.x * dx_phys
                state.deposit_y = state.y * dx_phys
                break

            # Record trajectory
            if step % record_every == 0:
                res.trajectory_x.append(state.x * dx_phys)
                res.trajectory_y.append(state.y * dx_phys)

        res.status = state.status
        res.deposit_x = state.deposit_x
        res.deposit_y = state.deposit_y
        res.age_steps = int(state.age)
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# Deposition map
# ---------------------------------------------------------------------------

def build_deposition_map(
    results: list[ParticleTrackResult],
    nx: int,
    ny: int,
    dx_phys: float = 1.0,
) -> dict:
    """Aggregate particle deposition positions into a density map.

    Returns a dict suitable for JSON serialisation / heatmap rendering.
    """
    deposited = [r for r in results if r.status == "deposited"]
    dep_x = [r.deposit_x for r in deposited if r.deposit_x is not None]
    dep_y = [r.deposit_y for r in deposited if r.deposit_y is not None]

    return {
        "n_total": len(results),
        "n_deposited": len(deposited),
        "n_escaped": sum(1 for r in results if r.status == "escaped"),
        "n_active": sum(1 for r in results if r.status == "active"),
        "deposition_fraction": len(deposited) / max(len(results), 1),
        "deposit_x": dep_x,
        "deposit_y": dep_y,
    }
