"""D2Q9 LBM simulation of plane Couette flow.

Physical setup
--------------
Two infinite parallel plates separated by distance *H*.  The bottom plate
is stationary; the top plate moves at constant velocity *U* in the +x
direction.  The analytical solution of the Navier-Stokes equations for
this configuration is a linear velocity profile:

    u_x(y) = U * y / H,   u_y = 0,   p = const.

This is one of the few exact solutions of the incompressible Navier-Stokes
equations and serves as a fundamental validation case for any CFD solver.

Lattice-Boltzmann implementation
---------------------------------
* **Lattice**: D2Q9
* **Collision**: BGK with relaxation time tau = 3*nu + 0.5
* **Bottom wall** (y = 0): standard bounce-back (no-slip, stationary)
* **Top wall** (y = ny-1): moving-wall bounce-back with velocity correction
* **x-direction**: periodic (natural with torch.roll streaming)

The moving-wall bounce-back follows the half-way rule with momentum
correction.  After streaming at the top wall the unknown populations
(f4, f7, f8 — those that came from outside the domain) are reconstructed as:

    f4 = f2
    f7 = f5 - (1/6) * rho * U
    f8 = f6 + (1/6) * rho * U

where rho is computed from the known populations.

Outputs
-------
* Velocity profile u_x(y) at the domain centre (x = nx//2)
* L2 error vs analytical linear profile
* No-slip error at both walls
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CouetteConfig:
    """Configuration for the plane Couette flow validation.

    Args:
        nx: Number of grid cells in x (streamwise, periodic).
        ny: Number of grid cells in y (wall-normal).
        u_top: Top wall velocity in lattice units.
        nu: Kinematic viscosity in lattice units.
        n_steps: Total number of time steps.
        n_warmup: Steps to discard before sampling (steady-state spin-up).
        device: Torch device string.
    """

    nx: int = 64
    ny: int = 64
    u_top: float = 0.05
    nu: float = 0.02
    n_steps: int = 5000
    n_warmup: int = 2000
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", self.device.lower())

    @property
    def tau(self) -> float:
        """BGK relaxation time tau = 3*nu + 0.5."""
        return 3.0 * self.nu + 0.5

    @property
    def re(self) -> float:
        """Reynolds number Re = U * H / nu."""
        return self.u_top * self.ny / self.nu

    def validate(self) -> None:
        if self.nx < 4:
            raise ValueError("nx must be >= 4")
        if self.ny < 4:
            raise ValueError("ny must be >= 4")
        if self.u_top <= 0:
            raise ValueError("u_top must be > 0")
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau:.4f} <= 0.5; increase nu")
        if self.n_steps <= self.n_warmup:
            raise ValueError("n_steps must be > n_warmup")


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def moving_wall_bounce_back_top(f: torch.Tensor, u_wall: float) -> torch.Tensor:
    """Moving-wall bounce-back for the top row (y = ny-1).

    After streaming at the top wall the unknown populations are f4, f7, f8
    (those that came from outside the domain, i.e. pointing downward).
    They are reconstructed using the half-way bounce-back rule with a
    momentum correction proportional to the wall velocity.

    D2Q9 direction convention:
    ``0=(0,0), 1=(E), 2=(N), 3=(W), 4=(S), 5=(NE), 6=(NW), 7=(SW), 8=(SE)``

    Formulas (with uy_wall = 0):
        rho = f0 + f1 + 2*f2 + f3 + 2*f5 + 2*f6
        f4  = f2
        f7  = f5 - (1/6) * rho * u_wall
        f8  = f6 + (1/6) * rho * u_wall

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)`` **after streaming**.
        u_wall: Prescribed x-velocity of the top wall.

    Returns:
        Updated distribution tensor (same shape).
    """
    # Density from known populations (those that came from inside the domain)
    rho_wall = (f[0, -1, :] + f[1, -1, :] + 2.0 * f[2, -1, :]
                + f[3, -1, :] + 2.0 * f[5, -1, :] + 2.0 * f[6, -1, :])
    rho_wall = rho_wall.clamp(min=1e-12)

    f_new = f.clone()
    f_new[4, -1, :] = f[2, -1, :]
    f_new[7, -1, :] = f[5, -1, :] - (1.0 / 6.0) * rho_wall * u_wall
    f_new[8, -1, :] = f[6, -1, :] + (1.0 / 6.0) * rho_wall * u_wall
    return f_new


def stationary_bounce_back_bottom(f: torch.Tensor) -> torch.Tensor:
    """Standard bounce-back for the bottom row (y = 0).

    After streaming, the unknown populations at the bottom wall are those
    pointing into the domain: directions 2 (N), 5 (NE), 6 (NW).
    These are set equal to the opposite-direction populations.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)`` **after streaming**.

    Returns:
        Updated distribution tensor.
    """
    f_new = f.clone()
    f_new[2, 0, :] = f[4, 0, :]
    f_new[5, 0, :] = f[7, 0, :]
    f_new[6, 0, :] = f[8, 0, :]
    return f_new


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def run_couette(cfg: CouetteConfig) -> dict:
    """Run the plane Couette flow simulation.

    Args:
        cfg: Simulation configuration.

    Returns:
        Dictionary with results:
            - ux_profile: 1D tensor of u_x values along y at x=nx//2
            - uy_profile: 1D tensor of u_y values along y at x=nx//2
            - y_coords: 1D tensor of y coordinates (lattice units)
            - u_analytical: analytical linear profile U*y/H
            - l2_error: L2 relative error of ux vs analytical
            - no_slip_bottom: |ux(y=0)| (should be ~0)
            - no_slip_top: |ux(y=ny-1) - U| (should be ~0)
            - n_steps: total steps run
            - re: Reynolds number
            - tau: relaxation time
    """
    cfg.validate()
    device = torch.device(cfg.device)
    nx, ny = cfg.nx, cfg.ny
    tau = cfg.tau
    u_top = cfg.u_top

    # Initialize distributions at equilibrium (fluid at rest)
    rho = torch.ones((ny, nx), dtype=torch.float32, device=device)
    ux = torch.zeros((ny, nx), dtype=torch.float32, device=device)
    uy = torch.zeros((ny, nx), dtype=torch.float32, device=device)
    f = equilibrium(rho, ux, uy)

    # Time stepping
    for step in range(cfg.n_steps):
        # Collision
        f = collide_bgk(f, tau)

        # Streaming
        f = stream(f)

        # Boundary conditions
        # Bottom wall: stationary bounce-back
        f = stationary_bounce_back_bottom(f)

        # Top wall: moving-wall bounce-back
        f = moving_wall_bounce_back_top(f, u_top)

    # Extract velocity profile at centre column
    x_mid = nx // 2
    rho_final, ux_final, uy_final = macroscopic(f)

    ux_profile = ux_final[:, x_mid].cpu()
    uy_profile = uy_final[:, x_mid].cpu()

    # y coordinates (lattice units, from 0 to ny-1)
    y_coords = torch.arange(ny, dtype=torch.float32)

    # Analytical solution: u(y) = U * y / H
    # With half-way bounce-back, the effective wall positions are at
    # y = -0.5 (bottom) and y = ny - 0.5 (top), so H_eff = ny - 1.
    # The analytical profile at node j is: u(j) = U * j / (ny - 1)
    h_eff = float(ny - 1)
    u_analytical = u_top * y_coords / h_eff

    # L2 error (relative)
    diff = ux_profile - u_analytical
    l2_norm = torch.sqrt(torch.sum(diff ** 2))
    ref_norm = torch.sqrt(torch.sum(u_analytical ** 2))
    l2_error = float(l2_norm / ref_norm) if ref_norm > 1e-15 else float(l2_norm)

    # No-slip errors
    no_slip_bottom = float(abs(ux_profile[0].item()))
    no_slip_top = float(abs(ux_profile[-1].item() - u_top))

    return {
        "ux_profile": ux_profile,
        "uy_profile": uy_profile,
        "y_coords": y_coords,
        "u_analytical": u_analytical,
        "l2_error": l2_error,
        "no_slip_bottom": no_slip_bottom,
        "no_slip_top": no_slip_top,
        "n_steps": cfg.n_steps,
        "n_warmup": cfg.n_warmup,
        "re": cfg.re,
        "tau": tau,
        "nx": nx,
        "ny": ny,
        "u_top": u_top,
        "nu": cfg.nu,
    }


def save_results(results: dict, output_dir: Path) -> Path:
    """Save results to JSON (excluding tensors)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: v for k, v in results.items()
        if not isinstance(v, torch.Tensor)
    }
    serializable["ux_profile"] = results["ux_profile"].tolist()
    serializable["uy_profile"] = results["uy_profile"].tolist()
    serializable["y_coords"] = results["y_coords"].tolist()
    serializable["u_analytical"] = results["u_analytical"].tolist()

    out_path = output_dir / "couette_results.json"
    with open(out_path, "w") as fh:
        json.dump(serializable, fh, indent=2)
    return out_path
