"""2-D D2Q9 open-channel flow (periodic in x, no-slip top/bottom).

A body-force-driven channel.  The steady state of the incompressible
Navier-Stokes equations with a uniform x-body force and no pressure gradient
is the Poiseuille parabola

    u(y) = (F / 2 nu) y (h - y),        u* = sqrt(F h / 2),

which is an *exact* solution, so the channel is initialised directly on that
profile and the LBM only has to maintain it.  The observable for the sediment
surface is the **wall-shear velocity** ``u* = sqrt(tau_b / rho)``, recovered
from the LBM velocity field via a finite-difference wall gradient.  This is
the flow-side input to the Rouse profile (C2) and the incipient-motion scan
(C1); the sediment response is computed by the Exner/MPM model, so the anchors
are not circular.

The solver reuses the repository's D2Q9 kernels (``tensorlbm.d2q9``,
``tensorlbm.solver``) and the bounce-back boundary (``tensorlbm.boundaries``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tensorlbm.boundaries import bounce_back_cells
from tensorlbm.d2q9 import C, W, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk_fused, stream

__all__ = ["ChannelConfig", "run_channel", "poiseuille_profile"]


@dataclass(frozen=True)
class ChannelConfig:
    """Parameters for the 2-D channel (lattice units)."""

    nx: int = 128
    ny: int = 42  # includes the two wall rows
    nu: float = 0.01  # lattice kinematic viscosity
    body_force: float = 3.75e-6  # x-directed body force (per unit mass)
    n_steps: int = 1500
    seed: int = 7

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def depth(self) -> int:
        """Wall-to-wall distance in lattice units (wall cell centres)."""
        return self.ny - 1


def poiseuille_profile(ny: int, body_force: float, nu: float) -> list[float]:
    """Analytic Poiseuille u(y) on the lattice rows (row i sits at y = i)."""
    h = ny - 1
    return [
        (body_force / (2.0 * nu)) * i * (h - i) if 0 < i < h else 0.0
        for i in range(ny)
    ]


def run_channel(cfg: ChannelConfig, device: str = "cpu") -> dict:
    """Run the channel and extract wall-shear / profile statistics.

    Returns a dict with:
        u_profile:  streamwise velocity per row (lattice units), x-averaged.
        u_profile_analytic: analytic Poiseuille profile on the same rows.
        u_star:     wall-shear velocity (lattice units), mean of both walls.
        u_mean:     volume-mean streamwise velocity (lattice units).
        profile_rel_err: max relative error of the LBM profile vs analytic.
        steps:      number of steps actually run.
    """
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    ny, nx = cfg.ny, cfg.nx
    h = cfg.depth

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    wall[0, :] = True
    wall[-1, :] = True

    # Initialise on the exact Poiseuille solution (exact steady state).
    u_an = poiseuille_profile(ny, cfg.body_force, cfg.nu)
    rho0 = torch.ones((ny, nx), device=dev, dtype=torch.float64)
    ux0 = torch.tensor(u_an, device=dev, dtype=torch.float64).unsqueeze(1).expand(ny, nx).clone()
    uy0 = torch.zeros((ny, nx), device=dev, dtype=torch.float64)
    f = equilibrium(rho0, ux0, uy0, device=dev)

    src_coeff = (W.to(dev) * C.to(dev)[:, 0] * cfg.body_force).view(9, 1, 1)
    for _ in range(cfg.n_steps):
        f = collide_bgk_fused(f, cfg.tau)
        f = stream(f)
        f = f + src_coeff * f.sum(dim=0, keepdim=True)
        f = bounce_back_cells(f, wall)

    rho, ux, uy = macroscopic(f)
    ux = ux.masked_fill(wall, 0.0)

    u_prof = ux[1:-1, :].mean(dim=1)
    u_mean = float(u_prof.mean().item())

    # Wall shear: tau_b = rho nu (du/dy)|_wall with (du/dy)|_wall ~ u_1 / 1
    # (wall cell at zero velocity, first fluid cell one lattice unit away).
    # Wall gradient from a parabolic fit u(y) = a y (h - y) over the fluid
    # rows (robust to the O(dy) near-wall discretisation of bounce-back).
    # The wall gradient is a h, so u* = sqrt(nu a h).
    y = torch.arange(1, ny - 1, device=dev, dtype=torch.float64)
    basis = y * (h - y)
    a_fit = float((u_prof * basis).sum().item() / (basis * basis).sum().item())
    u_star = math.sqrt(cfg.nu * a_fit * h)

    u_an_fluid = torch.tensor(u_an[1:-1], device=dev, dtype=torch.float64)
    rel = (u_prof - u_an_fluid).abs() / u_an_fluid.clamp_min(1e-12)
    return {
        "u_profile": u_prof.tolist(),
        "u_profile_analytic": u_an[1:-1],
        "u_star": u_star,
        "u_mean": u_mean,
        "profile_rel_err": float(rel.max().item()),
        "steps": cfg.n_steps,
        "nu": cfg.nu,
        "depth": h,
    }


