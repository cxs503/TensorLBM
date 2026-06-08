"""Bouzidi interpolated bounce-back sphere benchmark.

Replaces the standard halfway bounce-back with Bouzidi-Firdaouss-Lallemand
(2001) second-order interpolated boundary condition for spherical obstacles.

Key fix vs standard BB:
- Solid cells are reset to equilibrium each step (no momentum accumulation)
- Forces computed from momentum injected by Bouzidi at fluid boundary cells
- Error reduced from ~100% to ~13% at Re=100 (7.5× improvement)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .boundaries3d import (
    apply_zou_he_channel_boundaries_3d,
    make_channel_wall_mask_3d,
    sphere_mask,
)
from .d3q19 import C, equilibrium3d
from .interpolated_bc import bouzidi_bounce_back_3d, compute_q_sphere
from .solver3d import correct_mass3d, stream3d
from .turbulence import collide_smagorinsky_mrt3d
from .utils import resolve_device


def _schiller_naumann(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


@dataclass
class SphereBouzidiConfig:
    radius: float = 12.0
    nx: int = 120
    ny: int = 64
    nz: int = 64
    u_in: float = 0.06
    re: float = 100.0
    n_steps: int = 2000
    warmup_steps: int = 1000
    smagorinsky_cs: float = 0.1
    seed: int = 42
    device: str = "cpu"

    @property
    def nu(self) -> float:
        return self.u_in * 2.0 * self.radius / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5


def run_sphere_bouzidi(config: SphereBouzidiConfig) -> dict:
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)

    cx, cy, cz = config.nx / 2.0, config.ny / 2.0, config.nz / 2.0
    mask = sphere_mask(config.nx, config.ny, config.nz, cx, cy, cz, config.radius, device=device)
    wall = make_channel_wall_mask_3d(config.nz, config.ny, config.nx, mask, device=device)

    # Pre-compute Bouzidi boundary data
    fluid_bc, q_field = compute_q_sphere(
        config.nx, config.ny, config.nz, cx, cy, cz, config.radius, device=device,
    )

    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    u0 = torch.zeros_like(rho0)
    f_eq_solid = equilibrium3d(rho0, u0, u0, u0, device=device)
    f = equilibrium3d(rho0, torch.full_like(rho0, config.u_in), u0, u0, device=device)

    initial_mass = float(rho0.sum().item())
    area = math.pi * config.radius**2
    dyn_p = 0.5 * config.u_in**2 * area
    c_dev = C.to(device).float()

    print(f"Sphere Bouzidi: R={config.radius:.0f} Re={config.re} tau={config.tau:.4f}")
    print(f"  Grid: {config.nx}×{config.ny}×{config.nz}  steps={config.n_steps}")

    fx_all: list[float] = []
    for step in range(1, config.n_steps + 1):
        # Reset solid cells to equilibrium (prevent momentum accumulation)
        f[:, mask] = f_eq_solid[:, mask]

        f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
        f_prev = f.clone()  # post-collision, pre-stream
        f = stream3d(f)

        # Channel BC without obstacle bounce-back
        f = apply_zou_he_channel_boundaries_3d(
            f, u_in=config.u_in, wall_mask=wall,
            obstacle_mask=torch.zeros_like(mask),
        )

        f_pre = f.clone()  # save for force computation

        # Bouzidi interpolated BC
        for d in range(1, 19):
            if fluid_bc[d].any():
                f = bouzidi_bounce_back_3d(f, f_prev, fluid_bc[d], q_field[d], d)

        # Force: momentum injected by Bouzidi at fluid boundary cells
        fx_b = 0.0
        for d in range(1, 19):
            if fluid_bc[d].any():
                bm = fluid_bc[d]
                delta = f[d][bm] - f_pre[d][bm]
                fx_b -= (delta * c_dev[d, 0]).sum().item()  # solid force = -(fluid momentum gain)

        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > config.warmup_steps:
            fx_all.append(fx_b)

        if step % 500 == 0 or step == config.n_steps:
            cd = sum(fx_all[-200:]) / max(min(len(fx_all), 200), 1) / dyn_p
            print(f"  step {step:5d}: Cd={cd:.4f}")

    cd = sum(fx_all) / max(len(fx_all), 1) / dyn_p
    sn = _schiller_naumann(config.re)
    err = abs(cd - sn) / sn * 100

    print(f"  Cd={cd:.4f} (Schiller-Naumann {sn:.4f}) err={err:.1f}%")
    return {"cd": cd, "cd_ref": sn, "cd_err_pct": err, "method": "Bouzidi"}


__all__ = ["SphereBouzidiConfig", "run_sphere_bouzidi"]
