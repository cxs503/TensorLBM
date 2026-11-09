"""Boussinesq buoyancy coupling for LBM.

Implements the Boussinesq approximation for natural convection and buoyant
plumes: density variations are neglected everywhere except in the gravity
term, where ρ = ρ₀(1 - β(T - T_ref)) produces a buoyancy force.

Physical model:
- Flow: D2Q9 BGK with Guo forcing for buoyancy
- Scalar (temperature/concentration): D2Q5 advection-diffusion
- Buoyancy: F_b = +ρ₀ g β (T - T_ref) ĵ (upward for T > T_ref)

Applications:
- Natural convection cavities
- Buoyant plumes (chimney, smokestack, thermal plumes)
- Gas dispersion with buoyancy effects (heavy/light gas releases)

Key dimensionless numbers:
- Richardson number: Ri = g β ΔT H / U² (buoyancy vs inertia)
- Grashof number: Gr = g β ΔT H³ / ν²
- Rayleigh number: Ra = Gr · Pr

References:
- Guo Z., Zheng C., Shi B. (2002). "Discrete lattice effects on the forcing
  term in the lattice Boltzmann method." Phys Rev E 65, 046308.
- Shan X. (1993). "Lattice Boltzmann model for simulating magnetohydrodynamics."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .d2q9 import C, W, equilibrium, macroscopic
from .scalar_transport import (
    C5,
    W5,
    scalar_collision,
    scalar_equilibrium,
    scalar_macroscopic,
    scalar_stream,
)
from .solver import collide_bgk, stream


@dataclass
class BoussinesqConfig:
    """Configuration for Boussinesq buoyancy simulation."""
    nx: int = 128
    ny: int = 256
    tau_f: float = 0.6
    tau_s: float = 0.7
    g_beta: float = 1e-4  # gravity × expansion coefficient (positive)
    T_ref: float = 0.0
    T_source: float = 1.0
    source_x: int = 64
    source_y: int = 10
    source_width: int = 8
    U_inlet: float = 0.0
    n_steps: int = 10000
    output_interval: int = 500
    Richardson: float = 0.0


@dataclass
class BoussinesqState:
    """Runtime state for Boussinesq simulation."""
    f: torch.Tensor
    g: torch.Tensor
    step: int = 0
    history: list = field(default_factory=list)


def compute_buoyancy_force(
    rho: torch.Tensor,
    T: torch.Tensor,
    T_ref: float,
    g_beta: float,
    gravity_dir: str = "y",
) -> torch.Tensor:
    """Compute Boussinesq buoyancy force: F = +ρ g β (T - T_ref) ĵ.

    For T > T_ref, the force is upward (positive y). This represents
    the buoyancy force on a fluid parcel that is lighter than its surroundings.

    Args:
        rho: Density field (ny, nx)
        T: Temperature/scalar field (ny, nx)
        T_ref: Reference temperature
        g_beta: Combined gravity × thermal expansion coefficient (positive)
        gravity_dir: Direction of gravity ("y" for vertical)

    Returns:
        Force tensor (2, ny, nx) with Fx, Fy components
    """
    dT = T - T_ref
    F_buoy = rho * g_beta * dT  # positive when T > T_ref (upward)

    if gravity_dir == "y":
        Fx = torch.zeros_like(F_buoy)
        Fy = F_buoy
    elif gravity_dir == "x":
        Fx = F_buoy
        Fy = torch.zeros_like(F_buoy)
    else:
        raise ValueError(f"Unknown gravity direction: {gravity_dir}")

    return torch.stack([Fx, Fy], dim=0)


def apply_guo_forcing(
    f: torch.Tensor,
    F: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Apply Guo et al. (2002) forcing scheme to D2Q9 distributions."""
    device = f.device
    rho, ux, uy = macroscopic(f)
    rho_safe = rho.clamp(min=1e-12)

    ux_star = ux + 0.5 * F[0] / rho_safe
    uy_star = uy + 0.5 * F[1] / rho_safe

    feq = equilibrium(rho, ux_star, uy_star)
    f_post = f - (f - feq) / tau

    c = C.to(device)
    w = W.to(device=device, dtype=f.dtype).view(9, 1, 1)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    Fx = F[0].unsqueeze(0)
    Fy = F[1].unsqueeze(0)
    cF = cx * Fx + cy * Fy
    uF = ux_star * F[0] + uy_star * F[1]
    cu = cx * ux_star.unsqueeze(0) + cy * uy_star.unsqueeze(0)

    Fi = w * (1.0 - 0.5 / tau) * (3.0 * (cF - uF.unsqueeze(0)) + 9.0 * cu * cF)

    return f_post + Fi


def apply_scalar_source(
    g: torch.Tensor,
    T_source: float,
    x_center: int,
    y_center: int,
    width: int,
) -> torch.Tensor:
    """Apply a scalar (temperature) source at the bottom boundary."""
    g = g.clone()
    ny, nx = g.shape[1], g.shape[2]

    yy, xx = torch.meshgrid(
        torch.arange(ny, device=g.device),
        torch.arange(nx, device=g.device),
        indexing="ij",
    )
    mask = (
        (xx >= x_center - width) & (xx <= x_center + width) &
        (yy >= y_center - 1) & (yy <= y_center + 1)
    )

    w = W5.to(device=g.device, dtype=g.dtype).view(5, 1, 1)
    g_source = w * T_source
    m = mask.unsqueeze(0).expand_as(g)
    g = torch.where(m, g_source.expand_as(g), g)

    return g


def run_boussinesq_plume(
    cfg: BoussinesqConfig,
    device: torch.device | str = "cpu",
) -> BoussinesqState:
    """Run a buoyant plume simulation with Boussinesq coupling."""
    device = torch.device(device)
    nx, ny = cfg.nx, cfg.ny

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), cfg.U_inlet, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0)

    T0 = torch.zeros((ny, nx), device=device)
    g = scalar_equilibrium(T0, ux0, uy0)

    state = BoussinesqState(f=f, g=g)

    for step in range(cfg.n_steps):
        rho, ux, uy = macroscopic(state.f)
        T = scalar_macroscopic(state.g)

        F = compute_buoyancy_force(rho, T, cfg.T_ref, cfg.g_beta)

        state.f = apply_guo_forcing(state.f, F, cfg.tau_f)
        state.f = stream(state.f)

        # Top: convective outflow
        state.f[:, -1, :] = 2.0 * state.f[:, -2, :] - state.f[:, -3, :]

        rho, ux, uy = macroscopic(state.f)

        state.g = scalar_collision(state.g, T, ux, uy, cfg.tau_s)
        state.g = scalar_stream(state.g)

        state.g = apply_scalar_source(
            state.g, cfg.T_source,
            cfg.source_x, cfg.source_y, cfg.source_width
        )

        state.g[:, -1, :] = 2.0 * state.g[:, -2, :] - state.g[:, -3, :]

        state.step = step + 1

        if (step + 1) % cfg.output_interval == 0:
            T_field = scalar_macroscopic(state.g)
            _, uy_field = macroscopic(state.f)[1:]
            state.history.append({
                "step": step + 1,
                "T_max": float(T_field.max().item()),
                "T_total": float(T_field.sum().item()),
                "uy_max": float(uy_field.max().item()),
            })

    return state


def compute_volume_flux(
    T: torch.Tensor,
    uy: torch.Tensor,
    y_indices: list[int],
    T_threshold: float = 0.01,
) -> dict[int, float]:
    """Compute volume flux Q(z) at multiple heights for plume analysis."""
    fluxes = {}
    for y_idx in y_indices:
        if y_idx >= T.shape[0]:
            continue
        mask = T[y_idx] > T_threshold
        Q = float((uy[y_idx] * mask.float()).sum().item())
        fluxes[y_idx] = Q
    return fluxes


def fit_plume_exponent(
    heights: list[float],
    fluxes: list[float],
) -> dict[str, float]:
    """Fit power-law exponent to plume volume flux: Q(z) ∝ z^n."""
    import math

    valid = [(h, q) for h, q in zip(heights, fluxes) if h > 0 and q > 0]
    if len(valid) < 2:
        return {"exponent": float("nan"), "r_squared": float("nan")}

    log_h = [math.log(h) for h, _ in valid]
    log_q = [math.log(q) for _, q in valid]

    n = len(valid)
    sum_x = sum(log_h)
    sum_y = sum(log_q)
    sum_xy = sum(x * y for x, y in zip(log_h, log_q))
    sum_x2 = sum(x * x for x in log_h)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-15:
        return {"exponent": float("nan"), "r_squared": float("nan")}

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in log_q)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_h, log_q))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"exponent": slope, "r_squared": r_squared, "intercept": intercept}


__all__ = [
    "BoussinesqConfig",
    "BoussinesqState",
    "apply_guo_forcing",
    "apply_scalar_source",
    "compute_buoyancy_force",
    "compute_volume_flux",
    "fit_plume_exponent",
    "run_boussinesq_plume",
]
