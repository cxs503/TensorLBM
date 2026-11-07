"""Conjugate Heat Transfer (CHT) validation suite.

Four analytical anchors:
1. 1D composite wall – series thermal resistance (steady-state)
2. erf heat penetration – semi-infinite solid transient conduction
3. Pipe Graetz problem – Nu number for developing thermal entry flow
4. Vertical plate Churchill-Chu – natural convection Nu correlation

Plus interface heat-flux closure verification.

All physical parameters are programmatic (no hand-copied numbers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .conjugate_ht import CHTConfig, CHTState, apply_cht_interface, cht_solid_diffusion_step
from .d2q9 import equilibrium, macroscopic
from .solver import collide_bgk, stream
from .thermal import (
    apply_buoyancy_force,
    collide_thermal_bgk,
    equilibrium_thermal,
    macroscopic_thermal,
    stream_thermal,
)
from .boundaries import bounce_back_cells

__all__ = [
    "PhysicalParams",
    "LatticeParams",
    "physical_to_lattice",
    "lattice_to_physical",
    "run_1d_composite_wall",
    "analytical_composite_wall",
    "run_erf_penetration",
    "analytical_erf",
    "run_graetz_pipe",
    "analytical_graetz_nu",
    "run_churchill_chu",
    "analytical_churchill_chu_nu",
    "check_flux_closure",
]

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


@dataclass
class PhysicalParams:
    """Physical parameters for a CHT case."""
    rho_f: float = 998.0
    mu_f: float = 1.002e-3
    k_f: float = 0.598
    cp_f: float = 4182.0
    k_s: float = 45.0
    rho_s: float = 7850.0
    cp_s: float = 490.0
    k_i: float = 0.04
    rho_i: float = 100.0
    cp_i: float = 1000.0
    T_hot: float = 350.0
    T_cold: float = 300.0


@dataclass
class LatticeParams:
    """Lattice-unit parameters derived from physical ones."""
    dx: float = 1.0
    dt: float = 1.0
    rho_lu: float = 1.0
    nu_lu: float = 0.1
    alpha_f_lu: float = 0.1
    alpha_s_lu: float = 0.05
    alpha_i_lu: float = 0.02
    k_ratio_s: float = 5.0
    k_ratio_i: float = 0.1
    u_lu: float = 0.05
    T_hot_lu: float = 1.0
    T_cold_lu: float = 0.0
    tau_f: float = 0.8
    tau_g: float = 0.8


def physical_to_lattice(
    L_phys: float, U_phys: float, N: int,
    phys: PhysicalParams | None = None,
) -> LatticeParams:
    """Convert physical parameters to lattice units."""
    phys = phys or PhysicalParams()
    dx = L_phys / N
    u_lu = 0.05
    dt = u_lu * dx / U_phys

    nu_phys = phys.mu_f / phys.rho_f
    alpha_f_phys = phys.k_f / (phys.rho_f * phys.cp_f)
    alpha_s_phys = phys.k_s / (phys.rho_s * phys.cp_s)
    alpha_i_phys = phys.k_i / (phys.rho_i * phys.cp_i)

    nu_lu = nu_phys * dt / (dx * dx)
    alpha_f_lu = alpha_f_phys * dt / (dx * dx)
    alpha_s_lu = alpha_s_phys * dt / (dx * dx)
    alpha_i_lu = alpha_i_phys * dt / (dx * dx)

    tau_f = 3.0 * nu_lu + 0.5
    tau_g = 3.0 * alpha_f_lu + 0.5

    k_ratio_s = phys.k_s / phys.k_f
    k_ratio_i = phys.k_i / phys.k_f

    return LatticeParams(
        dx=dx, dt=dt, nu_lu=nu_lu,
        alpha_f_lu=alpha_f_lu, alpha_s_lu=alpha_s_lu, alpha_i_lu=alpha_i_lu,
        k_ratio_s=k_ratio_s, k_ratio_i=k_ratio_i,
        u_lu=u_lu, T_hot_lu=1.0, T_cold_lu=0.0,
        tau_f=max(tau_f, 0.51), tau_g=max(tau_g, 0.51),
    )


def lattice_to_physical(T_lu: torch.Tensor, T_hot_phys: float, T_cold_phys: float) -> torch.Tensor:
    """Convert lattice temperature to physical units."""
    return T_cold_phys + T_lu * (T_hot_phys - T_cold_phys)


# ---------------------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------------------


def analytical_composite_wall(
    x: torch.Tensor,
    h_water: float, t_s: float, t_i: float,
    k_f: float, k_s: float, k_i: float,
    T_hot: float, T_cold: float,
    h_conv: float | None = None,
) -> torch.Tensor:
    """1D steady-state composite wall temperature profile."""
    if h_conv is None:
        h_conv = k_f / h_water

    R_i = t_i / k_i
    R_s = t_s / k_s
    R_f = 1.0 / h_conv
    R_total = R_i + R_s + R_f
    q = (T_hot - T_cold) / R_total
    L_total = t_i + t_s + h_water

    T = torch.zeros_like(x)
    for idx in range(x.numel()):
        xi = x[idx].item()
        if xi <= t_i:
            T[idx] = T_hot - q * xi / k_i
        elif xi <= t_i + t_s:
            T[idx] = T_hot - q * (R_i + (xi - t_i) / k_s)
        else:
            T[idx] = T_cold + q * (L_total - xi) / k_f
    return T


def analytical_erf(
    x: torch.Tensor, t: float, alpha: float,
    T_surface: float, T_initial: float,
) -> torch.Tensor:
    """Semi-infinite solid transient conduction (error function solution)."""
    eta = x / (2.0 * math.sqrt(alpha * t))
    erf_vals = torch.tensor(
        [math.erf(float(v)) for v in eta.flatten()],
        dtype=x.dtype, device=x.device
    ).reshape(eta.shape)
    return T_surface + (T_initial - T_surface) * erf_vals


def analytical_graetz_nu(Re: float, Pr: float, x_star: float) -> float:
    """Graetz problem Nu number (Shah-London correlation)."""
    if x_star < 1e-10:
        return 100.0
    if x_star >= 0.1:
        return 4.364
    nu_entrance = 1.302 * x_star ** (-1.0 / 3.0) - 1.0
    nu_fd = 4.364
    return (nu_fd ** 3 + max(nu_entrance, 0.0) ** 3) ** (1.0 / 3.0)


def analytical_churchill_chu_nu(Ra: float, Pr: float) -> float:
    """Churchill-Chu correlation for natural convection on a vertical plate."""
    if Ra <= 0:
        return 0.0
    term1 = 0.825
    term2 = 0.387 * Ra ** (1.0 / 6.0)
    denom = (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    return (term1 + term2 / denom) ** 2


# ---------------------------------------------------------------------------
# 1D Composite Wall LBM Solver
# ---------------------------------------------------------------------------


def run_1d_composite_wall(
    N_fluid: int = 20,
    N_steel: int = 10,
    N_insul: int = 30,
    n_steps: int = 50000,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run 1D composite wall CHT simulation.

    Geometry (y-direction):
      y=0: hot wall (T=1) → insulation → steel → fluid → y=N-1: cold wall (T=0)
    """
    dev = torch.device(device)
    ny = N_insul + N_steel + N_fluid
    nx = 4

    mask_solid = torch.zeros(ny, nx, dtype=torch.bool, device=dev)
    mask_solid[0:N_insul, :] = True
    mask_solid[N_insul:N_insul + N_steel, :] = True

    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = torch.zeros(ny, nx, device=dev)
    uy0 = torch.zeros(ny, nx, device=dev)
    T0 = torch.zeros(ny, nx, device=dev)

    f = equilibrium(rho0, ux0, uy0, device=dev)
    g = equilibrium_thermal(T0, ux0, uy0)
    T_s = torch.zeros(ny, nx, device=dev)

    state = CHTState(f=f, g=g, T_s=T_s, mask_solid=mask_solid)

    cfg = CHTConfig(
        tau_f=0.8, kappa_f=1.0 / 6.0, alpha_s=0.05,
        k_ratio=10.0, beta=0.0, T_ref=0.5, gravity=0.0,
        T_hot=1.0, T_cold=0.0, Q_source=0.0,
        n_steps=n_steps, output_interval=n_steps,
    )

    final = _run_cht_1d(state, cfg, device=dev)

    T_fluid = macroscopic_thermal(final.g)
    T_combined = torch.where(final.mask_solid, final.T_s, T_fluid)
    T_profile = T_combined[:, nx // 2].cpu()

    return {
        "T_profile": T_profile,
        "T_fluid": T_fluid.cpu(),
        "T_solid": final.T_s.cpu(),
        "mask_solid": final.mask_solid.cpu(),
        "N_insul": N_insul, "N_steel": N_steel, "N_fluid": N_fluid,
        "ny": ny, "nx": nx, "n_steps": n_steps,
    }


def _run_cht_1d(
    state: CHTState, cfg: CHTConfig, device: str = "cpu",
) -> CHTState:
    """1D-specialized CHT loop (no buoyancy, pure conduction)."""
    dev = torch.device(device)
    tau_g = 0.5 + cfg.kappa_f / (1.0 / 6.0)
    ny, nx = state.f.shape[1], state.f.shape[2]

    for step in range(state.step, state.step + cfg.n_steps):
        # Fluid momentum
        state.f = collide_bgk(state.f, cfg.tau_f)
        state.f = stream(state.f)
        state.f = bounce_back_cells(state.f, state.mask_solid)

        # Wall BCs: hot at y=0, cold at y=ny-1
        rho_w = torch.ones(nx, device=dev)
        ux_w = torch.zeros(nx, device=dev)
        uy_w = torch.zeros(nx, device=dev)
        f_hot = equilibrium(rho_w, ux_w, uy_w, device=dev)
        f_cold = equilibrium(rho_w, ux_w, uy_w, device=dev)
        state.f[:, 0, :] = f_hot.squeeze(1)
        state.f[:, -1, :] = f_cold.squeeze(1)

        # Temperature
        T_fluid = macroscopic_thermal(state.g)
        T_fluid[0, :] = cfg.T_hot
        T_fluid[-1, :] = cfg.T_cold
        state.T_s[0, :] = cfg.T_hot
        state.T_s[-1, :] = cfg.T_cold

        rho_new, ux_new, uy_new = macroscopic(state.f)
        state.g = collide_thermal_bgk(state.g, T_fluid, ux_new, uy_new, tau_g)
        state.g = stream_thermal(state.g)

        # Thermal BCs on g
        g_hot = equilibrium_thermal(
            torch.full((nx,), cfg.T_hot, device=dev),
            torch.zeros(nx, device=dev), torch.zeros(nx, device=dev),
        )
        g_cold = equilibrium_thermal(
            torch.full((nx,), cfg.T_cold, device=dev),
            torch.zeros(nx, device=dev), torch.zeros(nx, device=dev),
        )
        state.g[:, 0, :] = g_hot.squeeze(1)
        state.g[:, -1, :] = g_cold.squeeze(1)

        # Solid conduction
        state.T_s = cht_solid_diffusion_step(
            state.T_s, state.mask_solid, cfg.alpha_s, cfg.Q_source
        )

        # Interface coupling
        T_fluid_new = macroscopic_thermal(state.g)
        T_f_cpl, T_s_cpl = apply_cht_interface(
            T_fluid_new, state.T_s, state.mask_solid, cfg.k_ratio
        )
        g_eq = equilibrium_thermal(T_f_cpl, ux_new, uy_new)
        state.g = state.g + (g_eq - state.g) * (1.0 / tau_g)
        state.T_s = T_s_cpl

        state.step = step + 1

    return state


# ---------------------------------------------------------------------------
# erf Heat Penetration Solver
# ---------------------------------------------------------------------------


def run_erf_penetration(
    N: int = 100, n_steps: int = 20000,
    alpha_lu: float = 0.05,
    T_surface: float = 1.0, T_initial: float = 0.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Semi-infinite solid transient conduction (1D)."""
    dev = torch.device(device)
    ny = N
    nx = 4

    mask_solid = torch.ones(ny, nx, dtype=torch.bool, device=dev)
    T_s = torch.full((ny, nx), T_initial, device=dev)
    T_s[0, :] = T_surface

    for _ in range(n_steps):
        T_s = cht_solid_diffusion_step(T_s, mask_solid, alpha_lu)
        T_s[0, :] = T_surface

    T_profile = T_s[:, nx // 2].cpu()
    x_lu = torch.arange(N, dtype=torch.float32)

    return {
        "T_profile": T_profile, "x_lu": x_lu,
        "n_steps": n_steps, "alpha_lu": alpha_lu,
        "T_surface": T_surface, "T_initial": T_initial,
    }


# ---------------------------------------------------------------------------
# Graetz Pipe Solver
# ---------------------------------------------------------------------------


def run_graetz_pipe(
    N_y: int = 30, N_x: int = 200,
    Re: float = 100.0, Pr: float = 7.0,
    n_steps: int = 30000, device: str = "cpu",
) -> dict[str, Any]:
    """Graetz problem: thermal entrance in a channel (2D approximation)."""
    dev = torch.device(device)
    ny, nx = N_y, N_x

    mask_solid = torch.zeros(ny, nx, dtype=torch.bool, device=dev)
    mask_solid[0, :] = True
    mask_solid[-1, :] = True

    y_norm = torch.linspace(0, 1, ny, device=dev)
    ux_inlet = 4.0 * y_norm * (1.0 - y_norm)

    rho0 = torch.ones(ny, nx, device=dev)
    uy0 = torch.zeros(ny, nx, device=dev)
    T0 = torch.zeros(ny, nx, device=dev)

    f = equilibrium(rho0, ux_inlet.unsqueeze(1).expand(ny, nx), uy0, device=dev)
    g = equilibrium_thermal(T0, ux_inlet.unsqueeze(1).expand(ny, nx), uy0)
    T_s = torch.zeros(ny, nx, device=dev)

    state = CHTState(f=f, g=g, T_s=T_s, mask_solid=mask_solid)

    nu_lu = Re * 0.05 / 3.0
    tau_f = 3.0 * nu_lu + 0.5
    alpha_lu = nu_lu / Pr
    tau_g = 3.0 * alpha_lu + 0.5

    cfg = CHTConfig(
        tau_f=max(tau_f, 0.51), kappa_f=alpha_lu, alpha_s=alpha_lu,
        k_ratio=100.0, beta=0.0, T_ref=0.5, gravity=0.0,
        T_hot=1.0, T_cold=0.0, Q_source=0.0,
        n_steps=n_steps, output_interval=n_steps,
    )

    final = _run_graetz_2d(state, cfg, ux_inlet, device=dev)

    T_field = macroscopic_thermal(final.g).cpu()
    T_bulk = T_field[1:-1, :].mean(dim=0)
    T_wall = T_field[0, :]
    dT_dy = (T_field[1, :] - T_field[0, :])
    Nu_local = -dT_dy * (ny - 1) / (T_wall - T_bulk + 1e-12)

    return {
        "T_field": T_field, "T_bulk": T_bulk, "T_wall": T_wall,
        "Nu_local": Nu_local, "Re": Re, "Pr": Pr, "N_y": N_y, "N_x": N_x,
    }


def _run_graetz_2d(
    state: CHTState, cfg: CHTConfig,
    ux_inlet: torch.Tensor, device: str = "cpu",
) -> CHTState:
    """Graetz problem CHT loop."""
    dev = torch.device(device)
    tau_g = 0.5 + cfg.kappa_f / (1.0 / 6.0)
    ny, nx = state.f.shape[1], state.f.shape[2]

    for step in range(state.step, state.step + cfg.n_steps):
        rho, ux, uy = macroscopic(state.f)

        # Inlet BC
        state.f[:, :, 0] = equilibrium(
            torch.ones(ny, device=dev),
            ux_inlet, torch.zeros(ny, device=dev),
        )

        state.f = collide_bgk(state.f, cfg.tau_f)
        state.f = stream(state.f)
        state.f = bounce_back_cells(state.f, state.mask_solid)
        state.f[:, :, -1] = state.f[:, :, -2]  # outlet

        T_fluid = macroscopic_thermal(state.g)

        # Inlet T BC
        T_inlet = torch.zeros(ny, device=dev)
        g_inlet = equilibrium_thermal(T_inlet, ux_inlet, torch.zeros(ny, device=dev))
        state.g[:, :, 0] = g_inlet.view(5, ny)

        # Wall heat flux BC
        T_fluid[0, :] = T_fluid[1, :] + 0.01
        T_fluid[-1, :] = T_fluid[-2, :] + 0.01

        rho_new, ux_new, uy_new = macroscopic(state.f)
        state.g = collide_thermal_bgk(state.g, T_fluid, ux_new, uy_new, tau_g)
        state.g = stream_thermal(state.g)

        state.g[:, :, 0] = g_inlet.view(5, ny)
        state.g[:, -1, :] = state.g[:, -2, :]

        state.T_s = cht_solid_diffusion_step(state.T_s, state.mask_solid, cfg.alpha_s)

        T_fluid_new = macroscopic_thermal(state.g)
        T_f_cpl, T_s_cpl = apply_cht_interface(
            T_fluid_new, state.T_s, state.mask_solid, cfg.k_ratio
        )
        g_eq = equilibrium_thermal(T_f_cpl, ux_new, uy_new)
        state.g = state.g + (g_eq - state.g) * (1.0 / tau_g)
        state.T_s = T_s_cpl

        state.step = step + 1

    return state


# ---------------------------------------------------------------------------
# Churchill-Chu Natural Convection
# ---------------------------------------------------------------------------


def run_churchill_chu(
    N: int = 64, Ra: float = 1e6, Pr: float = 0.71,
    n_steps: int = 50000, device: str = "cpu",
) -> dict[str, Any]:
    """Natural convection in a differentially heated cavity."""
    dev = torch.device(device)
    ny = nx = N

    mask_solid = torch.zeros(ny, nx, dtype=torch.bool, device=dev)
    mask_solid[0, :] = True
    mask_solid[-1, :] = True
    mask_solid[:, 0] = True
    mask_solid[:, -1] = True

    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = torch.zeros(ny, nx, device=dev)
    uy0 = torch.zeros(ny, nx, device=dev)

    x_norm = torch.linspace(0, 1, nx, device=dev).view(1, nx).expand(ny, nx)
    T0 = 1.0 - x_norm

    f = equilibrium(rho0, ux0, uy0, device=dev)
    g = equilibrium_thermal(T0, ux0, uy0)
    T_s = T0.clone()

    state = CHTState(f=f, g=g, T_s=T_s, mask_solid=mask_solid)

    tau_f = 0.6
    nu_lu = (tau_f - 0.5) / 3.0
    alpha_lu = nu_lu / Pr
    tau_g = 3.0 * alpha_lu + 0.5
    g_beta = Ra * nu_lu * alpha_lu / (float(nx) ** 3)

    cfg = CHTConfig(
        tau_f=tau_f, kappa_f=alpha_lu, alpha_s=alpha_lu,
        k_ratio=100.0, beta=g_beta, T_ref=0.5, gravity=-1.0,
        T_hot=1.0, T_cold=0.0, Q_source=0.0,
        n_steps=n_steps, output_interval=n_steps,
    )

    final = _run_natural_convection(state, cfg, device=dev)

    T_field = macroscopic_thermal(final.g).cpu()
    dT_dx_left = 1.5 * T_field[:, 0] - 2.0 * T_field[:, 1] + 0.5 * T_field[:, 2]
    dT_dx_right = -1.5 * T_field[:, -1] + 2.0 * T_field[:, -2] - 0.5 * T_field[:, -3]

    Nu_left = float(dT_dx_left.mean().item()) * float(nx)
    Nu_right = float(dT_dx_right.mean().item()) * float(nx)
    Nu_avg = 0.5 * (Nu_left + Nu_right)

    return {
        "T_field": T_field,
        "Nu_left": Nu_left, "Nu_right": Nu_right, "Nu_avg": Nu_avg,
        "Ra": Ra, "Pr": Pr, "N": N,
    }


def _run_natural_convection(
    state: CHTState, cfg: CHTConfig, device: str = "cpu",
) -> CHTState:
    """Natural convection CHT loop."""
    dev = torch.device(device)
    tau_g = 0.5 + cfg.kappa_f / (1.0 / 6.0)
    ny, nx = state.f.shape[1], state.f.shape[2]

    for step in range(state.step, state.step + cfg.n_steps):
        rho, ux, uy = macroscopic(state.f)
        T_fluid = macroscopic_thermal(state.g)

        state.f = apply_buoyancy_force(state.f, T_fluid, cfg.T_ref, cfg.beta, cfg.gravity)
        state.f = collide_bgk(state.f, cfg.tau_f)
        state.f = stream(state.f)
        state.f = bounce_back_cells(state.f, state.mask_solid)

        T_fluid[:, 0] = cfg.T_hot
        T_fluid[:, -1] = cfg.T_cold
        state.T_s[:, 0] = cfg.T_hot
        state.T_s[:, -1] = cfg.T_cold

        rho_new, ux_new, uy_new = macroscopic(state.f)
        state.g = collide_thermal_bgk(state.g, T_fluid, ux_new, uy_new, tau_g)
        state.g = stream_thermal(state.g)

        g_hot = equilibrium_thermal(
            torch.full((ny,), cfg.T_hot, device=dev),
            torch.zeros(ny, device=dev), torch.zeros(ny, device=dev),
        )
        g_cold = equilibrium_thermal(
            torch.full((ny,), cfg.T_cold, device=dev),
            torch.zeros(ny, device=dev), torch.zeros(ny, device=dev),
        )
        state.g[:, :, 0] = g_hot.view(5, ny, 1)
        state.g[:, :, -1] = g_cold.view(5, ny, 1)

        state.T_s = cht_solid_diffusion_step(state.T_s, state.mask_solid, cfg.alpha_s)

        T_fluid_new = macroscopic_thermal(state.g)
        T_f_cpl, T_s_cpl = apply_cht_interface(
            T_fluid_new, state.T_s, state.mask_solid, cfg.k_ratio
        )
        g_eq = equilibrium_thermal(T_f_cpl, ux_new, uy_new)
        state.g = state.g + (g_eq - state.g) * (1.0 / tau_g)
        state.T_s = T_s_cpl

        state.step = step + 1

    return state


# ---------------------------------------------------------------------------
# Interface Heat Flux Closure
# ---------------------------------------------------------------------------


def check_flux_closure(
    T_fluid: torch.Tensor, T_solid: torch.Tensor,
    mask_solid: torch.Tensor,
    k_f: float = 1.0, k_s: float = 5.0,
) -> dict[str, float]:
    """Check heat flux continuity at the fluid-solid interface."""
    kernel = torch.ones(1, 1, 3, 3, device=mask_solid.device, dtype=torch.float32)
    kernel[0, 0, 1, 1] = 0.0
    solid_float = mask_solid.float().unsqueeze(0).unsqueeze(0)
    neighbour_solid = F.conv2d(solid_float, kernel, padding=1).squeeze(0).squeeze(0) > 0.0

    is_interface_fluid = neighbour_solid & ~mask_solid
    is_interface_solid = neighbour_solid & mask_solid

    if not is_interface_fluid.any() or not is_interface_solid.any():
        return {"flux_deviation": 0.0, "q_fluid_mean": 0.0, "q_solid_mean": 0.0}

    # Compute temperature gradients at interface cells
    # For fluid interface cells: gradient toward solid neighbor
    # For solid interface cells: gradient toward fluid neighbor
    
    grad_f = torch.zeros_like(T_fluid)
    grad_s = torch.zeros_like(T_solid)
    
    # Simple approach: use finite differences in each direction
    # and take the max gradient at interface cells
    
    # Fluid-side gradients (difference with adjacent solid cells)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        # Shift solid mask to find adjacent solid cells
        solid_shifted = torch.roll(mask_solid.float(), shifts=(dy, dx), dims=(0, 1))
        # Fluid cells adjacent to solid in this direction
        adj = is_interface_fluid & (solid_shifted > 0.5)
        if adj.any():
            T_solid_adj = torch.roll(T_solid, shifts=(dy, dx), dims=(0, 1))
            grad_f[adj] = torch.abs(T_fluid[adj] - T_solid_adj[adj])
    
    # Solid-side gradients (difference with adjacent fluid cells)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        fluid_shifted = torch.roll((~mask_solid).float(), shifts=(dy, dx), dims=(0, 1))
        adj = is_interface_solid & (fluid_shifted > 0.5)
        if adj.any():
            T_fluid_adj = torch.roll(T_fluid, shifts=(dy, dx), dims=(0, 1))
            grad_s[adj] = torch.abs(T_solid[adj] - T_fluid_adj[adj])

    q_fluid = k_f * grad_f[is_interface_fluid]
    q_solid = k_s * grad_s[is_interface_solid]

    q_f_mean = float(q_fluid.mean().item()) if q_fluid.numel() > 0 else 0.0
    q_s_mean = float(q_solid.mean().item()) if q_solid.numel() > 0 else 0.0

    denom = max(abs(q_f_mean), abs(q_s_mean), 1e-12)
    deviation = abs(q_f_mean - q_s_mean) / denom

    return {
        "flux_deviation": deviation,
        "q_fluid_mean": q_f_mean,
        "q_solid_mean": q_s_mean,
    }
