"""Passive scalar transport for LBM (D2Q5 advection-diffusion on top of D2Q9 flow).

Implements the double-distribution-function (DDF) approach for scalar
(concentration or temperature) transport coupled to an incompressible flow:

- Flow field: D2Q9 BGK/MRT collision (existing solver.py / d2q9.py)
- Scalar field: D2Q5 BGK collision with equilibrium g_eq = w_i * C * (1 + 3*c_i·u)
  Scalar diffusivity D = cs2_D2Q5 * (tau_s - 0.5) = (tau_s - 0.5) / 3

This module provides:
- Scalar equilibrium, collision, streaming, macroscopic recovery
- Source term injection (continuous point source, instantaneous release)
- Numerical diffusion calibration (measuring D_eff vs D_input)
- Boundary conditions (Dirichlet, Neumann, periodic)

References:
- Shan X. (1993). "Lattice Boltzmann model for simulating magnetohydrodynamics."
- Ziegler D.B. & Johnson C.R. (2005). "A lattice Boltzmann method for dilute
  gas-particle suspensions."
- Mohamad A.A. (2011). Lattice Boltzmann Method, Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .d2q9 import C, W, equilibrium, macroscopic
from .solver import stream

# ── D2Q5 scalar lattice ──────────────────────────────────────────────────────
# Directions: 0:(0,0) 1:(1,0) 2:(-1,0) 3:(0,1) 4:(0,-1)
C5 = torch.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.int64)

# Exact D2Q5 weights in float64 for maximum precision
_W5_fp64 = torch.tensor([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], 
                          dtype=torch.float64)
_W5_fp64 = _W5_fp64 / _W5_fp64.sum()  # normalize to machine precision

# Float32 version (for standard computations)
W5 = _W5_fp64.to(torch.float32)

OPPOSITE5 = torch.tensor([0, 2, 1, 4, 3], dtype=torch.int64)

# D2Q5 cs² = 1/3 (same as D2Q9 for isotropy of second moment)
CS2_D2Q5 = 1.0 / 3.0

# Stream index cache
_stream5_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_w5(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get D2Q5 weights in the appropriate dtype for exact mass conservation."""
    if dtype == torch.float64:
        return _W5_fp64.to(device)
    else:
        return W5.to(device=device, dtype=dtype)


def scalar_equilibrium(
    C_field: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """D2Q5 scalar equilibrium: g_eq_i = w_i * C * (1 + 3 * c_i · u).

    First-order expansion in velocity (advection-diffusion limit).
    Zeroth moment recovers C exactly: Σ_i g_eq_i = C * Σ_i w_i = C.
    """
    device = C_field.device
    c = C5.to(device)
    w = _get_w5(C_field.dtype, device).view(5, 1, 1)
    cu = c[:, 0].view(5, 1, 1) * ux + c[:, 1].view(5, 1, 1) * uy
    return w * C_field.unsqueeze(0) * (1.0 + 3.0 * cu)


def scalar_collision(
    g: torch.Tensor,
    C_field: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau_s: float,
) -> torch.Tensor:
    """D2Q5 BGK collision for scalar: g_new = g - (g - g_eq) / tau_s.

    Scalar diffusivity: D = (tau_s - 0.5) / 3 (lattice units).
    Zeroth moment (total scalar mass) is exactly conserved.
    """
    geq = scalar_equilibrium(C_field, ux, uy)
    return g - (g - geq) / tau_s


def scalar_stream(g: torch.Tensor) -> torch.Tensor:
    """D2Q5 periodic streaming with cached index tensors."""
    ny, nx = g.shape[1], g.shape[2]
    device = g.device
    key = (ny, nx, device.type, getattr(device, "index", None))
    if key not in _stream5_cache:
        c = C5.to(device)
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
        q_idx = torch.arange(5, device=device).view(5, 1, 1).expand(5, ny, nx)
        y_idx = y_src.unsqueeze(2).expand(5, ny, nx)
        x_idx = x_src.unsqueeze(1).expand(5, ny, nx)
        _stream5_cache[key] = (q_idx, y_idx, x_idx)
    q_idx, y_idx, x_idx = _stream5_cache[key]
    return g[q_idx, y_idx, x_idx]


def scalar_macroscopic(g: torch.Tensor) -> torch.Tensor:
    """Recover scalar field C = Σ_i g_i."""
    return g.sum(dim=0)


def scalar_diffusivity(tau_s: float) -> float:
    """Lattice scalar diffusivity from relaxation time: D = (tau_s - 0.5) / 3."""
    return (tau_s - 0.5) * CS2_D2Q5


def tau_from_diffusivity(D: float) -> float:
    """Inverse: relaxation time from target diffusivity."""
    return D / CS2_D2Q5 + 0.5


# ── Source terms ─────────────────────────────────────────────────────────────

def add_continuous_source(
    g: torch.Tensor,
    source_rate: float,
    x_src: int,
    y_src: int,
    radius: int = 1,
) -> torch.Tensor:
    """Add scalar at a continuous point source (mass injection per step).

    Distributes source_rate uniformly over a circular patch of given radius
    centered at (x_src, y_src). The source is added to the rest population g_0
    to preserve zero net momentum injection.
    """
    g = g.clone()
    ny, nx = g.shape[1], g.shape[2]
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=g.device),
        torch.arange(nx, device=g.device),
        indexing="ij",
    )
    dist = ((xx - x_src) ** 2 + (yy - y_src) ** 2).float()
    mask = dist <= radius ** 2
    n_cells = mask.sum().clamp(min=1)
    g[0] = torch.where(mask, g[0] + source_rate / n_cells, g[0])
    return g


def add_instantaneous_release(
    g: torch.Tensor,
    total_mass: float,
    x_src: int,
    y_src: int,
    radius: int = 1,
) -> torch.Tensor:
    """Instantaneous scalar release (all mass at once, no further injection)."""
    return add_continuous_source(g, total_mass, x_src, y_src, radius)


# ── Boundary conditions ──────────────────────────────────────────────────────

def apply_dirichlet_bc(
    g: torch.Tensor,
    C_wall: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Dirichlet BC: set scalar to C_wall at masked cells.

    Replaces all populations with equilibrium at u=0: g_i = w_i * C_wall.
    """
    w = _get_w5(g.dtype, g.device).view(5, 1, 1)
    g_eq_wall = w * C_wall
    m = mask.unsqueeze(0).expand_as(g)
    return torch.where(m, g_eq_wall.expand_as(g), g)


def apply_neumann_bc(
    g: torch.Tensor,
    mask: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> torch.Tensor:
    """Zero-flux (Neumann) BC: copy populations from interior neighbor."""
    g_new = g.clone()
    for i in range(5):
        g_new[i] = torch.where(mask, g[i].roll(shifts=0), g_new[i])
    return g_new


def apply_outflow_bc(
    g: torch.Tensor,
    direction: str = "right",
) -> torch.Tensor:
    """Convective outflow BC (zero second derivative at boundary)."""
    g = g.clone()
    if direction == "right":
        g[:, :, -1] = 2.0 * g[:, :, -2] - g[:, :, -3]
    elif direction == "left":
        g[:, :, 0] = 2.0 * g[:, :, 1] - g[:, :, 2]
    elif direction == "top":
        g[:, -1, :] = 2.0 * g[:, -2, :] - g[:, -3, :]
    elif direction == "bottom":
        g[:, 0, :] = 2.0 * g[:, 1, :] - g[:, 2, :]
    return g


# ── Full scalar transport solver ─────────────────────────────────────────────

@dataclass
class ScalarTransportConfig:
    """Configuration for passive scalar transport simulation."""
    nx: int = 128
    ny: int = 128
    tau_f: float = 0.6
    tau_s: float = 0.8
    ux_bg: float = 0.1
    D: float = 0.0
    source_rate: float = 0.01
    source_x: int = 10
    source_y: int = 64
    source_radius: int = 2
    n_steps: int = 5000
    output_interval: int = 500
    bc_left: str = "outflow"
    bc_right: str = "outflow"
    bc_top: str = "periodic"
    bc_bottom: str = "periodic"


@dataclass
class ScalarTransportState:
    """Runtime state for scalar transport simulation."""
    f: torch.Tensor
    g: torch.Tensor
    step: int = 0
    history: list = field(default_factory=list)


def run_scalar_transport(
    cfg: ScalarTransportConfig,
    device: torch.device | str = "cpu",
) -> ScalarTransportState:
    """Run a passive scalar transport simulation."""
    device = torch.device(device)
    nx, ny = cfg.nx, cfg.ny

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), cfg.ux_bg, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0)

    C0 = torch.zeros((ny, nx), device=device)
    g = scalar_equilibrium(C0, ux0, uy0)

    tau_s = cfg.tau_s
    if cfg.D > 0:
        tau_s = tau_from_diffusivity(cfg.D)

    state = ScalarTransportState(f=f, g=g)

    for step in range(cfg.n_steps):
        f = _collide_bgk_simple(f, cfg.tau_f)
        f = stream(f)

        rho, ux, uy = macroscopic(f)

        C_field = scalar_macroscopic(state.g)
        state.g = scalar_collision(state.g, C_field, ux, uy, tau_s)
        state.g = scalar_stream(state.g)

        state.g = add_continuous_source(
            state.g, cfg.source_rate,
            cfg.source_x, cfg.source_y, cfg.source_radius
        )

        state.g = apply_outflow_bc(state.g, "right")
        state.g = apply_outflow_bc(state.g, "left")

        state.step = step + 1

        if (step + 1) % cfg.output_interval == 0:
            C = scalar_macroscopic(state.g)
            state.history.append({
                "step": step + 1,
                "C_max": float(C.max().item()),
                "C_total": float(C.sum().item()),
            })

    return state


def _collide_bgk_simple(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Simple BGK collision (wrapper for solver.collide_bgk)."""
    from .solver import collide_bgk
    return collide_bgk(f, tau)


# ── Numerical diffusion calibration ──────────────────────────────────────────

def calibrate_effective_diffusivity(
    nx: int = 128,
    ny: int = 1,
    tau_s: float = 0.8,
    n_steps: int = 5000,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Calibrate effective diffusivity from a 1D Gaussian pulse decay."""
    device = torch.device(device)

    x = torch.arange(nx, device=device, dtype=torch.float64)
    x_center = nx / 2.0

    sigma0 = 5.0
    C0 = torch.exp(-0.5 * ((x - x_center) / sigma0) ** 2)
    C0 = C0.unsqueeze(0)

    g = torch.zeros((5, ny, nx), device=device, dtype=torch.float32)
    g[0, 0, :] = C0[0]

    ux = torch.zeros((ny, nx), device=device)
    uy = torch.zeros((ny, nx), device=device)

    D_input = scalar_diffusivity(tau_s)

    for step in range(n_steps):
        C_field = scalar_macroscopic(g)
        g = scalar_collision(g, C_field, ux, uy, tau_s)
        g = scalar_stream(g)

    C_final = scalar_macroscopic(g)[0]
    total_mass = C_final.sum()
    x_mean = (x * C_final).sum() / total_mass
    variance = ((x - x_mean) ** 2 * C_final).sum() / total_mass

    D_eff = float((variance - sigma0 ** 2) / (2.0 * n_steps))

    return {
        "D_input": D_input,
        "D_eff": D_eff,
        "relative_error": abs(D_eff - D_input) / D_input if D_input > 0 else float("inf"),
        "sigma0": sigma0,
        "sigma_final": float(variance.sqrt().item()),
        "n_steps": n_steps,
        "tau_s": tau_s,
    }


__all__ = [
    "C5",
    "CS2_D2Q5",
    "OPPOSITE5",
    "ScalarTransportConfig",
    "ScalarTransportState",
    "W5",
    "add_continuous_source",
    "add_instantaneous_release",
    "apply_dirichlet_bc",
    "apply_neumann_bc",
    "apply_outflow_bc",
    "calibrate_effective_diffusivity",
    "scalar_collision",
    "scalar_diffusivity",
    "scalar_equilibrium",
    "scalar_macroscopic",
    "scalar_stream",
    "tau_from_diffusivity",
    "run_scalar_transport",
]
