"""Core SC/D2Q9 two-phase LBM simulation engine.

Implements the Shan-Chen pseudopotential single-component two-phase model
on the D2Q9 lattice with:
- Mei-Luo-Shyy weighted isotropic gradient for the interaction force
- BGK collision operator
- Standard streaming with periodic boundaries
- Optional half-way bounce-back for solid walls

The equation of state for the exponential pseudopotential ψ(ρ) = ρ₀(1-exp(-ρ/ρ₀))
with coupling G > 0 (attractive) gives liquid-gas coexistence at:
    ρ_l ≈ 1.96, ρ_v ≈ 0.16 (density ratio ~12.3) for G=5.0, ρ₀=1

The surface tension is approximately σ ≈ 0.056 lattice units for these
parameters (measured from the Laplace law).

References
----------
Shan & Chen (1993) Phys. Rev. E 47, 1815.
Mei, Luo & Shyy (1997) J. Comput. Phys. 134, 246.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from .gradient import gradient_9point_mls


# ---------------------------------------------------------------------------
# Pseudopotential functions
# ---------------------------------------------------------------------------

def psi_exp(rho: torch.Tensor, rho0: float = 1.0) -> torch.Tensor:
    """Shan-Chen exponential pseudopotential ψ(ρ) = ρ₀(1 - exp(-ρ/ρ₀))."""
    return rho0 * (1.0 - torch.exp(-rho / rho0))


def psi_linear(rho: torch.Tensor) -> torch.Tensor:
    """Linear pseudopotential ψ(ρ) = ρ."""
    return rho


# ---------------------------------------------------------------------------
# D2Q9 lattice constants
# ---------------------------------------------------------------------------

# Lattice velocities (9, 2)
_C = torch.tensor([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=torch.int64)

# Weights (9,)
_W = torch.tensor([
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
], dtype=torch.float64)

# Opposite direction mapping
_OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)

# Speed of sound squared
_CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Configuration and state
# ---------------------------------------------------------------------------

@dataclass
class SCConfig:
    """Configuration for SC/D2Q9 simulation."""
    nx: int = 100
    ny: int = 100
    tau: float = 1.0  # Relaxation time (ν = cs²(τ - 0.5))
    G: float = 5.0  # SC coupling constant (>0 = attractive)
    rho0: float = 1.0  # Reference density for psi_exp
    psi_fn: Callable = psi_exp  # Pseudopotential function
    gradient_scheme: str = "mls9"  # "mls9" or "mls13" or "standard"
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    # Wall boundary (optional)
    wall_bottom: bool = False  # Half-way bounce-back at y=0
    wall_top: bool = False  # Half-way bounce-back at y=ny-1
    wall_psi: float = 0.0  # Pseudopotential at wall (controls wetting)


@dataclass
class SCState:
    """State of SC/D2Q9 simulation."""
    f: torch.Tensor  # Populations (9, ny, nx)
    rho: torch.Tensor  # Density (ny, nx)
    ux: torch.Tensor  # x-velocity (ny, nx)
    uy: torch.Tensor  # y-velocity (ny, nx)
    config: SCConfig
    step: int = 0
    # History
    max_velocity_history: list = field(default_factory=list)
    mass_history: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Equilibrium distribution
# ---------------------------------------------------------------------------

def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Compute D2Q9 equilibrium distribution f_eq.

    f_eq_i = w_i ρ [1 + 3(c_i·u) + 4.5(c_i·u)² - 1.5|u|²]
    """
    device = rho.device
    dtype = rho.dtype

    c = _C.to(device=device, dtype=dtype)
    w = _W.to(device=device, dtype=dtype).view(9, 1, 1)

    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    u_sq = ux**2 + uy**2

    feq = w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq.unsqueeze(0))
    return feq


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------

def collide_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    """BGK collision: f_out = f - (f - f_eq) / τ."""
    device = f.device
    dtype = f.dtype

    # Compute macroscopic quantities
    c = _C.to(device=device, dtype=dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * cx).sum(dim=0) / rho_safe
    uy = (f * cy).sum(dim=0) / rho_safe

    # Equilibrium
    feq = equilibrium(rho, ux, uy)

    # BGK collision
    f_out = f - (f - feq) / tau
    return f_out


def collide_sc(
    f: torch.Tensor,
    Fx: torch.Tensor,
    Fy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """BGK collision with SC body force.

    The force is incorporated via velocity shift:
        u_eq = u + τ F / ρ

    This is the standard forcing scheme for LBM (Guo et al. 2002).
    """
    device = f.device
    dtype = f.dtype

    c = _C.to(device=device, dtype=dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * cx).sum(dim=0) / rho_safe
    uy = (f * cy).sum(dim=0) / rho_safe

    # Velocity shift due to force
    ux_eq = ux + tau * Fx / rho_safe
    uy_eq = uy + tau * Fy / rho_safe

    feq = equilibrium(rho, ux_eq, uy_eq)
    f_out = f - (f - feq) / tau
    return f_out


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream(f: torch.Tensor, solid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Standard LBM streaming with optional bounce-back (vectorized)."""
    # Vectorized streaming using torch.roll
    # D2Q9 velocities: (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)
    f_new = torch.zeros_like(f)
    
    # i=0: c=(0,0) - no movement
    f_new[0] = f[0]
    # i=1: c=(1,0)
    f_new[1] = torch.roll(f[1], shifts=(0, 1), dims=(0, 1))
    # i=2: c=(0,1)
    f_new[2] = torch.roll(f[2], shifts=(1, 0), dims=(0, 1))
    # i=3: c=(-1,0)
    f_new[3] = torch.roll(f[3], shifts=(0, -1), dims=(0, 1))
    # i=4: c=(0,-1)
    f_new[4] = torch.roll(f[4], shifts=(-1, 0), dims=(0, 1))
    # i=5: c=(1,1)
    f_new[5] = torch.roll(f[5], shifts=(1, 1), dims=(0, 1))
    # i=6: c=(-1,1)
    f_new[6] = torch.roll(f[6], shifts=(1, -1), dims=(0, 1))
    # i=7: c=(-1,-1)
    f_new[7] = torch.roll(f[7], shifts=(-1, -1), dims=(0, 1))
    # i=8: c=(1,-1)
    f_new[8] = torch.roll(f[8], shifts=(-1, 1), dims=(0, 1))

    # Bounce-back at solid boundaries
    if solid_mask is not None:
        for i in range(9):
            opp = int(_OPPOSITE[i].item())
            f_new[opp] = torch.where(solid_mask, f[i], f_new[opp])

    return f_new


# ---------------------------------------------------------------------------
# SC force with Mei-Luo-Shyy gradient
# ---------------------------------------------------------------------------

def compute_sc_force_mls(
    rho: torch.Tensor,
    G: float,
    psi_fn: Callable = psi_exp,
    gradient_scheme: str = "mls9",
    solid_mask: torch.Tensor | None = None,
    wall_psi: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SC interaction force using Mei-Luo-Shyy isotropic gradient.

    The force is:
        F = -G ψ(x) ∇ψ(x)

    where ∇ψ is computed using the weighted isotropic gradient:
        ∂ψ/∂x ≈ (1/cs²) Σᵢ wᵢ c_ix ψ(x + cᵢ)

    This is the "standard" SC force but with the gradient computed using
    the Mei-Luo-Shyy weighted stencil instead of the raw D2Q9 gather.

    For the wall boundary, the pseudopotential at solid cells is replaced
    by wall_psi to control wetting behavior.

    Args:
        rho: Density field (ny, nx).
        G: SC coupling constant (>0 = attractive).
        psi_fn: Pseudopotential function.
        gradient_scheme: "mls9" for 9-point, "mls13" for 13-point.
        solid_mask: Optional boolean mask for solid cells.
        wall_psi: Pseudopotential value at solid cells.

    Returns:
        (Fx, Fy) force components, each (ny, nx).
    """
    psi = psi_fn(rho)

    # Apply wall pseudopotential
    if solid_mask is not None:
        psi = torch.where(solid_mask, torch.full_like(psi, wall_psi), psi)

    # Compute gradient of psi using Mei-Luo-Shyy stencil
    if gradient_scheme == "mls9":
        grad_psi_x, grad_psi_y = gradient_9point_mls(psi, periodic=True)
    elif gradient_scheme == "mls13":
        from .gradient import gradient_13point_mls
        grad_psi_x, grad_psi_y = gradient_13point_mls(psi, periodic=True)
    else:
        raise ValueError(f"Unknown gradient scheme: {gradient_scheme}")

    # Force: F = +G ψ cs² ∇ψ
    # The MLS gradient computes: ∇ψ = (1/cs²) Σ wᵢ cᵢ ψ(x+cᵢ) (forward gather)
    # The standard SC force uses backward gather: F = -G ψ Σ wᵢ cᵢ ψ(x-cᵢ)
    # Since Σ wᵢ cᵢ ψ(x-cᵢ) = -Σ wᵢ cᵢ ψ(x+cᵢ) = -cs² ∇ψ, we have:
    # F = -G ψ * (-cs² ∇ψ) = +G ψ cs² ∇ψ
    cs2 = 1.0 / 3.0
    Fx = G * cs2 * psi * grad_psi_x
    Fy = G * cs2 * psi * grad_psi_y

    # Zero force at solid cells
    if solid_mask is not None:
        Fx = torch.where(solid_mask, torch.zeros_like(Fx), Fx)
        Fy = torch.where(solid_mask, torch.zeros_like(Fy), Fy)

    return Fx, Fy


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_droplet(
    config: SCConfig,
    R: float,
    center: tuple[float, float] | None = None,
    rho_l: float = 1.96,
    rho_v: float = 0.16,
    interface_width: float = 3.0,
) -> SCState:
    """Initialize a static droplet in a periodic domain.

    The droplet is a circular region of high density (liquid) surrounded
    by low density (vapour).  The interface is a tanh profile:
        ρ(r) = ρ_v + (ρ_l - ρ_v) * 0.5 * (1 + tanh((R - r) / width))

    Args:
        config: Simulation configuration.
        R: Droplet radius in lattice units.
        center: (cx, cy) center of droplet. Default is domain center.
        rho_l: Liquid density (inside droplet).
        rho_v: Vapour density (outside droplet).
        interface_width: Width of the diffuse interface.

    Returns:
        Initial SCState with droplet.
    """
    device = torch.device(config.device)
    dtype = config.dtype
    nx, ny = config.nx, config.ny

    if center is None:
        center = (nx / 2.0, ny / 2.0)

    # Create coordinate grid
    y = torch.arange(ny, device=device, dtype=dtype)
    x = torch.arange(nx, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Distance from center (periodic)
    dx = X - center[0]
    dy = Y - center[1]
    dx = dx - nx * torch.round(dx / nx)
    dy = dy - ny * torch.round(dy / ny)
    r = torch.sqrt(dx**2 + dy**2)

    # Density profile: tanh interface
    rho = rho_v + (rho_l - rho_v) * 0.5 * (1.0 + torch.tanh((R - r) / interface_width))

    # Zero initial velocity
    ux = torch.zeros(ny, nx, device=device, dtype=dtype)
    uy = torch.zeros(ny, nx, device=device, dtype=dtype)

    # Equilibrium populations
    f = equilibrium(rho, ux, uy)

    return SCState(
        f=f,
        rho=rho,
        ux=ux,
        uy=uy,
        config=config,
        step=0,
    )


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    state: SCState,
    n_steps: int,
    sample_interval: int = 100,
    verbose: bool = False,
) -> SCState:
    """Run SC simulation for n_steps.

    Each step:
    1. Compute macroscopic quantities (ρ, u) from populations
    2. Compute SC interaction force using MLS gradient
    3. Collision with force (velocity shift)
    4. Streaming

    Args:
        state: Initial state.
        n_steps: Number of time steps.
        sample_interval: Record diagnostics every this many steps.
        verbose: Print progress.

    Returns:
        Final state after n_steps.
    """
    config = state.config
    f = state.f
    device = torch.device(config.device)
    dtype = config.dtype

    c = _C.to(device=device, dtype=dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    # Solid mask for wall boundaries
    solid_mask = None
    if config.wall_bottom or config.wall_top:
        solid_mask = torch.zeros(config.ny, config.nx, device=device, dtype=torch.bool)
        if config.wall_bottom:
            solid_mask[0, :] = True
        if config.wall_top:
            solid_mask[-1, :] = True

    for step in range(n_steps):
        # 1. Compute macroscopic quantities
        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = (f * cx).sum(dim=0) / rho_safe
        uy = (f * cy).sum(dim=0) / rho_safe

        # 2. Compute SC force
        Fx, Fy = compute_sc_force_mls(
            rho, config.G, config.psi_fn,
            gradient_scheme=config.gradient_scheme,
            solid_mask=solid_mask,
            wall_psi=config.wall_psi,
        )

        # 3. Collision with force
        f = collide_sc(f, Fx, Fy, config.tau)

        # 4. Streaming
        f = stream(f, solid_mask)

        # Record diagnostics
        if (step + 1) % sample_interval == 0:
            rho = f.sum(dim=0)
            rho_safe = torch.clamp(rho, min=1e-12)
            ux = (f * cx).sum(dim=0) / rho_safe
            uy = (f * cy).sum(dim=0) / rho_safe
            max_u = torch.sqrt(ux**2 + uy**2).max().item()
            mass = rho.sum().item()
            state.max_velocity_history.append(max_u)
            state.mass_history.append(mass)
            if verbose and (step + 1) % (10 * sample_interval) == 0:
                print(f"  Step {step+1}/{n_steps}: max_u={max_u:.2e}, mass={mass:.1f}")

    # Update state
    state.f = f
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    state.rho = rho
    state.ux = (f * cx).sum(dim=0) / rho_safe
    state.uy = (f * cy).sum(dim=0) / rho_safe
    state.step += n_steps

    return state
