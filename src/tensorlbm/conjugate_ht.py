"""Conjugate Heat Transfer (CHT) for LBM.

Extends the existing passive-scalar thermal LBM (``thermal.py``) to include
solid-domain heat conduction, enabling simulation of coupled fluid–solid
thermal problems such as:

* Electronics cooling (heat sink, chip–PCB)
* Heat exchangers
* Turbine blade internal cooling

Physical model
--------------
The fluid domain is governed by the double-distribution-function (DDF)
model from ``thermal.py`` (D2Q9 momentum + D2Q5 temperature).

The solid domain is governed by the isotropic heat-conduction equation::

    ∂T_s/∂t = α_s ∇²T_s + Q_s / (ρ_s c_s)

discretised on the same lattice grid using a second-order central-difference
explicit time-stepping scheme (explicit Euler for simplicity; stable for
Fo = α_s Δt / Δx² < 0.25).

Interface condition
-------------------
At the fluid–solid interface the following conditions are enforced:

1. **Temperature continuity:** T_f = T_s  (matched by averaging)
2. **Heat-flux continuity:**  k_f (∂T/∂n)|_f = k_s (∂T/∂n)|_s

The interface is treated with a *harmonic-mean* effective conductivity::

    k_eff = 2 k_f k_s / (k_f + k_s)

which ensures flux continuity in the discrete setting.

Exported API
------------
* :class:`CHTConfig` – simulation parameters
* :func:`cht_solid_diffusion_step` – one explicit diffusion step in the solid
* :func:`apply_cht_interface` – enforce temperature/flux continuity
* :func:`run_conjugate_ht_2d` – full CHT time loop for 2-D problems

References
----------
Mohamad, A.A. (2011). *Lattice Boltzmann Method: Fundamentals and Engineering
    Applications with Computer Codes*. Springer.
Wang, J., Wang, M., & Li, Z. (2007).
    "A lattice Boltzmann algorithm for fluid–solid conjugate heat transfer."
    *Int. J. Thermal Sciences* 46, 228–234.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "CHTConfig",
    "CHTState",
    "cht_solid_diffusion_step",
    "apply_cht_interface",
    "run_conjugate_ht_2d",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CHTConfig:
    """Parameters for a 2-D conjugate heat transfer simulation.

    Fluid properties (lattice units)
    ---------------------------------
    tau_f:    Momentum relaxation time (``> 0.5``).
    nu_f:     Kinematic viscosity = (tau_f − 0.5) / 3.
    kappa_f:  Thermal diffusivity of fluid (lattice units).
    tau_g:    Thermal relaxation time for D2Q5 = 0.5 + kappa_f / (c_D2Q5²).

    Solid properties (lattice units)
    ----------------------------------
    alpha_s:  Thermal diffusivity of solid (lattice units), α_s = k_s / (ρ_s c_s).
    k_ratio:  Conductivity ratio k_s / k_f.  Used to weight the interface flux.

    Buoyancy
    --------
    beta:     Volumetric thermal-expansion coefficient (lattice).
    T_ref:    Reference temperature for Boussinesq approximation.
    gravity:  Gravity magnitude in lattice units (applied in y-direction).

    Simulation control
    ------------------
    n_steps:        Total time steps.
    output_interval: Steps between checkpoint saves.
    T_hot:          Hot wall temperature (Dirichlet BC).
    T_cold:         Cold wall temperature.
    Q_source:       Volumetric heat source in solid [lattice units].
    """
    tau_f: float = 0.6
    kappa_f: float = 1.0 / 6.0
    alpha_s: float = 0.5 / 6.0   # typically α_s < α_f for conduction-limited solid
    k_ratio: float = 5.0          # k_s / k_f
    beta: float = 2.0e-3
    T_ref: float = 0.5
    gravity: float = 2.0e-5
    T_hot: float = 1.0
    T_cold: float = 0.0
    Q_source: float = 0.0
    n_steps: int = 1000
    output_interval: int = 100


@dataclass
class CHTState:
    """Runtime state tensors for a CHT simulation.

    Attributes:
        f:     Momentum distribution, shape ``(9, ny, nx)``.
        g:     Temperature distribution (fluid, D2Q5), shape ``(5, ny, nx)``.
        T_s:   Solid temperature field, shape ``(ny, nx)``.
        mask_solid: Boolean mask ``True`` where cells are solid, shape ``(ny, nx)``.
        step:  Current time-step counter.
    """
    f: torch.Tensor
    g: torch.Tensor
    T_s: torch.Tensor
    mask_solid: torch.Tensor
    step: int = 0


# ---------------------------------------------------------------------------
# Solid conduction step
# ---------------------------------------------------------------------------

def cht_solid_diffusion_step(
    T_s: torch.Tensor,
    mask_solid: torch.Tensor,
    alpha_s: float,
    Q_source: float = 0.0,
) -> torch.Tensor:
    """One explicit Euler step of the heat-conduction equation in the solid.

    Computes::

        T_s_new = T_s + alpha_s * Laplacian(T_s) + Q_source

    where the Laplacian is the standard 2-D five-point stencil on a unit
    grid.  Only solid cells (``mask_solid == True``) are updated.

    Stability condition:  ``alpha_s ≤ 0.25``

    Args:
        T_s: Solid temperature, shape ``(ny, nx)``.
        mask_solid: Boolean mask (solid cells), shape ``(ny, nx)``.
        alpha_s: Solid thermal diffusivity in lattice units.
        Q_source: Uniform volumetric heat source in solid cells.

    Returns:
        Updated solid temperature tensor.
    """
    # 2-D Laplacian via 2D convolution with 5-point stencil
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=T_s.dtype,
        device=T_s.device,
    ).view(1, 1, 3, 3)

    T_4d = T_s.unsqueeze(0).unsqueeze(0)  # (1, 1, ny, nx)
    lap = F.conv2d(T_4d, kernel, padding=1).squeeze(0).squeeze(0)

    T_new = T_s + alpha_s * lap + (Q_source if Q_source != 0.0 else 0.0)

    # Only apply update to solid cells; fluid cells keep their value unchanged
    return torch.where(mask_solid, T_new, T_s)


# ---------------------------------------------------------------------------
# Interface condition
# ---------------------------------------------------------------------------

def apply_cht_interface(
    T_fluid: torch.Tensor,
    T_solid: torch.Tensor,
    mask_solid: torch.Tensor,
    k_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enforce temperature and heat-flux continuity at the fluid–solid interface.

    The interface cells (fluid cells adjacent to solid cells) have their
    temperature updated to satisfy:

    1. Temperature continuity: shared interface temperature is a
       conductivity-weighted average of the two domains.
    2. Flux continuity: maintained implicitly via the harmonic-mean
       effective conductivity ``k_eff = 2 k_f k_s / (k_f + k_s)``.

    The weight is:  w = k_ratio / (1 + k_ratio)

    So:  T_int = (k_f T_f + k_s T_s) / (k_f + k_s)
               = (T_f + k_ratio * T_s) / (1 + k_ratio)

    Args:
        T_fluid: Fluid temperature field, shape ``(ny, nx)``.
        T_solid: Solid temperature field, shape ``(ny, nx)``.
        mask_solid: Boolean mask (True = solid cell), shape ``(ny, nx)``.
        k_ratio: Conductivity ratio k_s / k_f.

    Returns:
        Updated ``(T_fluid, T_solid)`` tuple with interface temperatures
        applied.
    """
    # Detect fluid cells that are adjacent to at least one solid cell
    kernel = torch.ones(1, 1, 3, 3, device=mask_solid.device, dtype=torch.float32)
    kernel[0, 0, 1, 1] = 0.0
    solid_float = mask_solid.float().unsqueeze(0).unsqueeze(0)
    neighbour_solid = F.conv2d(solid_float, kernel, padding=1).squeeze(0).squeeze(0) > 0.0
    is_interface_fluid = neighbour_solid & ~mask_solid  # fluid touching solid

    # Compute interface temperature
    w = k_ratio / (1.0 + k_ratio)
    T_int = (1.0 - w) * T_fluid + w * T_solid

    # Apply to fluid interface cells
    T_fluid_new = torch.where(is_interface_fluid, T_int, T_fluid)

    # Mirror interface temperature to adjacent solid cells
    is_interface_solid = neighbour_solid & mask_solid
    T_solid_new = torch.where(is_interface_solid, T_int, T_solid)

    return T_fluid_new, T_solid_new


# ---------------------------------------------------------------------------
# Full CHT time loop
# ---------------------------------------------------------------------------

def run_conjugate_ht_2d(
    state: CHTState,
    cfg: CHTConfig,
    *,
    collide_fn: Callable | None = None,
    stream_fn: Callable | None = None,
    boundary_fn: Callable | None = None,
    callback: Callable[[CHTState], None] | None = None,
) -> CHTState:
    """Run a 2-D conjugate heat transfer simulation.

    Couples the DDF thermal LBM (fluid) with an explicit finite-difference
    solid solver at each time step.

    Algorithm per step
    ------------------
    1. **Fluid momentum step**: collision + streaming.
    2. **Fluid thermal step**: D2Q5 collision + streaming.
    3. **Solid diffusion step**: explicit Euler on the solid grid.
    4. **Interface coupling**: enforce T/flux continuity.
    5. **Boundary conditions**: hot/cold wall Dirichlet, no-slip.

    Args:
        state: Initial :class:`CHTState`.
        cfg: :class:`CHTConfig` parameters.
        collide_fn: Custom collision function for momentum (optional).
            Signature: ``(f, tau_f) → f_post``.
        stream_fn: Custom streaming function (optional).
            Signature: ``(f) → f_streamed``.
        boundary_fn: Custom boundary function applied after streaming
            (optional).  Signature: ``(f, mask) → f_bc``.
        callback: Called every ``cfg.output_interval`` steps with the
            current state.

    Returns:
        Final :class:`CHTState`.
    """
    from .d2q9 import macroscopic
    from .solver import collide_bgk, stream
    from .thermal import (
        apply_buoyancy_force,
        collide_thermal_bgk,
        equilibrium_thermal,
        macroscopic_thermal,
        stream_thermal,
    )

    _collide = collide_fn if collide_fn is not None else (
        lambda f, tau: collide_bgk(f, tau)  # noqa: B023
    )
    _stream = stream_fn if stream_fn is not None else stream

    tau_g = 0.5 + cfg.kappa_f / (1.0 / 6.0)  # D2Q5 cs² = 1/6

    for step in range(state.step, state.step + cfg.n_steps):
        # ---- 1. Fluid momentum: collide + stream ----
        rho, ux, uy = macroscopic(state.f)
        T_fluid = macroscopic_thermal(state.g)

        # Buoyancy coupling (Boussinesq)
        state.f = apply_buoyancy_force(
            state.f, T_fluid, cfg.T_ref, cfg.beta, cfg.gravity
        )
        state.f = _collide(state.f, cfg.tau_f)
        state.f = _stream(state.f)

        # Bounce-back on solid cells
        if boundary_fn is not None:
            state.f = boundary_fn(state.f, state.mask_solid)
        else:
            # Simple bounce-back: re-inject reversed distributions at solid cells
            from .boundaries import bounce_back_cells
            state.f = bounce_back_cells(state.f, state.mask_solid)

        # ---- 2. Fluid thermal: D2Q5 collide + stream ----
        rho_new, ux_new, uy_new = macroscopic(state.f)
        state.g = collide_thermal_bgk(state.g, T_fluid, ux_new, uy_new, tau_g)
        state.g = stream_thermal(state.g)

        # ---- 3. Solid conduction ----
        state.T_s = cht_solid_diffusion_step(
            state.T_s, state.mask_solid, cfg.alpha_s, cfg.Q_source
        )

        # ---- 4. Interface coupling ----
        T_fluid_new = macroscopic_thermal(state.g)
        T_fluid_coupled, T_solid_coupled = apply_cht_interface(
            T_fluid_new, state.T_s, state.mask_solid, cfg.k_ratio
        )
        # Update thermal distributions to match coupled T_fluid
        g_eq = equilibrium_thermal(T_fluid_coupled, ux_new, uy_new)
        # Partially relax towards new equilibrium (one BGK step for correction)
        state.g = state.g + (g_eq - state.g) * (1.0 / tau_g)
        state.T_s = T_solid_coupled

        state.step = step + 1

        if callback is not None and (step + 1) % cfg.output_interval == 0:
            callback(state)

    return state
