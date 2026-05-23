"""D2Q9 multiphase lattice Boltzmann models.

Implements four classes of multiphase LBM models for two-dimensional flows:

1. **Shan-Chen two-component (SCMC)** – two immiscible fluids driven apart by a
   repulsive pseudopotential interaction.  Suitable for dam-break, droplet, and
   liquid-gas simulations at moderate density ratios.

2. **Shan-Chen single-component (SCMP)** – one fluid with a non-linear EOS
   pseudopotential that generates spontaneous liquid/gas phase separation.

3. **Color-Gradient (CG)** – Gunstensen/Latva-Kokko-Rothman model.  Uses two
   distribution functions (red/blue) with a recoloring step to maintain sharp
   interfaces and an explicit surface-tension perturbation.

4. **Free-Energy (FE)** – simplified Swift et al. binary-fluid model.  A
   chemical-potential gradient drives interface dynamics via a modified
   equilibrium distribution, giving thermodynamically consistent interfaces.

References
----------
Shan & Chen (1993) Phys. Rev. E 47 1815
Shan & Chen (1994) Phys. Rev. E 49 2941
Gunstensen et al. (1991) Phys. Rev. A 43 4320
Latva-Kokko & Rothman (2005) Phys. Rev. E 71 056702
Swift et al. (1995) Phys. Rev. Lett. 75 830
Swift et al. (1996) Phys. Rev. E 54 5041
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from .d2q9 import C, W, equilibrium, macroscopic

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CS2 = 1.0 / 3.0  # lattice speed of sound squared


def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


def _w_on(device: torch.device) -> torch.Tensor:
    return W.to(device)


# ---------------------------------------------------------------------------
# Pseudopotential functions (for SCMP)
# ---------------------------------------------------------------------------

def psi_linear(rho: torch.Tensor) -> torch.Tensor:
    """Linear pseudopotential ψ(ρ) = ρ."""
    return rho


def psi_exp(rho: torch.Tensor, rho0: float = 1.0) -> torch.Tensor:
    """Shan-Chen original pseudopotential ψ(ρ) = ρ₀(1 − exp(−ρ/ρ₀)).

    Args:
        rho:  Density field (any shape).
        rho0: Reference density (default 1.0).

    Returns:
        Pseudopotential field of the same shape.
    """
    return rho0 * (1.0 - torch.exp(-rho / rho0))


def psi_power(rho: torch.Tensor, psi0: float = 4.0) -> torch.Tensor:
    """Power-law pseudopotential ψ(ρ) = ψ₀ exp(−ψ₀/ρ).

    This form can achieve higher density ratios than the linear variant.
    """
    return psi0 * torch.exp(-psi0 / torch.clamp(rho, min=1e-12))


# ---------------------------------------------------------------------------
# Shan-Chen neighborhood sum (shared by SCMC and SCMP)
# ---------------------------------------------------------------------------

def _sc_neighbor_weighted_sum(
    psi: torch.Tensor,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Σᵢ wᵢ ψ(x+cᵢ) cᵢ for the SC interaction force.

    Uses periodic rolls; non-periodic (wall) cells are handled by zeroing
    the pseudopotential at solid nodes before the sum so that walls do not
    inject spurious inter-component forces across the periodic boundaries.

    Args:
        psi:         Scalar field of shape ``(ny, nx)``.
        solid_mask:  Optional boolean mask of shape ``(ny, nx)``.  When
                     provided, ``psi`` is zeroed at solid/wall cells so that
                     they contribute neutral pseudopotential to the force sum.
                     This avoids instabilities caused by large density
                     gradients at corners when the domain uses periodic
                     streaming with closed-box bounce-back walls.

    Returns:
        Tuple ``(Fx_kernel, Fy_kernel)`` of shape ``(ny, nx)`` each – the
        weighted-sum *before* multiplication by −G ψ(x).
    """
    if solid_mask is not None:
        psi = psi.masked_fill(solid_mask, 0.0)

    device = psi.device
    c = _c_on(device)
    w = _w_on(device)

    Fx = torch.zeros_like(psi)
    Fy = torch.zeros_like(psi)
    for i in range(9):
        cx_i = int(c[i, 0].item())
        cy_i = int(c[i, 1].item())
        if cx_i == 0 and cy_i == 0:
            continue  # rest direction: c=0, no contribution
        w_i = float(w[i].item())
        psi_shifted = torch.roll(torch.roll(psi, cx_i, dims=-1), cy_i, dims=-2)
        Fx += w_i * psi_shifted * cx_i
        Fy += w_i * psi_shifted * cy_i

    return Fx, Fy


# ---------------------------------------------------------------------------
# Model 1 – Shan-Chen Two-Component (SCMC)
# ---------------------------------------------------------------------------

def sc_two_component_force(
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    G_12: float,
    gx: float = 0.0,
    gy: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Shan-Chen interaction + gravity forces for two immiscible components.

    The repulsive interaction (G_12 > 0) drives the two fluids apart and
    maintains a diffuse interface between them.

    The buoyancy force arises naturally: the heavier component sinks faster
    (or slower, depending on gy sign) than the lighter one.

    Args:
        rho1:        Density of component 1, shape ``(ny, nx)``.
        rho2:        Density of component 2, shape ``(ny, nx)``.
        G_12:        Coupling constant.  G_12 > 0 ↔ repulsive ↔ phase separation.
        gx:          Body-force acceleration in x (lattice units).
        gy:          Body-force acceleration in y (lattice units); negative = down.
        solid_mask:  Optional boolean mask of shape ``(ny, nx)``.  Solid/wall cells
                     are zeroed in the pseudopotential sum to prevent spurious
                     boundary forces when using periodic streaming.

    Returns:
        ``(Fx1, Fy1, Fx2, Fy2)`` – force fields for each component,
        each of shape ``(ny, nx)``.
    """
    # Interaction: F_σ = −G · ρ_σ · Σᵢ wᵢ ρ_σ'(x+cᵢ) cᵢ
    sum_x2, sum_y2 = _sc_neighbor_weighted_sum(rho2, solid_mask)
    Fx1 = -G_12 * rho1 * sum_x2 + rho1 * gx
    Fy1 = -G_12 * rho1 * sum_y2 + rho1 * gy

    sum_x1, sum_y1 = _sc_neighbor_weighted_sum(rho1, solid_mask)
    Fx2 = -G_12 * rho2 * sum_x1 + rho2 * gx
    Fy2 = -G_12 * rho2 * sum_y1 + rho2 * gy

    return Fx1, Fy1, Fx2, Fy2


def collide_sc_two_component(
    f1: torch.Tensor,
    f2: torch.Tensor,
    G_12: float = 0.9,
    tau1: float = 1.0,
    tau2: float = 1.0,
    gx: float = 0.0,
    gy: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shan-Chen two-component BGK collision step for D2Q9.

    Each component undergoes independent BGK relaxation towards its own
    equilibrium, but the equilibrium velocity is shifted by the inter-component
    repulsion force and the external body force (gravity).

    Equilibrium velocity for component σ:

        uᵉq_σ = uσ + τ_σ Fσ / ρ_σ

    Args:
        f1:          Distribution of component 1, shape ``(9, ny, nx)``.
        f2:          Distribution of component 2, shape ``(9, ny, nx)``.
        G_12:        SC coupling constant (> 0 for phase separation).
        tau1:        Relaxation time for component 1.
        tau2:        Relaxation time for component 2.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration (negative = downward).
        solid_mask:  Optional boolean mask ``(ny, nx)`` of solid/wall cells.
                     When provided, solid-cell densities are excluded from the
                     SC neighbour sum to avoid spurious forces from periodic
                     streaming across closed-box walls.

    Returns:
        Updated ``(f1, f2)`` after collision.
    """
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)

    Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(rho1, rho2, G_12, gx, gy, solid_mask)

    rho1_s = torch.clamp(rho1, min=1e-12)
    rho2_s = torch.clamp(rho2, min=1e-12)

    feq1 = equilibrium(rho1, ux1 + tau1 * Fx1 / rho1_s, uy1 + tau1 * Fy1 / rho1_s)
    feq2 = equilibrium(rho2, ux2 + tau2 * Fx2 / rho2_s, uy2 + tau2 * Fy2 / rho2_s)

    f1_out = f1 - (f1 - feq1) / tau1
    f2_out = f2 - (f2 - feq2) / tau2

    # Solid cells skip collision: their distributions are preserved so that the
    # subsequent bounce-back step can correctly reverse them after streaming.
    if solid_mask is not None:
        mask_3d = solid_mask.unsqueeze(0)  # (1, ny, nx)
        f1_out = torch.where(mask_3d, f1, f1_out)
        f2_out = torch.where(mask_3d, f2, f2_out)

    return f1_out, f2_out


# ---------------------------------------------------------------------------
# Model 2 – Shan-Chen Single-Component (SCMP)
# ---------------------------------------------------------------------------

def sc_single_component_force(
    rho: torch.Tensor,
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
    gx: float = 0.0,
    gy: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the SC self-interaction + gravity force for a single component.

    The self-interaction (G < 0, attractive) generates liquid–gas coexistence
    via a density-dependent pseudopotential.

    Args:
        rho:         Density field, shape ``(ny, nx)``.
        G:           SC self-coupling constant (< 0 for attraction / phase separation).
        psi_fn:      Pseudopotential function ψ(ρ).  Defaults to :func:`psi_exp`.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        solid_mask:  Optional boolean mask of wall/solid cells.

    Returns:
        ``(Fx, Fy)`` each of shape ``(ny, nx)``.
    """
    psi = psi_fn(rho)
    sum_x, sum_y = _sc_neighbor_weighted_sum(psi, solid_mask)
    Fx = -G * psi * sum_x + rho * gx
    Fy = -G * psi * sum_y + rho * gy
    return Fx, Fy


def collide_sc_single_component(
    f: torch.Tensor,
    G: float = -4.0,
    tau: float = 1.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
    gx: float = 0.0,
    gy: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shan-Chen single-component multiphase (SCMP) BGK collision for D2Q9.

    Use G < 0 to generate attractive self-interaction and spontaneous
    liquid–gas phase separation.

    Args:
        f:           Distribution tensor, shape ``(9, ny, nx)``.
        G:           SC self-coupling constant (< 0 for phase separation).
        tau:         Relaxation time.
        psi_fn:      Pseudopotential callable.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        solid_mask:  Optional boolean mask of wall/solid cells.

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy = macroscopic(f)
    Fx, Fy = sc_single_component_force(rho, G, psi_fn, gx, gy, solid_mask)
    rho_s = torch.clamp(rho, min=1e-12)
    feq = equilibrium(rho, ux + tau * Fx / rho_s, uy + tau * Fy / rho_s)
    f_out = f - (f - feq) / tau
    if solid_mask is not None:
        f_out = torch.where(solid_mask.unsqueeze(0), f, f_out)
    return f_out


# ---------------------------------------------------------------------------
# Model 3 – Color-Gradient (CG)
# ---------------------------------------------------------------------------
#
# Algorithm (Latva-Kokko & Rothman 2005):
#   1. Collision (BGK on total f) + surface-tension perturbation.
#   2. Recoloring step to restore phase separation.
#
# Distribution split:  f_total = f_r + f_b
# Phase field:         φ = (ρ_r − ρ_b) / (ρ_r + ρ_b)  ∈ [−1, 1]
# Interface normal:    n̂ = ∇φ / |∇φ|  (computed via central differences)

def _grad_phase_field(rho_r: torch.Tensor, rho_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the phase-field gradient via second-order central differences.

    Returns ``(phi, nx_hat, ny_hat)`` all of shape ``(ny, nx)``.
    The gradient is computed with periodic wrapping (boundaries are handled
    externally by bounce-back).
    """
    n = rho_r + rho_b
    n_safe = torch.clamp(n, min=1e-12)
    phi = (rho_r - rho_b) / n_safe

    # Central differences (periodic)
    dphi_dx = 0.5 * (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1))
    dphi_dy = 0.5 * (torch.roll(phi, -1, dims=-2) - torch.roll(phi, 1, dims=-2))

    mag = torch.sqrt(dphi_dx ** 2 + dphi_dy ** 2)
    mag_safe = torch.clamp(mag, min=1e-12)
    nx_hat = dphi_dx / mag_safe
    ny_hat = dphi_dy / mag_safe

    return phi, nx_hat, ny_hat


def color_gradient_step(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
    tau: float = 1.0,
    A: float = 0.04,
    beta: float = 0.7,
    gx: float = 0.0,
    gy: float = 0.0,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Color-Gradient two-phase step for D2Q9.

    Performs one full CG iteration:
      (a) BGK collision on the total distribution;
      (b) Surface-tension perturbation proportional to |∇φ|;
      (c) Recoloring to restore the two phases.

    Args:
        f_r:         Red (heavy) component distribution, shape ``(9, ny, nx)``.
        f_b:         Blue (light) component distribution, shape ``(9, ny, nx)``.
        tau:         Shared relaxation time (same for both components).
        A:           Surface-tension coefficient (larger → stronger tension).
        beta:        Recoloring parameter ∈ (0, 1].  β→1 gives sharp interfaces.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        solid_mask:  Optional boolean mask ``(ny, nx)`` of solid/wall cells.
                     Solid cells skip collision and the phase-field gradient is
                     computed with masked densities to prevent spurious surface
                     tension at sharp solid boundaries.

    Returns:
        Updated ``(f_r, f_b)`` after collision + recoloring.
    """
    device = f_r.device
    c = _c_on(device)
    w = _w_on(device)

    # --- 1. Total distribution and macroscopic quantities ---
    f_total = f_r + f_b
    rho_r = f_r.sum(dim=0)   # (ny, nx)
    rho_b = f_b.sum(dim=0)
    rho = rho_r + rho_b
    rho_s = torch.clamp(rho, min=1e-12)

    # Velocity from total momentum
    cx = c[:, 0].float().view(9, 1, 1)
    cy = c[:, 1].float().view(9, 1, 1)
    ux = (f_total * cx).sum(dim=0) / rho_s
    uy = (f_total * cy).sum(dim=0) / rho_s

    # Add body force to velocity used for equilibrium
    ux_eq = ux + tau * gx
    uy_eq = uy + tau * gy

    # --- 2. BGK collision on total distribution ---
    feq = equilibrium(rho, ux_eq, uy_eq)
    f_post = f_total - (f_total - feq) / tau

    # Solid cells skip collision (preserve pre-collision distributions)
    if solid_mask is not None:
        f_post = torch.where(solid_mask.unsqueeze(0), f_total, f_post)

    # --- 3. Surface-tension perturbation ---
    # Use masked densities to prevent spurious gradients at solid boundaries
    rho_r_safe = rho_r if solid_mask is None else rho_r.masked_fill(solid_mask, 0.0)
    rho_b_safe = rho_b if solid_mask is None else rho_b.masked_fill(solid_mask, 0.0)
    phi, nhat_x, nhat_y = _grad_phase_field(rho_r_safe, rho_b_safe)
    mag = torch.sqrt(
        (0.5 * (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1))) ** 2
        + (0.5 * (torch.roll(phi, -1, dims=-2) - torch.roll(phi, 1, dims=-2))) ** 2
    )

    # Perturbation: Δfᵢ = (A/2)|∇φ| wᵢ [(cᵢ·n̂)² − Bᵢ]
    # with Bᵢ = 1/3 for D2Q9 (isotropic contribution)
    ci_dot_n = cx * nhat_x.unsqueeze(0) + cy * nhat_y.unsqueeze(0)  # (9,ny,nx)
    B_iso = torch.tensor(1.0 / 3.0, device=device)
    w_view = w.view(9, 1, 1)
    perturbation = (A / 2.0) * mag.unsqueeze(0) * w_view * (ci_dot_n ** 2 - B_iso)
    f_post = f_post + perturbation

    # --- 4. Recoloring step (Latva-Kokko & Rothman 2005) ---
    feq_unit = equilibrium(torch.ones_like(rho), torch.zeros_like(ux), torch.zeros_like(uy))
    cos_theta = ci_dot_n
    recolor_amp = beta * (rho_r * rho_b / rho_s).unsqueeze(0) * cos_theta * feq_unit

    f_r_out = (rho_r / rho_s).unsqueeze(0) * f_post + recolor_amp
    f_b_out = (rho_b / rho_s).unsqueeze(0) * f_post - recolor_amp

    # Solid cells keep pre-collision distributions (will be bounce-backed later)
    if solid_mask is not None:
        mask_3d = solid_mask.unsqueeze(0)
        f_r_out = torch.where(mask_3d, f_r, f_r_out)
        f_b_out = torch.where(mask_3d, f_b, f_b_out)

    return f_r_out, f_b_out


# ---------------------------------------------------------------------------
# Model 4 – Free-Energy (FE)
# ---------------------------------------------------------------------------
#
# Simplified Swift et al. binary-fluid formulation.  The order parameter
# φ = (ρ₁ − ρ₂)/(ρ₁ + ρ₂) is advected and diffused by a separate distribution
# g (phase-field LBM), while the total density/momentum are governed by the
# usual f distribution with a modified pressure tensor.
#
# Chemical potential:  μ = −Aφ + Bφ³ − κ∇²φ
# Driving force:       F_φ = −φ ∇μ  (Korteweg force in the momentum eq.)


def _laplacian_2d(field: torch.Tensor) -> torch.Tensor:
    """2D Laplacian via second-order central differences (periodic)."""
    lap = (
        torch.roll(field, 1, dims=-1) + torch.roll(field, -1, dims=-1)
        + torch.roll(field, 1, dims=-2) + torch.roll(field, -1, dims=-2)
        - 4.0 * field
    )
    return lap


def free_energy_step(
    f: torch.Tensor,
    g: torch.Tensor,
    tau_f: float = 1.0,
    tau_g: float = 0.7,
    A: float = 0.1,
    B: float = 0.1,
    kappa: float = 0.02,
    Gamma: float = 0.5,
    gx: float = 0.0,
    gy: float = 0.0,
    rho_heavy: float | None = None,
    rho_light: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-Energy two-phase step for D2Q9.

    Two coupled LBM equations:
      • **f**: momentum distribution for total density n and velocity u.
      • **g**: order-parameter distribution that advects the phase field φ.

    The Korteweg stress drives interface dynamics.  When *rho_heavy* and
    *rho_light* are given the gravitational body force is scaled by the
    local effective density (Boussinesq buoyancy), enabling proper gravity-
    driven flow in the dam-break and water-entry benchmarks.

    Args:
        f:          Momentum distribution, shape ``(9, ny, nx)``.
        g:          Order-parameter distribution, shape ``(9, ny, nx)``;
                    its zeroth moment is the phase field φ = Σᵢ gᵢ.
        tau_f:      Relaxation time for momentum (ν = cs²(τ_f − ½)).
        tau_g:      Relaxation time for phase field (M = cs²(τ_g − ½)).
        A:          Double-well coefficient.
        B:          Quartic coefficient.
        kappa:      Interfacial-tension parameter (gradient penalty).
        Gamma:      Phase-field mobility coupling.
        gx:         x body-force acceleration.
        gy:         y body-force acceleration.
        rho_heavy:  Effective density for the φ=+1 phase (Boussinesq buoyancy).
        rho_light:  Effective density for the φ=−1 phase.

    Returns:
        Updated ``(f, g)`` after one collision step.
    """
    device = f.device
    c = _c_on(device)
    w = _w_on(device)
    cx = c[:, 0].float().view(9, 1, 1)
    cy = c[:, 1].float().view(9, 1, 1)
    w_v = w.view(9, 1, 1)

    # Macroscopic quantities
    rho, ux, uy = macroscopic(f)
    phi = g.sum(dim=0)  # order parameter

    # Effective density for buoyancy (Boussinesq approximation)
    if rho_heavy is not None and rho_light is not None:
        phi_c = phi.clamp(-1.0, 1.0)
        rho_eff = 0.5 * ((1.0 + phi_c) * rho_heavy + (1.0 - phi_c) * rho_light)
    else:
        rho_eff = rho

    # Chemical potential: μ = −Aφ + Bφ³ − κ∇²φ
    mu = -A * phi + B * phi ** 3 - kappa * _laplacian_2d(phi)

    # Korteweg (capillary) body force: F_cap = −φ ∇μ
    grad_mu_x = 0.5 * (torch.roll(mu, -1, dims=-1) - torch.roll(mu, 1, dims=-1))
    grad_mu_y = 0.5 * (torch.roll(mu, -1, dims=-2) - torch.roll(mu, 1, dims=-2))
    Fx = -phi * grad_mu_x + rho_eff * gx
    Fy = -phi * grad_mu_y + rho_eff * gy

    # Velocity for equilibrium (simple force shift)
    rho_s = torch.clamp(rho, min=1e-12)
    ux_eq = ux + tau_f * Fx / rho_s
    uy_eq = uy + tau_f * Fy / rho_s

    # Collision for f (BGK with capillary + buoyancy force)
    feq = equilibrium(rho, ux_eq, uy_eq)
    f_out = f - (f - feq) / tau_f

    # Equilibrium for g: geq_i advects φ and diffuses it via the chemical potential.
    # The diffusion term must be formulated using the *anisotropic* velocity basis so
    # that the zeroth moment of geq equals φ (conservation of the order parameter).
    # We use the correction: geq_i = w_i φ feq_factor_i + Γ w_i (c_i²/cs² - D) μ
    # where D = spatial dimension = 2, so that Σ geq_i = φ.
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    u_sq = (ux ** 2 + uy ** 2).unsqueeze(0)
    # Advection part: same form as feq for rho=phi
    geq_adv = w_v * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u_sq)
    # Diffusion part: Γ w_i (|c_i|² − D cs²) μ / cs⁴
    # |c_i|²/cs² for D2Q9: 0 for rest (i=0), 1 for face (i=1-4), 2 for diagonal (i=5-8)
    c_sq = (cx ** 2 + cy ** 2)  # |c_i|²
    diff_factor = (c_sq / _CS2 - 2.0)  # anisotropic, sums to 0 over w_i
    geq_diff = w_v * Gamma * diff_factor * mu.unsqueeze(0)
    geq = geq_adv + geq_diff
    g_out = g - (g - geq) / tau_g

    return f_out, g_out


def init_free_energy_g(
    phi: torch.Tensor,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Initialise the FE order-parameter distribution in equilibrium.

    Args:
        phi: Initial phase field, shape ``(ny, nx)``.
        ux:  Initial x-velocity (optional, defaults to zero).
        uy:  Initial y-velocity (optional, defaults to zero).

    Returns:
        Equilibrium distribution g, shape ``(9, ny, nx)``.
    """
    device = phi.device
    c = _c_on(device)
    w = _w_on(device).view(9, 1, 1)
    if ux is None:
        ux = torch.zeros_like(phi)
    if uy is None:
        uy = torch.zeros_like(phi)

    cx = c[:, 0].float().view(9, 1, 1)
    cy = c[:, 1].float().view(9, 1, 1)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    u_sq = (ux ** 2 + uy ** 2).unsqueeze(0)
    return w * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u_sq)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Pseudopotential functions
    "psi_linear",
    "psi_exp",
    "psi_power",
    # Model 1: Shan-Chen Two-Component
    "sc_two_component_force",
    "collide_sc_two_component",
    # Model 2: Shan-Chen Single-Component
    "sc_single_component_force",
    "collide_sc_single_component",
    # Model 3: Color-Gradient
    "color_gradient_step",
    # Model 4: Free-Energy
    "free_energy_step",
    "init_free_energy_g",
]
