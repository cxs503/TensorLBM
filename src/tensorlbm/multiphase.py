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

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from .d2q9 import C, W, equilibrium, macroscopic

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CS2 = 1.0 / 3.0  # lattice speed of sound squared

# Cache for SC neighbour-sum gather indices keyed by (ny, nx, device_type, device_index)
_sc2d_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}


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


def psi_carnahan_starling(rho: torch.Tensor) -> torch.Tensor:
    """Carnahan-Starling non-ideal EOS pseudopotential ψ = √(2(p_EOS − ρ·cs²)/(G·cs²)).

    This is the most commonly used realistic EOS for SCMP, as demonstrated
    by OpenLB's ``phaseSeparation3d`` example.  It produces significantly
    smaller spurious currents than Shan-Chen94 (psi_exp) and can achieve
    density ratios of 1000:1 with moderate G values (~-1.0 to -5.0).

    The EOS is:
        p(ρ) = ρ·RT·(1 + η + η² − η³)/(1 − η)³ − a·ρ²
    where η = b·ρ/4.  Default parameters (a=0.5, b=4, RT=1/3) give a
    liquid/vapor coexistence at ρ_l ≈ 0.45, ρ_v ≈ 0.05.

    Args:
        rho:  Density field (any shape).

    Returns:
        Pseudopotential field ψ(ρ) > 0, same shape as input.

    References
    ----------
    Yuan & Schaefer (2006) Phys. Fluids 18, 042101
    OpenLB ``phaseSeparation3d.cpp`` (CarnahanStarling EOS with G=-1.0)
    """
    a = 0.5  # attraction parameter
    b = 4.0  # repulsion parameter (≡ 1/ρ_c in Yuan & Schaefer)
    RT = 1.0 / 3.0
    cs2 = 1.0 / 3.0

    rho_c = torch.clamp(rho, min=1e-12)
    eta = b * rho_c / 4.0  # reduced density, 0 < η < 1
    eta_1 = 1.0 - eta
    eta_c = torch.clamp(eta_1, min=1e-8)

    # Carnahan-Starling pressure
    p = rho_c * RT * (1.0 + eta + eta**2 - eta**3) / (eta_c**3) - a * rho_c**2

    # ψ = √(2(p − ρ·cs²)/cs²)  — standard SCMP pseudopotential from EOS
    p_ideal = rho_c * cs2
    p_diff = torch.clamp(p - p_ideal, min=0.0)  # must be non-negative
    psi_val = torch.sqrt(2.0 * p_diff / cs2)
    return psi_val


def make_psi_carnahan_starling(
    a: float = 5.0, b: float = 4.0, RT: float = 1.0 / 3.0, cs2: float = 1.0 / 3.0
):
    """Build a **correct-sign** Carnahan–Starling EOS pseudopotential ψ(ρ).

    The historical :func:`psi_carnahan_starling` in this module computes
    ``√(2(p_EOS − ρcs²)/cs²)`` — the WRONG sign for the library's attractive
    convention (``G_lib > 0``), and it silently clamps ``p_EOS < ρcs²`` regions
    to ψ=0 (killing the vapour branch).  It also hard-codes a=0.5, b=4, RT=1/3,
    for which the CS EOS has **no van der Waals loop** (∂p/∂ρ > 0 everywhere),
    i.e. no spinodal → the model cannot phase-separate at all (this is why the
    historical ``--psi cs`` runs diverged at ~step 20).

    Correct construction.  With the library convention ``F = +G_lib·cs²·ψ∇ψ``
    (``G_lib > 0`` = attraction) the non-ideal pressure is
    ``p(ρ) = ρcs² − (G_lib·cs²/2)ψ²``.  Requiring ``p = p_EOS`` gives

        ψ(ρ) = √( 2(ρcs² − p_EOS(ρ)) / (G_lib·cs²) ) .

    Choosing the coupling ``G_lib = 1`` makes ψ independent of G, so this
    factory returns

        ψ(ρ) = √( 2(ρcs² − p_EOS(ρ)) / cs² ) ,   p_EOS < ρcs²  (ψ=0 elsewhere).

    Use it with ``G_lib = 1`` (any other G rescales the effective attraction
    and must be re-calibrated).  CS EOS:
        p_EOS = ρ·RT·(1 + η + η² − η³)/(1 − η)³ − a·ρ² ,  η = b·ρ/4 .

    Coexistence (Maxwell, this module's ``scripts/measure_coexistence.py``):
        a=5, b=4, RT=1/3 → ρ_l≈0.355, ρ_v≈0.0101, ratio ≈ 35  (≈3× psi_exp's 12.3)

    Args:
        a, b, RT, cs2: CS EOS parameters (defaults give a 35:1 two-phase loop).

    Returns:
        A callable ``psi(rho) -> tensor`` suitable as ``psi_fn``.
    """

    def psi(rho: torch.Tensor) -> torch.Tensor:
        rho_c = torch.clamp(rho, min=1e-12)
        eta = b * rho_c / 4.0
        eta_c = torch.clamp(1.0 - eta, min=1e-8)
        p_eos = rho_c * RT * (1.0 + eta + eta**2 - eta**3) / (eta_c**3) - a * rho_c**2
        p_ideal = rho_c * cs2
        val = 2.0 * (p_ideal - p_eos) / cs2
        return torch.sqrt(torch.clamp(val, min=0.0))

    return psi


def psi_peng_robinson(rho: torch.Tensor) -> torch.Tensor:
    """Peng-Robinson EOS pseudopotential ψ = √(2(p_EOS − ρ·cs²)/cs²).

    Achieves even higher density ratios (5000:1+) than Carnahan-Starling.
    The EOS is:
        p = ρ·RT/(1 − bρ) − a·α(T)·ρ²/(1 + 2bρ − b²ρ²)

    Default parameters tuned for water-like coexistence at RT=1/3, cs²=1/3.

    Args:
        rho:  Density field (any shape).

    Returns:
        Pseudopotential field, same shape as input.
    """
    a_val = 2.0 / 49.0
    b_val = 2.0 / 21.0
    RT = 1.0 / 3.0
    cs2 = 1.0 / 3.0

    rho_c = torch.clamp(rho, min=1e-12)
    one_m_bp = 1.0 - b_val * rho_c
    denom1 = torch.clamp(one_m_bp, min=1e-8)
    denom2 = torch.clamp(1.0 + 2.0 * b_val * rho_c - b_val**2 * rho_c**2, min=1e-8)

    p = rho_c * RT / denom1 - a_val * rho_c**2 / denom2
    p_ideal = rho_c * cs2
    p_diff = torch.clamp(p - p_ideal, min=0.0)
    psi_val = torch.sqrt(2.0 * p_diff / cs2)
    return psi_val


# ---------------------------------------------------------------------------
# Shan-Chen neighborhood sum (shared by SCMC and SCMP)
# ---------------------------------------------------------------------------


def _sc_neighbor_weighted_sum(
    psi: torch.Tensor,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
    square: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Σᵢ wᵢ ψ(x+cᵢ) cᵢ for the SC interaction force.

    Uses a vectorised gather (same strategy as :func:`~tensorlbm.solver.stream`)
    instead of a Python for-loop, eliminating all GPU→CPU synchronisations and
    reducing kernel launches to a small constant.  Index tensors are cached per
    (shape, device) to avoid re-allocation on every call.

    Args:
        psi:         Scalar field of shape ``(ny, nx)``.
        solid_mask:  Optional boolean mask of shape ``(ny, nx)``.  When
                     provided, the pseudopotential at solid/wall cells is
                     replaced before the force sum (see ``wall_psi``).
        wall_psi:    Pseudopotential value attributed to solid/wall cells in the
                     neighbour sum.  ``None`` (default) reproduces the historical
                     behaviour of zeroing ``psi`` at the wall (a fully non-wetting
                     "dry" wall).  A positive value ``0 < wall_psi < psi(rho_l)``
                     models a partially wetting wall (finite contact angle) and
                     removes the artificial density-depleted layer that the dry
                     wall creates on a no-slip floor.  See
                     ``benchmarks/pending/dam_break_sc``.
        square:      When ``True``, the (masked) field is squared *before* the
                     gather, returning ``Σᵢ wᵢ ψ²(x−cᵢ) cᵢ``.  This is the
                     isotropic-stencil evaluation of the pressure-tensor form
                     ``F = −½∇(ψ²)`` used by ``scheme="pressure_tensor"``
                     (leading-order-equivalent to the standard force, but a
                     symmetric divergence form with Σₓ F = 0 exactly).

    Returns:
        Tuple ``(Fx_kernel, Fy_kernel)`` of shape ``(ny, nx)`` each – the
        weighted-sum *before* multiplication by −G ψ(x).
    """
    if solid_mask is not None:
        psi = psi.masked_fill(solid_mask, 0.0 if wall_psi is None else float(wall_psi))
    if square:
        psi = psi * psi

    device = psi.device
    ny, nx = psi.shape[-2], psi.shape[-1]
    c = _c_on(device)  # (9, 2)  int64
    w = _w_on(device)  # (9,)    float32

    # Build and cache gather index tensors (one-time cost per unique shape/device)
    cache_key = (ny, nx, device.type, device.index)
    if cache_key not in _sc2d_cache:
        cy = c[:, 1]  # (9,)
        cx = c[:, 0]  # (9,)
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - cy.unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - cx.unsqueeze(1)) % nx
        # y_idx: (9, ny, 1)  x_idx: (9, 1, nx)  → broadcast to (9, ny, nx)
        _sc2d_cache[cache_key] = (
            y_src.unsqueeze(2),  # (9, ny, 1)
            x_src.unsqueeze(1),  # (9, 1, nx)
        )

    y_idx, x_idx = _sc2d_cache[cache_key]
    # psi_shifts: (9, ny, nx) – all shifted copies gathered in one operation
    psi_shifts = psi[y_idx, x_idx]  # advanced-index gather, no Python loop

    # w * cx and w * cy: (9, 1, 1) for broadcasting over (ny, nx)
    cx_float = c[:, 0].float().view(9, 1, 1)
    cy_float = c[:, 1].float().view(9, 1, 1)
    w_3d = w.view(9, 1, 1)

    Fx = (w_3d * cx_float * psi_shifts).sum(0)  # (ny, nx)
    Fy = (w_3d * cy_float * psi_shifts).sum(0)  # (ny, nx)
    return Fx, Fy


def _psi_sq_central_gradient(
    psi: torch.Tensor,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discrete gradient of ψ² by *central differences* (exactly curl-free).

    This is the core of the Ramshaw–Phathanapirom pressure-tensor force form.
    The standard Shan–Chen force samples ψ on the (forward/backward) D2Q9
    stencil; the resulting discrete operator is **not** a discrete gradient,
    so its discrete curl is O(h²) ≠ 0 and it can inject a *non-gradient*
    (vortical) spurious force at the interface — the mechanism that biases the
    late-time dam-break front.

    Writing the (leading-order) SC force as a pressure-tensor divergence,

        F_α = −G_eff ψ ∂_α ψ = −(G_eff/2) ∂_α(ψ²) = −∂_β P_αβ ,
        P_αβ = (G_eff/2) ψ² δ_αβ   (isotropic part)

    the force becomes an *exact* discrete gradient when the ψ² derivative is
    taken with central differences: central differences on a rectangular grid
    satisfy ∂_x ∂_y = ∂_y ∂_x **exactly**, so the discrete curl is identically
    zero and the force cannot generate spurious vorticity at the interface.
    Additionally, Σ_x D(ψ²) = 0 telescopically for a periodic box, i.e. the
    total force vanishes **exactly** (Newton's third law holds at machine
    precision) — the defining property of the pressure-tensor form.

    Calibration is identical to the standard scheme at leading order:
    with ``G_lib`` the library coupling (``G_eff = −G_lib``) the returned
    field equals ``(G_lib·cs²/2)·D(ψ²)``, reproducing ``G_lib·cs²·ψ∇ψ`` to
    O(h²) so the coexistence densities / EOS are unchanged.

    Args:
        psi:         Pseudopotential field of shape ``(ny, nx)``.
        solid_mask:  Optional boolean wall mask (same convention as
                     :func:`_sc_neighbor_weighted_sum`).
        wall_psi:    ψ attributed to wall cells (``None`` → dry wall, ψ=0).

    Returns:
        ``(Dx, Dy)`` of shape ``(ny, nx)`` — the central-difference gradient
        of the (masked) ψ² field, periodic wrap.
    """
    if solid_mask is not None:
        psi = psi.masked_fill(solid_mask, 0.0 if wall_psi is None else float(wall_psi))
    psi2 = psi * psi
    dpx = 0.5 * (torch.roll(psi2, -1, dims=-1) - torch.roll(psi2, 1, dims=-1))
    dpy = 0.5 * (torch.roll(psi2, -1, dims=-2) - torch.roll(psi2, 1, dims=-2))
    return dpx, dpy


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
    u_eq: str = "self",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shan-Chen two-component BGK collision step for D2Q9.

    Each component undergoes BGK relaxation towards an equilibrium whose
    velocity is shifted by the inter-component interaction force and the
    external body force (gravity).

    Equilibrium velocity for component σ (``u_eq="self"``, the default):

        uᵉq_σ = uσ + τ_σ Fσ / ρ_σ

    With ``u_eq="mixture"`` the equilibrium is built on the mixture
    centre-of-mass velocity (Shan & Doolen 1995):

        u_mix  = (ρ₁u₁ + ρ₂u₂) / (ρ₁ + ρ₂)
        uᵉq_σ  = u_mix + τ_σ Fσ / ρ_σ

    The mixture form couples the components' momentum: each component
    relaxes towards the barycentric velocity, which transmits the viscous
    stress across the diffuse interface (per-component relaxation towards
    u_mix adds exactly (ρ_σ/τ_σ)(u_mix − u_σ) of momentum exchange per
    step; for τ₁ = τ₂ the exchanges sum to zero, so total momentum is
    conserved while the *relative* component velocity decays).  The "self"
    form has no such coupling: two components may carry different
    velocities through the interface.

    In both forms the force shift τ_σ F_σ/ρ_σ transfers exactly F_σ of
    momentum per collision step (the τ_σ cancels: Σᵢ cᵢ fᵉq(ρ, u+τF/ρ) =
    ρu + τF, and the BGK update then yields M' = M + F), matching the
    library's velocity-shift forcing convention.

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
        u_eq:        Equilibrium-velocity convention: ``"self"`` (default,
                     legacy behaviour, bit-identical to the previous
                     implementation) or ``"mixture"`` (Shan–Doolen
                     centre-of-mass coupling).

    Returns:
        Updated ``(f1, f2)`` after collision.

    Note
    ----
    With unequal relaxation times the mixture form carries a net momentum
    source (1/tau2 - 1/tau1) * rho1 * rho2 * (u1 - u2) / (rho1 + rho2)
    per collision step (zero for equal taus), which is structurally
    destabilising at strong segregation: with tau = (1.0, 0.75) the
    mixture form diverges for G_12 <= -1.5 while the "self" form remains
    stable.

    References
    ----------
    Shan & Doolen (1995) J. Stat. Phys. 81:379 — multicomponent LBM with the
    barycentric (mixture) velocity in the equilibrium.
    """
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)

    Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(rho1, rho2, G_12, gx, gy, solid_mask)

    rho1_s = torch.clamp(rho1, min=1e-12)
    rho2_s = torch.clamp(rho2, min=1e-12)

    if u_eq == "self":
        feq1 = equilibrium(rho1, ux1 + tau1 * Fx1 / rho1_s, uy1 + tau1 * Fy1 / rho1_s)
        feq2 = equilibrium(rho2, ux2 + tau2 * Fx2 / rho2_s, uy2 + tau2 * Fy2 / rho2_s)
    elif u_eq == "mixture":
        rho_tot = rho1_s + rho2_s
        u_mix_x = (rho1_s * ux1 + rho2_s * ux2) / rho_tot
        u_mix_y = (rho1_s * uy1 + rho2_s * uy2) / rho_tot
        feq1 = equilibrium(rho1, u_mix_x + tau1 * Fx1 / rho1_s, u_mix_y + tau1 * Fy1 / rho1_s)
        feq2 = equilibrium(rho2, u_mix_x + tau2 * Fx2 / rho2_s, u_mix_y + tau2 * Fy2 / rho2_s)
    else:
        raise ValueError(f"u_eq must be 'self' or 'mixture', got {u_eq!r}")

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
    wall_psi: float | None = None,
    scheme: str = "standard",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the SC self-interaction + gravity force for a single component.

    The self-interaction (G < 0, attractive) generates liquid–gas coexistence
    via a density-dependent pseudopotential.

    Two interaction-force discretisations are available (``scheme``):

    ``"standard"`` (default, historical behaviour)
        F = −G ψ(x) Σᵢ wᵢ cᵢ ψ(x+cᵢ)  — the SC force sampled on the D2Q9
        stencil.  Its discrete curl is O(h²) ≠ 0, so it is *not* an exact
        discrete gradient and can generate spurious vorticity at interfaces.

    ``"pressure_tensor"`` (Ramshaw–Phathanapirom form)
        F = −∂·P with P_αβ = (G_eff/2) ψ² δ_αβ + …; evaluated here as
        (G·cs²/2)·D(ψ²) with **central differences** D.  Central differences
        on a rectangular grid commute, so the discrete curl is *identically
        zero* (no spurious vorticity) and Σₓ F = 0 exactly (exact momentum
        conservation / Newton's third law).  Leading-order calibration is
        identical to ``"standard"`` (same EOS / coexistence densities).

    Args:
        rho:         Density field, shape ``(ny, nx)``.
        G:           SC self-coupling constant (< 0 for attraction / phase separation).
        psi_fn:      Pseudopotential function ψ(ρ).  Defaults to :func:`psi_exp`.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        solid_mask:  Optional boolean mask of wall/solid cells.
        wall_psi:    Pseudopotential attributed to solid cells (see
                     :func:`_sc_neighbor_weighted_sum`); ``None`` = dry wall.
        scheme:      ``"standard"`` or ``"pressure_tensor"``.

    Returns:
        ``(Fx, Fy)`` each of shape ``(ny, nx)``.
    """
    psi = psi_fn(rho)
    if scheme == "standard":
        sum_x, sum_y = _sc_neighbor_weighted_sum(psi, solid_mask, wall_psi)
        Fx = -G * psi * sum_x + rho * gx
        Fy = -G * psi * sum_y + rho * gy
    elif scheme == "pressure_tensor":
        # Ramshaw–Phathanapirom divergence form on the *isotropic* D2Q9 stencil:
        #     F = −G_eff ψ ∇ψ = −(G_eff/2) ∇(ψ²)  →  −(G/2) Σᵢ wᵢ ψ²(x−cᵢ) cᵢ
        # (library sign G_lib = −G_eff).  Leading-order identical to the standard
        # branch, but the force is a symmetric divergence: Σₓ F ≡ 0 exactly
        # because Σᵢ wᵢ cᵢ = 0 → exact momentum conservation (Newton's third law).
        # This keeps the standard D2Q9 gather stencil (no central-difference
        # odd-even/checkerboard decoupling → numerically robust).
        sumsq_x, sumsq_y = _sc_neighbor_weighted_sum(psi, solid_mask, wall_psi, square=True)
        Fx = -0.5 * G * sumsq_x + rho * gx
        Fy = -0.5 * G * sumsq_y + rho * gy
    elif scheme == "pressure_tensor_cd":
        # Exactly curl-free variant: F = (G·cs²/2)·D(ψ²) with *central*
        # differences D.  Central differences commute on a rectangular grid, so
        # the discrete curl is identically zero (no interface spurious
        # vorticity) — the strictest reading of the pressure-tensor form.
        # WARNING: central differences on a colocated grid are prone to
        # odd-even (checkerboard) decoupling; this variant was found UNSTABLE on
        # the dam-break at a=80 (diverges ~step 320).  Kept for the record only.
        dpx, dpy = _psi_sq_central_gradient(psi, solid_mask, wall_psi)
        Fx = G * _CS2 * 0.5 * dpx + rho * gx
        Fy = G * _CS2 * 0.5 * dpy + rho * gy
    else:
        raise ValueError(
            f"unknown scheme {scheme!r}; expected 'standard', 'pressure_tensor' "
            f"or 'pressure_tensor_cd'"
        )
    return Fx, Fy


def collide_sc_single_component(
    f: torch.Tensor,
    G: float = -4.0,
    tau: float = 1.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
    gx: float = 0.0,
    gy: float = 0.0,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
    forcing: str = "velocity_shift",
    scheme: str = "standard",
) -> torch.Tensor:
    """Shan-Chen single-component multiphase (SCMP) BGK collision for D2Q9.

    Use G < 0 to generate attractive self-interaction and spontaneous
    liquid–gas phase separation.

    Two forcing schemes are available (selected by ``forcing``):

    ``"velocity_shift"`` (default, historical behaviour)
        The interaction/gravity force is folded into the *equilibrium velocity*:

            uᵉq = u + τ·F/ρ ,   f_out = f − (f − fᵉq(uᵉq))/τ

        This is the scheme used by the ``laplace_droplet`` benchmark.  On a
        curved interface it injects a *spurious* force whose magnitude depends
        on the local acceleration ``g`` (proved by the g-dependence of the
        simulated front), which biases the late-time dam-break front low.

    ``"guo"`` (Guo 2002 forcing, F as a source term)
        The force enters as an explicit source term and the physical velocity
        carries the half-force correction:

            u_phys = u + F/(2ρ)
            fᵉq    = fᵉq(u_phys)
            Sᵢ     = wᵢ (1 − 1/(2τ)) [ (cᵢ − u_phys)·F / cs²
                                      + (cᵢ·u_phys)(cᵢ·F) / cs⁴ ]
            f_out  = f − (f − fᵉq)/τ + S

        The standard mass-conserving source term (Σᵢ Sᵢ = 0) is used — the
        ``−(u_phys·F)/cs²`` contribution is essential; omitting it (as in early
        prototypes) leaks mass at ~10% over a dam-break run.  After the step
        the lattice momentum is ``ρ·u + F/2``, so the next ``macroscopic(f)``
        call returns ``u_phys − F/(2ρ)`` and the half-force correction is
        applied consistently each step.  This is the scheme that removes the
        interface spurious force and is the lever for the ``dam_break_sc``
        late-time front (see ``benchmarks/pending/dam_break_sc``).

    Args:
        f:           Distribution tensor, shape ``(9, ny, nx)``.
        G:           SC self-coupling constant (< 0 for phase separation).
        tau:         Relaxation time.
        psi_fn:      Pseudopotential callable.
        gx:          x body-force acceleration.
        gy:          y body-force acceleration.
        solid_mask:  Optional boolean mask ``(ny, nx)`` of solid/wall cells.
        wall_psi:    Pseudopotential value attributed to solid cells.  ``None``
                     (default) keeps the historical dry-wall behaviour; a
                     positive value gives a partially wetting wall.
        forcing:     ``"velocity_shift"`` (default) or ``"guo"``.
        scheme:      interaction-force discretisation, ``"standard"`` (default,
                     historical) or ``"pressure_tensor"`` (Ramshaw–Phathanapirom
                     form: exactly curl-free, exactly momentum-conserving).  The
                     two are independent: any ``forcing`` × ``scheme`` combination
                     is allowed.

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy = macroscopic(f)
    Fx, Fy = sc_single_component_force(rho, G, psi_fn, gx, gy, solid_mask, wall_psi, scheme)
    rho_s = torch.clamp(rho, min=1e-12)

    if forcing == "velocity_shift":
        feq = equilibrium(rho, ux + tau * Fx / rho_s, uy + tau * Fy / rho_s)
        f_out = f - (f - feq) / tau
    elif forcing == "guo":
        # Physical velocity includes the half-force correction u + F/(2ρ).
        uxp = ux + 0.5 * Fx / rho_s
        uyp = uy + 0.5 * Fy / rho_s
        feq = equilibrium(rho, uxp, uyp)
        device = f.device
        c = _c_on(device).float()
        w = _w_on(device).float()
        cx = c[:, 0].view(9, 1, 1)
        cy = c[:, 1].view(9, 1, 1)
        w3 = w.view(9, 1, 1)
        cf = cx * Fx.unsqueeze(0) + cy * Fy.unsqueeze(0)  # cᵢ·F
        cu = cx * uxp.unsqueeze(0) + cy * uyp.unsqueeze(0)  # cᵢ·u_phys
        uf = uxp * Fx + uyp * Fy  # u_phys·F
        # Standard Guo (2002) source term, Σᵢ Sᵢ = 0 (mass-conserving):
        #   Sᵢ = wᵢ (1 − 1/(2τ)) [ (cᵢ − u_phys)·F / cs² + (cᵢ·u_phys)(cᵢ·F) / cs⁴ ]
        S = w3 * (1.0 - 1.0 / (2.0 * tau)) * ((cf - uf.unsqueeze(0)) / _CS2 + cu * cf / _CS2**2)
        f_out = f - (f - feq) / tau + S
    else:
        raise ValueError(f"unknown forcing {forcing!r}; expected 'velocity_shift' or 'guo'")

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


def _grad_phase_field(
    rho_r: torch.Tensor,
    rho_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the phase-field gradient via second-order central differences.

    Returns ``(phi, nx_hat, ny_hat, mag)`` all of shape ``(ny, nx)``.
    The gradient is computed with periodic wrapping (boundaries are handled
    externally by bounce-back).  ``mag`` is returned so callers can reuse it
    without recomputing the four :func:`torch.roll` operations.
    """
    n = rho_r + rho_b
    n_safe = torch.clamp(n, min=1e-12)
    phi = (rho_r - rho_b) / n_safe

    # Central differences (periodic)
    dphi_dx = 0.5 * (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1))
    dphi_dy = 0.5 * (torch.roll(phi, -1, dims=-2) - torch.roll(phi, 1, dims=-2))

    mag = torch.sqrt(dphi_dx**2 + dphi_dy**2)
    mag_safe = torch.clamp(mag, min=1e-12)
    nx_hat = dphi_dx / mag_safe
    ny_hat = dphi_dy / mag_safe

    return phi, nx_hat, ny_hat, mag


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
    rho_r = f_r.sum(dim=0)  # (ny, nx)
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
    # Use masked densities to prevent spurious gradients at solid boundaries.
    # _grad_phase_field now returns mag directly, avoiding 4 extra torch.roll calls.
    rho_r_safe = rho_r if solid_mask is None else rho_r.masked_fill(solid_mask, 0.0)
    rho_b_safe = rho_b if solid_mask is None else rho_b.masked_fill(solid_mask, 0.0)
    _phi, nhat_x, nhat_y, mag = _grad_phase_field(rho_r_safe, rho_b_safe)

    # Perturbation: Δfᵢ = (A/2)|∇φ| wᵢ [(cᵢ·n̂)² − Bᵢ]
    # with Bᵢ = 1/3 for D2Q9 (isotropic contribution)
    ci_dot_n = cx * nhat_x.unsqueeze(0) + cy * nhat_y.unsqueeze(0)  # (9,ny,nx)
    B_iso = torch.tensor(1.0 / 3.0, device=device)
    w_view = w.view(9, 1, 1)
    perturbation = (A / 2.0) * mag.unsqueeze(0) * w_view * (ci_dot_n**2 - B_iso)
    f_post = f_post + perturbation

    # --- 4. Recoloring step (Latva-Kokko & Rothman 2005) ---
    # feq at zero velocity with unit density equals wᵢ, so reuse w_view directly
    # instead of allocating new tensors via equilibrium(ones, zeros, zeros).
    cos_theta = ci_dot_n
    recolor_amp = beta * (rho_r * rho_b / rho_s).unsqueeze(0) * cos_theta * w_view

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
        torch.roll(field, 1, dims=-1)
        + torch.roll(field, -1, dims=-1)
        + torch.roll(field, 1, dims=-2)
        + torch.roll(field, -1, dims=-2)
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
    mu = -A * phi + B * phi**3 - kappa * _laplacian_2d(phi)

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
    u_sq = (ux**2 + uy**2).unsqueeze(0)
    # Advection part: same form as feq for rho=phi
    geq_adv = w_v * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)
    # Diffusion part: Γ w_i (|c_i|² − D cs²) μ / cs⁴
    # |c_i|²/cs² for D2Q9: 0 for rest (i=0), 1 for face (i=1-4), 2 for diagonal (i=5-8)
    c_sq = cx**2 + cy**2  # |c_i|²
    diff_factor = c_sq / _CS2 - 2.0  # anisotropic, sums to 0 over w_i
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
    u_sq = (ux**2 + uy**2).unsqueeze(0)
    return w * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Pseudopotential functions
    "psi_linear",
    "psi_exp",
    "psi_power",
    "psi_carnahan_starling",
    "psi_peng_robinson",
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
