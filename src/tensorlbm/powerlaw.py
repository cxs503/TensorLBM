"""Power-law (generalised Newtonian) viscosity model for D2Q9.

Implements the variable-viscosity BGK collision for power-law fluids whose
rheology is described by the Ostwald–de Waele constitutive law

    τ = K · γ̇ⁿ          (shear stress)
    ν(γ̇) = K · γ̇^(n-1)   (local kinematic viscosity)

with K the consistency index, n the flow index, and γ̇ the local shear rate

    γ̇ = sqrt(2·S:S),   S_αβ = ½(∂_α u_β + ∂_β u_α)   (strain-rate tensor).

n = 1     → Newtonian fluid (ν = K constant).
n < 1     → shear-thinning  (viscosity decreases with shear rate).
n > 1     → shear-thickening (viscosity increases with shear rate).

At every cell the velocity gradients are estimated by **central differences**
(second-order accurate in the interior), the strain-rate tensor is assembled,
and the local relaxation time follows from the standard BGK relation

    τ_eff = 3·ν(γ̇) + ½ .

The collision then relaxes the population towards the local equilibrium with
the per-cell relaxation time, which is the standard lattice-Boltzmann
treatment of generalised-Newtonian fluids (e.g. OpenLB ``powerLaw2d``).

Notes on the implementation
---------------------------
* ``central_difference`` uses the interior stencil (f[i+1] − f[i−1])/2 and
  edge-replicated (clamped) one-sided stencils on the two boundary rows,
  which are solid/wall cells in channel setups and never enter the
  collision.  This differs from ``tensorlbm.non_newtonian``, which relies on
  ``torch.gradient`` (one-sided stencils at the edges, same interior).
* Because γ̇ → 0 at the centreline of a channel, ν diverges for n < 1 and
  vanishes for n > 1; the physically singular region is clipped with
  ``nu_min``/``nu_max`` (and ``tau_min``/``tau_max``) for stability.  With
  reasonable bounds the clipped region is confined to the plug-like core of
  the channel where the velocity error it introduces is O(0.1 %) (see the
  powerlaw_channel benchmark).
* ``apply_body_force_shift`` implements the Guo-style body-force scheme in
  its velocity-shift form, generalised to a per-cell relaxation time — the
  exact discretisation validated for the Newtonian case in
  ``tests/test_verification.py`` (steady profile u(y) = fₓ/(2ν)·y(ny−1−y)).
"""

from __future__ import annotations

import torch

from .d2q9 import C, W, equilibrium, macroscopic

__all__ = [
    "central_difference",
    "velocity_gradients_2d",
    "strain_rate_shear_rate_2d",
    "powerlaw_viscosity",
    "tau_from_viscosity",
    "collide_powerlaw_bgk",
    "guo_force_term",
    "collide_powerlaw_bgk_forced",
    "apply_body_force_shift",
]


def central_difference(field: torch.Tensor, dim: int) -> torch.Tensor:
    """∂field/∂x_dim by second-order central differences (unit spacing).

    Interior cells use (f[i+1] − f[i−1]) / 2.  The two boundary rows use an
    edge-replicated (clamped) stencil, i.e. a one-sided difference of unit
    spacing; those rows are solid/wall cells in the channel benchmarks and
    are never used by the collision.

    Args:
        field: Tensor of shape (ny, nx).
        dim:   0 → derivative along y, 1 → derivative along x.

    Returns:
        Tensor of the same shape as ``field``.
    """
    if dim == 0:
        fp = torch.cat([field[1:], field[-1:]], dim=0)  # f[i+1], last row replicated
        fm = torch.cat([field[:1], field[:-1]], dim=0)  # f[i-1], first row replicated
    elif dim == 1:
        fp = torch.cat([field[:, 1:], field[:, -1:]], dim=1)
        fm = torch.cat([field[:, :1], field[:, :-1]], dim=1)
    else:
        msg = f"dim must be 0 or 1, got {dim}"
        raise ValueError(msg)
    grad = (fp - fm) / 2.0
    # Boundary rows: replicated edge gives half-value one-sided step; fix with
    # true one-sided differences (f[1]-f[0] and f[-1]-f[-2], unit spacing).
    if dim == 0:
        grad[0] = field[1] - field[0]
        grad[-1] = field[-1] - field[-2]
    else:
        grad[:, 0] = field[:, 1] - field[:, 0]
        grad[:, -1] = field[:, -1] - field[:, -2]
    return grad


def velocity_gradients_2d(
    ux: torch.Tensor, uy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Velocity-gradient tensor ∂_β u_α via central differences.

    Returns (du_dx, du_dy, dv_dx, dv_dy) with du_dx = ∂uₓ/∂x, du_dy = ∂uₓ/∂y,
    dv_dx = ∂u_y/∂x, dv_dy = ∂u_y/∂y, each of shape (ny, nx).
    """
    du_dx = central_difference(ux, 1)
    du_dy = central_difference(ux, 0)
    dv_dx = central_difference(uy, 1)
    dv_dy = central_difference(uy, 0)
    return du_dx, du_dy, dv_dx, dv_dy


def strain_rate_shear_rate_2d(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Local shear rate γ̇ = sqrt(2·S:S) from the strain-rate tensor.

    S_αβ = ½(∂_α u_β + ∂_β u_α), so for a 2-D field

        S_xx = ∂uₓ/∂x,  S_yy = ∂u_y/∂y,  S_xy = ½(∂uₓ/∂y + ∂u_y/∂x)
        S:S  = S_xx² + S_yy² + 2·S_xy²
        γ̇    = sqrt(2·(S_xx² + S_yy² + 2·S_xy²))

    For a 1-D channel flow uₓ(y) this reduces to γ̇ = |∂uₓ/∂y|.

    Args:
        ux, uy: Velocity fields of shape (ny, nx).

    Returns:
        Shear-rate field of shape (ny, nx), γ̇ ≥ 0.
    """
    du_dx, du_dy, dv_dx, dv_dy = velocity_gradients_2d(ux, uy)
    s_xx = du_dx
    s_yy = dv_dy
    s_xy = 0.5 * (du_dy + dv_dx)
    second_invariant = s_xx * s_xx + s_yy * s_yy + 2.0 * s_xy * s_xy
    return torch.sqrt(torch.clamp(2.0 * second_invariant, min=0.0))


def powerlaw_viscosity(
    shear_rate: torch.Tensor,
    consistency_index: float,
    flow_index: float,
    nu_min: float | None = None,
    nu_max: float | None = None,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Local kinematic viscosity of a power-law fluid: ν(γ̇) = K·γ̇^(n−1).

    The shear rate is floored at ``shear_rate_floor`` before the power is
    taken (γ̇ = 0 would make the exponent blow up for n < 1).  The result is
    optionally clipped to [nu_min, nu_max].

    Args:
        shear_rate: Local shear-rate field γ̇ ≥ 0, shape (ny, nx).
        consistency_index: Consistency index K > 0 (lattice units).
        flow_index: Flow index n > 0.  n=1 Newtonian, n<1 shear-thinning,
            n>1 shear-thickening.
        nu_min: Optional lower viscosity bound (stability at γ̇ → 0 for n>1).
        nu_max: Optional upper viscosity bound (stability at γ̇ → 0 for n<1).
        shear_rate_floor: Small positive floor applied to γ̇.

    Returns:
        Viscosity field of shape (ny, nx).
    """
    if consistency_index <= 0.0:
        msg = "consistency_index must be > 0"
        raise ValueError(msg)
    if flow_index <= 0.0:
        msg = "flow_index must be > 0"
        raise ValueError(msg)
    if shear_rate_floor <= 0.0:
        msg = "shear_rate_floor must be > 0"
        raise ValueError(msg)

    gamma_safe = torch.clamp(shear_rate, min=shear_rate_floor)
    nu = consistency_index * torch.pow(gamma_safe, flow_index - 1.0)
    if nu_min is not None:
        nu = torch.clamp(nu, min=nu_min)
    if nu_max is not None:
        nu = torch.clamp(nu, max=nu_max)
    return nu


def tau_from_viscosity(
    nu: torch.Tensor,
    tau_min: float = 0.501,
    tau_max: float | None = None,
) -> torch.Tensor:
    """Local BGK relaxation time τ_eff = 3ν + ½, optionally clipped.

    Args:
        nu: Local viscosity field, shape (ny, nx).
        tau_min: Lower bound on τ (must be > 0.5, stability at high ν).
        tau_max: Optional upper bound (stability at very low ν).

    Returns:
        Relaxation-time field of shape (ny, nx).
    """
    if tau_min <= 0.5:
        msg = "tau_min must be > 0.5"
        raise ValueError(msg)
    if tau_max is not None and tau_max < tau_min:
        msg = "tau_max must be >= tau_min"
        raise ValueError(msg)
    tau = 3.0 * nu + 0.5
    tau = torch.clamp(tau, min=tau_min, max=tau_max if tau_max is not None else torch.inf)
    return tau


def collide_powerlaw_bgk(
    f: torch.Tensor,
    consistency_index: float,
    flow_index: float,
    nu_min: float = 1e-5,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    tau_field: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variable-viscosity BGK collision for power-law fluids (D2Q9).

    Pipeline per cell:
        1. macroscopic velocity (uₓ, u_y) from the populations,
        2. velocity gradients by central differences → strain-rate tensor S,
        3. shear rate γ̇ = sqrt(2·S:S),
        4. local viscosity ν(γ̇) = K·γ̇^(n−1) (clipped to [nu_min, nu_max]),
        5. local relaxation time τ_eff = 3ν + ½ (clipped to [tau_min, tau_max]),
        6. BGK relaxation towards the local equilibrium with τ_eff per cell.

    Args:
        f: Distribution tensor of shape (9, ny, nx).
        consistency_index: Power-law consistency index K > 0.
        flow_index: Power-law flow index n > 0.
        nu_min: Lower viscosity bound (default 1e-5).
        nu_max: Upper viscosity bound (default 0.3, τ ≤ 1.4).
        tau_min: Lower relaxation-time bound, must be > 0.5 (default 0.501).
        tau_max: Optional upper relaxation-time bound.
        tau_field: Optional precomputed per-cell τ_eff field (ny, nx).
            When given, steps 2–5 are skipped and this field is used
            directly (useful when the caller needs the same τ for the body
            force, as in the powerlaw_channel benchmark).

    Returns:
        Updated distribution tensor of the same shape.
    """
    if f.dim() != 3 or f.shape[0] != 9:
        msg = "f must have shape (9, ny, nx)"
        raise ValueError(msg)
    if tau_min <= 0.5:
        msg = "tau_min must be > 0.5"
        raise ValueError(msg)
    if tau_max is not None and tau_max < tau_min:
        msg = "tau_max must be >= tau_min"
        raise ValueError(msg)

    rho, ux, uy = macroscopic(f)
    tau = tau_field if tau_field is not None else tau_from_viscosity(
        powerlaw_viscosity(
            strain_rate_shear_rate_2d(ux, uy),
            consistency_index=consistency_index,
            flow_index=flow_index,
            nu_min=nu_min,
            nu_max=nu_max,
        ),
        tau_min=tau_min,
        tau_max=tau_max,
    )

    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau.unsqueeze(0)


def guo_force_term(
    f: torch.Tensor,
    ax: float | torch.Tensor,
    tau_field: torch.Tensor,
    ux_star: torch.Tensor | None = None,
) -> torch.Tensor:
    """Guo (2002) discrete body-force term for D2Q9 with per-cell τ.

    For a constant body force in +x with acceleration ``ax`` (force per unit
    mass, so the force density is G = ρ·a), the discrete force is

        F_i = w_i·(1 − 1/(2τ))·ρ·a·[ 3(c_x − u*_x) + 9·c_x·(c_i·u*) ]

    with c_s² = 1/3 and ``u*`` the force-corrected macroscopic velocity
    (see :func:`collide_powerlaw_bgk_forced`).  Combined with the collision
    towards the equilibrium at ``u*``, this injects exactly ρ·a of momentum
    per step — *independent of the local relaxation time* — so the recovered
    steady momentum balance is d(ν_eff·∂_y u)/dy = −a, which is the correct
    generalised-Newtonian balance (constant driving, viscosity-dependent
    response).  This is the standard force treatment in non-Newtonian LBM
    (e.g. OpenLB power-law setups).

    Args:
        f: Distribution tensor of shape (9, ny, nx).
        ax: Acceleration in +x (scalar or field of shape (ny, nx)).
        tau_field: Per-cell relaxation-time field (ny, nx).
        ux_star: Optional precomputed corrected velocity field (ny, nx).
            When omitted it is recomputed as ux + ax/2.

    Returns:
        Force contribution F_i of shape (9, ny, nx) (add to the collided f).
    """
    rho, ux, uy = macroscopic(f)
    ust = ux_star if ux_star is not None else ux + 0.5 * ax
    a_t = torch.as_tensor(ax, dtype=f.dtype, device=f.device)
    if a_t.ndim == 0:
        a_t = a_t.expand(ux.shape)
    c = C.to(f.device).to(f.dtype)
    w = W.to(f.device).to(f.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    cu = 3.0 * (cx * ust + cy * uy)  # 3·(c_i·u*), c_s² = 1/3
    pref = (1.0 - 0.5 / tau_field).unsqueeze(0)  # (1, ny, nx)
    return (
        w.view(9, 1, 1)
        * rho.unsqueeze(0)
        * a_t.unsqueeze(0)
        * pref
        * (3.0 * (cx - ust.unsqueeze(0)) + 9.0 * cx * cu)
    )


def collide_powerlaw_bgk_forced(
    f: torch.Tensor,
    ax: float | torch.Tensor,
    consistency_index: float,
    flow_index: float,
    nu_min: float = 1e-5,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    tau_field: torch.Tensor | None = None,
) -> torch.Tensor:
    """Power-law BGK collision + Guo (2002) body force (per-cell τ).

    One complete collide step for a pressure-gradient/body-force driven
    power-law flow:

        1. macroscopic velocity, velocity gradients (central differences),
           shear rate γ̇ = sqrt(2·S:S),
        2. local viscosity ν(γ̇) = K·γ̇^(n−1) and τ_eff = 3ν + ½ (clipped),
        3. force-corrected velocity u* = u + a·Δt/2 (Guo),
        4. BGK relaxation towards feq(ρ, u*) with per-cell τ,
        5. add the Guo discrete force F_i (momentum source ρ·a per step,
           independent of the local τ).

    Args:
        f: Distribution tensor of shape (9, ny, nx).
        ax: Body-force acceleration in +x (scalar or field).
        consistency_index: Power-law consistency index K > 0.
        flow_index: Power-law flow index n > 0.
        nu_min: Lower viscosity bound (default 1e-5).
        nu_max: Upper viscosity bound (default 0.3).
        tau_min: Lower relaxation-time bound, must be > 0.5 (default 0.501).
        tau_max: Optional upper relaxation-time bound.
        tau_field: Optional precomputed per-cell τ_eff (ny, nx); when given
            steps 1–2 are skipped.

    Returns:
        Updated distribution tensor of the same shape.
    """
    if f.dim() != 3 or f.shape[0] != 9:
        msg = "f must have shape (9, ny, nx)"
        raise ValueError(msg)

    rho, ux, uy = macroscopic(f)
    tau = tau_field if tau_field is not None else tau_from_viscosity(
        powerlaw_viscosity(
            strain_rate_shear_rate_2d(ux, uy),
            consistency_index=consistency_index,
            flow_index=flow_index,
            nu_min=nu_min,
            nu_max=nu_max,
        ),
        tau_min=tau_min,
        tau_max=tau_max,
    )
    ux_star = ux + 0.5 * ax
    feq = equilibrium(rho, ux_star, uy)
    fi = guo_force_term(f, ax, tau, ux_star=ux_star)
    return f - (f - feq) / tau.unsqueeze(0) + fi


def apply_body_force_shift(
    f: torch.Tensor,
    fx: float | torch.Tensor,
    tau_field: torch.Tensor,
) -> torch.Tensor:
    """Guo-style body force in velocity-shift form with per-cell τ.

    Implements the force discretisation validated for the Newtonian case in
    ``tests/test_verification.py`` (steady Poiseuille profile
    u(y) = fₓ/(2ν)·y(ny−1−y)):

        δu = 2·fₓ·τ_eff
        f ← f + (feq(ρ, u + δu) − feq(ρ, u))·(1 − 1/(2·τ_eff))

    .. warning::
        This form injects momentum ρ·fₓ·(2τ_eff − 1) = 6·ν_eff·fₓ per step,
        which depends on the **local** viscosity.  With a constant τ
        (Newtonian fluid) the source is uniform and the scheme reproduces the
        validated Poiseuille solution (with the effective force fₓ·(2τ−1));
        with the per-cell τ of a power-law fluid the source varies with
        ν(γ̇) and the steady state is corrupted (the profile becomes
        parabolic for any n).  Use :func:`collide_powerlaw_bgk_forced` /
        :func:`guo_force_term` for variable-viscosity flows.

    Args:
        f: Distribution tensor of shape (9, ny, nx).
        fx: Body-force density in +x (scalar or field of shape (ny, nx)).
        tau_field: Per-cell relaxation-time field (ny, nx).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy = macroscopic(f)
    fx_t = torch.as_tensor(fx, dtype=f.dtype, device=f.device)
    if fx_t.ndim == 0:
        fx_t = fx_t.expand(ux.shape)
    du = 2.0 * fx_t * tau_field
    feq_new = equilibrium(rho, ux + du, uy)
    feq_old = equilibrium(rho, ux, uy)
    coeff = (1.0 - 0.5 / tau_field).unsqueeze(0)
    return f + (feq_new - feq_old) * coeff
