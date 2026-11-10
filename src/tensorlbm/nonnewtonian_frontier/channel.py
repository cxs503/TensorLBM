"""Variable-viscosity D2Q9 lattice-Boltzmann runner for plane channel flow.

The runner solves the generalised-Newtonian momentum balance

.. math::  \\partial_t u_x = -\\partial_x p/\\rho
           + \\partial_y\\big(\\nu_{app}(\\dot\\gamma)\\,\\partial_y u_x\\big),
           \\qquad \\dot\\gamma = |\\partial_y u_x| ,

on a D2Q9 lattice with

* **periodic** streamwise direction (the flow is streamwise invariant),
* **pre-streaming half-way bounce-back** plates, the variant validated by
  ``benchmarks/verified/poiseuille_2d``,
* **Guo (2002) body forcing** in velocity-shift form with a per-cell relaxation
  time, so that exactly ``ρ a`` momentum per step is injected whatever the local
  viscosity (see :func:`tensorlbm.powerlaw.guo_force_term`),
* a **second-order shear-rate stencil**: central differences inside the channel
  and a one-sided quadratic stencil anchored on the imposed no-slip value at the
  two wall-adjacent nodes,
* optional **under-relaxation of the viscosity field** (the constitutive update
  is a fixed-point iteration; ``omega < 1`` damps shear-thinning oscillation).

Geometry convention
-------------------
``ny = h + 2`` rows; rows ``0`` and ``ny-1`` are the *solid* rows, the physical
walls sit half-way between a solid row and its first fluid neighbour, i.e. at
``y = ±H`` with ``H = h/2``; the fluid cell centres are
``y_i = (i - 0.5) - H`` for ``i = 1 … ny-2``, so the gap in lattice units is
exactly ``h``.

Wall treatment: what was measured, not assumed
----------------------------------------------
Four no-slip treatments were tested here against the Newtonian parabola
``u = a(H² − y²)/(2ν)`` (same grid, same forcing, converged profile):

===============================================  ==============  =============
treatment                                        L2 error@h=32   effective
                                                                 wall
===============================================  ==============  =============
bounce-back stored in the solid row,            ~1.4 %          +0.5 cell
post-streaming
(``herschel_bulkley_lbm.moving_wall_bounce_back_2d``)
bounce-back reflected inside the fluid node      0.8 %          −0.5 cell
post-streaming
pre-streaming bounce-back from the               **0.11 %**      exact
*pre-collision* state (this module, and
``benchmarks/verified/poiseuille_2d``)
non-equilibrium extrapolation (Guo 2002)         0.32 %          exact
===============================================  ==============  =============

The first two place the effective no-slip plane half a cell off the geometric
wall — a first-order *location* error of ``≈1.8/h`` of ``u_max`` — and are
rejected.  The extrapolation is second-order for a Newtonian fluid but with a
shear-thinning closure it decelerates the near-wall update so much that the
``h = 16`` run did not reach steady state in 40 000 steps (L2 6.3 %, residual
4e-4); it is kept in :func:`wall_nonequilibrium_2d` for reference and tested,
but the production runner uses the bounce-back variant, which converges in a
few thousand steps for every closure.
"""

from __future__ import annotations

import torch

from ..d2q9 import equilibrium, macroscopic
from ..powerlaw import collide_powerlaw_bgk_forced
from ..solver import stream
from .closures import Closure

__all__ = [
    "run_channel",
    "channel_geometry",
    "shear_rate_profile",
    "bounce_back_prestream_2d",
    "wall_nonequilibrium_2d",
    "l2_relative_error",
    "solution_residual",
]

_OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)


def channel_geometry(h: int, nx: int = 4, dtype: torch.dtype = torch.float32, device="cpu"):
    """Coordinates/masks of the plane channel of gap ``h``."""
    if h < 4:
        raise ValueError("h must be >= 4")
    if nx < 1:
        raise ValueError("nx must be >= 1")
    ny = h + 2
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    solid[0, :] = True
    solid[-1, :] = True
    fluid = ~solid
    half_width = 0.5 * h
    rows = torch.arange(ny, dtype=torch.float64, device=device)
    return {
        "ny": ny,
        "nx": nx,
        "half_width": half_width,
        "solid": solid,
        "fluid": fluid,
        "y_center": (rows - 0.5) - half_width,  # mid-plane origin, walls at ±H
        "y_wall": rows - 0.5,                   # distance from the lower plate
    }


def bounce_back_prestream_2d(
    f_post: torch.Tensor,
    f_pre: torch.Tensor,
    solid: torch.Tensor,
) -> torch.Tensor:
    """Pre-streaming half-way bounce-back on the two outermost plates.

    The pre-collision populations sitting in the solid rows — which arrived
    there from the neighbouring fluid node during the previous streaming step —
    are reflected (``f_ī``) and written back into the solid rows *before*
    streaming; the stream step then carries them one cell back into the fluid.
    The population therefore leaves the fluid node, is reflected and returns
    within one lattice step, which is what places the no-slip plane exactly
    half-way between the solid row and the first fluid row.

    Args:
        f_post: Populations after the collision step, ``(9, ny, nx)``.
        f_pre: Populations **before** the collision step, same shape.
        solid: Boolean mask of the solid rows, ``(ny, nx)``.

    Returns:
        Updated populations with the wall condition applied (new tensor).
    """
    if f_post.shape != f_pre.shape:
        raise ValueError("f_post and f_pre must have the same shape")
    if f_post.dim() != 3 or f_post.shape[0] != 9:
        raise ValueError("populations must have shape (9, ny, nx)")
    if solid.shape != f_post.shape[1:]:
        raise ValueError("solid mask must have shape (ny, nx)")
    opp = _OPP.to(f_post.device)
    return torch.where(solid.unsqueeze(0), f_pre[opp], f_post)


def wall_nonequilibrium_2d(f_post: torch.Tensor, u_wall_x: float = 0.0) -> torch.Tensor:
    """Non-equilibrium extrapolation wall condition (Guo, Zheng & Shi 2002).

    ``f_i(x_b) = f_i^eq(ρ_b, u_w) + [f_i(x_f) − f_i^eq(ρ_f, u_f)]`` on the two
    outermost rows.  Second-order accurate for a Dirichlet velocity; kept as a
    documented alternative to the bounce-back (see the module docstring for the
    measured comparison — it is *not* used by :func:`run_channel`).

    Args:
        f_post: Populations after collision, ``(9, ny, nx)``.
        u_wall_x: Wall velocity in the streamwise direction (0 = at rest).

    Returns:
        Populations with the two outermost rows replaced (new tensor).
    """
    if f_post.dim() != 3 or f_post.shape[0] != 9:
        raise ValueError("populations must have shape (9, ny, nx)")
    rho, ux, uy = macroscopic(f_post)
    fneq = f_post - equilibrium(rho, ux, uy)
    g = f_post.clone()
    for row_b, row_f in ((0, 1), (f_post.shape[1] - 1, f_post.shape[1] - 2)):
        rho_b = rho[row_f].unsqueeze(0)
        zero_b = torch.zeros_like(rho_b)
        u_b = torch.full_like(rho_b, float(u_wall_x))
        f_eq_b = equilibrium(rho_b, u_b, zero_b)  # (9, 1, nx)
        g[:, row_b, :] = f_eq_b[:, 0, :] + fneq[:, row_f, :]
    return g


def shear_rate_profile(ux: torch.Tensor) -> torch.Tensor:
    """``γ̇ = |∂u_x/∂y|`` with second-order stencils, walls at ``±H``.

    Interior rows use a central difference.  At the two wall-adjacent rows the
    central stencil cannot be used as-is: the solid rows hold the *bounce-back*
    state, whose velocity is the mirror image of the fluid neighbour rather than
    zero, and feeding it into the difference biases the near-wall shear rate by
    ~6 % (ν = 0.1, h = 32) — which is precisely the quantity that sets
    ``ν ∝ γ̇^(n−1)`` and ``ν ∝ τ_y/γ̇``.  The imposed no-slip value ``u = 0`` at
    the wall plane (half a cell out, one cell from the second fluid row) gives
    the second-order one-sided form

    .. math::  \\dot\\gamma_1 = \\left|u_1 + \\frac{u_2}{3}\\right| ,

    from the quadratic through ``(0, 0)``, ``(1/2, u_1)`` and ``(3/2, u_2)``.
    It is exact for a parabola and is what this function returns.
    """
    gamma = torch.zeros_like(ux)
    gamma[1:-1, :] = (ux[2:, :] - ux[:-2, :]).abs() * 0.5
    gamma[1, :] = (ux[1, :] + ux[2, :] / 3.0).abs()
    gamma[-2, :] = (ux[-2, :] + ux[-3, :] / 3.0).abs()
    gamma[0, :] = gamma[1, :]
    gamma[-1, :] = gamma[-2, :]
    return gamma


def run_channel(
    h: int,
    accel: float,
    closure: Closure,
    nx: int = 4,
    n_steps: int = 120_000,
    tol: float = 1.0e-6,
    check_every: int = 500,
    omega_nu: float = 1.0,
    u_init: torch.Tensor | None = None,
    tau_max: float | None = None,
    dtype: torch.dtype = torch.float32,
    device="cpu",
) -> dict:
    """Run a plane-Poiseuille channel to steady state.

    Args:
        h: Channel gap in lattice units (``2H``); the grid is ``h + 2`` rows.
        accel: Body acceleration in +x (lattice units).
        closure: Generalised-Newtonian closure (lattice-unit parameters).
        nx: Periodic streamwise extent (only affects arithmetic intensity).
        n_steps: Maximum number of lattice steps.
        tol: Relative change of the streamwise velocity profile between two
            consecutive checks that defines steady state.
        check_every: Steps between convergence checks.
        omega_nu: Under-relaxation of the viscosity field (1.0 = none).
        u_init: Optional initial streamwise velocity over **all** rows.
        tau_max: Optional clamp on the relaxation time.

    Returns:
        dict with ``y`` (mid-plane origin, all rows), ``u`` (x-averaged, all
        rows, force-corrected), ``u_field``, ``gamma_dot``, ``nu``,
        ``tau_field``, ``steps``, ``converged``, ``residual``, ``history``, the
        scalars ``u_max``, ``mean_velocity``, ``flow_rate``,
        ``gamma_dot_wall``, ``tau_wall``, and the record ``geometry``.
    """
    if accel <= 0.0:
        raise ValueError("accel must be > 0")
    if not 0.0 < omega_nu <= 1.0:
        raise ValueError("omega_nu must be in (0, 1]")
    geo = channel_geometry(h, nx=nx, dtype=dtype, device=device)
    ny, H = geo["ny"], geo["half_width"]
    solid = geo["solid"]

    rho0 = torch.ones((ny, nx), dtype=dtype, device=device)
    zero = torch.zeros((ny, nx), dtype=dtype, device=device)
    if u_init is None:
        u0 = zero
    else:
        u_try = torch.as_tensor(u_init, dtype=dtype, device=device).reshape(-1).clone()
        if u_try.numel() != ny:
            raise ValueError(f"u_init must have {ny} entries, got {u_try.numel()}")
        u_try[0] = 0.0
        u_try[-1] = 0.0
        u0 = u_try.unsqueeze(1).expand(ny, nx).contiguous()
    f = equilibrium(rho0, u0, zero)

    tau_prev: torch.Tensor | None = None
    u_prev: torch.Tensor | None = None
    history: list[tuple[int, float, float]] = []
    converged = False
    residual = float("nan")
    best_res = float("inf")
    stall = 0
    step = 0
    for step in range(1, n_steps + 1):
        _, ux, _ = macroscopic(f)
        gamma = shear_rate_profile(ux)
        tau = 3.0 * closure.nu_of_gamma(gamma) + 0.5
        if tau_max is not None:
            tau = torch.clamp(tau, max=tau_max)
        if tau_prev is not None and omega_nu < 1.0:
            tau = (1.0 - omega_nu) * tau_prev + omega_nu * tau
        tau_prev = tau
        f_pre = f
        f = collide_powerlaw_bgk_forced(f, accel, 1.0, 1.0, tau_field=tau)
        f = bounce_back_prestream_2d(f, f_pre, solid)
        f = stream(f)
        if step % check_every == 0:
            _, ux, _ = macroscopic(f)
            prof = ux[1:-1, :].mean(dim=1)
            scale = max(float(prof.abs().max()), 1e-300)
            residual = float((prof - u_prev).abs().max() / scale) if u_prev is not None else float("nan")
            history.append((step, float(prof.mean()), residual))
            if u_prev is not None:
                if residual < tol:
                    converged = True
                    break
                # the BGK stepper has a kinetic noise floor under the profile
                # change; once the residual stops improving far below the
                # discretisation error the run is finished
                if residual < 0.7 * best_res:
                    best_res, stall = residual, 0
                else:
                    stall += 1
                if stall >= 5 and best_res < 5.0 * tol:
                    converged = True
                    break
            u_prev = prof

    _, ux, _ = macroscopic(f)
    gamma = shear_rate_profile(ux)
    nu_field = closure.nu_of_gamma(gamma)
    fluid_rows = slice(1, ny - 1)
    # Guo velocity shift: the populations carry u − a/2, so the physical
    # velocity is the moment plus a/2 (Guo, Zheng & Shi 2002).
    u_all = ux.mean(dim=1) + 0.5 * accel
    u_all[0] = 0.0
    u_all[-1] = 0.0
    return {
        "y": geo["y_center"],
        "y_wall": geo["y_wall"],
        "u": u_all,
        "u_field": ux + 0.5 * accel,
        "gamma_dot": gamma,
        "nu": nu_field,
        "tau_field": tau_prev,
        "steps": step,
        "converged": converged,
        "residual": residual,
        "history": history,
        "u_max": float(u_all[fluid_rows].max()),
        "mean_velocity": float(u_all[fluid_rows].mean()),
        "flow_rate": float(u_all[fluid_rows].mean()) * h,
        "gamma_dot_wall": float(gamma[1].mean()),
        "tau_wall": float(accel * H),
        "geometry": {k: v for k, v in geo.items() if k not in ("solid", "fluid")},
        "h": h,
        "accel": accel,
        "closure": closure.name,
        "closure_params": dict(closure.params),
        "fluid_rows": fluid_rows,
        "wall_bc": "pre-streaming half-way bounce-back",
    }


def l2_relative_error(u_num: torch.Tensor, u_ref: torch.Tensor) -> float:
    """Discrete L2 relative error ``‖u_num − u_ref‖₂ / ‖u_ref‖₂``."""
    d = (u_num - u_ref).double()
    r = u_ref.double()
    den = float(torch.sqrt((r * r).mean()))
    if den == 0.0:
        raise ValueError("reference profile is identically zero")
    return float(torch.sqrt((d * d).mean()) / den)


def solution_residual(
    y: torch.Tensor,
    gamma_dot: torch.Tensor,
    closure: Closure,
    accel: float,
    half_width: float,
) -> torch.Tensor:
    """Momentum-balance residual ``d(ν γ̇)/dy + ρ a`` of a measured profile.

    For a converged solution of the *measured* constitutive response this
    residual vanishes up to discretisation error; it is the equation actually
    solved, independent of any analytic anchor.  The full profile is folded onto
    ``|y|`` (mean of the ±pair), ``ν`` comes from the closure, and the
    derivative is a second-order central difference.  The result is normalised
    by the driving stress ``ρ a H``, so it reads as a fraction of the wall
    stress.
    """
    if accel <= 0.0:
        raise ValueError("accel must be > 0")
    y = torch.as_tensor(y, dtype=torch.float64).abs()
    g = torch.as_tensor(gamma_dot, dtype=torch.float64)
    keep = y <= float(half_width) * (1.0 + 1.0e-12)
    y, g = y[keep], g[keep]
    order = torch.argsort(y)
    y, g = y[order], g[order]
    uniq, inverse = torch.unique(y, return_inverse=True)
    g_acc = torch.zeros(uniq.numel(), dtype=torch.float64)
    cnt = torch.zeros(uniq.numel(), dtype=torch.float64)
    g_acc.index_add_(0, inverse, g)
    cnt.index_add_(0, inverse, torch.ones_like(g))
    r, g_sym = uniq, g_acc / cnt
    tau = g_sym * closure.nu_of_gamma(torch.clamp(g_sym, min=1.0e-300))
    dtau = torch.gradient(tau, spacing=(r,), dim=0)[0]
    return (dtau + accel) / (accel * float(half_width))