"""D2Q9 free-surface LBM with an exactly conservative VOF mass ledger (batched).

Why not the repository's Körner kernels
---------------------------------------
``free_surface_lbm.free_surface_step`` (D3Q19) and
``free_surface_lbm_2d.free_surface_step_2d`` (D2Q9) are the production
free-surface models, but both need 10²–10³ torch ops per call, so their cost is
per-call overhead dominated (measured: 27 ms for a 560-cell tile, 1570 ms for
35 840 cells — i.e. call overhead rather than work), which makes them unusable as
the inner kernel of a 15–45-call-per-step AMR hierarchy.  They also keep two
independent mass representations (LBM density on ``LIQUID`` cells, ledger mass on
``INTERFACE`` cells), so a *representation switch* (restriction/prolongation
across refinement levels) moves the conserved quantity — measured 1–11 % of the
total mass per AMR step at the verification geometry.

This module keeps a **single** conserved scalar per cell (``mass``), which makes
the AMR exchange the only possible source of mass error, and fits in ~120 torch
ops so tiles stay cheap.

Model
-----
Momentum: D2Q9 BGK + Guo gravity on ``LIQUID``/``INTERFACE`` cells, half-way
bounce-back on solid, fill-weighted anti-bounce-back reconstruction on every gas
link (Körner et al. 2005 gas-link rule with the gas-side weight ``1-ε``), all on
a batched ``(..., 9, ny, nx)`` layout so every refinement level costs one call.

Interface: the liquid volume fraction ``ε = m/ρ_l`` advanced by a finite-volume,
flux-form VOF advection with second-order limited (minmod) interface values and a
donor-cell bound, driven by the LBM velocity extrapolated one cell into the dry
region.  Every face flux leaves one cell and enters the other, so ``Σm`` is
conserved to round-off; the limiter keeps ``0 ≤ ε ≤ 1`` while ``Σ|u| < 1``.

Known limitation (measured, see ``docs/AMR_FREE_SURFACE_REPORT.md``): the split
between the LBM momentum field and the VOF mass field means the surface is not a
true zero-pressure boundary, so the collapse is slower than the shallow-water
reference.  It is adequate for what this module is used for — an AMR verification
face where the *adaptive* result must reproduce the *fixed-grid* result of the
same kernel — but it is not a replacement for the Körner model.

References
----------
Körner C., Thürey N., Ulrich R. (2005) *Lattice Boltzmann model for free surface
flow for modeling foaming.* J. Comput. Phys. 202, 68–89.
Thürey N. (2007) *A lattice Boltzmann method for single-phase free surface
flows.* PhD thesis, Universität Erlangen-Nürnberg.
"""

from __future__ import annotations

import torch

from .d2q9 import C, OPPOSITE, W

__all__ = ["free_surface_vof_step", "GAS", "LIQUID", "INTERFACE", "SOLID"]

GAS = 0
LIQUID = 1
INTERFACE = 2
SOLID = 3

_CX = [int(C[q, 0]) for q in range(9)]
_CY = [int(C[q, 1]) for q in range(9)]
_OPP = torch.tensor([int(OPPOSITE[q]) for q in range(9)])
_W = W.to(torch.float32)
_CX_T = torch.tensor([float(c) for c in _CX])
_CY_T = torch.tensor([float(c) for c in _CY])
_CXW3 = torch.tensor([3.0 * float(c) * float(w) for c, w in zip(_CX, W)])
_CYW3 = torch.tensor([3.0 * float(c) * float(w) for c, w in zip(_CY, W)])


def _roll(field: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Shift so that the value at ``x`` comes from ``x − (dx, dy)``."""
    if dx == 0 and dy == 0:
        return field
    return torch.roll(field, shifts=(-dy, -dx), dims=(-2, -1))


def _dir_view(vec: torch.Tensor, ndim: int) -> torch.Tensor:
    return vec.view(*([1] * (ndim - 3)), 9, 1, 1)


def _equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium on batched ``(..., ny, nx)`` fields."""
    ndim = rho.dim() + 1
    u_sq = ux * ux + uy * uy
    cu = _dir_view(_CX_T, ndim) * ux.unsqueeze(-3) + _dir_view(_CY_T, ndim) * uy.unsqueeze(-3)
    return (
        _dir_view(_W, ndim)
        * rho.unsqueeze(-3)
        * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(-3))
    )


def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(
        a * b > 0.0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a)
    )


def _dilate4(mask: torch.Tensor) -> torch.Tensor:
    out = mask.clone()
    out[..., 1:, :] |= mask[..., :-1, :]
    out[..., :-1, :] |= mask[..., 1:, :]
    out[..., :, 1:] |= mask[..., :, :-1]
    out[..., :, :-1] |= mask[..., :, 1:]
    return out


def _face_flux_axis(
    eps: torch.Tensor, vel: torch.Tensor, solid: torch.Tensor, axis: int
) -> torch.Tensor:
    """Conservative, positivity-preserving flux through the high-side faces of *axis*."""
    if axis == 0:
        e_l, e_r = eps[..., :-1, :], eps[..., 1:, :]
        v = 0.5 * (vel[..., :-1, :] + vel[..., 1:, :])
        e_ll = torch.roll(eps, shifts=1, dims=-2)[..., :-1, :]
        e_rr = torch.roll(eps, shifts=-1, dims=-2)[..., 1:, :]
        s_l, s_r = solid[..., :-1, :], solid[..., 1:, :]
    else:
        e_l, e_r = eps[..., :, :-1], eps[..., :, 1:]
        v = 0.5 * (vel[..., :, :-1] + vel[..., :, 1:])
        e_ll = torch.roll(eps, shifts=1, dims=-1)[..., :, :-1]
        e_rr = torch.roll(eps, shifts=-1, dims=-1)[..., :, 1:]
        s_l, s_r = solid[..., :, :-1], solid[..., :, 1:]
    hi_left = e_l + 0.5 * _minmod(e_l - e_ll, e_r - e_l)
    hi_right = e_r - 0.5 * _minmod(e_r - e_l, e_rr - e_r)
    face = torch.where(v > 0.0, hi_left, hi_right)
    face = face.clamp(min=torch.minimum(e_l, e_r), max=torch.maximum(e_l, e_r))
    # donor bound: a cell never gives away more than it holds at that face
    face = torch.where(v > 0.0, torch.minimum(face, e_l), torch.minimum(face, e_r))
    flux = v * face
    return torch.where(s_l | s_r, torch.zeros_like(flux), flux)


def free_surface_vof_step(
    f: torch.Tensor,
    mass: torch.Tensor,
    flags: torch.Tensor,
    solid: torch.Tensor,
    *,
    tau: float = 0.8,
    gravity: tuple[float, float] = (0.0, 0.0),
    rho_liquid: float = 1.0,
    rho_gas: float = 1e-3,
    u_max: float = 0.2,
):
    """One free-surface (LBM + conservative VOF) timestep.

    Args:
        f: populations ``(..., 9, ny, nx)`` (direction axis ``-3``).
        mass: liquid mass per cell ``(..., ny, nx)`` — the conserved quantity.
        flags: ``GAS/LIQUID/INTERFACE/SOLID`` ``(..., ny, nx)``.
        solid: solid mask ``(..., ny, nx)``.
        tau: BGK relaxation time on the fluid cells.
        gravity: lattice acceleration ``(gx, gy)`` (applied to the fluid cells).
        rho_liquid: bulk liquid density.
        rho_gas: gas density on the gas side of a surface link.
        u_max: velocity clamp (incompressibility guard).

    Returns:
        ``(f, mass, flags)``.
    """
    gx, gy = gravity
    fluid = (flags == LIQUID) | (flags == INTERFACE)
    gas = flags == GAS
    iface = flags == INTERFACE
    fluid_d = fluid.unsqueeze(-3)
    zero = torch.zeros_like(f)
    cx = _dir_view(_CX_T, f.dim())
    cy = _dir_view(_CY_T, f.dim())
    f = torch.where(fluid_d, f, zero)

    # ---- 1. macroscopic state -------------------------------------------
    rho = f.sum(dim=-3)
    rho_f = torch.where(fluid, rho.clamp(min=0.2 * rho_liquid), rho_liquid * torch.ones_like(rho))
    ux = (f * cx).sum(dim=-3) / rho_f
    uy = (f * cy).sum(dim=-3) / rho_f
    ux = torch.where(fluid, ux, torch.zeros_like(ux)).clamp(-u_max, u_max)
    uy = torch.where(fluid, uy, torch.zeros_like(uy)).clamp(-u_max, u_max)

    # ---- 2. fill-weighted gas-link reconstruction -----------------------
    f_eq_l = _equilibrium(torch.full_like(ux, rho_liquid), ux, uy)
    f_eq_g = _equilibrium(torch.full_like(ux, rho_gas), ux, uy)
    f_opp = f[..., _OPP, :, :]
    eps_c = (mass / rho_liquid).clamp(0.0, 1.0).unsqueeze(-3)
    for q in range(1, 9):
        hit = (_roll(gas, -_CX[q], -_CY[q]) & iface).unsqueeze(-3)
        mirror = (
            f_eq_g[..., q : q + 1, :, :]
            + f_eq_l[..., _OPP[q] : _OPP[q] + 1, :, :]
            - f_opp[..., q : q + 1, :, :]
        )
        blended = (1.0 - eps_c) * f_eq_g[..., q : q + 1, :, :] + eps_c * mirror
        f[..., q : q + 1, :, :] = torch.where(hit, blended, f[..., q : q + 1, :, :])

    # ---- 3. collision (BGK + Guo source) --------------------------------
    rho = f.sum(dim=-3)
    rho_f = torch.where(fluid, rho.clamp(min=0.2 * rho_liquid), rho_liquid * torch.ones_like(rho))
    ux = (f * cx).sum(dim=-3) / rho_f + 0.5 * gx
    uy = (f * cy).sum(dim=-3) / rho_f + 0.5 * gy
    ux = torch.where(fluid, ux, torch.zeros_like(ux)).clamp(-u_max, u_max)
    uy = torch.where(fluid, uy, torch.zeros_like(uy)).clamp(-u_max, u_max)
    f_eq = _equilibrium(rho_f, ux, uy)
    guo = (1.0 - 0.5 / tau) * rho_f.unsqueeze(-3)
    force = guo * (_dir_view(_CXW3, f.dim()) * gx + _dir_view(_CYW3, f.dim()) * gy)
    f_post = torch.where(fluid_d, f - (f - f_eq) / tau + force, zero)

    # ---- 4. streaming with half-way bounce-back -------------------------
    f_new = torch.empty_like(f_post)
    f_new[..., 0, :, :] = f_post[..., 0, :, :]
    f_opp_post = f_post[..., _OPP, :, :]
    for q in range(1, 9):
        arrived = _roll(f_post[..., q : q + 1, :, :], _CX[q], _CY[q])
        src_solid = _roll(solid, _CX[q], _CY[q]).unsqueeze(-3)
        f_new[..., q : q + 1, :, :] = torch.where(
            src_solid, f_opp_post[..., q : q + 1, :, :], arrived
        )
    f_new = torch.where(fluid_d, f_new, zero)

    # ---- 5. VOF advection (flux form, exactly conservative) -------------
    ux_fluid = torch.where(fluid, ux, torch.zeros_like(ux))
    uy_fluid = torch.where(fluid, uy, torch.zeros_like(uy))
    fluid_f = fluid.to(torch.float32)
    cnt = (
        _roll(fluid_f, 1, 0) + _roll(fluid_f, -1, 0) + _roll(fluid_f, 0, 1) + _roll(fluid_f, 0, -1)
    )
    gx_sum = (
        _roll(ux_fluid, 1, 0) + _roll(ux_fluid, -1, 0) + _roll(ux_fluid, 0, 1) + _roll(ux_fluid, 0, -1)
    )
    gy_sum = (
        _roll(uy_fluid, 1, 0) + _roll(uy_fluid, -1, 0) + _roll(uy_fluid, 0, 1) + _roll(uy_fluid, 0, -1)
    )
    inv = torch.where(cnt > 0, 1.0 / cnt.clamp(min=1.0), torch.zeros_like(cnt))
    ux_v = torch.where(fluid, ux, gx_sum * inv)
    uy_v = torch.where(fluid, uy, gy_sum * inv)
    eps = mass / rho_liquid
    flux_y = _face_flux_axis(eps, uy_v, solid, 0)
    flux_x = _face_flux_axis(eps, ux_v, solid, 1)
    mass_new = mass.clone()
    mass_new[..., :-1, :] -= flux_y
    mass_new[..., 1:, :] += flux_y
    mass_new[..., :, :-1] -= flux_x
    mass_new[..., :, 1:] += flux_x
    negative = (mass_new < 0.0) & ~solid
    if bool(negative.any()):
        clipped = torch.where(negative, -mass_new, torch.zeros_like(mass_new)).sum()
        mass_new = mass_new.clamp(min=0.0)
        room = fluid & ~solid
        pool = mass_new[room].sum().clamp(min=1e-30)
        mass_new = mass_new * (1.0 + clipped / pool)

    # ---- 6. flags ------------------------------------------------------
    eps_new = mass_new / rho_liquid
    newflags = torch.full_like(flags, GAS)
    newflags[eps_new >= 1.0 - 1e-6] = LIQUID
    newflags[(eps_new > 1e-6) & (eps_new < 1.0 - 1e-6)] = INTERFACE
    envelope = _dilate4(newflags == LIQUID)
    newly_wet = (flags == GAS) & ~(newflags == GAS)
    newflags[(newflags == GAS) & envelope & ~solid] = INTERFACE
    newflags[solid] = SOLID
    if bool(newly_wet.any()):
        f_new = torch.where(
            newly_wet.unsqueeze(-3),
            _equilibrium(
                torch.full_like(ux, rho_liquid), torch.zeros_like(ux), torch.zeros_like(uy)
            ),
            f_new,
        )
    return f_new, mass_new, newflags