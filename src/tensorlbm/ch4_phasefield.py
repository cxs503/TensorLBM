"""CH4: single-distribution Cahn-Hilliard (fourth-order) phase-field LBE.

He-Shan-Doolen proportional-control scheme for a binary (liquid/gas) phase
field, coupled to a D3Q19 incompressible fluid.

Model
-----
* Order parameter ``phi`` in ``[-1, 1]`` (``+1`` liquid, ``-1`` gas).
* Free energy ``F = int [kappa |grad phi|^2 / 2 + (A/4) (phi^2 - 1)^2] dV``.
* Chemical potential ``mu = dF/dphi = A (phi^3 - phi) - kappa laplacian(phi)``.
* Cahn-Hilliard (fourth order in space)::

      dphi/dt = M laplacian(mu)

  carried by a **single** D3Q19 distribution ``g_i`` whose first moment is the
  diffusive flux ``J = -M grad mu``.  The equilibrium is the He-Shan-Doolen
  *proportional control* closure (the non-equilibrium part is proportional to
  the flux, no integral term)::

      g_i^eq = w_i [ 1 + (c_i . J)/c_s^2 + (c_i . J)^2 / (2 c_s^4) ]

  so that ``sum g_i = 1`` and ``sum c_i g_i = J``.
* Fluid: D3Q19 BGK at a **constant** reference density ``rho0`` (incompressible
  form) with the chemical-potential (surface-tension) force
  ``F = -force_scale * rho_phase grad mu + rho_phase g`` (proportional to the
  phase-field gradient) and the phase density
  ``rho_phase = rho_g + (rho_l - rho_g) (phi + 1) / 2``.  The force is folded
  into the equilibrium through the shifted velocity.

The module is device agnostic (CPU and the SDAA accelerator) and uses only
``roll``-based central differences plus elementwise arithmetic.
"""

from __future__ import annotations

import functools

import torch

from .d3q19 import C, W_EXACT64

# (dz, dy, dx) pull shifts for the 19 directions, field order (z, y, x).
_SHIFTS = tuple((int(C[q, 2]), int(C[q, 1]), int(C[q, 0])) for q in range(19))


@functools.cache
def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


@functools.cache
def _w_on(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return W_EXACT64.to(device=device, dtype=dtype)


def central_grad(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order central gradient (periodic) of a scalar field."""
    gx = (field.roll(-1, dims=2) - field.roll(1, dims=2)) * 0.5
    gy = (field.roll(-1, dims=1) - field.roll(1, dims=1)) * 0.5
    gz = (field.roll(-1, dims=0) - field.roll(1, dims=0)) * 0.5
    return gx, gy, gz


def laplacian(field: torch.Tensor) -> torch.Tensor:
    """Second-order central Laplacian (periodic) of a scalar field."""
    return (
        field.roll(-1, dims=2) + field.roll(1, dims=2)
        + field.roll(-1, dims=1) + field.roll(1, dims=1)
        + field.roll(-1, dims=0) + field.roll(1, dims=0)
        - 6.0 * field
    )


def divergence(jx: torch.Tensor, jy: torch.Tensor, jz: torch.Tensor) -> torch.Tensor:
    """Second-order central divergence (periodic) of a vector field."""
    return (
        (jx.roll(-1, dims=2) - jx.roll(1, dims=2)) * 0.5
        + (jy.roll(-1, dims=1) - jy.roll(1, dims=1)) * 0.5
        + (jz.roll(-1, dims=0) - jz.roll(1, dims=0)) * 0.5
    )


def chemical_potential(phi: torch.Tensor, A: float, kappa: float) -> torch.Tensor:
    """Chemical potential ``mu = A (phi^3 - phi) - kappa laplacian(phi)``."""
    return A * (phi * phi * phi - phi) - kappa * laplacian(phi)


def phase_density(phi: torch.Tensor, rho_g: float, rho_l: float) -> torch.Tensor:
    """Phase density ``rho = rho_g + (rho_l - rho_g) (phi + 1) / 2``."""
    return rho_g + (rho_l - rho_g) * (phi + 1.0) * 0.5


def ch4_equilibrium(jx: torch.Tensor, jy: torch.Tensor, jz: torch.Tensor) -> torch.Tensor:
    """He-Shan-Doolen proportional-control equilibrium for the flux carrier.

    ``g_i^eq = w_i [1 + 3 (c_i . J) + 7.5 (c_i . J)^2]`` for D3Q19.
    """
    device = jx.device
    c = _c_on(device)
    w = _w_on(device, jx.dtype).view(19, 1, 1, 1)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * jx
        + c[:, 1].view(19, 1, 1, 1) * jy
        + c[:, 2].view(19, 1, 1, 1) * jz
    )
    return w * (1.0 + 3.0 * cu + 7.5 * cu * cu)


def _stream19(dist: torch.Tensor) -> torch.Tensor:
    """D3Q19 periodic streaming (pull scheme) via ``torch.roll``."""
    out = torch.empty_like(dist)
    for q in range(19):
        sz, sy, sx = _SHIFTS[q]
        if sz == 0 and sy == 0 and sx == 0:
            out[q] = dist[q]
        else:
            out[q] = torch.roll(dist[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def ch4_flux(mu: torch.Tensor, M: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diffusive flux ``J = -M grad mu``."""
    gx, gy, gz = central_grad(mu)
    return -M * gx, -M * gy, -M * gz


def ch4_step(
    g: torch.Tensor,
    phi: torch.Tensor,
    A: float,
    kappa: float,
    M: float,
    tau_g: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Cahn-Hilliard (fourth-order) single-distribution step.

    The conservative ``phi`` update is driven by the target flux
    ``J = -M grad mu``; the single distribution ``g`` carries the LBE
    structure (its first moment relaxes toward ``J``).
    """
    mu = chemical_potential(phi, A, kappa)
    jx, jy, jz = ch4_flux(mu, M)
    g_eq = ch4_equilibrium(jx, jy, jz)
    g = g - (g - g_eq) / tau_g
    g = _stream19(g)
    phi = phi - divergence(jx, jy, jz)
    return g, phi


def surface_force(
    rho_phase: torch.Tensor, mu: torch.Tensor, force_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chemical-potential (surface-tension) force ``-force_scale * rho_phase grad mu``."""
    gx, gy, gz = central_grad(mu)
    return (
        -force_scale * rho_phase * gx,
        -force_scale * rho_phase * gy,
        -force_scale * rho_phase * gz,
    )


def ch4_fluid_step(
    f: torch.Tensor,
    rho_phase: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    tau_f: float,
    rho0: float = 1.0,
) -> torch.Tensor:
    """One D3Q19 BGK fluid step with an external force (shifted-velocity form).

    The fluid density is the actual distribution moment ``rho_f = sum f``; the
    force is folded into the equilibrium through the shifted velocity
    ``u = (mom + F/2) / rho_f``.
    """
    device = f.device
    c = _c_on(device)
    w = _w_on(device, f.dtype).view(19, 1, 1, 1)
    rho_f = f.sum(dim=0)
    mom_x = (f * c[:, 0].view(19, 1, 1, 1)).sum(dim=0)
    mom_y = (f * c[:, 1].view(19, 1, 1, 1)).sum(dim=0)
    mom_z = (f * c[:, 2].view(19, 1, 1, 1)).sum(dim=0)
    rho_safe = torch.clamp(rho_f, min=1e-12)
    ux = (mom_x + 0.5 * fx) / rho_safe
    uy = (mom_y + 0.5 * fy) / rho_safe
    uz = (mom_z + 0.5 * fz) / rho_safe
    u_sq = ux * ux + uy * uy + uz * uz
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux
        + c[:, 1].view(19, 1, 1, 1) * uy
        + c[:, 2].view(19, 1, 1, 1) * uz
    )
    f_eq = w * rho_f.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0)
    )
    f = f - (f - f_eq) / tau_f
    f = _stream19(f)
    return f


def ch4_init(
    phi: torch.Tensor,
    A: float,
    kappa: float,
    M: float,
    rho_g: float,
    rho_l: float,
    rho0: float = 1.0,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
    uz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise the fluid ``f`` and flux carrier ``g`` from a phase field.

    ``f`` is the D3Q19 equilibrium at the constant reference density ``rho0``
    and the given velocity (zero by default); ``g`` is the flux-carrier
    equilibrium at the initial flux ``J = -M grad mu``.
    """
    device = phi.device
    if ux is None:
        ux = torch.zeros_like(phi)
    if uy is None:
        uy = torch.zeros_like(phi)
    if uz is None:
        uz = torch.zeros_like(phi)
    c = _c_on(device)
    w = _w_on(device, phi.dtype).view(19, 1, 1, 1)
    u_sq = ux * ux + uy * uy + uz * uz
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux
        + c[:, 1].view(19, 1, 1, 1) * uy
        + c[:, 2].view(19, 1, 1, 1) * uz
    )
    f = w * rho0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
    mu = chemical_potential(phi, A, kappa)
    jx, jy, jz = ch4_flux(mu, M)
    g = ch4_equilibrium(jx, jy, jz)
    return f, g


def ch4_coupled_step(
    f: torch.Tensor,
    g: torch.Tensor,
    phi: torch.Tensor,
    A: float,
    kappa: float,
    M: float,
    tau_g: float,
    tau_f: float,
    gz: float,
    force_scale: float = 1.0,
    rho_g: float = 1.0,
    rho_l: float = 1000.0,
    rho0: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One coupled CH4 phase-field + D3Q19 fluid step.

    Order: (1) CH step updating ``g`` and ``phi``; (2) recompute ``mu`` and the
    phase density from the new ``phi``; (3) chemical-potential + gravity force
    (scaled by the phase density); (4) fluid step at the constant ``rho0``.
    """
    g, phi = ch4_step(g, phi, A, kappa, M, tau_g)
    mu = chemical_potential(phi, A, kappa)
    rho_phase = phase_density(phi, rho_g, rho_l)
    fx, fy, fz = surface_force(rho_phase, mu, force_scale)
    fz = fz + rho_phase * gz
    f = ch4_fluid_step(f, rho_phase, fx, fy, fz, tau_f, rho0)
    return f, g, phi
