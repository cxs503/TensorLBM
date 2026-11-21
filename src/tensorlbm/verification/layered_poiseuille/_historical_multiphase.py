"""Frozen compatibility kernels for the migrated layered-Poiseuille validation.

These functions retain the historical validation implementation and are not the
production multiphase API. They can be retired after the archived validation is
requalified against the canonical kernels.
"""
from __future__ import annotations

from typing import Callable

import torch

from ...d2q9 import C, W, equilibrium, macroscopic

CS2 = 1.0 / 3.0

OPPOSITE = [0, 3, 4, 1, 2, 7, 8, 5, 6]
_C_INT = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
WALL_MODES = ("mirror", "dry")

def psi_exp(rho: torch.Tensor, rho0: float = 1.0) -> torch.Tensor:
    """Shan-Chen pseudopotential psi(rho) = rho0 * (1 - exp(-rho/rho0))."""
    return rho0 * (1.0 - torch.exp(-rho / rho0))


def _sc_neighbor_sum(
    psi: torch.Tensor,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sum_i w_i * psi(x + c_i) * c_i for the SC force."""
    device = psi.device
    ny, nx = psi.shape
    c = C.to(device)
    w = W.to(device)

    # Apply wall pseudopotential if solid mask provided
    if solid_mask is not None and wall_psi is not None:
        psi = psi.masked_fill(solid_mask, float(wall_psi))
    elif solid_mask is not None:
        psi = psi.masked_fill(solid_mask, 0.0)

    fx = torch.zeros_like(psi)
    fy = torch.zeros_like(psi)

    for i in range(9):
        cx, cy = int(c[i, 0]), int(c[i, 1])
        if cx == 0 and cy == 0:
            continue
        psi_shift = torch.roll(psi, shifts=(-cy, -cx), dims=(0, 1))
        fx = fx + w[i] * cx * psi_shift
        fy = fy + w[i] * cy * psi_shift

    return fx, fy


def _shifted(field: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """field(x + d) under periodic wrap."""
    return torch.roll(field, shifts=(-dy, -dx), dims=(0, 1))


def stream_bounce_back(f_post: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
    """Streaming with half-way bounce-back at fluid cells facing a solid.

    For each direction i: f_i(x) = f_post,i(x - c_i) unless x - c_i is solid,
    in which case f_i(x) = f_post,opp(i)(x) (wall sits half-way between nodes).
    Solid cells keep their post-collision state.
    """
    out = torch.empty_like(f_post)
    for i in range(9):
        c = _C_INT[i]
        src_solid = _shifted(solid, -int(c[0]), -int(c[1]))
        streamed = _shifted(f_post[i], -int(c[0]), -int(c[1]))
        out[i] = torch.where(src_solid, f_post[OPPOSITE[i]], streamed)
    # Solid cells keep their post-collision state
    if solid.any():
        out = torch.where(solid.unsqueeze(0), f_post, out)
    return out


def _wall_extend(field: torch.Tensor, fill: float, wall_mode: str) -> torch.Tensor:
    """Extend ``field`` through the first/last row for the interaction stencil.

    ``wall_mode="mirror"`` mirrors the adjacent fluid row (neutral wetting: the
    wall prefers neither species); ``"dry"`` fills the wall row with ``fill``.
    The dry variant leaves a spurious adsorption layer of the minority species
    within a few cells of the wall (measured in the XVAL layered-Poiseuille
    two-component study), which the mirror variant removes.
    """
    if wall_mode == "mirror":
        ext = field.clone()
        ext[0, :] = field[1, :]
        ext[-1, :] = field[-2, :]
        return ext
    if wall_mode == "dry":
        ext = field.clone()
        ext[0, :] = fill
        ext[-1, :] = fill
        return ext
    raise ValueError(f"unknown wall_mode {wall_mode!r}; choose from {WALL_MODES}")


def sc_two_component_force(
    rho_a: torch.Tensor,
    rho_b: torch.Tensor,
    G: float = 2.5,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float = 0.0,
    wall_mode: str = "mirror",
    gy: float = 0.0,
    gx: float = 0.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    g_self: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-species Shan-Chen force of a two-component (MCMP) model.

    Library convention for the *two-component* force (note that it differs from
    the single-component :func:`sc_force`):

        F_a = -G * psi(rho_a) * sum_i w_i psi(rho_b)(x+c_i) c_i
        F_b = -G * psi(rho_b) * sum_i w_i psi(rho_a)(x+c_i) c_i
        F_sigma += rho_sigma * (gx, gy)

    ``G > 0`` is **repulsive** (demixing), ``G < 0`` attractive; no ``cs^2``
    prefactor is applied (the neighbour sum already carries the lattice weights).
    Solid neighbours are passed through the ``wall_mode`` extension of the
    pseudopotential, and the body force is applied as an acceleration ``(gx, gy)``
    common to both species, so that a layered co-flow has ``u'' = -g/nu`` inside
    every layer.

    Returns
    -------
    (fx_a, fy_a, fx_b, fy_b) : force per unit volume of each component
    """
    _psi = psi_fn if psi_fn is not None else psi_exp
    psi_a = _psi(rho_a)
    psi_b = _psi(rho_b)
    if solid_mask is not None:
        zero = torch.zeros_like(psi_a)
        psi_a = torch.where(solid_mask, zero, psi_a)
        psi_b = torch.where(solid_mask, zero, psi_b)
    ext_a = _wall_extend(psi_a, wall_psi, wall_mode)
    ext_b = _wall_extend(psi_b, wall_psi, wall_mode)

    fx_a = torch.zeros_like(psi_a)
    fy_a = torch.zeros_like(psi_a)
    fx_b = torch.zeros_like(psi_b)
    fy_b = torch.zeros_like(psi_b)
    for i in range(1, 9):
        dx, dy = _C_INT[i]
        nb_b = _shifted(ext_b, dx, dy)
        nb_a = _shifted(ext_a, dx, dy)
        fx_a = fx_a + W[i] * dx * nb_b
        fy_a = fy_a + W[i] * dy * nb_b
        fx_b = fx_b + W[i] * dx * nb_a
        fy_b = fy_b + W[i] * dy * nb_a

    fx_a = -G * psi_a * fx_a + rho_a * gx
    fy_a = -G * psi_a * fy_a + rho_a * gy
    fx_b = -G * psi_b * fx_b + rho_b * gx
    fy_b = -G * psi_b * fy_b + rho_b * gy

    # Self-interaction (optional, for full MCMP with g_ss != 0)
    if g_self[0] != 0.0:
        sx_a, sy_a = _sc_neighbor_sum(psi_a, solid_mask, wall_psi if solid_mask is not None else None)
        fx_a = fx_a + g_self[0] * CS2 * psi_a * sx_a
        fy_a = fy_a + g_self[0] * CS2 * psi_a * sy_a
    if g_self[1] != 0.0:
        sx_b, sy_b = _sc_neighbor_sum(psi_b, solid_mask, wall_psi if solid_mask is not None else None)
        fx_b = fx_b + g_self[1] * CS2 * psi_b * sx_b
        fy_b = fy_b + g_self[1] * CS2 * psi_b * sy_b

    if solid_mask is not None:
        zero = torch.zeros_like(fx_a)
        fx_a = torch.where(solid_mask, zero, fx_a)
        fy_a = torch.where(solid_mask, zero, fy_a)
        fx_b = torch.where(solid_mask, zero, fx_b)
        fy_b = torch.where(solid_mask, zero, fy_b)
    return fx_a, fy_a, fx_b, fy_b


def collide_edm_common(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """EDM collision of one component of a mixture, at a *supplied* velocity.

    ``collide_edm`` recovers the velocity from ``f`` itself, which is only right
    for a single-component model.  In a two-component model both components
    relax towards equilibrium at the common (viscosity-weighted) velocity
    ``(ux, uy)`` while the force shift stays component-specific
    (``F_sigma / rho_sigma``)::

        f^+ = f - (f - feq(rho, u)) / tau + [feq(rho, u + F/rho) - feq(rho, u)]

    ``rho`` is the density of the component that owns ``f``.
    """
    rho_safe = torch.clamp(rho, min=1e-12)
    u2 = torch.zeros_like(rho)
    feq = equilibrium(rho, ux, uy)
    feq_d = equilibrium(rho, ux + fx / rho_safe, uy + fy / rho_safe)
    del u2
    return f - (f - feq) / tau + (feq_d - feq)
