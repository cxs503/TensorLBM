"""Shan-Chen single-component multiphase LBM (SCMP) for D2Q9.

Implements the pseudopotential model with:
- psi_exp (Shan-Chen 1994) pseudopotential
- Guo forcing scheme (second-order accurate)
- Wall wetting boundary condition via wall pseudopotential
- Half-way bounce-back on solid walls

Conventions (locked by the library):
- G > 0 is attractive (backward-gather sign convention)
- G = 5.0 with psi_exp and tau = 1 gives coexistence:
  rho_l = 1.957, rho_v = 0.1596, sigma = 0.0561 (lattice units)

References
----------
Shan & Chen (1993) Phys. Rev. E 47, 1815
Shan & Chen (1994) Phys. Rev. E 49, 2941
Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308 (Guo forcing)
"""

from __future__ import annotations

import torch

from ..d2q9 import C, W, equilibrium, macroscopic, CS2

# ---------------------------------------------------------------------------
# Pseudopotential
# ---------------------------------------------------------------------------

#: Discrete liquid/vapour coexistence for G=5, psi_exp, tau=1 (measured)
RHO_L_DEFAULT = 1.957
RHO_V_DEFAULT = 0.1596
SIGMA_DEFAULT = 0.05611


def psi_exp(rho: torch.Tensor, rho0: float = 1.0) -> torch.Tensor:
    """Shan-Chen pseudopotential psi(rho) = rho0 * (1 - exp(-rho/rho0))."""
    return rho0 * (1.0 - torch.exp(-rho / rho0))


# ---------------------------------------------------------------------------
# SC interaction force with wall wetting
# ---------------------------------------------------------------------------

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


def sc_force(
    rho: torch.Tensor,
    G: float = 5.0,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
    gx: float = 0.0,
    gy: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shan-Chen interaction force.

    With the library convention (G > 0 = attractive):
        F = G * cs^2 * psi * sum_i w_i * psi(x+c_i) * c_i

    Args:
        rho:        Density field (ny, nx)
        G:          SC coupling constant (> 0 = attractive)
        solid_mask: Boolean mask of solid cells
        wall_psi:   Pseudopotential value for solid cells (wetting control)
        gx, gy:     External body force acceleration

    Returns:
        (Fx, Fy): Force per unit volume fields
    """
    psi = psi_exp(rho)
    sx, sy = _sc_neighbor_sum(psi, solid_mask, wall_psi)

    fx = G * CS2 * psi * sx
    fy = G * CS2 * psi * sy

    # Add external body force
    if gx != 0.0:
        fx = fx + rho * gx
    if gy != 0.0:
        fy = fy + rho * gy

    return fx, fy


# ---------------------------------------------------------------------------
# Guo forcing scheme (Guo, Zheng & Shi 2002)
# ---------------------------------------------------------------------------

def collide_guo(
    f: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Guo forcing scheme collision.

    Second-order accurate forcing that correctly recovers the Navier-Stokes
    equations with body force.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)

    # Guo source term
    device = f.device
    c = C.to(device).float()
    w = W.to(device).view(9, 1, 1)

    # u' = u + F*dt/(2*rho)  (midpoint velocity)
    rho_safe = torch.clamp(rho, min=1e-12)
    uxp = ux + fx / (2.0 * rho_safe)
    uyp = uy + fy / (2.0 * rho_safe)

    # Source term: S_i = w_i * (1 - 1/(2*tau)) * [(c_i - u)/cs^2 + (c_i*u)*c_i/cs^4] * F
    cf = c[:, 0].view(9, 1, 1) * fx.unsqueeze(0) + c[:, 1].view(9, 1, 1) * fy.unsqueeze(0)
    cu = c[:, 0].view(9, 1, 1) * uxp.unsqueeze(0) + c[:, 1].view(9, 1, 1) * uyp.unsqueeze(0)
    uf = uxp * fx + uyp * fy

    src = w * (1.0 - 1.0 / (2.0 * tau)) * ((cf - uf.unsqueeze(0)) / CS2 + cu * cf / CS2 ** 2)

    return f - (f - feq) / tau + src


# ---------------------------------------------------------------------------
# Streaming with bounce-back
# ---------------------------------------------------------------------------

#: Opposite direction mapping for D2Q9
OPPOSITE = [0, 3, 4, 1, 2, 7, 8, 5, 6]
#: D2Q9 velocity components as int tuples
_C_INT = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))


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


def stream_periodic(f: torch.Tensor) -> torch.Tensor:
    """Periodic streaming (no walls)."""
    from ..d2q9 import stream
    return stream(f)


# ---------------------------------------------------------------------------
# Full step
# ---------------------------------------------------------------------------

def step_sc(
    f: torch.Tensor,
    G: float = 5.0,
    tau: float = 1.0,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
    gx: float = 0.0,
    gy: float = 0.0,
) -> torch.Tensor:
    """One time step: collide (Guo) + stream + bounce-back.

    Parameters
    ----------
    f : Distribution function (9, ny, nx)
    G : SC coupling
    tau : Relaxation time
    solid_mask : Solid cell mask
    wall_psi : Wall pseudopotential
    gx, gy : Body force

    Returns
    -------
    f_new : Updated distribution
    """
    rho = f.sum(dim=0)
    fx, fy = sc_force(rho, G, solid_mask, wall_psi, gx, gy)
    f_post = collide_guo(f, fx, fy, tau)

    if solid_mask is not None:
        return stream_bounce_back(f_post, solid_mask)
    else:
        return stream_periodic(f_post)
