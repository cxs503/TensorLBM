"""Shan-Chen single-component multiphase LBM (SCMP) for D2Q9.

Implements the pseudopotential model with:
- psi_exp (Shan-Chen 1994) pseudopotential
- Pluggable forcing schemes: velocity_shift, EDM, Guo
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
Kupershtokh (2004) — Exact Difference Method (EDM)
Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308 (Guo forcing)
"""

from __future__ import annotations

from typing import Callable, Literal

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
    psi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shan-Chen interaction force.

    With the library convention (G > 0 = attractive):
        F = G * psi * sum_i w_i * psi(x+c_i) * c_i
    (c_s^2 enters only through the neighbour sum, which approximates
    sum_i w_i c_i psi(x+c_i) ~ c_s^2 grad(psi); an extra c_s^2 prefactor
    was a defect: it made the effective coupling G/3 too weak to establish
    the documented coexistence -- see the XVAL Laplace-pressure report.)

    Args:
        rho:        Density field (ny, nx)
        G:          SC coupling constant (> 0 = attractive)
        solid_mask: Boolean mask of solid cells
        wall_psi:   Pseudopotential value for solid cells (wetting control)
        gx, gy:     External body force acceleration

    Returns:
        (Fx, Fy): Force per unit volume fields
    """
    _psi = psi_fn if psi_fn is not None else psi_exp
    psi = _psi(rho)
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

    Reference: Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308.

    Formula:
        u* = u + F/(2ρ)
        feq = feq(ρ, u*)
        S_i = w_i (1 - 1/(2τ)) [(c_i - u*)·F/cs² + (c_i·u*)(c_i·F)/cs⁴]
        f_out = f - (f - feq)/τ + S
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
# Velocity-shift forcing (Shan & Chen 1993)
# ---------------------------------------------------------------------------

def collide_velocity_shift(
    f: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Velocity-shift forcing collision (Shan & Chen 1993).

    First-order accurate in force. The equilibrium is evaluated at a shifted
    velocity that includes the force contribution.

    Reference: Shan & Chen (1993) Phys. Rev. E 47, 1815.

    Formula:
        u_eq = u + τ·F/ρ
        feq = feq(ρ, u_eq)
        f_out = f - (f - feq)/τ
    """
    rho, ux, uy = macroscopic(f)
    rho_safe = torch.clamp(rho, min=1e-12)

    # Shifted velocity for equilibrium
    uxeq = ux + tau * fx / rho_safe
    uyeq = uy + tau * fy / rho_safe

    feq = equilibrium(rho, uxeq, uyeq)
    return f - (f - feq) / tau


# ---------------------------------------------------------------------------
# Exact Difference Method (EDM, Kupershtokh 2004)
# ---------------------------------------------------------------------------

def collide_edm(
    f: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Exact Difference Method (EDM) collision (Kupershtokh 2004).

    Adds the exact difference of equilibrium distributions to account for
    the force, ensuring exact momentum conservation per time step.

    Reference: Kupershtokh (2004) "New method of incorporating a body force
    term into the lattice Boltzmann equation".

    Formula:
        feq_u = feq(ρ, u)
        feq_d = feq(ρ, u + F/ρ)
        f_out = f - (f - feq_u)/τ + (feq_d - feq_u)
    """
    rho, ux, uy = macroscopic(f)
    rho_safe = torch.clamp(rho, min=1e-12)

    feq_u = equilibrium(rho, ux, uy)

    # Shifted velocity
    uxd = ux + fx / rho_safe
    uyd = uy + fy / rho_safe
    feq_d = equilibrium(rho, uxd, uyd)

    return f - (f - feq_u) / tau + (feq_d - feq_u)


# ---------------------------------------------------------------------------
# Pluggable SCMP collision (unified interface)
# ---------------------------------------------------------------------------

#: Supported forcing scheme names
ForcingScheme = Literal["velocity_shift", "edm", "guo"]

#: Mapping from scheme name to collision function
_FORCING_SCHEMES: dict[str, Callable] = {
    "velocity_shift": collide_velocity_shift,
    "edm": collide_edm,
    "guo": collide_guo,
}


def collide_sc_single_component(
    f: torch.Tensor,
    G: float = 5.0,
    tau: float = 1.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
    gx: float = 0.0,
    gy: float = 0.0,
    forcing: ForcingScheme = "guo",
    scheme: Literal["standard", "pressure_tensor"] = "standard",
) -> torch.Tensor:
    """Pluggable SCMP collision with selectable forcing scheme.

    This is the unified entry point for single-component multiphase LBM
    simulations. It computes the SC interaction force and applies the
    selected forcing scheme.

    Parameters
    ----------
    f : torch.Tensor
        Distribution function, shape (9, ny, nx).
    G : float
        Shan-Chen coupling constant (> 0 = attractive).
    tau : float
        Relaxation time.
    psi_fn : callable, optional
        Pseudopotential function. Defaults to psi_exp.
    solid_mask : torch.Tensor, optional
        Boolean mask of solid cells.
    wall_psi : float, optional
        Pseudopotential value for solid cells (wetting control).
    gx, gy : float
        External body force acceleration.
    forcing : str
        Forcing scheme: "velocity_shift", "edm", or "guo".
    scheme : str
        Force discretisation: "standard" or "pressure_tensor".
        Currently only "standard" is implemented.

    Returns
    -------
    f_post : torch.Tensor
        Post-collision distribution function.

    References
    ----------
    - velocity_shift: Shan & Chen (1993) Phys. Rev. E 47, 1815
    - edm: Kupershtokh (2004)
    - guo: Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308
    """
    if forcing not in _FORCING_SCHEMES:
        raise ValueError(
            f"forcing must be one of {list(_FORCING_SCHEMES.keys())}, got '{forcing}'"
        )
    if scheme != "standard":
        raise ValueError(
            f"scheme must be 'standard', got '{scheme}' "
            "(pressure_tensor not yet implemented for pluggable interface)"
        )

    rho = f.sum(dim=0)

    # Compute SC force (uses psi_exp by default, or custom psi_fn)
    if psi_fn is not None:
        psi = psi_fn(rho)
        sx, sy = _sc_neighbor_sum(psi, solid_mask, wall_psi)
        fx = G * CS2 * psi * sx
        fy = G * CS2 * psi * sy
    else:
        fx, fy = sc_force(rho, G, solid_mask, wall_psi, gx, gy)

    # Add external body force
    if gx != 0.0:
        fx = fx + rho * gx
    if gy != 0.0:
        fy = fy + rho * gy

    collide_fn = _FORCING_SCHEMES[forcing]
    return collide_fn(f, fx, fy, tau)


def sc_single_component_force(
    rho: torch.Tensor,
    G: float = 5.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
    gx: float = 0.0,
    gy: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SC interaction force with optional custom pseudopotential.

    Parameters
    ----------
    rho : torch.Tensor
        Density field (ny, nx).
    G : float
        Shan-Chen coupling constant (> 0 = attractive).
    psi_fn : callable, optional
        Pseudopotential function. Defaults to psi_exp.
    solid_mask : torch.Tensor, optional
        Boolean mask of solid cells.
    wall_psi : float, optional
        Pseudopotential value for solid cells.
    gx, gy : float
        External body force acceleration.

    Returns
    -------
    (Fx, Fy) : tuple[torch.Tensor, torch.Tensor]
        Force per unit volume fields.
    """
    if psi_fn is not None:
        psi = psi_fn(rho)
        sx, sy = _sc_neighbor_sum(psi, solid_mask, wall_psi)
        fx = G * CS2 * psi * sx
        fy = G * CS2 * psi * sy
    else:
        fx, fy = sc_force(rho, G, solid_mask, wall_psi, gx, gy)

    if gx != 0.0:
        fx = fx + rho * gx
    if gy != 0.0:
        fy = fy + rho * gy

    return fx, fy


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
    forcing: ForcingScheme = "guo",
) -> torch.Tensor:
    """One time step: collide (pluggable forcing) + stream + bounce-back.

    Parameters
    ----------
    f : Distribution function (9, ny, nx)
    G : SC coupling
    tau : Relaxation time
    solid_mask : Solid cell mask
    wall_psi : Wall pseudopotential
    gx, gy : Body force
    forcing : Forcing scheme ("velocity_shift", "edm", or "guo")

    Returns
    -------
    f_new : Updated distribution
    """
    rho = f.sum(dim=0)
    fx, fy = sc_force(rho, G, solid_mask, wall_psi, gx, gy)

    collide_fn = _FORCING_SCHEMES[forcing]
    f_post = collide_fn(f, fx, fy, tau)

    if solid_mask is not None:
        return stream_bounce_back(f_post, solid_mask)
    else:
        return stream_periodic(f_post)


# ---------------------------------------------------------------------------
# Two-component (MCMP) extensions
# ---------------------------------------------------------------------------
#: Wall treatment of the cross-species interaction stencil
WALL_MODES = ("mirror", "dry")


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


# ---------------------------------------------------------------------------
# Two-component (MCMP) extensions — pluggable psi, self-interaction, Guo
# ---------------------------------------------------------------------------

def psi_linear(rho: torch.Tensor, rho0: float = 1.0) -> torch.Tensor:
    """Linear pseudopotential psi(rho) = rho (simplest MCMP choice)."""
    return rho


def guo_force_term(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Guo (2002) source term S_i for one component, at a *supplied* velocity.

    Returns the source term (shape (9, ny, nx)) to be *added* to the
    post-collision distribution.  The caller is responsible for the BGK
    relaxation part.

    Reference: Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308.
    """
    device = rho.device
    c = C.to(device).to(rho.dtype)
    w = W.to(device).to(rho.dtype).view(9, 1, 1)

    cf = c[:, 0].view(9, 1, 1) * fx.unsqueeze(0) + c[:, 1].view(9, 1, 1) * fy.unsqueeze(0)
    cu = c[:, 0].view(9, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(9, 1, 1) * uy.unsqueeze(0)
    uf = ux * fx + uy * fy

    src = w * (1.0 - 1.0 / (2.0 * tau)) * ((cf - uf.unsqueeze(0)) / CS2 + cu * cf / CS2 ** 2)
    return src


def collide_sc_two_component(
    f1: torch.Tensor,
    f2: torch.Tensor,
    G_12: float = 6.0,
    tau1: float = 1.0,
    tau2: float = 1.0,
    u_eq: str = "mixture",
    force_scheme: str = "guo",
    psi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    g_self: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-component Shan-Chen BGK collision with Guo forcing.

    Both components relax towards equilibrium at the common mixture velocity.
    The force is computed with the pluggable pseudopotential and optional
    self-interaction.

    Parameters
    ----------
    f1, f2 : distribution functions (9, ny, nx)
    G_12 : cross-interaction coupling (positive = repulsive/demixing)
    tau1, tau2 : relaxation times
    u_eq : velocity for equilibrium (only "mixture" supported)
    force_scheme : "guo" (only Guo supported for MCMP)
    psi_fn : pseudopotential function (default: psi_exp)
    g_self : (g_11, g_22) self-interaction strengths (library sign: positive=attractive)

    Returns
    -------
    (f1_post, f2_post)
    """
    if psi_fn is None:
        psi_fn = psi_exp

    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)

    # Cross-component force
    Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(rho1, rho2, G_12, psi_fn=psi_fn)

    # Add self-interaction (same formula as single-component, applied per-species)
    if g_self[0] != 0.0:
        fx_self, fy_self = sc_force(rho1, g_self[0], psi_fn=psi_fn)
        Fx1 = Fx1 + fx_self
        Fy1 = Fy1 + fy_self
    if g_self[1] != 0.0:
        fx_self, fy_self = sc_force(rho2, g_self[1], psi_fn=psi_fn)
        Fx2 = Fx2 + fx_self
        Fy2 = Fy2 + fy_self

    # Mixture velocity (mass-weighted)
    r1 = rho1.clamp(min=1e-12)
    r2 = rho2.clamp(min=1e-12)
    rt = r1 + r2
    umx = (r1 * ux1 + r2 * ux2) / rt
    umy = (r1 * uy1 + r2 * uy2) / rt

    # BGK relaxation at mixture velocity + Guo source term
    feq1 = equilibrium(rho1, umx, umy)
    feq2 = equilibrium(rho2, umx, umy)

    if force_scheme == "guo":
        src1 = guo_force_term(rho1, umx, umy, Fx1, Fy1, tau1)
        src2 = guo_force_term(rho2, umx, umy, Fx2, Fy2, tau2)
        f1_post = f1 - (f1 - feq1) / tau1 + src1
        f2_post = f2 - (f2 - feq2) / tau2 + src2
    else:
        raise ValueError(f"force_scheme must be 'guo', got '{force_scheme}'")

    return f1_post, f2_post
