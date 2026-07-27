"""Common free-surface (Volume-of-Fluid) LBM module — composable VOF step.

Extracts the free-surface (VOF) physics into a reusable module that can be
composed with *any* D3Q19 collision / streaming solver.  Unlike the full
Körner mass-tracking model in :mod:`tensorlbm.free_surface_lbm`, this module
uses a simple **volume-fraction (VOF) advection** approach:

* A single D3Q19 distribution ``f`` carries momentum for the blended fluid.
* A scalar volume-fraction field ``phi`` (0 = gas, 1 = liquid,
  0 < phi < 1 = interface) is advected with the local velocity via an
  **upwind finite-difference** scheme.
* Density is blended:  ``rho = rho_l·phi + rho_g·(1 − phi)``.
* **Gravity** is applied as a Guo body force, only in fluid cells
  (``phi > 0.5``).
* **Surface tension** uses the Continuum Surface Force (CSF) model of
  Brackbill et al. (1992):  ``F_st = σ·κ·∇φ``  where ``κ = −∇·n̂`` is the
  mean curvature and ``n̂ = ∇φ/|∇φ|`` the interface normal.

Design principles
-----------------
* **No solver hot-path changes** — standalone composable step function,
  mirroring :mod:`tensorlbm.passive_scalar_common`.
* **D3Q19 momentum lattice** — velocity extracted from ``f``; ``phi`` is
  advected by finite differences (no second LBM distribution needed).
* **Composable** — call after the momentum collision / streaming::

      for step in range(n_steps):
          f, phi = free_surface_vof_step(
              f, phi, tau=0.8, gx=0.0, gy=-0.001, sigma=0.0,
              rho_liquid=1.0, rho_gas=0.01, solid=solid,
          )

References
----------
Brackbill, J. U., Kothe, D. B., & Zemach, C. (1992).
    A continuum method for modeling surface tension.
    *J. Comput. Phys.* 100, 335–354.

Körner, C., Thies, M., Thurey, E., & Rüde, U. (2005).
    Lattice Boltzmann simulation of free-surface flows.
    *J. Stat. Phys.* 121, 179–207.
"""
from __future__ import annotations

import functools
import math
from typing import Any

import torch

__all__ = [
    "vof_advect_upwind_3d",
    "interface_normal_3d",
    "mean_curvature_3d",
    "surface_tension_force_3d",
    "gravity_force_3d",
    "guo_force_delta_3d",
    "free_surface_vof_collide_3d",
    "free_surface_vof_step",
    "init_phi_block_3d",
    "init_phi_column_3d",
    "init_phi_tilted_3d",
    "init_phi_bubble_3d",
    "init_phi_rayleigh_taylor_3d",
    "front_position_3d",
    "wave_height_at_wall_3d",
    "mixing_layer_thickness_3d",
    "bubble_centroid_velocity_3d",
]

from .d3q19 import C as _C19, W as _W19, equilibrium3d, macroscopic3d

_CS2 = 1.0 / 3.0  # lattice speed of sound squared


@functools.cache
def _c_on(device: torch.device) -> torch.Tensor:
    return _C19.to(device)


# --------------------------------------------------------------------------- #
# 1. VOF advection (upwind finite-difference)                                 #
# --------------------------------------------------------------------------- #


def vof_advect_upwind_3d(
    phi: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Advect the volume fraction ``phi`` by one lattice time-step.

    Uses first-order **upwind** finite differences for stability::

        phi^{t+1} = phi^t − (u·∇phi)_upwind

    In lattice units ``dt = dx = 1``.  The upwind stencil chooses the
    backward or forward difference depending on the sign of the local
    velocity component, guaranteeing monotone (non-oscillatory) transport.

    Boundary handling: **zero-gradient** (Neumann) at all domain edges
    via reflective padding, so no mass wraps around periodically.

    Args:
        phi: Volume fraction, shape ``(nz, ny, nx)``.
        ux:  x-velocity, shape ``(nz, ny, nx)``.
        uy:  y-velocity, shape ``(nz, ny, nx)``.
        uz:  z-velocity, shape ``(nz, ny, nx)``.

    Returns:
        Updated volume fraction, shape ``(nz, ny, nx)``, clamped to
        ``[0, 1]``.
    """
    # Pad phi with edge replication (zero-gradient / Neumann BC) so that
    # the upwind stencil at domain boundaries does not wrap periodically.
    pad_phi = torch.nn.functional.pad(phi, (1, 1, 1, 1, 1, 1), mode="replicate")

    # Upwind differences: backward if u>0, forward if u<0
    # x-direction (dim=2 in (nz,ny,nx) layout); in padded array dim=2
    dphi_dx = torch.where(
        ux > 0,
        pad_phi[1:-1, 1:-1, 1:-1] - pad_phi[1:-1, 1:-1, 0:-2],    # backward
        pad_phi[1:-1, 1:-1, 2:]   - pad_phi[1:-1, 1:-1, 1:-1],   # forward
    )
    # y-direction (dim=1)
    dphi_dy = torch.where(
        uy > 0,
        pad_phi[1:-1, 1:-1, 1:-1] - pad_phi[1:-1, 0:-2, 1:-1],
        pad_phi[1:-1, 2:,   1:-1] - pad_phi[1:-1, 1:-1, 1:-1],
    )
    # z-direction (dim=0)
    dphi_dz = torch.where(
        uz > 0,
        pad_phi[1:-1, 1:-1, 1:-1] - pad_phi[0:-2, 1:-1, 1:-1],
        pad_phi[2:,   1:-1, 1:-1] - pad_phi[1:-1, 1:-1, 1:-1],
    )

    phi_new = phi - (ux * dphi_dx + uy * dphi_dy + uz * dphi_dz)
    return phi_new.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# 2. Interface geometry (normal, curvature, surface tension)                  #
# --------------------------------------------------------------------------- #


def interface_normal_3d(phi: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Compute the interface normal ``n̂ = ∇φ / |∇φ|`` and its magnitude.

    Uses second-order **central** differences for the gradient.

    Args:
        phi: Volume fraction, shape ``(nz, ny, nx)``.

    Returns:
        ``(nx, ny, nz, mag)`` — the three normal components and the
        gradient magnitude ``|∇φ|``, each of shape ``(nz, ny, nx)``.
    """
    # Central differences (periodic via torch.roll)
    dphi_dx = (torch.roll(phi, shifts=-1, dims=2) - torch.roll(phi, shifts=1, dims=2)) * 0.5
    dphi_dy = (torch.roll(phi, shifts=-1, dims=1) - torch.roll(phi, shifts=1, dims=1)) * 0.5
    dphi_dz = (torch.roll(phi, shifts=-1, dims=0) - torch.roll(phi, shifts=1, dims=0)) * 0.5

    mag = torch.sqrt(dphi_dx ** 2 + dphi_dy ** 2 + dphi_dz ** 2).clamp(min=1e-12)
    return dphi_dx / mag, dphi_dy / mag, dphi_dz / mag, mag


def mean_curvature_3d(phi: torch.Tensor) -> torch.Tensor:
    """Compute the mean interface curvature ``κ = −∇·n̂``.

    The normal ``n̂ = ∇φ/|∇φ|`` is computed first, then its divergence
    gives the mean curvature (with the sign convention ``κ = −∇·n̂`` so
    that a convex liquid droplet has positive curvature).

    Args:
        phi: Volume fraction, shape ``(nz, ny, nx)``.

    Returns:
        Curvature field, shape ``(nz, ny, nx)``.
    """
    nx_n, ny_n, nz_n, _ = interface_normal_3d(phi)

    # Divergence of n_hat via central differences
    dnx_dx = (torch.roll(nx_n, shifts=-1, dims=2) - torch.roll(nx_n, shifts=1, dims=2)) * 0.5
    dny_dy = (torch.roll(ny_n, shifts=-1, dims=1) - torch.roll(ny_n, shifts=1, dims=1)) * 0.5
    dnz_dz = (torch.roll(nz_n, shifts=-1, dims=0) - torch.roll(nz_n, shifts=1, dims=0)) * 0.5

    kappa = -(dnx_dx + dny_dy + dnz_dz)
    return kappa


def surface_tension_force_3d(
    phi: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuum Surface Force (CSF) surface-tension body force.

    Following Brackbill et al. (1992)::

        F_st = σ · κ · ∇φ

    where ``κ`` is the mean curvature and ``∇φ`` acts as the interface
    delta function (concentrating the force at the interface).  The force
    is zero away from the interface because ``∇φ → 0`` in pure fluid/gas
    regions.

    Args:
        phi:   Volume fraction, shape ``(nz, ny, nx)``.
        sigma: Surface-tension coefficient (lattice units).

    Returns:
        ``(Fx, Fy, Fz)`` — surface-tension force components, each of
        shape ``(nz, ny, nx)``.
    """
    if sigma == 0.0:
        z = torch.zeros_like(phi)
        return z, z, z

    kappa = mean_curvature_3d(phi)

    # grad(phi) — same central differences as in interface_normal_3d
    dphi_dx = (torch.roll(phi, shifts=-1, dims=2) - torch.roll(phi, shifts=1, dims=2)) * 0.5
    dphi_dy = (torch.roll(phi, shifts=-1, dims=1) - torch.roll(phi, shifts=1, dims=1)) * 0.5
    dphi_dz = (torch.roll(phi, shifts=-1, dims=0) - torch.roll(phi, shifts=1, dims=0)) * 0.5

    coef = sigma * kappa
    return coef * dphi_dx, coef * dphi_dy, coef * dphi_dz


# --------------------------------------------------------------------------- #
# 3. Gravity body force                                                       #
# --------------------------------------------------------------------------- #


def gravity_force_3d(
    phi: torch.Tensor,
    rho: torch.Tensor,
    gx: float,
    gy: float,
    gz: float,
    fluid_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gravity body force ``F_g = ρ·g``, applied only in fluid cells.

    The force is masked to cells where ``phi > fluid_threshold`` so that
    gas cells (which have negligible density in the VOF model) do not
    receive spurious acceleration.

    Args:
        phi:             Volume fraction, shape ``(nz, ny, nx)``.
        rho:             Local density, shape ``(nz, ny, nx)``.
        gx, gy, gz:      Gravity acceleration components (lattice units).
        fluid_threshold: Volume fraction above which a cell is "fluid".

    Returns:
        ``(Fx, Fy, Fz)`` — gravity force components, each of shape
        ``(nz, ny, nx)``.
    """
    mask = (phi > fluid_threshold).to(rho.dtype)
    Fx = rho * gx * mask
    Fy = rho * gy * mask
    Fz = rho * gz * mask
    return Fx, Fy, Fz


# --------------------------------------------------------------------------- #
# 4. Guo (2002) forcing for D3Q19 BGK collision                               #
# --------------------------------------------------------------------------- #


def guo_force_delta_3d(
    Fx: torch.Tensor,
    Fy: torch.Tensor,
    Fz: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    tau: float,
    device: torch.device,
) -> torch.Tensor:
    """Guo (2002) second-order force correction for D3Q19.

    ::

        Δfᵢ = (1 − 1/(2τ)) · wᵢ · [(cᵢ − u)/cs² + (cᵢ·u)·cᵢ/cs⁴] · F

    This is added to the post-collision distribution to incorporate body
    forces (gravity, surface tension) with second-order accuracy.

    Args:
        Fx, Fy, Fz: Force components, shape ``(nz, ny, nx)``.
        ux, uy, uz: Velocity components, shape ``(nz, ny, nx)``.
        tau:        Relaxation time.
        device:     Torch device.

    Returns:
        Force correction ``Δf``, shape ``(19, nz, ny, nx)``.
    """
    c = _c_on(device)  # (19, 3)
    w = _W19.to(device).view(19, 1, 1, 1)

    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    # c_i · u  and  c_i · F
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    cF = cx * Fx.unsqueeze(0) + cy * Fy.unsqueeze(0) + cz * Fz.unsqueeze(0)

    inv_cs2 = 1.0 / _CS2
    inv_cs4 = 1.0 / (_CS2 * _CS2)

    # (c_i - u)/cs^2 · F  +  (c_i·u)(c_i·F)/cs^4
    # = cF/cs^2 - (u·F)/cs^2 + cu*cF/cs^4
    uF = (ux * Fx + uy * Fy + uz * Fz).unsqueeze(0)  # (1, nz, ny, nx)

    delta = (cF * inv_cs2 - uF * inv_cs2 + cu * cF * inv_cs4)
    coef = (1.0 - 1.0 / (2.0 * tau)) * w
    return coef * delta


# --------------------------------------------------------------------------- #
# 5. VOF collision (BGK + Guo forcing)                                        #
# --------------------------------------------------------------------------- #


def free_surface_vof_collide_3d(
    f: torch.Tensor,
    phi: torch.Tensor,
    tau: float,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    sigma: float = 0.0,
    rho_liquid: float = 1.0,
    rho_gas: float = 0.01,
    solid: torch.Tensor | None = None,
) -> torch.Tensor:
    """BGK collision with gravity + surface-tension Guo forcing.

    The density is blended from ``phi``::

        rho = rho_l · phi + rho_g · (1 − phi)

    and the equilibrium is computed with this blended density and the
    local velocity.  Gravity is applied only in fluid cells
    (``phi > 0.5``); surface tension is applied at the interface via CSF.

    Args:
        f:           D3Q19 distribution, shape ``(19, nz, ny, nx)``.
        phi:         Volume fraction, shape ``(nz, ny, nx)``.
        tau:         Relaxation time.
        gx, gy, gz:  Gravity acceleration (lattice units).
        sigma:       Surface-tension coefficient (lattice units).
        rho_liquid:  Liquid density (lattice units).
        rho_gas:     Gas density (lattice units).
        solid:       Optional solid mask, shape ``(nz, ny, nx)``.

    Returns:
        Post-collision distribution, shape ``(19, nz, ny, nx)``.
    """
    device = f.device
    rho, ux, uy, uz = macroscopic3d(f)

    # Blend density from phi
    rho_blend = rho_liquid * phi + rho_gas * (1.0 - phi)

    # Zero velocity in solid cells
    if solid is not None:
        ux = ux.masked_fill(solid, 0.0)
        uy = uy.masked_fill(solid, 0.0)
        uz = uz.masked_fill(solid, 0.0)

    # Equilibrium with blended density
    feq = equilibrium3d(rho_blend, ux, uy, uz, device=device)

    # Body forces
    Fx_g, Fy_g, Fz_g = gravity_force_3d(phi, rho_blend, gx, gy, gz)
    Fx_st, Fy_st, Fz_st = surface_tension_force_3d(phi, sigma)
    Fx = Fx_g + Fx_st
    Fy = Fy_g + Fy_st
    Fz = Fz_g + Fz_st

    # BGK collision + Guo forcing
    f_post = f - (f - feq) / tau + guo_force_delta_3d(Fx, Fy, Fz, ux, uy, uz, tau, device)

    # Clamp for stability (prevent negative populations in gas cells)
    f_post = f_post.clamp(min=0.0)

    return f_post


# --------------------------------------------------------------------------- #
# 6. Composable free-surface VOF step                                         #
# --------------------------------------------------------------------------- #


def free_surface_vof_step(
    f: torch.Tensor,
    phi: torch.Tensor,
    tau: float,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    sigma: float = 0.0,
    rho_liquid: float = 1.0,
    rho_gas: float = 0.01,
    solid: torch.Tensor | None = None,
    stream_fn=None,
    bounce_back_fn=None,
    target_phi_sum: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One composable free-surface VOF LBM step (collision + stream + advect).

    Order of operations:
      1. Collision with gravity + surface-tension Guo forcing
      2. Streaming (D3Q19 pull scheme)
      3. Bounce-back at solid walls (half-way)
      4. VOF advection of ``phi`` with the post-stream velocity
      5. Global mass conservation correction for ``phi``

    The density coupling is through the **equilibrium** in the collision
    step: ``feq`` is computed with ``rho_blend = rho_l·phi + rho_g·(1−phi)``,
    so the collision naturally drives ``f`` toward the phi-determined
    density.  No explicit density rescaling is applied (which was found to
    amplify numerical errors and break mass conservation).

    Designed to be inserted into *any* time loop::

        for step in range(n_steps):
            f, phi = free_surface_vof_step(
                f, phi, tau=0.8, gy=-1e-3, sigma=0.0,
                rho_liquid=1.0, rho_gas=0.01, solid=solid,
            )

    Args:
        f:           D3Q19 distribution, shape ``(19, nz, ny, nx)``.
        phi:         Volume fraction, shape ``(nz, ny, nx)``.
        tau:         Relaxation time.
        gx, gy, gz:  Gravity acceleration (lattice units).
        sigma:       Surface-tension coefficient (lattice units).
        rho_liquid:  Liquid density.
        rho_gas:     Gas density.
        solid:       Optional solid mask, shape ``(nz, ny, nx)``.
        stream_fn:   Custom streaming function (default: D3Q19 pull).
        bounce_back_fn: Custom bounce-back function
            ``fn(f, solid) -> f``.  If None and ``solid`` is given, uses
            :func:`tensorlbm.boundaries3d.bounce_back_cells_3d`.
        target_phi_sum: Target total fluid volume (sum of phi over non-solid
            cells).  If given, phi is globally rescaled each step to
            preserve this total, correcting upwind advection mass drift.

    Returns:
        ``(f_updated, phi_updated)``.
    """
    # 1. Collision with forcing
    f = free_surface_vof_collide_3d(
        f, phi, tau, gx, gy, gz, sigma, rho_liquid, rho_gas, solid
    )

    # 2. Streaming
    if stream_fn is not None:
        f = stream_fn(f)
    else:
        from .solver3d import stream3d
        f = stream3d(f)

    # 3. Bounce-back at solid walls
    if solid is not None and bounce_back_fn is not None:
        f = bounce_back_fn(f, solid)
    elif solid is not None:
        from .boundaries3d import bounce_back_cells_3d
        f = bounce_back_cells_3d(f, solid)

    # 4. Extract post-stream velocity and advect phi
    rho, ux, uy, uz = macroscopic3d(f)
    if solid is not None:
        ux = ux.masked_fill(solid, 0.0)
        uy = uy.masked_fill(solid, 0.0)
        uz = uz.masked_fill(solid, 0.0)

    phi = vof_advect_upwind_3d(phi, ux, uy, uz)

    # 5. Global mass conservation: rescale phi to preserve total fluid volume
    if target_phi_sum is not None and target_phi_sum > 1e-10:
        if solid is not None:
            phi_sum = phi[~solid].sum().item()
        else:
            phi_sum = phi.sum().item()
        if phi_sum > 1e-10:
            scale_phi = target_phi_sum / phi_sum
            phi = (phi * scale_phi).clamp(0.0, 1.0)

    # 6. Clamp f to non-negative for stability in gas cells
    f = f.clamp(min=0.0)

    return f, phi


# --------------------------------------------------------------------------- #
# 7. Initial-condition helpers                                                #
# --------------------------------------------------------------------------- #


def init_phi_block_3d(
    nz: int,
    ny: int,
    nx: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    device: torch.device,
    z0: int = 0,
    z1: int | None = None,
) -> torch.Tensor:
    """Initialise ``phi`` for a rectangular fluid block.

    Args:
        nz, ny, nx: Grid dimensions.
        x0, x1:     x-range of the fluid block (x0 inclusive, x1 exclusive).
        y0, y1:     y-range of the fluid block.
        z0:         z-start (default 0).
        z1:         z-end (default ``nz``).
        device:     Torch device.

    Returns:
        Volume-fraction field, shape ``(nz, ny, nx)``.
    """
    if z1 is None:
        z1 = nz
    phi = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    phi[z0:z1, y0:y1, x0:x1] = 1.0
    return phi


def init_phi_column_3d(
    nz: int,
    ny: int,
    nx: int,
    width: int,
    height: int,
    device: torch.device,
) -> torch.Tensor:
    """Initialise ``phi`` for a dam-break water column at the bottom-left.

    The column occupies ``x ∈ [1, 1+width)`` and ``y ∈ [0, height)``
    (y = 0 is the floor).  All z-layers are filled.

    Args:
        nz, ny, nx: Grid dimensions.
        width:      Column width in x (lattice units).
        height:     Column height in y.
        device:     Torch device.

    Returns:
        Volume-fraction field, shape ``(nz, ny, nx)``.
    """
    return init_phi_block_3d(nz, ny, nx, 1, 1 + width, 0, height, device)


def init_phi_tilted_3d(
    nz: int,
    ny: int,
    nx: int,
    fill_frac: float,
    angle_deg: float,
    device: torch.device,
) -> torch.Tensor:
    """Initialise ``phi`` for a tilted free surface (sloshing IC).

    The surface is a plane tilted by ``angle_deg`` about the z-axis,
    with mean fill level ``fill_frac · ny``.  The tilt creates an
    initial potential that drives sloshing.

    Args:
        nz, ny, nx:  Grid dimensions.
        fill_frac:   Mean fill fraction (0–1) of the y-height.
        angle_deg:   Tilt angle (degrees).
        device:      Torch device.

    Returns:
        Volume-fraction field, shape ``(nz, ny, nx)``.
    """
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_c = (nx - 1) / 2.0
    y_mean = fill_frac * ny
    angle = math.radians(angle_deg)
    # Surface height: h(x) = y_mean + (x - x_c) * tan(angle)
    h = y_mean + (xx - x_c) * math.tan(angle)
    phi = (yy < h).float()
    # Smooth the interface over 2 cells for stability
    phi = torch.where(
        (yy >= h - 1) & (yy < h + 1),
        torch.clamp(h + 1 - yy, 0.0, 1.0),
        phi,
    )
    # Expand to 3D
    phi = phi.unsqueeze(0).expand(nz, ny, nx).contiguous()
    return phi


def init_phi_bubble_3d(
    nz: int,
    ny: int,
    nx: int,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    """Initialise ``phi`` for a gas bubble (phi=0) in liquid (phi=1).

    Args:
        nz, ny, nx: Grid dimensions.
        cx, cy, cz: Bubble centre.
        radius:     Bubble radius.
        device:     Torch device.

    Returns:
        Volume-fraction field (1 = liquid, 0 = gas bubble),
        shape ``(nz, ny, nx)``.
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    # Smooth interface over 1 cell
    phi = torch.clamp((dist - radius + 0.5), 0.0, 1.0)
    return phi


def init_phi_rayleigh_taylor_3d(
    nz: int,
    ny: int,
    nx: int,
    interface_frac: float,
    amplitude: float,
    wavelength: float,
    device: torch.device,
) -> torch.Tensor:
    """Initialise ``phi`` for a Rayleigh–Taylor instability.

    Heavy fluid (phi=1) sits on top of light fluid (phi=0), with a
    perturbed interface at ``y = interface_frac·ny``.

    Args:
        nz, ny, nx:        Grid dimensions.
        interface_frac:    Mean interface position as fraction of ny.
        amplitude:         Perturbation amplitude.
        wavelength:        Perturbation wavelength (in x).
        device:            Torch device.

    Returns:
        Volume-fraction field (1 = heavy top, 0 = light bottom),
        shape ``(nz, ny, nx)``.
    """
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    y_iface = interface_frac * ny + amplitude * torch.sin(2.0 * math.pi * xx / wavelength)
    phi = (yy < y_iface).float()
    # Smooth interface
    phi = torch.where(
        (yy >= y_iface - 1) & (yy < y_iface + 1),
        torch.clamp(y_iface + 1 - yy, 0.0, 1.0),
        phi,
    )
    phi = phi.unsqueeze(0).expand(nz, ny, nx).contiguous()
    return phi


# --------------------------------------------------------------------------- #
# 8. Diagnostics                                                               #
# --------------------------------------------------------------------------- #


def front_position_3d(phi: torch.Tensor, threshold: float = 0.5) -> float:
    """Measure the dam-break front position (max x with phi > threshold).

    Averages over the z and y dimensions to find the furthest x-extent
    of the fluid.

    Args:
        phi:       Volume fraction, shape ``(nz, ny, nx)``.
        threshold:  Fluid threshold (default 0.5).

    Returns:
        Front position in x (lattice units).
    """
    fluid = (phi > threshold)
    if not fluid.any():
        return 0.0
    # Max x index where fluid exists
    x_mask = fluid.any(dim=0).any(dim=0)  # (nx,)
    indices = torch.where(x_mask)[0]
    if indices.numel() == 0:
        return 0.0
    return float(indices.max().item())


def wave_height_at_wall_3d(
    phi: torch.Tensor,
    wall: str = "left",
    threshold: float = 0.5,
) -> float:
    """Measure the wave height at a side wall.

    Finds the highest y-index where ``phi > threshold`` at the specified
    wall (left = x=0, right = x=nx-1).

    Args:
        phi:       Volume fraction, shape ``(nz, ny, nx)``.
        wall:      ``"left"`` or ``"right"``.
        threshold: Fluid threshold.

    Returns:
        Wave height in y (lattice units).
    """
    nz, ny, nx = phi.shape
    if wall == "left":
        col = phi[:, :, 1]  # first interior cell
    else:
        col = phi[:, :, -2]
    fluid = (col > threshold)
    if not fluid.any():
        return 0.0
    y_mask = fluid.any(dim=0)  # (ny,)
    indices = torch.where(y_mask)[0]
    if indices.numel() == 0:
        return 0.0
    return float(indices.max().item())


def mixing_layer_thickness_3d(
    phi: torch.Tensor,
    low: float = 0.1,
    high: float = 0.9,
) -> float:
    """Measure the Rayleigh–Taylor mixing-layer thickness.

    Computes the y-extent over which ``low < phi < high`` (the mixed
    region), averaged over x and z.

    Args:
        phi:  Volume fraction, shape ``(nz, ny, nx)``.
        low:  Lower threshold for "mixed".
        high: Upper threshold for "mixed".

    Returns:
        Mixing-layer thickness in y (lattice units).
    """
    mixed = (phi > low) & (phi < high)
    # For each y-slice, count mixed cells
    y_mixed = mixed.any(dim=0).any(dim=1)  # (ny,)
    indices = torch.where(y_mixed)[0]
    if indices.numel() == 0:
        return 0.0
    return float(indices.max().item() - indices.min().item())


def bubble_centroid_velocity_3d(
    phi: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float, float, float, float]:
    """Measure the bubble centroid and its velocity.

    The bubble is the gas region (``phi < threshold``).  The centroid is
    the volume-weighted centre of the gas region, and the velocity is the
    average velocity in the gas region.

    Args:
        phi:       Volume fraction, shape ``(nz, ny, nx)``.
        ux, uy, uz: Velocity components.
        threshold: Fluid threshold (bubble = phi < threshold).

    Returns:
        ``(cx, cy, cz, vx, vy, vz)`` — centroid position and velocity.
    """
    gas = (phi < threshold).to(torch.float32)
    total = gas.sum()
    if total < 1.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    nz, ny, nx = phi.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=phi.device, dtype=torch.float32),
        torch.arange(ny, device=phi.device, dtype=torch.float32),
        torch.arange(nx, device=phi.device, dtype=torch.float32),
        indexing="ij",
    )
    cx = float((gas * xx).sum().item() / total.item())
    cy = float((gas * yy).sum().item() / total.item())
    cz = float((gas * zz).sum().item() / total.item())

    vx = float((gas * ux).sum().item() / total.item())
    vy = float((gas * uy).sum().item() / total.item())
    vz = float((gas * uz).sum().item() / total.item())

    return cx, cy, cz, vx, vy, vz


# --------------------------------------------------------------------------- #
# Module-level imports needed by helpers                                       #
# --------------------------------------------------------------------------- #
