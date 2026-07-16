"""Common sliding-mesh module for 3-D LBM (D3Q19 / D3Q27).

Implements a multi-zone sliding-mesh technique for 3-D LBM.  The domain is
split into a **static outer zone** and a **rotating inner zone**; at each time
step the velocity distribution functions (DFs) are interpolated across the
interface using bilinear interpolation after applying a coordinate rotation.

This module provides ``sliding_mesh_step(f, rotor_mask, omega, ...)`` as a
**post-collision / pre-streaming** (or post-streaming) boundary update that is
collision-agnostic: it can be freely combined with any collision operator
(BGK, MRT, CG, …), turbulence model (Smagorinsky, WALE, RANS, …), or IBM
immersed-boundary method.

Supported lattices: **D3Q19** and **D3Q27**.

References
----------
Krause et al. (2017) "Fluid flow simulation and optimisation with lattice
    Boltzmann methods on high performance computers". KIT Scientific Publishing.
Latt et al. (2021) "Palabos: parallel lattice Boltzmann solver".
    *Computers & Mathematics with Applications* 81, 334–350.

Hot-path invariants
-------------------
* No GPU→CPU syncs (``.item()``, ``float(tensor)``) inside the step path.
* The rotation and interpolation are fully tensorised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import equilibrium27, macroscopic27

__all__ = [
    "SlidingMeshParams",
    "sliding_mesh_step",
    "rotate_velocity_field_3d",
    "interpolate_interface_3d",
    "apply_sliding_mesh_bc_3d",
]

RotationAxis = Literal["x", "y", "z"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SlidingMeshParams:
    """Parameters for the 3-D sliding-mesh boundary condition.

    Attributes
    ----------
    omega : float
        Angular velocity of the rotor (rad/step).
    theta : float
        Accumulated rotation angle (radians).
    tau : float
        BGK relaxation time used for the moving-wall equilibrium relaxation.
    axis : str
        Rotation axis: ``"x"``, ``"y"``, or ``"z"`` (default ``"z"``).
    cx, cy, cz : float
        Rotor centre in lattice coordinates.
    rotor_radius : float
        Rotor (interface) radius in lattice units.
    interface_width : float
        Half-width of the interface annulus in lattice units (default 1.5).
    """

    omega: float = 0.01
    theta: float = 0.0
    tau: float = 1.0
    axis: RotationAxis = "z"
    cx: float = 0.0
    cy: float = 0.0
    cz: float = 0.0
    rotor_radius: float = 10.0
    interface_width: float = 1.5


# --------------------------------------------------------------------------- #
# 3-D velocity-field rotation
# --------------------------------------------------------------------------- #

def rotate_velocity_field_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    theta: float,
    axis: RotationAxis = "z",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate a 3-D velocity *vector* field by angle *theta* about *axis*.

    Applies a rigid-body rotation:

    - axis ``"z"``: rotation in the x-y plane
        ``ux' = ux cos θ − uy sin θ``
        ``uy' = ux sin θ + uy cos θ``
        ``uz' = uz``
    - axis ``"x"``: rotation in the y-z plane
    - axis ``"y"``: rotation in the z-x plane

    Args:
        ux, uy, uz: Velocity component fields, each of shape ``(nz, ny, nx)``.
        theta: Rotation angle in radians (counter-clockwise positive).
        axis: Rotation axis (``"x"``, ``"y"``, or ``"z"``).

    Returns:
        Rotated ``(ux', uy', uz')`` tuple.
    """
    c = math.cos(theta)
    s = math.sin(theta)

    if axis == "z":
        ux_rot = c * ux - s * uy
        uy_rot = s * ux + c * uy
        uz_rot = uz
    elif axis == "x":
        uy_rot = c * uy - s * uz
        uz_rot = s * uy + c * uz
        ux_rot = ux
    elif axis == "y":
        uz_rot = c * uz - s * ux
        ux_rot = s * uz + c * ux
        uy_rot = uy
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")

    return ux_rot, uy_rot, uz_rot


# --------------------------------------------------------------------------- #
# 3-D interface interpolation
# --------------------------------------------------------------------------- #

def interpolate_interface_3d(
    f_inner: torch.Tensor,
    f_outer: torch.Tensor,
    interface_mask: torch.Tensor,
    theta: float,
    axis: RotationAxis = "z",
) -> torch.Tensor:
    """Interpolate DFs across the 3-D sliding interface after rotation.

    For each interface cell, the DF is blended from the outer zone value and
    the inner zone value (evaluated at the rotated position via trilinear
    interpolation using ``grid_sample``).

    Args:
        f_inner: DF of the inner (rotating) zone, shape ``(Q, nz, ny, nx)``.
        f_outer: DF of the outer (static) zone, same shape.
        interface_mask: Boolean mask of interface cells, shape ``(nz, ny, nx)``.
        theta: Current rotation angle of the inner zone (radians).
        axis: Rotation axis.

    Returns:
        Blended DF at interface cells; non-interface cells keep ``f_outer``.
    """
    nq, nz, ny, nx = f_inner.shape
    device = f_inner.device

    # Normalised grid coordinates [0, 1]
    zz, yy, xx = torch.meshgrid(
        torch.linspace(0, 1, nz, device=device),
        torch.linspace(0, 1, ny, device=device),
        torch.linspace(0, 1, nx, device=device),
        indexing="ij",
    )

    cx = cy = cz = 0.5
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)

    if axis == "z":
        dx = xx - cx
        dy = yy - cy
        x_rot = cx + cos_t * dx - sin_t * dy
        y_rot = cy + sin_t * dx + cos_t * dy
        z_rot = zz
    elif axis == "x":
        dy = yy - cy
        dz = zz - cz
        y_rot = cy + cos_t * dy - sin_t * dz
        z_rot = cz + sin_t * dy + cos_t * dz
        x_rot = xx
    else:  # axis == "y"
        dz = zz - cz
        dx = xx - cx
        z_rot = cz + cos_t * dz - sin_t * dx
        x_rot = cx + sin_t * dz + cos_t * dx
        y_rot = yy

    # Map to [-1, 1] for grid_sample (D, H, W ordering)
    grid = torch.stack(
        [x_rot * 2.0 - 1.0, y_rot * 2.0 - 1.0, z_rot * 2.0 - 1.0],
        dim=-1,
    )  # (nz, ny, nx, 3)
    grid = grid.reshape(1, nz, ny, nx, 3)

    f_inner_5d = f_inner.unsqueeze(0)  # (1, Q, nz, ny, nx)
    f_sampled = torch.nn.functional.grid_sample(
        f_inner_5d.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0)  # (Q, nz, ny, nx)

    mask_exp = interface_mask.unsqueeze(0).expand(nq, -1, -1, -1)
    return torch.where(mask_exp, f_sampled, f_outer)


# --------------------------------------------------------------------------- #
# 3-D sliding-mesh BC application
# --------------------------------------------------------------------------- #

def apply_sliding_mesh_bc_3d(
    f: torch.Tensor,
    interface_mask: torch.Tensor,
    theta: float,
    omega: float,
    rho: torch.Tensor,
    tau: float,
    *,
    axis: RotationAxis = "z",
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    lattice: str = "D3Q19",
) -> torch.Tensor:
    """Apply sliding-mesh boundary condition for one time step (3-D).

    At the sliding interface, the rotor solid-body velocity (``u = ω × r``)
    is enforced via a moving-wall equilibrium relaxation on the interface
    cells.

    Args:
        f: Full domain DF, shape ``(Q, nz, ny, nx)``.
        interface_mask: Boolean mask of interface annulus, shape ``(nz, ny, nx)``.
        theta: Accumulated rotation angle (radians).
        omega: Angular velocity (rad/step).
        rho: Density field, shape ``(nz, ny, nx)``.
        tau: BGK relaxation time.
        axis: Rotation axis.
        cx, cy, cz: Rotor centre in lattice coordinates.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.

    Returns:
        Updated DF with sliding-mesh BC applied.
    """
    nq, nz, ny, nx = f.shape
    device = f.device
    lattice_u = lattice.upper()

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    r_x = xx - cx
    r_y = yy - cy
    r_z = zz - cz

    # Tangential velocity: u_wall = ω × r
    if axis == "z":
        u_wall_x = -omega * r_y
        u_wall_y = omega * r_x
        u_wall_z = torch.zeros_like(r_x)
    elif axis == "x":
        u_wall_y = -omega * r_z
        u_wall_z = omega * r_y
        u_wall_x = torch.zeros_like(r_x)
    else:  # axis == "y"
        u_wall_z = -omega * r_x
        u_wall_x = omega * r_z
        u_wall_y = torch.zeros_like(r_x)

    # Enforce moving-wall equilibrium on interface cells
    u_eq_x = torch.where(interface_mask, u_wall_x, torch.zeros_like(u_wall_x))
    u_eq_y = torch.where(interface_mask, u_wall_y, torch.zeros_like(u_wall_y))
    u_eq_z = torch.where(interface_mask, u_wall_z, torch.zeros_like(u_wall_z))

    if lattice_u == "D3Q19":
        f_eq = equilibrium3d(rho, u_eq_x, u_eq_y, u_eq_z)
    elif lattice_u == "D3Q27":
        f_eq = equilibrium27(rho, u_eq_x, u_eq_y, u_eq_z)
    else:
        raise ValueError(
            f"lattice must be 'D3Q19' or 'D3Q27', got {lattice!r}"
        )

    mask_exp = interface_mask.unsqueeze(0).expand(nq, -1, -1, -1)
    omega_relax = 1.0 / tau
    f_interface = f - omega_relax * (f - f_eq)
    return torch.where(mask_exp, f_interface, f)


# --------------------------------------------------------------------------- #
# Unified dispatch: sliding_mesh_step
# --------------------------------------------------------------------------- #

def sliding_mesh_step(
    f: torch.Tensor,
    rotor_mask: torch.Tensor,
    omega: float,
    *,
    theta: float = 0.0,
    tau: float = 1.0,
    axis: RotationAxis = "z",
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    rotor_radius: float = 10.0,
    interface_width: float = 1.5,
    lattice: str = "auto",
) -> torch.Tensor:
    """Apply the sliding-mesh interface BC for one time step (D3Q19 / D3Q27).

    This is a **boundary update** that enforces the rotor solid-body velocity
    (``u = ω × r``) at the sliding interface via a moving-wall equilibrium
    relaxation.  It is collision-agnostic and can be combined with any
    collision operator, turbulence model, or IBM method.

    The ``rotor_mask`` marks the interface annulus where the sliding-mesh BC
    is applied.  The caller is responsible for:

    1. Collision (any operator).
    2. Calling ``sliding_mesh_step`` to update the interface.
    3. Bounce-back on solid cells (if any).
    4. Streaming.

    Args:
        f: Distribution tensor of shape ``(Q, nz, ny, nx)`` where *Q* is 19
           or 27.
        rotor_mask: Boolean mask of the interface annulus, shape
                    ``(nz, ny, nx)``.
        omega: Angular velocity (rad/step).
        theta: Accumulated rotation angle (radians).
        tau: BGK relaxation time for the moving-wall equilibrium.
        axis: Rotation axis (``"x"``, ``"y"``, or ``"z"``).
        cx, cy, cz: Rotor centre in lattice coordinates.
        rotor_radius: Rotor (interface) radius in lattice units (informational).
        interface_width: Half-width of the interface annulus (informational).
        lattice: ``"D3Q19"``, ``"D3Q27"``, or ``"auto"`` (inferred from
                 ``f.shape[0]``).

    Returns:
        Updated distribution tensor of the same shape as *f*.

    Raises:
        ValueError: If the lattice is unsupported or the shape is wrong.
    """
    if lattice == "auto":
        q = f.shape[0]
        if q == 19:
            lattice = "D3Q19"
        elif q == 27:
            lattice = "D3Q27"
        else:
            raise ValueError(
                f"Cannot auto-detect lattice from f.shape[0]={q}; "
                f"expected 19 (D3Q19) or 27 (D3Q27)"
            )

    lattice_u = lattice.upper()
    expected_q = 19 if lattice_u == "D3Q19" else 27
    if f.ndim != 4 or f.shape[0] != expected_q:
        raise ValueError(
            f"{lattice_u} populations must have shape ({expected_q}, nz, ny, nx), "
            f"got {tuple(f.shape)}"
        )

    # Compute density field from f
    if lattice_u == "D3Q19":
        rho, _, _, _ = macroscopic3d(f)
    else:
        rho, _, _, _ = macroscopic27(f)

    return apply_sliding_mesh_bc_3d(
        f,
        interface_mask=rotor_mask,
        theta=theta,
        omega=omega,
        rho=rho,
        tau=tau,
        axis=axis,
        cx=cx,
        cy=cy,
        cz=cz,
        lattice=lattice_u,
    )
