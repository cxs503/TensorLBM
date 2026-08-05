"""Common Fluid-Structure Interaction (FSI) module — composes IBM + 6-DOF + spring-mass.

This module provides solver-agnostic FSI steps that combine the common IBM
direct-forcing interface (:mod:`tensorlbm.ibm_common`) with the common 6-DOF
rigid-body integrator (:mod:`tensorlbm.sixdof_common`) and a spring-mass-damper
system for vortex-induced vibration (VIV).  It can be inserted into **any**
collision → stream → boundary loop and composed with arbitrary turbulence or
multiphase models.

Three coupling modes are provided:

1. ``fsi_step`` — IBM direct-forcing + 6-DOF rigid-body (original, tested).
2. ``spring_mass_step`` — spring-mass-damper integrator for VIV / galloping.
3. ``fsi_step_drag`` — drag-based force (pressure + friction integration) +
   structure update (spring-mass or rigid-body), for moving-boundary FSI.

Public contract
----------------
``fsi_step(f, structure_state, mask, *, body, lattice, kernel, dt, ...)``

    * ``f``               – distribution tensor ``(Q, nz, ny, nx)``.
    * ``structure_state`` – :class:`RigidBodyState` of the moving body.
    * ``mask``            – solid mask ``(nz, ny, nx)``; ``True`` inside body.
    Returns ``(f_updated, structure_updated, force)``:
        - ``f_updated``        – distribution with IBM body-force correction.
        - ``structure_updated`` – advanced :class:`RigidBodyState`.
        - ``force``            – ``(6,)`` force/moment on the body (fluid → solid).

``spring_mass_step(state, force, dt, *, smd) -> SpringMassState``

    Advance a spring-mass-damper oscillator by one explicit step:
        m·ÿ + c·ẏ + k·y = F
    Natural frequency: f_n = (1/2π)·√(k/m)
    Damping ratio:     ζ = c / (2·√(k·m))

``fsi_step_drag(f, mesh, dpS, nu, structure, *, smd=None, body=None, dt=1.0)``

    Compute fluid force via drag_pressure_integration + drag_friction_integration,
    then advance the structure (spring-mass or rigid-body).

The body force on the fluid is the IBM direct-forcing field; the force on the
**body** is its Newton-third-law reaction (negative of the summed IBM force,
resolved about the body centre of mass).  This is a one-step explicit coupling
(no sub-iteration); it is the standard explicit FSI scheme used in
direct-forcing IBM.

This module does **not** modify the solver hot path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .ibm_common import (
    IBMLatticeName,
    IBMKernelName,
    derive_surface_markers_3d,
    ibm_direct_forcing_3d_common,
    macroscopic_velocity_3d,
)
from .sixdof_common import RigidBodyState, rigid_body_step
from .sixdof import SixDOFBody

__all__ = [
    "FSILatticeName",
    "FSICouplingName",
    "FSICapabilityWithheldError",
    "FSIResult",
    "FSIDragResult",
    "SpringMassDamper",
    "SpringMassState",
    "spring_mass_step",
    "fsi_step",
    "fsi_step_drag",
    "shift_solid_mask",
]

FSILatticeName = Literal["D3Q19", "D3Q27"]
FSICouplingName = Literal["one_way_explicit", "two_way_explicit"]


class FSICapabilityWithheldError(NotImplementedError):
    """Raised when an FSI capability request lacks a validated composition."""


@dataclass
class FSIResult:
    """Output of :func:`fsi_step`.

    Attributes:
        f_updated:         Distribution with IBM body-force correction applied.
        structure_updated:  Advanced rigid-body state.
        force_on_body:     ``(6,)`` force/moment ``[fx, fy, fz, mx, my, mz]``
                           exerted by the fluid on the body (SI / lattice units).
        force_on_fluid:    ``(3, nz, ny, nx)`` Eulerian IBM body-force field.
    """

    f_updated: torch.Tensor
    structure_updated: RigidBodyState
    force_on_body: torch.Tensor
    force_on_fluid: torch.Tensor


@dataclass
class FSIDragResult:
    """Output of :func:`fsi_step_drag`.

    Attributes:
        structure_updated: Advanced structure state (SpringMassState or
                           RigidBodyState, depending on which was passed).
        force:            ``(3,)`` force ``[fx, fy, fz]`` exerted by the fluid
                          on the body (lattice units).
        cd_pressure:      ``(3,)`` pressure-drag coefficients ``[Cdpx, Cdpy, Cdpz]``.
        cd_friction:      ``(3,)`` friction-drag coefficients.
        cd_total:         ``(3,)`` total drag coefficients.
    """

    structure_updated: object
    force: torch.Tensor
    cd_pressure: tuple
    cd_friction: tuple
    cd_total: tuple


# --------------------------------------------------------------------------- #
# Spring-mass-damper system (for VIV / galloping)
# --------------------------------------------------------------------------- #


@dataclass
class SpringMassDamper:
    """Spring-mass-damper oscillator properties for VIV / galloping.

    Models the 1-DOF (or multi-DOF) equation of motion:
        m·ÿ + c·ẏ + k·y = F_fluid

    Attributes:
        mass:       Oscillator mass [lattice units].
        stiffness:  Spring stiffness k [lattice units].
        damping:    Damping coefficient c [lattice units].
        n_dof:      Number of DOFs (1 = transverse only, 2 = x+y, 3 = xyz).
        gravity:    Gravity vector ``(3,)`` [lattice units], default zero.
    """

    mass: float
    stiffness: float = 0.0
    damping: float = 0.0
    n_dof: int = 1
    gravity: tuple[float, ...] = (0.0, 0.0, 0.0)

    @property
    def natural_frequency(self) -> float:
        """Natural frequency f_n = (1/2π)·√(k/m) [Hz, lattice units]."""
        if self.mass <= 0 or self.stiffness <= 0:
            return 0.0
        return (1.0 / (2.0 * math.pi)) * math.sqrt(self.stiffness / self.mass)

    @property
    def damping_ratio(self) -> float:
        """Damping ratio ζ = c / (2·√(k·m))."""
        if self.mass <= 0 or self.stiffness <= 0:
            return 0.0
        return self.damping / (2.0 * math.sqrt(self.stiffness * self.mass))

    @classmethod
    def from_mass_ratio_freq(
        cls,
        mass_ratio: float,
        rho_f: float,
        D: float,
        u_in: float,
        f_n: float,
        zeta: float,
        n_dof: int = 1,
    ) -> "SpringMassDamper":
        """Build SMD from non-dimensional VIV parameters.

        m* = m / (ρ_f · D^n), f_n = natural frequency, ζ = damping ratio.

        For 2D (n=2): m = m* · ρ_f · D²
        For 3D (n=3): m = m* · ρ_f · D³

        k = m · (2π·f_n)²
        c = 2·ζ·√(k·m)
        """
        n = 2 if n_dof <= 2 else 3
        mass = mass_ratio * rho_f * D ** n
        omega_n = 2.0 * math.pi * f_n
        k = mass * omega_n ** 2
        c = 2.0 * zeta * math.sqrt(k * mass)
        return cls(mass=mass, stiffness=k, damping=c, n_dof=n_dof)


@dataclass
class SpringMassState:
    """State of a spring-mass-damper oscillator.

    Attributes:
        disp:  Displacement ``(n_dof,)`` [lattice units].
        vel:   Velocity ``(n_dof,)`` [lattice units / step].
    """

    disp: torch.Tensor
    vel: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.disp, torch.Tensor):
            self.disp = torch.tensor(self.disp, dtype=torch.float64)
        if not isinstance(self.vel, torch.Tensor):
            self.vel = torch.tensor(self.vel, dtype=torch.float64)
        if self.disp.shape != self.vel.shape:
            raise ValueError(
                f"disp and vel must have the same shape; got "
                f"{tuple(self.disp.shape)} and {tuple(self.vel.shape)}."
            )

    def clone(self) -> "SpringMassState":
        return SpringMassState(self.disp.clone(), self.vel.clone())

    @classmethod
    def zero(cls, n_dof: int = 1, dtype: torch.dtype = torch.float64) -> "SpringMassState":
        return cls(
            torch.zeros(n_dof, dtype=dtype),
            torch.zeros(n_dof, dtype=dtype),
        )


def spring_mass_step(
    state: SpringMassState,
    force: torch.Tensor,
    dt: float,
    *,
    smd: SpringMassDamper,
) -> SpringMassState:
    """Advance a spring-mass-damper oscillator by one explicit step.

    Solves:  m·ÿ + c·ẏ + k·y = F

    Using Symplectic Euler (semi-implicit):
        1. a = (F - c·ẏ - k·y) / m
        2. v_new = v + a·dt
        3. y_new = y + v_new·dt

    Args:
        state: Current :class:`SpringMassState`.
        force: External force ``(n_dof,)`` (fluid force on body).
        dt:    Time step.
        smd:   :class:`SpringMassDamper` properties.

    Returns:
        Advanced :class:`SpringMassState`.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive; got {dt}.")
    m = smd.mass
    k = smd.stiffness
    c = smd.damping
    f = force.detach().to(torch.float64)
    if f.shape != state.disp.shape:
        f = f.reshape(state.disp.shape)

    # Spring + damping restoring force
    f_restore = -k * state.disp - c * state.vel
    # Gravity (if any)
    grav = torch.tensor(smd.gravity[: f.shape[0]], dtype=torch.float64)

    a = (f + f_restore + m * grav) / m
    vel_new = state.vel + a * dt
    disp_new = state.disp + vel_new * dt
    return SpringMassState(disp=disp_new, vel=vel_new)


# --------------------------------------------------------------------------- #
# Moving-boundary mask shifting
# --------------------------------------------------------------------------- #


def shift_solid_mask(
    solid: torch.Tensor,
    dx: int,
    dy: int,
    dz: int = 0,
) -> torch.Tensor:
    """Shift a solid mask by integer lattice displacements.

    Used for moving-boundary FSI where the body translates.  Cells that
    leave the domain are clipped; fresh cells (previously solid, now fluid)
    are filled with the equilibrium distribution by the caller.

    Args:
        solid: Solid mask ``(nz, ny, nx)``.
        dx, dy, dz: Integer shift in lattice cells.

    Returns:
        Shifted solid mask of the same shape.
    """
    nz, ny, nx = solid.shape
    shifted = torch.zeros_like(solid)
    # Source and destination ranges (clipped to domain)
    z0s = max(0, -dz); z1s = min(nz, nz - dz)
    y0s = max(0, -dy); y1s = min(ny, ny - dy)
    x0s = max(0, -dx); x1s = min(nx, nx - dx)
    z0d = max(0, dz); z1d = z0d + (z1s - z0s)
    y0d = max(0, dy); y1d = y0d + (y1s - y0s)
    x0d = max(0, dx); x1d = x0d + (x1s - x0s)
    if z1s > z0s and y1s > y0s and x1s > x0s:
        shifted[z0d:z1d, y0d:y1d, x0d:x1d] = solid[z0s:z1s, y0s:y1s, x0s:x1s]
    return shifted


# --------------------------------------------------------------------------- #
# Drag-based FSI step (pressure + friction integration → structure update)
# --------------------------------------------------------------------------- #


def fsi_step_drag(
    f: torch.Tensor,
    mesh,
    dpS: float,
    nu: float,
    structure,
    *,
    smd: SpringMassDamper | None = None,
    body: SixDOFBody | None = None,
    dt: float = 1.0,
    force_axis: int = 1,
    extrap: str = "none",
    p0_method: str = "far_field",
    solid: torch.Tensor | None = None,
    friction_formula: str = "standard",
) -> FSIDragResult:
    """One FSI step using drag integration for force + structure update.

    Computes the fluid force on the body via
    :func:`drag_pressure_integration` + :func:`drag_friction_integration`,
    then advances the structure (spring-mass-damper or rigid-body).

    Args:
        f:          Distribution tensor ``(Q, nz, ny, nx)``.
        mesh:       :class:`SurfaceMesh` with precomputed normals.
        dpS:        Dynamic pressure scale (0.5·ρ·U²·A).
        nu:         Lattice kinematic viscosity.
        structure:  Current structure state (:class:`SpringMassState` or
                    :class:`RigidBodyState`).
        smd:        Spring-mass-damper properties (required for SMD structure).
        body:       :class:`SixDOFBody` (required for rigid-body structure).
        dt:         Time step.
        force_axis: Axis index (0=x, 1=y, 2=z) of the transverse force driving
                    the spring-mass oscillator (for 1-DOF VIV).
        extrap:     Pressure extrapolation method ('none', 'linear', 'quadratic').
        p0_method:  Background pressure method.
        solid:      Solid mask (required for p0_method != 'near_wall').
        friction_formula: Friction formula ('standard', '2nd_order', etc.).

    Returns:
        :class:`FSIDragResult` with advanced structure state and force/CD.
    """
    from .drag_pressure import (
        drag_pressure_integration,
        drag_friction_integration,
    )

    # 1. Compute fluid force via drag integration.
    fx_p, fy_p, fz_p = drag_pressure_integration(
        f, mesh, dpS, extrap=extrap, p0_method=p0_method, solid=solid,
    )
    fx_f, fy_f, fz_f = drag_friction_integration(
        f, mesh, dpS, nu, formula=friction_formula,
    )
    cd_p = (fx_p, fy_p, fz_p)
    cd_f = (fx_f, fy_f, fz_f)
    cd_t = (fx_p + fx_f, fy_p + fy_f, fz_p + fz_f)

    # Force in lattice units: F = Cd · dpS  (dpS = 0.5·ρ·U²·A)
    # For spring-mass, we need the actual force, not just Cd.
    force_vec = torch.tensor(
        [cd_t[0] * dpS, cd_t[1] * dpS, cd_t[2] * dpS],
        dtype=torch.float64,
    )

    # 2. Advance the structure.
    if isinstance(structure, SpringMassState):
        if smd is None:
            raise ValueError("smd (SpringMassDamper) is required for SpringMassState.")
        if smd.n_dof == 1:
            # 1-DOF: use only the transverse force component
            f_drive = force_vec[force_axis].reshape(1)
        else:
            f_drive = force_vec[: smd.n_dof]
        struct_new = spring_mass_step(structure, f_drive, dt, smd=smd)
    elif isinstance(structure, RigidBodyState):
        if body is None:
            raise ValueError("body (SixDOFBody) is required for RigidBodyState.")
        force_6 = torch.tensor(
            [cd_t[0] * dpS, cd_t[1] * dpS, cd_t[2] * dpS, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        struct_new = rigid_body_step(structure, force_6, dt, body=body)
    else:
        raise TypeError(
            f"structure must be SpringMassState or RigidBodyState; "
            f"got {type(structure).__name__}."
        )

    return FSIDragResult(
        structure_updated=struct_new,
        force=force_vec,
        cd_pressure=cd_p,
        cd_friction=cd_f,
        cd_total=cd_t,
    )


def _normalise_lattice(lattice: str) -> FSILatticeName:
    value = lattice.upper()
    if value not in {"D3Q19", "D3Q27"}:
        raise FSICapabilityWithheldError(
            f"WITHHELD_UNKNOWN_LATTICE: {lattice!r} is not an audited FSI lattice."
        )
    return value  # type: ignore[return-value]


def _normalise_coupling(coupling: str) -> str:
    value = coupling.lower().replace("-", "_")
    aliases = {
        "one_way": "one_way_explicit",
        "one_way_explicit": "one_way_explicit",
        "explicit": "one_way_explicit",
        "two_way": "two_way_explicit",
        "two_way_explicit": "two_way_explicit",
    }
    if value not in aliases:
        raise FSICapabilityWithheldError(
            f"WITHHELD_UNKNOWN_COUPLING: {coupling!r} is not an audited FSI coupling mode."
        )
    return aliases[value]


def _body_centroid(mask: torch.Tensor) -> tuple[float, float, float]:
    """Return the (x, y, z) centroid of the solid mask in lattice coordinates."""
    if not mask.any():
        return 0.0, 0.0, 0.0
    iz, iy, ix = torch.where(mask)
    return (
        float(ix.float().mean()),
        float(iy.float().mean()),
        float(iz.float().mean()),
    )


def fsi_step(
    f: torch.Tensor,
    structure_state: RigidBodyState,
    mask: torch.Tensor,
    *,
    body: SixDOFBody,
    lattice: FSILatticeName = "D3Q19",
    kernel: IBMKernelName = "hat",
    dt: float = 1.0,
    coupling: FSICouplingName = "one_way_explicit",
    markers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    u_target: torch.Tensor | None = None,
) -> FSIResult:
    """Perform one explicit FSI step: IBM direct forcing + 6-DOF rigid-body advance.

    The step proceeds as:

    1. **IBM force**: compute the direct-forcing body force on the fluid needed
       to enforce the body velocity (no-slip) at the immersed boundary markers.
       The target marker velocity is the body's translational velocity
       (``structure_state.vel``) unless ``u_target`` is explicitly provided.
    2. **Reaction force**: the force on the body is the negative of the summed
       IBM fluid force, resolved about the body centroid to produce moments.
    3. **Rigid-body advance**: advance the 6-DOF state by ``dt`` using the
       reaction force (plus gravity from ``body``).

    Args:
        f:               Distribution tensor ``(Q, nz, ny, nx)``.
        structure_state: Current :class:`RigidBodyState`.
        mask:            Solid mask ``(nz, ny, nx)``.
        body:            :class:`SixDOFBody` physical properties.
        lattice:         ``"D3Q19"`` or ``"D3Q27"``.
        kernel:          IBM delta kernel: ``"hat"`` or ``"4pt"``.
        dt:              Time step [s] (lattice units if ``dt=1``).
        coupling:        ``"one_way_explicit"`` (default) or
                         ``"two_way_explicit"``.  Both use the same explicit
                         one-step scheme; ``two_way`` re-applies the advanced
                         body velocity as the target for a second IBM pass.
        markers:         Optional explicit marker positions.
        u_target:        Optional explicit target marker velocity.  When
                         ``None``, the body's translational velocity is used.

    Returns:
        :class:`FSIResult` with the updated distribution, advanced rigid-body
        state, and force/moment on the body.
    """
    lattice_name = _normalise_lattice(lattice)
    coupling_name = _normalise_coupling(coupling)

    # 1. Resolve target marker velocity from the body state.
    if u_target is None:
        # Body translational velocity in the world frame → uniform target.
        u_target_resolved = structure_state.vel.detach().to(f.dtype).clone()
    else:
        u_target_resolved = u_target

    # 2. IBM direct forcing on the fluid.
    force_on_fluid, f_corrected = ibm_direct_forcing_3d_common(
        f, mask, u_target_resolved,
        lattice=lattice_name, kernel=kernel, markers=markers,
    )

    # 3. Reaction force on the body = −Σ IBM fluid force.
    #    Sum over the Eulerian grid (force is (3, nz, ny, nx)).
    fx_total = float(force_on_fluid[0].sum().item())
    fy_total = float(force_on_fluid[1].sum().item())
    fz_total = float(force_on_fluid[2].sum().item())
    force_on_body = torch.tensor(
        [-fx_total, -fy_total, -fz_total, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )

    # Resolve moments about the body centroid.
    cx, cy, cz = _body_centroid(mask)
    nz, ny, nx = mask.shape
    iz_grid, iy_grid, ix_grid = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    dx = ix_grid - cx
    dy = iy_grid - cy
    dz = iz_grid - cz
    # M = r × F; for each grid cell: M = r × F_cell, summed.
    mx_total = float((dy * force_on_fluid[2].double() - dz * force_on_fluid[1].double()).sum().item())
    my_total = float((dz * force_on_fluid[0].double() - dx * force_on_fluid[2].double()).sum().item())
    mz_total = float((dx * force_on_fluid[1].double() - dy * force_on_fluid[0].double()).sum().item())
    force_on_body[3] = -mx_total
    force_on_body[4] = -my_total
    force_on_body[5] = -mz_total

    # 4. Advance the rigid body.
    structure_updated = rigid_body_step(
        structure_state, force_on_body, dt, body=body,
    )

    # 5. Two-way explicit: re-apply IBM with the advanced body velocity.
    if coupling_name == "two_way_explicit":
        u_target_2 = structure_updated.vel.detach().to(f.dtype).clone()
        force_on_fluid_2, f_corrected = ibm_direct_forcing_3d_common(
            f, mask, u_target_2,
            lattice=lattice_name, kernel=kernel, markers=markers,
        )
        # Recompute reaction force with the second pass.
        fx2 = float(force_on_fluid_2[0].sum().item())
        fy2 = float(force_on_fluid_2[1].sum().item())
        fz2 = float(force_on_fluid_2[2].sum().item())
        mx2 = float((dy * force_on_fluid_2[2].double() - dz * force_on_fluid_2[1].double()).sum().item())
        my2 = float((dz * force_on_fluid_2[0].double() - dx * force_on_fluid_2[2].double()).sum().item())
        mz2 = float((dx * force_on_fluid_2[1].double() - dy * force_on_fluid_2[0].double()).sum().item())
        force_on_body = torch.tensor(
            [-fx2, -fy2, -fz2, -mx2, -my2, -mz2],
            dtype=torch.float64,
        )
        force_on_fluid = force_on_fluid_2

    return FSIResult(
        f_updated=f_corrected,
        structure_updated=structure_updated,
        force_on_body=force_on_body,
        force_on_fluid=force_on_fluid,
    )
