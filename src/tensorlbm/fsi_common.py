"""Common Fluid-Structure Interaction (FSI) module — composes IBM + 6-DOF.

This module provides a single solver-agnostic ``fsi_step`` that combines the
common IBM direct-forcing interface (:mod:`tensorlbm.ibm_common`) with the
common 6-DOF rigid-body integrator (:mod:`tensorlbm.sixdof_common`).  It can
be inserted into **any** collision → stream → boundary loop and composed with
arbitrary turbulence or multiphase models.

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

The body force on the fluid is the IBM direct-forcing field; the force on the
**body** is its Newton-third-law reaction (negative of the summed IBM force,
resolved about the body centre of mass).  This is a one-step explicit coupling
(no sub-iteration); it is the standard explicit FSI scheme used in
direct-forcing IBM.

This module does **not** modify the solver hot path.
"""
from __future__ import annotations

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
    "fsi_step",
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
