"""Free-surface LBM with ice lid (rigid top cover) boundary conditions.

This module extends the Körner free-surface model to support rigid top covers
(ice lids) with various slip conditions:

  (a) Full slip (specular / mirror) — no tangential momentum exchange
  (b) Specified slip velocity — moving ice lid with prescribed velocity
  (c) Navier partial slip — parameterized by slip length Ls

The ice lid is applied at the top boundary (y=ny-1) as a wall BC after
streaming, replacing the default free-surface treatment at that location.

This is the SI-1 coupling precursor: ice-under flow = top slip BC + free
surface evolution in the interior.

References
----------
- Körner et al. (2005). "Lattice Boltzmann model for free surface flows".
- SI-0A audit (scenarios/suboff_icebreaking/audits/SI-0A-TensorLBM.md).
- waLBerla FreeSlip / NoSlip boundary patterns.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch

from .boundaries3d import (
    free_slip_cells_3d,
)
from .d3q19 import C, W
from .free_surface_lbm import (
    free_surface_step,
    init_fill_rectangular,
    init_flags_from_fill,
    init_mass_from_fill,
)
from .navier_slip_bc import navier_slip_cells_3d


class IceLidMode(Enum):
    """Ice lid boundary condition modes."""

    FULL_SLIP = "full_slip"  # Specular reflection (mirror)
    SPECIFIED_VELOCITY = "specified"  # Moving lid with prescribed velocity
    NAVIER_SLIP = "navier"  # Partial slip, parameterized by Ls


def apply_ice_lid_bc(
    f: torch.Tensor,
    lid_mask: torch.Tensor,
    mode: IceLidMode = IceLidMode.FULL_SLIP,
    u_lid: float = 0.0,
    Ls: float = 0.0,
) -> torch.Tensor:
    """Apply ice lid boundary condition at top wall (y=ny-1).

    Args:
        f:        Distribution tensor, shape (19, nz, ny, nx).
        lid_mask: Boolean mask (nz, ny, nx) of lid cells (typically y=ny-1 plane).
        mode:     Ice lid BC mode (full_slip, specified, or navier).
        u_lid:    Prescribed lid velocity (x-direction) for 'specified' mode.
        Ls:       Slip length in lattice units for 'navier' mode.

    Returns:
        Updated distribution tensor with ice lid BC applied.
    """
    if mode == IceLidMode.FULL_SLIP:
        # Specular reflection: flip cy, keep cx, cz
        return free_slip_cells_3d(f, lid_mask, axis=1)

    elif mode == IceLidMode.SPECIFIED_VELOCITY:
        # Moving lid: specular reflection + velocity correction
        # First apply specular reflection
        f_new = free_slip_cells_3d(f, lid_mask, axis=1)
        # Then add linearised equilibrium shift for moving wall
        if u_lid != 0.0:
            device = f.device
            c_dev = C.to(device).float()
            w_dev = W.to(device).float()
            rho_lid = f_new.sum(dim=0)
            # δf_q = w_q * ρ * 3 * (c_q · u_lid) for incoming populations
            # For y-high wall, incoming are cy > 0 directions
            for q in range(19):
                if int(C[q, 1]) > 0:  # cy > 0: incoming at top wall
                    cq_dot_u = c_dev[q, 0] * u_lid  # c_q · u_lid (x-direction)
                    delta_f = w_dev[q] * rho_lid * 3.0 * cq_dot_u
                    f_new[q] = torch.where(lid_mask, f_new[q] + delta_f, f_new[q])
        return f_new

    elif mode == IceLidMode.NAVIER_SLIP:
        # Navier partial slip: interpolation between BB and specular
        return navier_slip_cells_3d(f, lid_mask, axis=1, Ls=Ls, u_wall=u_lid)

    else:
        raise ValueError(f"Unknown ice lid mode: {mode}")


def free_surface_with_ice_lid_step(
    f,
    fill,
    flags,
    solid_mask,
    mass=None,
    tau=1.0,
    gx=0.0,
    gy=0.0,
    gz=0.0,
    rho_liquid=1.0,
    rho_gas=1.0,
    surface_tension=0.0,
    C_s=0.0,
    sgs_model="smagorinsky",
    free_slip_y=False,
    y_wall_mask=None,
    bubble_pressure=None,
    collision="bgk",
    wall_function=False,
    near_mask=None,
    y_val=0.5,
    wf_force_coef=None,
    mass_ledger=None,
    freeze_topology=False,
    runtime_ledger=None,
    paired_liquid_interface_debit=False,
    ownership_ledger=None,
    inventory_reconciliation_ledger=None,
    conversion_density_audit_ledger=None,
    enable_i_to_g_ownership_closure=False,
    capture_replay_stages=False,
    replay_capture=None,
    # Ice lid parameters
    ice_lid_mode: IceLidMode = IceLidMode.FULL_SLIP,
    ice_lid_mask: Optional[torch.Tensor] = None,
    ice_lid_velocity: float = 0.0,
    ice_lid_slip_length: float = 0.0,
):
    """One free-surface LBM timestep with ice lid boundary at top.

    This is a wrapper around free_surface_step that applies an ice lid BC
    at the top boundary (y=ny-1) after the standard free-surface step.

    The ice lid replaces the default top-wall treatment with one of:
      - Full slip (specular reflection)
      - Specified velocity (moving lid)
      - Navier partial slip (parameterized by Ls)

    All other parameters are identical to free_surface_step.

    Args:
        (same as free_surface_step, plus:)
        ice_lid_mode:         Ice lid BC mode (full_slip, specified, or navier).
        ice_lid_mask:         Boolean mask for lid cells (y=ny-1 plane).
        ice_lid_velocity:     Prescribed lid velocity for 'specified' mode.
        ice_lid_slip_length:  Slip length for 'navier' mode.

    Returns:
        Tuple (f, fill, flags, mass) after one timestep.
    """
    # Call the base free_surface_step
    f, fill, flags, mass, df = free_surface_step(
        f=f,
        fill=fill,
        flags=flags,
        solid_mask=solid_mask,
        mass=mass,
        tau=tau,
        gx=gx,
        gy=gy,
        gz=gz,
        rho_liquid=rho_liquid,
        rho_gas=rho_gas,
        surface_tension=surface_tension,
        C_s=C_s,
        sgs_model=sgs_model,
        free_slip_y=free_slip_y,
        y_wall_mask=y_wall_mask,
        bubble_pressure=bubble_pressure,
        collision=collision,
        wall_function=wall_function,
        near_mask=near_mask,
        y_val=y_val,
        wf_force_coef=wf_force_coef,
        mass_ledger=mass_ledger,
        freeze_topology=freeze_topology,
        runtime_ledger=runtime_ledger,
        paired_liquid_interface_debit=paired_liquid_interface_debit,
        ownership_ledger=ownership_ledger,
        inventory_reconciliation_ledger=inventory_reconciliation_ledger,
        conversion_density_audit_ledger=conversion_density_audit_ledger,
        enable_i_to_g_ownership_closure=enable_i_to_g_ownership_closure,
        capture_replay_stages=capture_replay_stages,
        replay_capture=replay_capture,
    )

    # Apply ice lid BC if mask is provided
    if ice_lid_mask is not None:
        f = apply_ice_lid_bc(
            f,
            ice_lid_mask,
            mode=ice_lid_mode,
            u_lid=ice_lid_velocity,
            Ls=ice_lid_slip_length,
        )

    return f, fill, flags, mass


def init_ice_lid_domain(
    nz: int,
    ny: int,
    nx: int,
    fluid_height: int,
    device: torch.device,
    column_width: Optional[int] = None,
) -> dict:
    """Initialize a domain with free surface and ice lid at top.

    Creates a rectangular fluid column with:
      - Bottom wall (y=0): solid
      - Top wall (y=ny-1): ice lid (to be configured via BC mode)
      - Side walls (x=0, x=nx-1, z=0, z=nz-1): solid
      - Fluid occupies y ∈ [1, fluid_height]

    Args:
        nz, ny, nx:    Domain dimensions.
        fluid_height:  Height of fluid column (in lattice units).
        device:        Torch device.
        column_width:  Width of fluid column (default: nx-2, full width).

    Returns:
        Dictionary with keys: f, fill, flags, mass, solid_mask, ice_lid_mask.
    """
    if column_width is None:
        column_width = nx - 2

    # Initialize fill field for rectangular column
    fill, solid_mask = init_fill_rectangular(nz, ny, nx, column_width, fluid_height, device)

    # Initialize flags and mass
    flags = init_flags_from_fill(fill, solid_mask)
    mass = init_mass_from_fill(fill, flags, rho_liquid=1.0)

    # Initialize distributions to equilibrium (rest)
    from .d3q19 import equilibrium3d

    rho_init = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux_init = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    uy_init = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    uz_init = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    f = equilibrium3d(rho_init, ux_init, uy_init, uz_init)

    # Create ice lid mask (top wall y=ny-1)
    ice_lid_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    ice_lid_mask[:, -1, :] = True

    return {
        "f": f,
        "fill": fill,
        "flags": flags,
        "mass": mass,
        "solid_mask": solid_mask,
        "ice_lid_mask": ice_lid_mask,
    }
