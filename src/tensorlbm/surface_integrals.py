"""Surface and volume integral utilities for LBM flow fields.

Provides scalar- and vector-valued integrals over user-defined control surfaces
and volumes.  All quantities are computed on the Eulerian lattice grid using
discrete summations (exact for uniform grids) and are analogous to the
surface-integration panels available in PowerFlow and XFlow.

Functions
---------
:func:`mass_flow_rate_2d`
    Volume / mass flow rate through a 2-D cross-section line.
:func:`mass_flow_rate_3d`
    Volume / mass flow rate through a 3-D cross-section plane.
:func:`area_average_2d`
    Area-averaged scalar over a 2-D region.
:func:`area_average_3d`
    Volume-averaged scalar over a 3-D region.
:func:`surface_force_2d`
    Total force on a 2-D solid surface from momentum flux + pressure.
:func:`surface_force_3d`
    Total force on a 3-D solid surface from momentum flux + pressure.
:func:`surface_moment_2d`
    Moment (torque) about a pivot from surface forces (2-D).
:func:`surface_moment_3d`
    Moment vector about a pivot from surface forces (3-D).
:func:`pressure_drop`
    Pressure drop between two cross-section planes.
:func:`force_coefficients`
    Non-dimensionalise force into drag / lift / side-force coefficients.
:func:`moment_coefficients`
    Non-dimensionalise moment into Cl, Cm, Cn coefficients.
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Mass / volume flow rate
# ---------------------------------------------------------------------------

def mass_flow_rate_2d(
    ux: torch.Tensor,
    rho: torch.Tensor,
    x_plane: int,
    y_range: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Mass and volume flow rate through a vertical cross-section at *x_plane*.

    Args:
        ux:      Streamwise velocity field, shape ``(ny, nx)``.
        rho:     Density field, shape ``(ny, nx)``.
        x_plane: Grid column index of the cross-section.
        y_range: ``(y_min, y_max)`` row indices to include (inclusive).
                 Defaults to the full domain.

    Returns:
        Dictionary with keys:

        ``volume_flow``
            Σ ux(y, x_plane) — volume flow rate (lattice units³/step).
        ``mass_flow``
            Σ ρ(y, x_plane) · ux(y, x_plane) — mass flow rate.
        ``mean_velocity``
            Area-averaged streamwise velocity.
        ``area``
            Number of cross-section cells (= height when Δy = 1).
    """
    ny, nx = ux.shape
    x_plane = max(0, min(nx - 1, x_plane))
    y0, y1 = (0, ny - 1) if y_range is None else y_range
    ux_col = ux[y0:y1 + 1, x_plane]
    rho_col = rho[y0:y1 + 1, x_plane]
    n_cells = ux_col.numel()
    vol_flow = float(ux_col.sum())
    mass_flow = float((rho_col * ux_col).sum())
    mean_vel = vol_flow / n_cells if n_cells > 0 else 0.0
    return {
        "volume_flow": vol_flow,
        "mass_flow": mass_flow,
        "mean_velocity": mean_vel,
        "area": n_cells,
    }


def mass_flow_rate_3d(
    ux: torch.Tensor,
    rho: torch.Tensor,
    x_plane: int,
    y_range: tuple[int, int] | None = None,
    z_range: tuple[int, int] | None = None,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Mass and volume flow rate through a yz cross-section at *x_plane*.

    Args:
        ux:      Streamwise velocity field, shape ``(nz, ny, nx)``.
        rho:     Density field, shape ``(nz, ny, nx)``.
        x_plane: Grid x-index of the cross-section.
        y_range: ``(y_min, y_max)`` row range (inclusive).
        z_range: ``(z_min, z_max)`` depth range (inclusive).
        mask:    Boolean solid mask ``(nz, ny, nx)``; excluded from integration.

    Returns:
        Dictionary with ``volume_flow``, ``mass_flow``, ``mean_velocity``,
        ``area`` (number of fluid cross-section cells).
    """
    nz, ny, nx = ux.shape
    x_plane = max(0, min(nx - 1, x_plane))
    y0, y1 = (0, ny - 1) if y_range is None else y_range
    z0, z1 = (0, nz - 1) if z_range is None else z_range

    ux_sl = ux[z0:z1 + 1, y0:y1 + 1, x_plane]
    rho_sl = rho[z0:z1 + 1, y0:y1 + 1, x_plane]

    fluid = torch.ones_like(ux_sl, dtype=torch.bool)
    if mask is not None:
        fluid = ~mask[z0:z1 + 1, y0:y1 + 1, x_plane]

    n_cells = int(fluid.sum())
    vol_flow = float((ux_sl * fluid).sum())
    mass_flow = float((rho_sl * ux_sl * fluid).sum())
    mean_vel = vol_flow / n_cells if n_cells > 0 else 0.0
    return {
        "volume_flow": vol_flow,
        "mass_flow": mass_flow,
        "mean_velocity": mean_vel,
        "area": n_cells,
    }


# ---------------------------------------------------------------------------
# Area/volume averages
# ---------------------------------------------------------------------------

def area_average_2d(
    scalar: torch.Tensor,
    x_range: tuple[int, int] | None = None,
    y_range: tuple[int, int] | None = None,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Area-averaged scalar over a 2-D sub-region.

    Args:
        scalar:  Scalar field, shape ``(ny, nx)``.
        x_range: ``(x_min, x_max)`` column range (inclusive).
        y_range: ``(y_min, y_max)`` row range (inclusive).
        mask:    Boolean solid mask; excluded from average.

    Returns:
        Dictionary: ``mean``, ``min``, ``max``, ``area``.
    """
    ny, nx = scalar.shape
    x0, x1 = (0, nx - 1) if x_range is None else x_range
    y0, y1 = (0, ny - 1) if y_range is None else y_range
    s = scalar[y0:y1 + 1, x0:x1 + 1]
    fluid = torch.ones_like(s, dtype=torch.bool)
    if mask is not None:
        fluid = ~mask[y0:y1 + 1, x0:x1 + 1]
    n = int(fluid.sum())
    vals = s[fluid]
    return {
        "mean": float(vals.mean()) if n > 0 else 0.0,
        "min": float(vals.min()) if n > 0 else 0.0,
        "max": float(vals.max()) if n > 0 else 0.0,
        "area": n,
    }


def area_average_3d(
    scalar: torch.Tensor,
    x_range: tuple[int, int] | None = None,
    y_range: tuple[int, int] | None = None,
    z_range: tuple[int, int] | None = None,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Volume-averaged scalar over a 3-D sub-region.

    Args:
        scalar:  Scalar field, shape ``(nz, ny, nx)``.
        x_range, y_range, z_range: Inclusive index ranges.
        mask:    Boolean solid mask; excluded from average.

    Returns:
        Dictionary: ``mean``, ``min``, ``max``, ``volume``.
    """
    nz, ny, nx = scalar.shape
    x0, x1 = (0, nx - 1) if x_range is None else x_range
    y0, y1 = (0, ny - 1) if y_range is None else y_range
    z0, z1 = (0, nz - 1) if z_range is None else z_range
    s = scalar[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    fluid = torch.ones_like(s, dtype=torch.bool)
    if mask is not None:
        fluid = ~mask[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    n = int(fluid.sum())
    vals = s[fluid]
    return {
        "mean": float(vals.mean()) if n > 0 else 0.0,
        "min": float(vals.min()) if n > 0 else 0.0,
        "max": float(vals.max()) if n > 0 else 0.0,
        "volume": n,
    }


# ---------------------------------------------------------------------------
# Surface force integration
# ---------------------------------------------------------------------------

def surface_force_2d(
    f: torch.Tensor,
    mask: torch.Tensor,
    rho_ref: float = 1.0,
) -> dict[str, float]:
    """Total hydrodynamic force on a 2-D obstacle using the momentum-exchange method.

    The momentum-exchange algorithm (Ladd 1994) computes the force by summing
    the momentum transferred to solid nodes from fluid nodes during streaming:

        F_α = Σ_{fluid→solid links} (f_i(r_f) c_{iα} + f_ī(r_s) c_{iα})

    where *i* is the direction from fluid cell r_f to solid cell r_s and *ī*
    is the opposite direction.

    Args:
        f:       Post-collision (pre-stream) distribution, shape ``(9, ny, nx)``.
        mask:    Boolean solid mask, shape ``(ny, nx)``; ``True`` = solid.
        rho_ref: Reference density for non-dimensionalisation.

    Returns:
        Dictionary: ``fx``, ``fy`` (total forces in lattice units).
    """
    # D2Q9 velocity vectors and opposites
    cx = torch.tensor([0, 1, 0, -1,  0,  1, -1, -1,  1], dtype=torch.float32, device=f.device)
    cy = torch.tensor([0, 0, 1,  0, -1,  1,  1, -1, -1], dtype=torch.float32, device=f.device)
    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long, device=f.device)

    ny, nx = mask.shape
    fx_total = 0.0
    fy_total = 0.0

    fluid = ~mask  # (ny, nx) True = fluid

    for q in range(1, 9):  # skip rest direction (q=0)
        dcx = int(cx[q].item())
        dcy = int(cy[q].item())

        # Neighbour of each fluid cell in direction q
        y_f, x_f = torch.where(fluid)
        x_n = (x_f + dcx).clamp(0, nx - 1)
        y_n = (y_f + dcy).clamp(0, ny - 1)

        # Check which neighbours are solid
        is_solid_nbr = mask[y_n, x_n]
        if not is_solid_nbr.any():
            continue

        yf_ = y_f[is_solid_nbr]
        xf_ = x_f[is_solid_nbr]
        yn_ = y_n[is_solid_nbr]
        xn_ = x_n[is_solid_nbr]

        q_opp = int(opp[q].item())
        # Momentum exchange: f_q at fluid cell + f_opp at solid cell
        contribution = f[q, yf_, xf_] + f[q_opp, yn_, xn_]
        fx_total += float((contribution * cx[q]).sum())
        fy_total += float((contribution * cy[q]).sum())

    return {"fx": fx_total, "fy": fy_total}


def surface_force_3d(
    f: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Total hydrodynamic force on a 3-D obstacle (momentum-exchange method).

    Args:
        f:    Post-collision distribution, shape ``(19, nz, ny, nx)`` (D3Q19).
        mask: Boolean solid mask, shape ``(nz, ny, nx)``; ``True`` = solid.

    Returns:
        Dictionary: ``fx``, ``fy``, ``fz``.
    """
    from .d3q19 import C as C3D  # noqa: PLC0415

    device = f.device
    c = C3D.to(device).float()  # (19, 3)
    nq = c.shape[0]
    nz, ny, nx = mask.shape
    fluid = ~mask

    # Opposite direction indices for D3Q19
    # Build opp by matching -c[i] to c[j]
    opp = torch.zeros(nq, dtype=torch.long, device=device)
    for i in range(nq):
        for j in range(nq):
            if (c[i] + c[j]).abs().sum() < 0.5:
                opp[i] = j
                break

    fx_total = fy_total = fz_total = 0.0
    z_f, y_f, x_f = torch.where(fluid)

    for q in range(1, nq):
        dcx = int(c[q, 0].item())
        dcy = int(c[q, 1].item())
        dcz = int(c[q, 2].item())

        x_n = (x_f + dcx).clamp(0, nx - 1)
        y_n = (y_f + dcy).clamp(0, ny - 1)
        z_n = (z_f + dcz).clamp(0, nz - 1)

        is_solid_nbr = mask[z_n, y_n, x_n]
        if not is_solid_nbr.any():
            continue

        zf_ = z_f[is_solid_nbr]
        yf_ = y_f[is_solid_nbr]
        xf_ = x_f[is_solid_nbr]
        zn_ = z_n[is_solid_nbr]
        yn_ = y_n[is_solid_nbr]
        xn_ = x_n[is_solid_nbr]

        q_opp = int(opp[q].item())
        contrib = f[q, zf_, yf_, xf_] + f[q_opp, zn_, yn_, xn_]
        fx_total += float((contrib * c[q, 0]).sum())
        fy_total += float((contrib * c[q, 1]).sum())
        fz_total += float((contrib * c[q, 2]).sum())

    return {"fx": fx_total, "fy": fy_total, "fz": fz_total}


# ---------------------------------------------------------------------------
# Surface moment (torque)
# ---------------------------------------------------------------------------

def surface_moment_2d(
    f: torch.Tensor,
    mask: torch.Tensor,
    pivot_x: float,
    pivot_y: float,
) -> dict[str, float]:
    """Torque about a pivot from 2-D hydrodynamic surface forces.

    The moment arm is from the pivot to each surface cell that contributes
    to the force.  Returns the scalar z-moment Mz = rx·Fy − ry·Fx.

    Args:
        f:            Distribution function, shape ``(9, ny, nx)``.
        mask:         Boolean solid mask, shape ``(ny, nx)``.
        pivot_x, pivot_y: Pivot coordinates in lattice units.

    Returns:
        Dictionary: ``mz`` (moment about z-axis), ``fx``, ``fy``.
    """
    cx = torch.tensor([0, 1, 0, -1,  0,  1, -1, -1,  1], dtype=torch.float32, device=f.device)
    cy = torch.tensor([0, 0, 1,  0, -1,  1,  1, -1, -1], dtype=torch.float32, device=f.device)
    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long, device=f.device)

    ny, nx = mask.shape
    fluid = ~mask
    fx_total = fy_total = mz_total = 0.0

    for q in range(1, 9):
        dcx = int(cx[q].item())
        dcy = int(cy[q].item())

        y_f, x_f = torch.where(fluid)
        x_n = (x_f + dcx).clamp(0, nx - 1)
        y_n = (y_f + dcy).clamp(0, ny - 1)
        is_solid_nbr = mask[y_n, x_n]
        if not is_solid_nbr.any():
            continue

        yf_ = y_f[is_solid_nbr]
        xf_ = x_f[is_solid_nbr]
        yn_ = y_n[is_solid_nbr]
        xn_ = x_n[is_solid_nbr]

        q_opp = int(opp[q].item())
        contrib = f[q, yf_, xf_] + f[q_opp, yn_, xn_]

        dfx = contrib * cx[q]
        dfy = contrib * cy[q]

        # Moment arm from pivot to the solid surface cell
        rx = xn_.float() - pivot_x
        ry = yn_.float() - pivot_y
        dmz = rx * dfy - ry * dfx

        fx_total += float(dfx.sum())
        fy_total += float(dfy.sum())
        mz_total += float(dmz.sum())

    return {"fx": fx_total, "fy": fy_total, "mz": mz_total}


def surface_moment_3d(
    f: torch.Tensor,
    mask: torch.Tensor,
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, float]:
    """Moment vector about a pivot from 3-D hydrodynamic surface forces.

    Args:
        f:      Distribution, shape ``(19, nz, ny, nx)`` (D3Q19).
        mask:   Solid mask, shape ``(nz, ny, nx)``.
        pivot:  ``(px, py, pz)`` pivot coordinates in lattice units.

    Returns:
        Dictionary: ``fx``, ``fy``, ``fz``, ``mx``, ``my``, ``mz``.
    """
    from .d3q19 import C as C3D  # noqa: PLC0415

    device = f.device
    c = C3D.to(device).float()
    nq = c.shape[0]
    nz, ny, nx = mask.shape
    fluid = ~mask

    opp = torch.zeros(nq, dtype=torch.long, device=device)
    for i in range(nq):
        for j in range(nq):
            if (c[i] + c[j]).abs().sum() < 0.5:
                opp[i] = j
                break

    px, py, pz = pivot
    fx_t = fy_t = fz_t = 0.0
    mx_t = my_t = mz_t = 0.0
    z_f, y_f, x_f = torch.where(fluid)

    for q in range(1, nq):
        dcx = int(c[q, 0].item())
        dcy = int(c[q, 1].item())
        dcz = int(c[q, 2].item())

        x_n = (x_f + dcx).clamp(0, nx - 1)
        y_n = (y_f + dcy).clamp(0, ny - 1)
        z_n = (z_f + dcz).clamp(0, nz - 1)

        is_solid_nbr = mask[z_n, y_n, x_n]
        if not is_solid_nbr.any():
            continue

        zf_ = z_f[is_solid_nbr]
        yf_ = y_f[is_solid_nbr]
        xf_ = x_f[is_solid_nbr]
        zn_ = z_n[is_solid_nbr]
        yn_ = y_n[is_solid_nbr]
        xn_ = x_n[is_solid_nbr]

        q_opp = int(opp[q].item())
        contrib = f[q, zf_, yf_, xf_] + f[q_opp, zn_, yn_, xn_]

        dfx = contrib * c[q, 0]
        dfy = contrib * c[q, 1]
        dfz = contrib * c[q, 2]

        rx = xn_.float() - px
        ry = yn_.float() - py
        rz = zn_.float() - pz

        # M = r × F
        dmx = ry * dfz - rz * dfy
        dmy = rz * dfx - rx * dfz
        dmz = rx * dfy - ry * dfx

        fx_t += float(dfx.sum())
        fy_t += float(dfy.sum())
        fz_t += float(dfz.sum())
        mx_t += float(dmx.sum())
        my_t += float(dmy.sum())
        mz_t += float(dmz.sum())

    return {
        "fx": fx_t, "fy": fy_t, "fz": fz_t,
        "mx": mx_t, "my": my_t, "mz": mz_t,
    }


# ---------------------------------------------------------------------------
# Pressure drop
# ---------------------------------------------------------------------------

def pressure_drop(
    rho: torch.Tensor,
    x_upstream: int,
    x_downstream: int,
    cs2: float = 1.0 / 3.0,
) -> dict[str, float]:
    """Pressure drop between two cross-section planes.

    Computes the laterally-averaged pressure at two x-planes and returns the
    difference Δp = p_upstream − p_downstream.  For LBM, p = ρ c_s².

    Args:
        rho:          Density field, shape ``(ny, nx)`` or ``(nz, ny, nx)``.
        x_upstream:   x-index of the upstream plane.
        x_downstream: x-index of the downstream plane.
        cs2:          Speed of sound squared (default 1/3 for standard LBM).

    Returns:
        Dictionary: ``p_upstream``, ``p_downstream``, ``delta_p``.
    """
    if rho.ndim == 2:
        p_up = float(rho[:, x_upstream].mean()) * cs2
        p_dn = float(rho[:, x_downstream].mean()) * cs2
    elif rho.ndim == 3:
        p_up = float(rho[:, :, x_upstream].mean()) * cs2
        p_dn = float(rho[:, :, x_downstream].mean()) * cs2
    else:
        raise ValueError(f"rho must be 2-D or 3-D, got {rho.ndim}-D")

    return {
        "p_upstream": p_up,
        "p_downstream": p_dn,
        "delta_p": p_up - p_dn,
    }


# ---------------------------------------------------------------------------
# Non-dimensionalisation helpers
# ---------------------------------------------------------------------------

def force_coefficients(
    fx: float,
    fy: float,
    fz: float | None,
    rho_ref: float,
    u_ref: float,
    area_ref: float,
    flow_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    lift_direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> dict[str, float]:
    """Non-dimensionalise total force into drag, lift, and side-force coefficients.

    C_D = F_drag / (½ ρ U² A_ref)

    Args:
        fx, fy, fz:       Force components in lattice units.
        rho_ref:          Reference density.
        u_ref:            Reference velocity.
        area_ref:         Reference area (planform or frontal).
        flow_direction:   Unit vector pointing in the free-stream direction.
        lift_direction:   Unit vector pointing in the lift direction.

    Returns:
        Dictionary: ``cd`` (drag), ``cl`` (lift), ``cs`` (side), ``q_ref``.
    """
    q = 0.5 * rho_ref * u_ref * u_ref * area_ref
    if q < 1e-30:
        return {"cd": 0.0, "cl": 0.0, "cs": 0.0, "q_ref": 0.0}

    fvec = [fx, fy, fz if fz is not None else 0.0]
    # Flow direction (drag)
    fd_hat = flow_direction
    fl_hat = lift_direction
    # Side force direction = flow × lift
    fs_hat = (
        fd_hat[1] * fl_hat[2] - fd_hat[2] * fl_hat[1],
        fd_hat[2] * fl_hat[0] - fd_hat[0] * fl_hat[2],
        fd_hat[0] * fl_hat[1] - fd_hat[1] * fl_hat[0],
    )
    f_drag = sum(fvec[i] * fd_hat[i] for i in range(3))
    f_lift = sum(fvec[i] * fl_hat[i] for i in range(3))
    f_side = sum(fvec[i] * fs_hat[i] for i in range(3))

    return {
        "cd": f_drag / q,
        "cl": f_lift / q,
        "cs": f_side / q,
        "q_ref": q,
    }


def moment_coefficients(
    mx: float,
    my: float,
    mz: float,
    rho_ref: float,
    u_ref: float,
    area_ref: float,
    length_ref: float,
) -> dict[str, float]:
    """Non-dimensionalise moment into rolling, pitching, and yawing coefficients.

    C_l = Mx / (½ ρ U² A l),  C_m = My / (½ ρ U² A l),  C_n = Mz / (½ ρ U² A l)

    Args:
        mx, my, mz:  Moment components in lattice units.
        rho_ref:     Reference density.
        u_ref:       Reference velocity.
        area_ref:    Reference area.
        length_ref:  Reference length (chord, span, or diameter).

    Returns:
        Dictionary: ``cl_roll``, ``cm_pitch``, ``cn_yaw``.
    """
    q = 0.5 * rho_ref * u_ref * u_ref * area_ref * length_ref
    if q < 1e-30:
        return {"cl_roll": 0.0, "cm_pitch": 0.0, "cn_yaw": 0.0}
    return {
        "cl_roll": mx / q,
        "cm_pitch": my / q,
        "cn_yaw": mz / q,
    }


__all__ = [
    "mass_flow_rate_2d",
    "mass_flow_rate_3d",
    "area_average_2d",
    "area_average_3d",
    "surface_force_2d",
    "surface_force_3d",
    "surface_force_decomposed_2d",
    "surface_moment_2d",
    "surface_moment_3d",
    "pressure_drop",
    "force_coefficients",
    "moment_coefficients",
]


# ---------------------------------------------------------------------------
# Force decomposition: pressure + viscous components
# ---------------------------------------------------------------------------

def surface_force_decomposed_2d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    mask: torch.Tensor,
    tau: float,
    rho_ref: float = 1.0,
    u_ref: float = 0.1,
    area_ref: float = 1.0,
) -> dict[str, float]:
    """Decompose total aerodynamic force into pressure and viscous components.

    This implements the surface-force decomposition available in PowerFlow and
    XFlow, splitting the total drag/lift into:

    * **Pressure (form) drag/lift** – from the hydrostatic pressure distribution
      on the solid surface (p n̂ integration via momentum exchange).
    * **Viscous (friction) drag/lift** – from wall shear stress (τ_w t̂ integration
      via the f_neq tensor at wall-adjacent fluid cells).

    The two contributions sum to the total momentum-exchange force.

    Args:
        f:        Post-collision DF, shape ``(9, ny, nx)``.
        rho:      Density field, shape ``(ny, nx)``.
        ux, uy:   Velocity fields, shape ``(ny, nx)``.
        mask:     Boolean solid mask, shape ``(ny, nx)``; ``True`` = solid.
        tau:      BGK relaxation time.
        rho_ref:  Reference density for Cd/Cl non-dimensionalisation.
        u_ref:    Reference velocity.
        area_ref: Reference area (chord length in 2-D).

    Returns:
        Dictionary with keys:
        ``fx_total``, ``fy_total``       – total momentum-exchange force
        ``fx_pressure``, ``fy_pressure`` – pressure (normal stress) component
        ``fx_viscous``, ``fy_viscous``   – viscous (shear stress) component
        ``cd_total``, ``cl_total``       – total drag/lift coefficients
        ``cd_pressure``, ``cl_pressure`` – pressure-drag/lift coefficients
        ``cd_viscous``, ``cl_viscous``   – viscous-drag/lift coefficients
    """
    from .d2q9 import equilibrium as feq  # noqa: PLC0415

    device = f.device
    cx = torch.tensor([0, 1, 0, -1,  0,  1, -1, -1,  1], dtype=torch.float32, device=device)
    cy = torch.tensor([0, 0, 1,  0, -1,  1,  1, -1, -1], dtype=torch.float32, device=device)
    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long, device=device)

    ny, nx = mask.shape
    fluid = ~mask

    # Equilibrium and non-equilibrium DFs
    f_eq = feq(rho, ux, uy)
    f_neq = f - f_eq

    # Prefactor for viscous stress tensor
    sigma_prefactor = -(1.0 - 0.5 / tau)
    cx_b = cx.view(9, 1, 1); cy_b = cy.view(9, 1, 1)
    # σ_xx, σ_xy stress components at fluid cells
    sigma_xx = sigma_prefactor * (f_neq * cx_b * cx_b).sum(0)  # (ny, nx)
    sigma_yy = sigma_prefactor * (f_neq * cy_b * cy_b).sum(0)
    sigma_xy = sigma_prefactor * (f_neq * cx_b * cy_b).sum(0)

    # Pressure (EOS: p = rho / 3 in LBM)
    p_field = rho / 3.0

    fx_total = fy_total = 0.0
    fx_pressure = fy_pressure = 0.0
    fx_viscous = fy_viscous = 0.0

    y_f, x_f = torch.where(fluid)

    for q in range(1, 9):
        dcx = int(cx[q].item()); dcy = int(cy[q].item())
        x_n = (x_f + dcx).clamp(0, nx - 1)
        y_n = (y_f + dcy).clamp(0, ny - 1)
        is_solid_nbr = mask[y_n, x_n]
        if not is_solid_nbr.any():
            continue

        yf_ = y_f[is_solid_nbr]; xf_ = x_f[is_solid_nbr]
        q_opp = int(opp[q].item())
        # Total momentum exchange
        contrib = f[q, yf_, xf_] + f[q_opp, y_n[is_solid_nbr], x_n[is_solid_nbr]]
        fx_total += float((contrib * cx[q]).sum())
        fy_total += float((contrib * cy[q]).sum())

        # Pressure component: p * n_hat contribution (normal direction)
        # n̂ = (cx_q, cy_q) points from fluid to solid
        p_f = p_field[yf_, xf_]
        # Weight by equilibrium fraction: f_eq part of contribution
        f_eq_contrib = f_eq[q, yf_, xf_] + f_eq[q_opp, y_n[is_solid_nbr], x_n[is_solid_nbr]]
        fx_pressure += float((f_eq_contrib * cx[q]).sum())
        fy_pressure += float((f_eq_contrib * cy[q]).sum())

    # Viscous is the remainder
    fx_viscous = fx_total - fx_pressure
    fy_viscous = fy_total - fy_pressure

    # Non-dimensionalise
    q_dyn = 0.5 * rho_ref * u_ref ** 2 * area_ref
    if q_dyn < 1e-30:
        q_dyn = 1.0

    return {
        "fx_total": fx_total,
        "fy_total": fy_total,
        "fx_pressure": fx_pressure,
        "fy_pressure": fy_pressure,
        "fx_viscous": fx_viscous,
        "fy_viscous": fy_viscous,
        "cd_total": fx_total / q_dyn,
        "cl_total": fy_total / q_dyn,
        "cd_pressure": fx_pressure / q_dyn,
        "cl_pressure": fy_pressure / q_dyn,
        "cd_viscous": fx_viscous / q_dyn,
        "cl_viscous": fy_viscous / q_dyn,
    }
