"""Momentum / energy budget diagnostics for the Santini case-3 water entry.

Read-only companion of :mod:`benchmarks.water_entry.single_drop.free_surface_probe`.
The module never touches the solver state; it only *reads* the fill fraction,
the phase flags, the stored mass and the macroscopic velocity field and turns
them into the budget terms of the pre-registered K2-BUDGET-B ledger.

Unit system
-----------
Everything is computed in **lattice units** (dx = dt = 1, reference liquid
density = 1) so that the terms can be added without hidden conversion factors.
``tools/k2_budget_verdict.py`` converts the final numbers to SI with
``rho_l * dx**3`` (energy) and ``rho_l * dx**4 / dt`` (impulse) using the same
``scales()`` block the probe writes into ``result.json``.

Definitions (one line each, all evaluated on the *post-step* state)
------------------------------------------------------------------
``ke_liq``            1/2 * sum_i m_i |u_i|^2 over LIQUID+INTERFACE cells.
``ke_axial``          1/2 * sum_i m_i u_z,i^2   (water-column / axial motion).
``ke_lat``            1/2 * sum_i m_i (u_x^2 + u_y^2)  (lateral / jet motion).
``ke_rad``            radial part of ``ke_lat`` (u_r = e_r . u in the x-y plane).
``ke_lat_roi``        ``ke_lat`` restricted to the crater footprint (r <= 1.5 D,
                      below the initial free surface) - the "lateral jet" store.
``ke_drop``           1/2 * sum_i m_marker |u_marker|^2 over the tracked
                      drop-material markers (item 5 of the task ledger).
``area_total``        sum_i |grad phi| (smeared-interface area estimator,
                      lattice area units).
``area_cav``          the part of ``area_total`` carried by the interface cells
                      that bound the headspace-connected cavity gas below the
                      initial free surface (the crater wall).
``cav_volume``        number of cavity-gas cells (lattice volume units).
``p_amb``             ambient gas pressure carried by the anti-bounce-back
                      reconstruction, rho_gas/3 (lattice pressure).
``p_lap``             Laplace part of the same reconstruction, sigma * kappa,
                      with kappa = div(n) exactly as in ``free_surface_lbm``.
``gas_force_wall_*``  force of the cavity gas on the liquid, item 1 (momentum).
``gas_power_wall``    work rate of the cavity gas on the crater wall, item 1
                      (energy).  Positive = the gas pushes the wall away from
                      the liquid.
``gas_power_free``    work rate of the ambient pressure over the *whole* free
                      surface (the ledger term; in a rigid domain the crater
                      term and the far-field term cancel).
``grav_power``        work rate of gravity on the liquid.
``diss_rate``         sum over the liquid of 2 mu |S|^2 - the continuum viscous
                      dissipation rate (item 3), split into the interface
                      neighbourhood and the bulk.
``conv_loss``         kinetic energy carried away by liquid mass that the mass
                      exchange converts to gas (a conversion channel, not part
                      of the five task items).
``conv_gain``         kinetic energy that appears when gas mass is converted
                      back to liquid.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

GAS = 0
LIQUID = 1
INTERFACE = 2
SOLID = 3


def _grad(field):
    """Non-periodic gradient of the interior fill field, axis order (z, y, x)."""
    gz, gy, gx = np.gradient(field)
    return gz, gy, gx


def _neighbour_any(mask):
    """True where a 6-neighbour of ``mask`` is set."""
    out = np.zeros_like(mask)
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def _cavity_body(gas, surf_z, roi):
    """Headspace-connected gas body below the initial surface (the cavity)."""
    seeds = np.zeros_like(gas)
    seeds[-1] = gas[-1]
    headspace = ndimage.binary_propagation(seeds, mask=gas)
    below = np.arange(gas.shape[0])[:, None, None] < surf_z
    body = gas & below & roi[None]
    if not body.any():
        return body
    labels, _ = ndimage.label(body)
    depths = np.where(body, np.arange(body.shape[0])[:, None, None], 10**6)
    deepest = int(depths.min())
    keep = labels == labels[np.unravel_index(int(depths.argmin()), labels.shape)]
    return keep


def budget_snapshot(
    field,
    flags,
    mass,
    ux,
    uy,
    uz,
    level,
    n,
    sigma,
    mu,
    g,
    p_amb,
    u_limit=0.0,
    markers=None,
    marker_state=None,
    prev=None,
):
    """Evaluate every budget term on the current state.

    Parameters
    ----------
    field, flags, mass
        interior fill fraction, phase flags and stored mass (lattice units).
    ux, uy, uz
        interior macroscopic velocity (lattice units).
    level
        sub-cell position of the initial free surface in field coordinates.
    n
        cells per drop diameter.
    sigma, mu, g, p_amb
        lattice surface tension, dynamic viscosity, gravity and ambient gas
        pressure.
    markers
        optional :class:`DropMaterialMarkers` instance (drop material).
    marker_state
        optional dict with the marker mass, used for the drop-material KE.
    prev
        optional dict with the previous ``(mass, u2)`` used for the conversion
        channels.

    Returns
    -------
    dict of floats, all in lattice units (energy) or lattice force units.
    """
    side = field.shape[1]
    center = (side - 1) / 2
    yy, xx = np.indices(field.shape[1:])
    rad2 = (yy - center) ** 2 + (xx - center) ** 2
    roi = rad2 <= (1.5 * n) ** 2
    r = np.sqrt(rad2)
    z_index = np.arange(field.shape[0])[:, None, None]

    liquid_like = (flags == LIQUID) | (flags == INTERFACE)
    gas = flags == GAS
    m = np.where(liquid_like, mass, 0.0)
    u2 = ux**2 + uy**2 + uz**2

    ke_liq = float(0.5 * np.sum(m[liquid_like] * u2[liquid_like]))
    ke_axial = float(0.5 * np.sum(m[liquid_like] * uz[liquid_like] ** 2))
    ke_lat = float(0.5 * np.sum(m[liquid_like] * (ux[liquid_like] ** 2 + uy[liquid_like] ** 2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ur = np.where(r > 0, (ux * (yy - center) + uy * (xx - center)) / np.maximum(r, 1e-12), 0.0)
    ke_rad = float(0.5 * np.sum(m[liquid_like] * ur[liquid_like] ** 2))
    jet_zone = liquid_like & roi[None] & (z_index < level)
    ke_lat_roi = float(0.5 * np.sum(m[jet_zone] * (ux[jet_zone] ** 2 + uy[jet_zone] ** 2)))

    # --- smeared-interface area -------------------------------------------
    gz, gy, gx = _grad(field)
    grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)
    area_total = float(np.sum(grad_mag))

    # --- cavity geometry ----------------------------------------------------
    body = _cavity_body(gas, level, roi)
    cav_volume = float(body.sum())
    wall = np.zeros_like(body)
    if body.any():
        wall = _neighbour_any(body) & liquid_like & roi[None] & (z_index < level)

    # --- free surface (interface cells that see gas: the ABB set) -----------
    free = (flags == INTERFACE) & _neighbour_any(gas)
    gn = np.stack([gz, gy, gx], axis=0)
    mag = np.maximum(grad_mag, 1e-12)
    nhat = -gn / mag[None]          # points from the liquid into the gas
    kappa = 0.5 * (
        (np.roll(nhat[2], -1, axis=2) - np.roll(nhat[2], 1, axis=2))
        + (np.roll(nhat[1], -1, axis=1) - np.roll(nhat[1], 1, axis=1))
        + (np.roll(nhat[0], -1, axis=0) - np.roll(nhat[0], 1, axis=0))
    )
    p_lap = sigma * kappa
    area_free = grad_mag[free]
    p_lap_free = p_lap[free]
    n_free = nhat[:, free]
    u_free = np.stack([ux[free], uy[free], uz[free]], axis=0)
    un_free = np.sum(n_free * u_free, axis=0)

    # The ambient pressure acts on the whole free surface; the surface
    # tension part is booked through the interface-area change (dE_surf) and
    # is therefore excluded from the ledger term and reported separately.
    force_free_amb = -p_amb * np.sum(n_free * area_free[None, :], axis=1)
    power_free_amb = -p_amb * float(np.sum(un_free * area_free))
    force_free_lap = -np.sum(p_lap_free[None, :] * n_free * area_free[None, :], axis=1)
    p_lap_mean = float(np.mean(p_lap_free)) if p_lap_free.size else 0.0

    # --- item 1: the cavity gas on the crater wall -------------------------
    gas_force_wall = np.zeros(3)
    gas_force_wall_lap = np.zeros(3)
    gas_power_wall = 0.0
    area_cav = 0.0
    if wall.any():
        area_cav = float(np.sum(grad_mag[wall]))
        w = np.stack([ux[wall], uy[wall], uz[wall]], axis=0)
        n_w = nhat[:, wall]
        a_w = grad_mag[wall]
        un_w = np.sum(n_w * w, axis=0)
        gas_force_wall = -p_amb * np.sum(n_w * a_w[None, :], axis=1)
        gas_force_wall_lap = -np.sum(p_lap[wall][None, :] * n_w * a_w[None, :], axis=1)
        gas_power_wall = -p_amb * float(np.sum(un_w * a_w))

    # --- gravity work rate --------------------------------------------------
    grav_power = float(-g * np.sum(m[liquid_like] * uz[liquid_like]))
    grav_force_z = float(-g * np.sum(m[liquid_like]))

    # --- viscous dissipation rate -----------------------------------------
    s_zz = np.gradient(uz, axis=0)
    s_yy = np.gradient(uy, axis=1)
    s_xx = np.gradient(ux, axis=2)
    s_zy = 0.5 * (np.gradient(uz, axis=1) + np.gradient(uy, axis=0))
    s_zx = 0.5 * (np.gradient(uz, axis=2) + np.gradient(ux, axis=0))
    s_yx = 0.5 * (np.gradient(uy, axis=2) + np.gradient(ux, axis=1))
    s_norm2 = (s_xx**2 + s_yy**2 + s_zz**2) + 2 * (s_yx**2 + s_zx**2 + s_zy**2)
    iface_zone = _neighbour_any(flags == INTERFACE)
    diss_rate = float(2 * mu * np.sum(m[liquid_like] * s_norm2[liquid_like]))
    diss_iface = float(
        2 * mu * np.sum(m[liquid_like & iface_zone] * s_norm2[liquid_like & iface_zone])
    )
    diss_bulk = diss_rate - diss_iface
    jump = float(np.mean(np.sqrt(u2)[flags == INTERFACE])) if (flags == INTERFACE).any() else 0.0
    # pseudo-current limiter census: interface cells that sit exactly on the
    # velocity cap are the cells the solver clipped this step.
    speed_lbm = np.sqrt(u2)
    at_cap = int(np.sum((flags == INTERFACE) & (speed_lbm >= 0.999 * u_limit)))
    ke_at_cap = float(0.5 * np.sum(m[(flags == INTERFACE) & (speed_lbm >= 0.999 * u_limit)]
                                  * u2[(flags == INTERFACE) & (speed_lbm >= 0.999 * u_limit)]))

    # --- conversion (mass exchange) channels -------------------------------
    conv_loss = 0.0
    conv_gain = 0.0
    conv_mom = np.zeros(3)
    if prev is not None:
        dm = m - prev["mass"]
        loss = np.clip(dm, None, 0.0)
        gain = np.clip(dm, 0.0, None)
        conv_loss = float(-0.5 * np.sum(loss * prev["u2"]))
        conv_gain = float(0.5 * np.sum(gain * u2))
        conv_mom = np.array([
            float(np.sum(dm * ux)),
            float(np.sum(dm * uy)),
            float(np.sum(dm * uz)),
        ])

    # --- drop-material kinetic energy and momentum -------------------------
    ke_drop = 0.0
    p_drop = np.zeros(3)
    if markers is not None and len(markers) and marker_state is not None:
        pos = markers.pos
        cells = np.floor(pos).astype(np.int64)
        np.clip(cells[:, 0], 0, field.shape[0] - 1, out=cells[:, 0])
        np.clip(cells[:, 1], 0, field.shape[1] - 1, out=cells[:, 1])
        np.clip(cells[:, 2], 0, field.shape[2] - 1, out=cells[:, 2])
        um = np.stack([ux[cells[:, 0], cells[:, 1], cells[:, 2]],
                        uy[cells[:, 0], cells[:, 1], cells[:, 2]],
                        uz[cells[:, 0], cells[:, 1], cells[:, 2]]], axis=0)
        mm = marker_state["mass"]
        ke_drop = float(0.5 * np.sum(mm * np.sum(um**2, axis=0)))
        p_drop = np.sum(mm[None] * um, axis=1)

    # --- liquid momentum ----------------------------------------------------
    p_liq = np.array([
        float(np.sum(m[liquid_like] * ux[liquid_like])),
        float(np.sum(m[liquid_like] * uy[liquid_like])),
        float(np.sum(m[liquid_like] * uz[liquid_like])),
    ])

    return {
        "vol_liq": float(np.sum(field)),
        "ke_liq": ke_liq,
        "ke_axial": ke_axial,
        "ke_lat": ke_lat,
        "ke_rad": ke_rad,
        "ke_lat_roi": ke_lat_roi,
        "ke_drop": ke_drop,
        "area_total": area_total,
        "area_cav": area_cav,
        "cav_volume": cav_volume,
        "p_lap_mean": p_lap_mean,
        "gas_force_wall_x": float(gas_force_wall[0]),
        "gas_force_wall_y": float(gas_force_wall[1]),
        "gas_force_wall_z": float(gas_force_wall[2]),
        "gas_force_wall_lap_z": float(gas_force_wall_lap[2]),
        "gas_power_wall": gas_power_wall,
        "gas_power_free": power_free_amb,
        "gas_force_free_z": float(force_free_amb[2]),
        "gas_force_free_lap_z": float(force_free_lap[2]),
        "grav_power": grav_power,
        "grav_force_z": grav_force_z,
        "diss_rate": diss_rate,
        "diss_iface": diss_iface,
        "diss_bulk": diss_bulk,
        "jump_iface_mean": jump,
        "n_at_cap": at_cap,
        "ke_at_cap": ke_at_cap,
        "conv_loss": conv_loss,
        "conv_gain": conv_gain,
        "conv_mom_z": float(conv_mom[2]),
        "p_liq_x": p_liq[0],
        "p_liq_y": p_liq[1],
        "p_liq_z": p_liq[2],
        "p_drop_x": float(p_drop[0]),
        "p_drop_y": float(p_drop[1]),
        "p_drop_z": float(p_drop[2]),
    }


def integrate(rows, keys):
    """Trapezoidal time integration of the per-step rate terms."""
    out = {}
    for key in keys:
        total = 0.0
        for i, row in enumerate(rows):
            value = row.get(key)
            if value is None:
                value = 0.0
            if i == 0:
                total += 0.5 * value
            else:
                prev = rows[i - 1].get(key)
                total += 0.5 * ((prev or 0.0) + value)
        out[key] = float(total)
    return out
