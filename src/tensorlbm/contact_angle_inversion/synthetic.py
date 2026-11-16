"""Analytic sessile-cap fields for auditing the contact-angle estimator.

The calibration-based inversion of Huang et al. (2007) is only as good as the
angle *measurement* that produced the calibration curve.  Every number in a
calibration sweep comes from :func:`tensorlbm.sc_surface_tension.
contact_angle_from_mask`, a discrete estimator that fits a circle to the
``rho > (rho_l + rho_v) / 2`` mask row by row.  That estimator has never been
checked against a *known* angle, so its discretisation bias was unknown.

This module builds the missing ground truth: a sessile drop whose interface is
an exact circular arc meeting the wall at a prescribed angle, rendered with the
same diffuse ``tanh`` profile as a Shan-Chen interface.  The mask boundary of
such a field is, by construction, the analytic arc, so any difference between
``contact_angle_from_mask`` and the prescribed angle is estimator error and
nothing else.

Geometry
--------
A circular arc of radius ``R`` centred at ``(x_c, y_c)`` meets the wall
``y = y_wall`` at the contact angle ``theta`` (measured through the liquid)
when ``y_c = y_wall - R * cos(theta)``; the cap above the wall has half width
``R sin(theta)`` and height ``R (1 - cos(theta))``.

This is a *verification* utility: it produces no physics, it only makes the
measurement operator falsifiable.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

__all__ = ["sessile_cap_field", "cap_geometry", "grid_for_cap"]


def cap_geometry(theta_deg: float, radius: float, y_wall: float) -> Dict[str, float]:
    """Analytic geometry of a sessile cap of radius ``radius`` on ``y = y_wall``.

    Returns a dict with the circle centre ``y_c``, the cap height ``h`` and the
    half width at the wall ``a``.  ``cos(theta) = -(y_c - y_wall) / radius``.
    """
    th = math.radians(theta_deg)
    y_c = y_wall - radius * math.cos(th)
    return {
        "theta_deg": float(theta_deg),
        "y_c": float(y_c),
        "height": float(radius * (1.0 - math.cos(th))),
        "half_width": float(radius * math.sin(th)),
    }


def grid_for_cap(radius: float, theta_deg: float, wall_rows: int = 2,
                 margin: float = 4.0) -> Dict[str, int]:
    """Smallest sensible grid for a cap: the same shape rule the LB cells use.

    ``nx`` allows the full ``2 R sin(theta)`` footprint at the wall (the widest
    cap is ``theta = 90``), ``ny`` allows the full ``2 R`` height (``theta =
    180``).  ``margin`` is added on each side so the drop is never squeezed.
    """
    g = cap_geometry(theta_deg, radius, 0.0)
    # domain ratio of the calibration grids: 64x32 at R=8, 128x64 at R=15
    ny = int(math.ceil(max(4.0 * radius, wall_rows + g["height"] + margin)))
    nx = int(math.ceil(max(8.0 * radius, 2.0 * g["half_width"] + margin)))
    return {"nx": nx, "ny": ny,
            "fits": bool(g["height"] + margin <= ny - wall_rows
                         and g["half_width"] * 2.0 + margin <= nx)}


def sessile_cap_field(
    theta_deg: float,
    radius: float,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    wall_rows: int = 2,
    rho_l: float = 1.957,
    rho_v: float = 0.1596,
    interface_width: float = 2.5,
    x_phase: float = 0.0,
) -> Dict[str, object]:
    """Render a diffuse-interface sessile cap of *exactly known* contact angle.

    Args:
        theta_deg: Prescribed contact angle (through the liquid), degrees.
        radius: Cap radius in lattice units.
        nx, ny: Grid size; defaults from :func:`grid_for_cap`.
        wall_rows: Number of solid rows at the bottom.
        rho_l, rho_v: Coexisting densities used to scale the ``tanh`` profile.
        interface_width: ``tanh`` width in lattice units (Shan-Chen is ~2-4).
        x_phase: Sub-lattice shift of the cap centre; sweeping it exposes the
            grid-alignment noise of the estimator.

    Returns:
        ``{"rho", "solid", "y_wall", "geometry"}`` with ``rho``/``solid`` torch
        tensors of shape ``(ny, nx)``.
    """
    default = grid_for_cap(radius, theta_deg, wall_rows)
    nx = int(nx if nx is not None else default["nx"])
    ny = int(ny if ny is not None else default["ny"])
    y_wall = float(wall_rows) - 0.5          # centre of the last solid row + 0.5

    geom = cap_geometry(theta_deg, radius, y_wall)
    x_c = 0.5 * (nx - 1) + x_phase
    y_c = geom["y_c"]

    ys = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    xs = torch.arange(nx, dtype=torch.float64).view(1, nx)
    r = torch.sqrt((xs - x_c) ** 2 + (ys - y_c) ** 2)
    # signed distance from the analytic arc: positive inside the liquid
    d = radius - r
    # below the wall everything is solid: force vapour so it can never leak in
    rho = rho_v + (rho_l - rho_v) * 0.5 * (1.0 + torch.tanh(2.0 * d / interface_width))
    solid = (ys < float(wall_rows)).expand(ny, nx).clone()
    rho = torch.where(solid, torch.full_like(rho, rho_v), rho)

    return {
        "rho": rho.to(torch.float32),
        "solid": solid,
        "y_wall": y_wall,
        "geometry": geom,
    }