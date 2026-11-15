"""Contact-angle convention for the geometrical (gradient-rotation) wetting BC.

The problem
-----------
``openLBMPM``'s ``updateColorGradientOnWettingNew`` (and therefore the faithful
port in :mod:`tensorlbm.cg_wetting_geometric`) rewrites the colour gradient at
wall-adjacent fluid nodes so that

    angle( g_hat , n_s ) = theta_arg ,      g_hat = -grad(phi)/|grad(phi)|   (1)

with ``n_s`` the unit wall normal pointing *from the fluid into the solid*,
``phi > 0`` in the wetting phase, and ``g_hat`` therefore the **outward** normal
of the wetting-phase body.

What a real sessile drop requires
---------------------------------
Put the wall plane at ``y = y_wall`` with the solid below it (so
``n_s = (0, -1)``) and a drop of the wetting phase sitting on it with contact
angle ``theta`` measured *inside* the drop.  At the left contact line the
interface tangent, walking up into the drop, is ``t = (cos theta, sin theta)``
and the drop interior lies to the right of ``t``, so the outward normal is
``g_hat = (-sin theta, cos theta)``.  Hence

    g_hat . n_s = -cos(theta)   ==>   angle(g_hat, n_s) = 180 deg - theta .  (2)

``theta = 90`` is the only fixed point of (2), and ``theta = 0`` / ``180`` map
onto each other: **the angle in (1) is not the physical contact angle, it is its
mirror**.  This is exactly why the reference implementation only exposes
``theta < 90`` and internally passes ``180 - theta`` (fluid 1 is always the
non-wetting phase there): the reference's ``theta`` argument is the angle of the
*other* phase.

Consequences
------------
* Feeding a physical contact angle straight into the port realises
  ``theta_realised = 180 - theta_phys``.  The error is invisible at ``90`` deg,
  which is why a single 90 deg arm can never catch it, and it is also invisible
  to the two round-1 tests: comparing against a literal transcription of the
  reference kernel proves faithfulness (not correctness), and reading the angle
  back out of the rotated gradient with ``acos(g_hat . n_s)`` measures the very
  quantity the kernel sets -- a circular check.
* The independent check is the **analytic cap**: an equilibrium sessile drop is
  a spherical cap, so its own outward normal is a geometric truth that the
  kernel must reproduce at the contact line.  :func:`cap_convention_error`
  builds that cap, runs the kernel and reports the mismatch.  For the physical
  convention the residual is the stencil/discretisation floor (a few degrees);
  for the reference convention it is ``|180 - 2 theta|``.

Public API
----------
:func:`physical_to_reference_angle`
    ``theta_arg = 180 - theta_phys`` (identity for ``convention="reference"``).
:func:`cap_phase_field`, :func:`cap_outward_normal_field`
    exact spherical cap and its normal field on a lattice.
:func:`cap_convention_error`
    analytic-cap verification of one convention at one contact angle.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .cg_wetting_geometric import (
    isotropic_color_gradient,
    rotate_wall_color_gradient,
    wall_normal_field,
)

CONVENTIONS = ("physical", "reference")

#: discretisation floor of the analytic-cap check (stencil + diffuse interface)
CAP_FLOOR_DEG = 15.0


def physical_to_reference_angle(theta_deg: float, convention: str = "physical") -> float:
    """Map a *physical* contact angle to the angle argument of the ported kernel.

    Args:
        theta_deg: contact angle in degrees measured inside the ``phi > 0``
            phase, ``0 < theta < 180``.
        convention: ``"physical"`` (default) applies the ``180 - theta`` map of
            Eq. (2); ``"reference"`` is the raw pass-through used by round 1 of
            ``XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE`` (kept for exact
            reproduction only -- it realises the mirror angle).

    Returns:
        The angle to hand to :func:`tensorlbm.cg_wetting_geometric.rotate_wall_color_gradient`.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    theta = float(theta_deg)
    if not (0.0 < theta < 180.0):
        raise ValueError(f"contact angle must be inside (0, 180) deg, got {theta}")
    return 180.0 - theta if convention == "physical" else theta


def angle_between_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Angle between two 2-D vectors in degrees (``nan`` for a zero vector)."""
    na = math.hypot(*a)
    nb = math.hypot(*b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def cap_phase_field(ny: int, nx: int, theta_deg: float, radius: float, xi: float,
                    y_wall: float = 0.5, solid_bottom: bool = True) -> tuple[np.ndarray, dict[str, float]]:
    """Exact spherical cap as a phase field (``+1`` inside the drop).

    ``phi = tanh(d/xi)`` with ``d`` the signed distance to the cap of contact
    angle ``theta_deg``; the cap meets the wall at ``x_c +- R sin(theta)`` so
    the footprint is ``2 R sin(theta)`` and the apex ``R (1 - cos(theta))``.
    """
    th = math.radians(float(theta_deg))
    x_c = 0.5 * nx
    y_c = y_wall - radius * math.cos(th)
    y = np.arange(ny, dtype=np.float64)[:, None]
    x = np.arange(nx, dtype=np.float64)[None, :]
    r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    d = radius - r
    if solid_bottom:
        d[0, :] = -np.inf
    phi = np.tanh(np.clip(d, -50.0, 50.0) / xi)
    geo = {
        "x_centre": float(x_c),
        "y_centre": float(y_c),
        "radius": float(radius),
        "footprint": 2.0 * radius * math.sin(th),
        "apex": radius * (1.0 - math.cos(th)),
        "y_wall": float(y_wall),
        "contact_x_left": float(x_c - radius * math.sin(th)),
        "contact_x_right": float(x_c + radius * math.sin(th)),
    }
    return phi, geo


def cap_outward_normal_field(ny: int, nx: int, geo: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normal of the analytic cap at every node."""
    y = np.arange(ny, dtype=np.float64)[:, None]
    x = np.arange(nx, dtype=np.float64)[None, :]
    dx = x - geo["x_centre"]
    dy = y - geo["y_centre"]
    r = np.sqrt(dx * dx + dy * dy)
    r_safe = np.where(r > 0.0, r, 1.0)
    return dx / r_safe, dy / r_safe


def cap_convention_error(theta_deg: float, convention: str = "physical", *,
                         radius: float = 20.0, xi: float = 2.0, ny: int = 80, nx: int = 96,
                         y_wall: float = 0.5, contact_band: float = 3.0,
                         mag_tol: float = 1e-3) -> dict[str, Any]:
    """Compare one convention against the analytic cap normal at the contact line.

    Returns a dict with

    ``err_vs_cap_normal_deg_mean/max``
        angle between the rotated ``g_hat`` and the analytic cap normal over the
        contact-line nodes (the physical soundness of the convention);
    ``angle_vs_ns_deg``
        ``angle(g_hat_out, n_s)`` -- the quantity the kernel sets exactly;
    ``mirror_prediction_deg``
        ``|180 - 2 theta|``: what the mean mismatch must be if the convention
        realises the mirror angle.
    """
    phi_np, geo = cap_phase_field(ny, nx, theta_deg, radius, xi, y_wall)
    phi = torch.as_tensor(phi_np, dtype=torch.float64)
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    theta_arg = physical_to_reference_angle(theta_deg, convention)

    gx, gy = isotropic_color_gradient(phi, solid, periodic=(True, False))
    rx, ry = rotate_wall_color_gradient(gx, gy, solid, theta_arg, periodic=(True, False))
    ntx, nty = cap_outward_normal_field(ny, nx, geo)
    _, nsx, nsy = wall_normal_field(solid, periodic=(True, False))

    errs: list[float] = []
    ang_ns: list[float] = []
    for j in (1, 2):
        for i in range(nx):
            if min(abs(i - geo["contact_x_left"]), abs(i - geo["contact_x_right"])) > contact_band:
                continue
            gm = math.hypot(float(gx[j, i]), float(gy[j, i]))
            if gm < mag_tol or abs(float(phi_np[j, i])) > 0.9:
                continue
            rxm, rym = float(rx[j, i]), float(ry[j, i])
            rm = math.hypot(rxm, rym)
            if rm < mag_tol:
                continue
            g_out = (-rxm / rm, -rym / rm)  # kernel convention: g_hat = -g/|g|
            errs.append(angle_between_deg(g_out, (float(ntx[j, i]), float(nty[j, i]))))
            ang_ns.append(angle_between_deg(g_out, (float(nsx[j, i]), float(nsy[j, i]))))
    out: dict[str, Any] = {
        "theta_phys_deg": float(theta_deg),
        "theta_arg_deg": float(theta_arg),
        "convention": convention,
        "n_samples": len(errs),
        "kernel_exact_angle_vs_ns_deg": float(np.mean(ang_ns)) if ang_ns else float("nan"),
        "cap_required_angle_vs_ns_deg": 180.0 - float(theta_deg),
        "err_vs_cap_normal_deg_mean": float(np.mean(errs)) if errs else float("nan"),
        "err_vs_cap_normal_deg_max": float(np.max(errs)) if errs else float("nan"),
        "mirror_prediction_deg": abs(180.0 - 2.0 * float(theta_deg)),
        "geometry": geo,
    }
    out["physical_ok"] = bool(
        math.isfinite(out["err_vs_cap_normal_deg_mean"])
        and out["err_vs_cap_normal_deg_mean"] < CAP_FLOOR_DEG
    )
    return out