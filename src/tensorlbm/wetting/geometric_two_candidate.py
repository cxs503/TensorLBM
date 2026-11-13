"""Geometrical wetting condition, *two-candidate* variant (openLBMPM recipe).

Why a second variant
--------------------
:mod:`tensorlbm.cg_wetting2d` realises the geometrical wetting condition of
Akai et al. (2018) in closed form: the ghost phase in the solid cell is

    phi_s = phi_f + cot(theta) * |dphi/dt|_f                        (closed form)

i.e. the wall-normal difference is *always* ``+|dphi/dt| cot(theta)``, so the
sign of the imposed wall-normal gradient is fixed by ``theta`` alone.  Both
open-source references that were read for this anchor resolve that sign
*dynamically* instead, by choosing among two candidate interface normals:

* ``PorousMediaSimulation/openLBMPM`` projects the near-wall colour gradient
  onto "the nearer of two candidate unit vectors constructed from ``cosTheta``
  and the wall normal" (``RKCG2D/AcceleratedRKGPU2D.py:2430-2492``, older
  branch ``:1639-1678``; recorded in the platform OSS inventory
  ``docs/oss-inventory-openLBMPM.md`` §4.2 and in ``oss_solutions.id=5``
  (``NEW:CAP-LBM-VALIDATION-SURFACE`` / ``geometric-wetting-contact-angle``)).
  The solid colour value is produced beforehand by ``calColorValueOnSolid``
  (``:1560``) using a prescribed ``SolidColorDiff`` (``IniFiles/
  RKtwophasesetup2D.ini:42``), so ``cosTheta`` is imposed directly and no
  Shan-Chen ``G_s`` calibration is needed;
* ``lanl/MF-LBM`` uses the same idea with an explicit ``distP``/``distM``
  candidate-distance test (``Phase_gradient.F90:238-259``) -- the branch that
  :mod:`tensorlbm.cg_wetting2d` documents as *not* reproduced.

This module adds the candidate selection, i.e. it closes the gap recorded in
``docs/XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING.md`` §六 (improvement item 4).

The selection rule, derived
---------------------------
Let ``g = (g_t, g_n)`` be the (isotropic 9-point) colour gradient at the
wall-face fluid node, ``t`` the wall tangent and ``n_w`` the direction pointing
*from the fluid node into the wall*.  A unit normal at contact angle ``theta``
(measured inside the ``phi > 0`` phase) has tangential component
``s_t sin(theta)`` with ``s_t = sign(g_t)`` fixed by the actual interface, and
wall-normal component of *unknown sign* ``+-cos(theta)``.  The two candidates
are therefore

    n_+ = ( s_t sin(theta), +cos(theta) ),
    n_- = ( s_t sin(theta), -cos(theta) ),

and the nearer one to the measured unit gradient ``g/|g|`` is picked.  Because
only the second component differs, the distance test is equivalent to

    sign(g_n') = sign(g_n),

so the ghost value

    phi_s = phi_f + |g_t| * cot(theta) * sign(g_n)                   (candidate)

differs from the closed form of :mod:`tensorlbm.cg_wetting2d`
(``+|g_t| cot(theta)``) exactly on the wall faces where
``sign(g_n) * sign(cot theta) < 0``.  Those are the faces where the interface
approaches the wall "from the other side"; a closed-form/fixed-sign rule
silently imposes the wrong wall-normal gradient there.  The sign-selected form
stays continuous with the local interface orientation, which is the reason both
reference codes carry the candidate branch.

Where ``|g| < gate`` (the reference's ``|C| > 1e-6`` interface guard) no
interface reaches the wall node and the solid value falls back to the mirror
value plus ``solid_color_offset`` -- the stand-in for openLBMPM's prescribed
``SolidColorDiff`` ("SolidColorDiff ~0.5 sets solid colour density difference",
``oss_solutions.id=5`` ``conventions``).

Contact-angle measurement, reference recipe
-------------------------------------------
``cap_geometry_contact_angle`` implements the *measurement* side recorded for
openLBMPM (``oss_solutions.id=5``, mirrored by
:func:`tensorlbm.mcmp_verification.cap_geometry_contact_angle`): base width
``b`` and apex height ``h`` of the sessile cap give

    R = (b^2 + 4 h^2) / (8 h),   theta = atan2(b/2, R - h),

which needs two geometric extents only and no circle fit -- a much more
pseudo-current-robust estimator than a full contour fit.

References
----------
* T. Akai et al., Phys. Rev. E **98**, 043302 (2018) -- geometrical wetting.
* PorousMediaSimulation/openLBMPM, ``RKCG2D/AcceleratedRKGPU2D.py`` (see the
  line pointers above; no licence file in that repository, so **no code was
  copied** -- the construction is re-derived from the documented recipe).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from ..cg_wetting2d import WALL_DIRECTIONS, _nb, _wall_face

__all__ = [
    "D2Q9_EX",
    "D2Q9_EY",
    "D2Q9_W",
    "isotropic_color_gradient",
    "solid_color_mirror",
    "geometric_wetting_ghost_candidates",
    "cap_geometry_contact_angle",
]

#: D2Q9 lattice vectors in the ``tensorlbm.d2q9.C`` ordering
#: (0 rest, 1 E, 2 N, 3 W, 4 S, 5 NE, 6 NW, 7 SW, 8 SE).
D2Q9_EX: Tuple[int, ...] = (0, 1, 0, -1, 0, 1, -1, -1, 1)
D2Q9_EY: Tuple[int, ...] = (0, 0, 1, 0, -1, 1, 1, -1, -1)
D2Q9_W: Tuple[float, ...] = (
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
)


def isotropic_color_gradient(phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Isotropic 9-point colour gradient ``grad phi = 3 sum_i w_i phi(x+c_i) c_i``.

    The prefactor 3 (``1/c_s^2``) and the isotropic weight set are the ones
    recorded for openLBMPM (``RKCG2D/AcceleratedRKGPU2D.py:1584-1632``).  The
    field is sampled with periodic shifts, so solid cells must already carry
    their solid colour value (see :func:`solid_color_mirror`).
    """
    gx = torch.zeros_like(phi)
    gy = torch.zeros_like(phi)
    for ex, ey, w in zip(D2Q9_EX, D2Q9_EY, D2Q9_W):
        val = _nb(phi, ey, ex) if (ex or ey) else phi
        gx = gx + w * ex * val
        gy = gy + w * ey * val
    return 3.0 * gx, 3.0 * gy


def solid_color_mirror(
    phi: torch.Tensor,
    solid: torch.Tensor,
    dy: int,
    dx: int,
    offset: float = 0.0,
) -> torch.Tensor:
    """Solid colour value for a wall with normal ``(dy, dx)`` (fluid -> solid).

    ``calColorValueOnSolid`` analogue: the solid cell takes the colour value of
    its wall-normal fluid neighbour (zero normal difference = neutral wall) plus
    a constant ``offset``, which stands for openLBMPM's preset
    ``SolidColorDiff``.  Using the *normal* neighbour (not the nearest fluid
    cell of any direction) keeps the fill single-valued on corners.
    """
    mir = _nb(phi, -dy, -dx)
    return torch.where(solid, mir + offset, phi)


def geometric_wetting_ghost_candidates(
    phi: torch.Tensor,
    solid: torch.Tensor,
    theta_deg: float,
    *,
    gate: float = 1e-6,
    solid_color_offset: float = 0.0,
    phi_bounds: Tuple[float, float] = (-1.0, 1.0),
    periodic: Tuple[bool, bool] = (True, False),
    theta_by_direction: Dict[Tuple[int, int], float] | None = None,
    stencil: str = "central2",
    return_diagnostics: bool = False,
):
    """Ghost phase in the solid cells with the openLBMPM two-candidate rule.

    Args:
        phi: ``(ny, nx)`` phase field, ``phi > 0`` = wetting phase.
        solid: ``(ny, nx)`` boolean solid mask.
        theta_deg: prescribed static contact angle inside the ``phi > 0``
            phase, ``0 < theta < 180``.
        gate: interface guard ``|g| > gate`` (openLBMPM: ``|C| > 1e-6``);
            wall faces below it get the neutral (mirror + ``solid_color_offset``)
            value.
        solid_color_offset: constant added to the mirrored solid colour value
            where the geometric constraint is inactive (``SolidColorDiff``).
        phi_bounds: clamp for the ghost/colour values.
        periodic: ``(x, y)`` periodicity used to clip phantom wall faces.
        theta_by_direction: optional ``{(dy, dx): theta_deg}`` override.
        stencil: ``"central2"`` (default) uses the two-point central tangential
            difference of the wall-adjacent row -- the *same* stencil the
            closed-form condition of :mod:`tensorlbm.cg_wetting2d` and its
            read-back use, so the imposed wall-normal difference and the
            interface orientation the solver sees are consistent.
            ``"isotropic9"`` uses openLBMPM's isotropic 9-point colour gradient
            (``:1584-1632``) evaluated on the mirror-filled field; it is kept as
            a *sensitivity* arm because transplanting that stencil into a
            mirror-filled solver biases the tangential component (measured in
            ``docs/XVAL-LBM-OPENLBMPM-GEOMETRIC-WETTING.md`` §5.2).
        return_diagnostics: also return the candidate-selection statistics
            (how many active wall faces exist and on how many of them the
            candidate rule picks the opposite sign to the fixed-sign closed
            form of :func:`tensorlbm.cg_wetting2d.geometrical_wetting_ghost`).

    Returns:
        ``(ny, nx)`` ghost/colour field for the solid cells; or
        ``(field, diag)`` with ``diag`` as described above.
    """
    if not (0.0 < theta_deg < 180.0):
        raise ValueError(f"theta_deg must lie in (0, 180), got {theta_deg}")
    if phi.shape != solid.shape:
        raise ValueError(f"shape mismatch: phi {tuple(phi.shape)} vs solid {tuple(solid.shape)}")

    lo, hi = phi_bounds
    ghost_acc = torch.zeros_like(phi)
    count = torch.zeros_like(phi)
    n_face = 0
    n_active = 0
    n_flip = 0
    n_degenerate = 0
    flips = []

    for dy, dx in WALL_DIRECTIONS:
        face = _wall_face(solid, dy, dx, periodic)
        if not bool(face.any()):
            continue
        th = theta_deg if theta_by_direction is None else theta_by_direction.get((dy, dx), theta_deg)
        if not (0.0 < th < 180.0):
            raise ValueError(f"theta for direction {(dy, dx)} must lie in (0, 180), got {th}")
        theta = math.radians(th)
        cot = math.cos(theta) / math.sin(theta)

        ext = solid_color_mirror(phi, solid, dy, dx, offset=solid_color_offset)
        # tangent t = (-dx, dy) and wall normal n_w = (dx, dy), both in (row, col)
        if stencil == "central2":
            g_t = 0.5 * (_nb(ext, -dx, dy) - _nb(ext, dx, -dy))
            # one-sided measured wall-normal difference across the face
            g_n_meas = _nb(ext, dy, dx) - phi
            stencil_norm = g_t.abs()
        elif stencil == "isotropic9":
            gx, gy = isotropic_color_gradient(ext)
            g_t = gx * dy + gy * (-dx)
            g_n_meas = gx * dx + gy * dy
            stencil_norm = torch.sqrt(g_t * g_t + g_n_meas * g_n_meas)
        else:
            raise ValueError(f"stencil must be 'central2' or 'isotropic9', got {stencil!r}")

        active = face & (stencil_norm > gate)
        mag = (g_t.abs() * abs(cot)).abs()          # |g_t| |cot theta|
        # two-candidate rule: the sign of the imposed wall-normal difference
        # follows the sign of the *measured* one-sided difference; a mirror
        # fill (solid_color_offset = 0) makes that difference vanish and the
        # rule degenerates to the fixed-sign closed form.
        fixed = -1.0 if cot < 0 else 1.0
        sign_choice = torch.where(
            g_n_meas > 0,
            torch.ones_like(g_n_meas),
            torch.where(g_n_meas < 0, -torch.ones_like(g_n_meas), torch.full_like(g_n_meas, fixed)),
        )
        # where no interface reaches the wall node the prescribed solid colour
        # difference (openLBMPM SolidColorDiff) is used instead of the rotation
        impose = torch.where(
            active, mag * sign_choice, torch.full_like(g_n_meas, solid_color_offset)
        )
        neigh = torch.where(face, phi + impose, phi + solid_color_offset)
        ghost_face = neigh.clamp(min=lo, max=hi)
        face_solid = _nb(face.to(phi.dtype), -dy, -dx) > 0.5
        ghost_acc = torch.where(face_solid, ghost_acc + _nb(ghost_face, -dy, -dx), ghost_acc)
        count = torch.where(face_solid, count + 1.0, count)

        n_face += int(face.sum())
        n_active += int(active.sum())
        if bool(active.any()):
            flip = active & (sign_choice != torch.full_like(sign_choice, fixed))
            n_flip += int(flip.sum())
            if bool(flip.any()):
                flips.append(mag[flip].detach())
        degenerate = active & (mag < gate)
        n_degenerate += int(degenerate.sum())

    has = count > 0
    out = torch.where(has, ghost_acc / count.clamp(min=1.0), torch.zeros_like(phi))
    out = out.clamp(min=lo, max=hi)

    if not return_diagnostics:
        return out

    diag = {
        "theta_deg": float(theta_deg),
        "n_wall_face_nodes": float(n_face),
        "n_interface_face_nodes": float(n_active),
        "n_sign_flips": float(n_flip),
        "sign_flip_fraction": float(n_flip) / float(n_active) if n_active else 0.0,
        "stencil": stencil,
        "n_degenerate_face_nodes": float(n_degenerate),
        "n_ghost_cells": float(int(has.sum())),
        "ghost_mean": float(out[has].mean()) if bool(has.any()) else 0.0,
        "flip_magnitude_mean": float(torch.cat(flips).mean()) if flips else 0.0,
    }
    return out, diag


def cap_geometry_contact_angle(
    phi,
    y_wall: float = 0.5,
    level: float = 0.0,
    x_center: float | None = None,
) -> dict:
    """Contact angle from cap base width ``b`` and apex height ``h`` (openLBMPM).

    Two scalar extents only: ``b`` = chord length of the ``phi = level`` contour
    at the wall (linearly extrapolated from the two wall-adjacent fluid rows,
    which removes the half-cell offset of the first row), ``h`` = apex height
    above the wall of the same contour.  For a circular cap

        R = (b^2 + 4 h^2) / (8 h),  theta = atan2(b/2, R - h).

    Accepts an ``(ny, nx)`` tensor or ndarray and returns a dict with ``theta_deg``,
    ``b``, ``h``, ``R`` and a ``status`` field (``ok`` / ``no_interface`` /
    ``bracket_failed``) so a degenerate field never raises inside a campaign.
    """
    import numpy as np

    if isinstance(phi, torch.Tensor):
        arr = phi.detach().cpu().numpy().astype(float)
    else:
        arr = np.asarray(phi, dtype=float)
    ny, nx = arr.shape
    out = {"theta_deg": float("nan"), "b": float("nan"), "h": float("nan"), "R": float("nan"), "status": "ok"}
    xc = (nx - 1) / 2.0 if x_center is None else float(x_center)

    def _crossings(row: np.ndarray):
        """x positions where ``row`` crosses ``level`` (linear interpolation)."""
        xs = []
        for j in range(nx - 1):
            a, b = row[j], row[j + 1]
            if (a - level) * (b - level) < 0.0:
                xs.append(j + (level - a) / (b - a))
        return xs

    def _row(i: int) -> np.ndarray:
        return arr[int(i), :]

    # ~ index of the first fluid row: the wall is at y_wall, rows are at integers
    i0 = int(math.floor(y_wall)) + 1
    if i0 + 1 >= ny:
        out["status"] = "no_interface"
        return out
    right = []
    left = []
    for i in (i0, i0 + 1):
        xs = _crossings(_row(i))
        if len(xs) < 2:
            out["status"] = "no_interface"
            return out
        left.append(min(xs, key=lambda x: abs(x - xc)))
        right.append(max(xs, key=lambda x: abs(x - xc)))
    # linear extrapolation of each contact point to the wall plane
    y1, y2 = float(i0), float(i0 + 1)
    xr = right[0] + (right[1] - right[0]) * (y_wall - y1) / (y2 - y1)
    xl = left[0] + (left[1] - left[0]) * (y_wall - y1) / (y2 - y1)
    b = xr - xl
    if not (b > 0.0):
        out["status"] = "bracket_failed"
        return out

    # apex height: topmost crossing along the centre column
    jc = int(round(xc))
    col = arr[:, jc]
    ys = []
    for i in range(ny - 1):
        a, b2 = col[i], col[i + 1]
        if (a - level) * (b2 - level) < 0.0:
            ys.append(i + (level - a) / (b2 - a))
    if not ys:
        out["status"] = "no_interface"
        return out
    y_apex = max(ys)
    h = y_apex - y_wall
    if not (h > 0.0):
        out["status"] = "bracket_failed"
        return out

    R = (b * b + 4.0 * h * h) / (8.0 * h)
    theta = math.degrees(math.atan2(b / 2.0, R - h))
    out.update({"theta_deg": theta, "b": b, "h": h, "R": R})
    return out