"""Geometrical wetting boundary condition for the D2Q9 color-gradient solver.

What this module adds
---------------------
``tensorlbm.color_gradient2d`` has no wetting model: the solid cells are
masked to zero density before the color gradient is taken, which is exactly
the *neutral* (90 deg) wall condition.  This module supplies the missing
piece -- the **geometrical wetting** condition of Akai et al. (2018), i.e.
the color-gradient normal at a wall-adjacent node is rotated so that it meets
the wall under the prescribed static contact angle, and the phase value of
the adjacent solid cell ("ghost" value) is set to be consistent with that
rotated normal.

The condition used here
-----------------------
Let ``phi = (rho_r - rho_b)/rho`` in ``[-1, 1]`` (``phi > 0`` = wetting /
"red" phase), let ``t`` be the wall tangent and ``n_w`` the direction that
points *from the fluid node into the wall* (one lattice step).  A static
interface meeting the wall under contact angle ``theta`` (measured inside the
``phi > 0`` phase) has, near the contact line,

    d(phi)/d(n_w) = cot(theta) * |d(phi)/dt|                        (*)

because the interface normal ``n = (sin theta, -cos theta)`` (in wall-tangent
/ wall-normal coordinates, ``n`` pointing from the non-wetting into the
wetting phase) satisfies ``|n_n| / |n_t| = cot(theta)``.  Integrating (*)
over the single lattice step from the last fluid node ``f`` to the adjacent
solid cell ``s`` gives the ghost value

    phi(s) = phi(f) + cot(theta) * |d(phi)/dt|(f)                    (**)

so that the one-sided wall-normal difference the solver sees,
``phi(f) - phi(s)``, reproduces the rotated normal.  ``theta = 90 deg``
gives ``phi(s) = phi(f)``, i.e. the mirror (neutral) wall of the stock
solver; ``theta -> 0`` pushes ``phi`` strongly positive (complete wetting)
and ``theta -> 180`` strongly negative.

The magnitude (``|d(phi)/dt|``, not the signed derivative) is used because a
wetting phase climbs the wall on *both* flanks of a drop; only the ratio of
the two gradient components carries the angle information.  This is the
"rotation" step of the reference implementation; see
``docs/XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING.md`` for the point-by-point
comparison with ``lanl/MF-LBM`` (``Phase_gradient.F90:209-265`` @ 11be324),
including the ``distP``/``distM`` candidate selection that this port does
*not* reproduce and the ``theta < 90 deg`` restriction that it does not
inherit.

Reference
---------
T. Akai, S. M. Al Zarooni, C. B. ... "Wetting boundary condition for the
color-gradient lattice Boltzmann method: Simulation and theoretical insight",
Phys. Rev. E 98, 043302 (2018) -- the method inherited by ``lanl/MF-LBM``
(``multiphase_3D/0.src/Phase_gradient.F90:209-265``, BSD-3-Clause).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch

__all__ = [
    "OPP",
    "WALL_DIRECTIONS",
    "phase_field",
    "geometrical_wetting_ghost",
    "wall_local_contact_angle",
]

#: D2Q9 opposite-direction table for the ``tensorlbm.d2q9.C`` ordering
#: (0 rest, 1 E, 2 N, 3 W, 4 S, 5 NE, 6 NW, 7 SW, 8 SE).
OPP: Tuple[int, ...] = (0, 3, 2, 1, 4, 7, 8, 5, 6)

#: Lattice directions that can act as a "wall normal" (fluid -> solid) in 2-D.
WALL_DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))


def phase_field(rho_r: torch.Tensor, rho_b: torch.Tensor) -> torch.Tensor:
    """``phi = (rho_r - rho_b) / (rho_r + rho_b)`` in ``[-1, 1]``."""
    rho = (rho_r + rho_b).clamp(min=1e-12)
    return (rho_r - rho_b) / rho


def _nb(a: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """``out[..., i, j] = a[..., i + dy, j + dx]`` (periodic, like the solver)."""
    return torch.roll(a, shifts=(-dy, -dx), dims=(-2, -1))


def _extend_into_solid(phi: torch.Tensor, solid: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Copy the fluid neighbour's ``phi`` into solid cells one step against ``(dy, dx)``.

    Only used to keep the *tangential* stencil from sampling meaningless
    values inside the wall; at a flat wall the tangential stencil never
    touches a solid cell at all, so this is a corner-safety measure.
    """
    return torch.where(solid, _nb(phi, -dy, -dx), phi)


def _wall_face(solid: torch.Tensor, dy: int, dx: int, periodic: Tuple[bool, bool]) -> torch.Tensor:
    """Fluid cells whose ``(dy, dx)`` neighbour is solid, without wrap-around artefacts.

    ``periodic`` is ``(x, y)``; the solver streams periodically in both
    directions, but a wall that is only imposed on one side must not pick up a
    phantom face from the wrapped row/column.
    """
    face = (~solid) & _nb(solid, dy, dx)
    if not periodic[1]:
        if dy > 0:
            face[-1, :] = False
        elif dy < 0:
            face[0, :] = False
    if not periodic[0]:
        if dx > 0:
            face[:, -1] = False
        elif dx < 0:
            face[:, 0] = False
    return face


def geometrical_wetting_ghost(
    phi: torch.Tensor,
    solid: torch.Tensor,
    theta_deg: float,
    tol_grad: float = 1e-8,
    phi_bounds: Tuple[float, float] = (-1.0, 1.0),
    periodic: Tuple[bool, bool] = (True, False),
    theta_by_direction: Dict[Tuple[int, int], float] | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    """Ghost phase values in the solid cells enforcing a static contact angle.

    Args:
        phi:        ``(ny, nx)`` phase field; values inside ``solid`` are ignored.
        solid:      ``(ny, nx)`` boolean solid mask (``True`` = solid).
        theta_deg:  prescribed static contact angle **inside the ``phi > 0``
                    phase**.  ``90`` reproduces the neutral stock wall;
                    values outside ``(0, 180)`` are rejected.
        tol_grad:   nodes whose tangential phase gradient is below this
                    magnitude carry no interface and get the neutral ghost
                    ``phi = 0`` (the stock masked-wall behaviour); this is the
                    analogue of the reference code's ``|C| > 1e-6`` guard.
        phi_bounds: clamp for the ghost value (keeps the ghost state physical);
        periodic:   ``(x, y)`` periodicity of the domain; the wall faces of a
                    non-periodic direction are clipped so that a one-sided wall
                    cannot pick up a phantom face from the wrapped row.
        theta_by_direction: optional ``{(dy, dx): theta_deg}`` override, used to
                    keep one wall neutral (90 deg) while another wets.

    Returns:
        ``(ny, nx)`` tensor of ghost phase values; solid cells that carry no
        wetting constraint hold ``0.0`` (neutral).  With
        ``return_diagnostics=True`` also a dict with the number of constrained
        cells and the tangential-gradient statistics at the wall.
    """
    if not (0.0 < theta_deg < 180.0):
        raise ValueError(f"theta_deg must lie in (0, 180), got {theta_deg}")
    if phi.shape != solid.shape:
        raise ValueError(f"shape mismatch: phi {tuple(phi.shape)} vs solid {tuple(solid.shape)}")

    lo, hi = phi_bounds

    ghost_acc = torch.zeros_like(phi)
    count = torch.zeros_like(phi)
    n_active = 0
    n_faces = 0
    grad_samples = []

    for dy, dx in WALL_DIRECTIONS:
        wall_face = _wall_face(solid, dy, dx, periodic)
        if not bool(wall_face.any()):
            continue
        th_dir = theta_deg if theta_by_direction is None else theta_by_direction.get((dy, dx), theta_deg)
        if not (0.0 < th_dir < 180.0):
            raise ValueError(f"theta for direction {(dy, dx)} must lie in (0, 180), got {th_dir}")
        cot = math.cos(math.radians(th_dir)) / math.sin(math.radians(th_dir))
        phi_ext = _extend_into_solid(phi, solid, dy, dx)
        # wall tangent t = (-dx, dy)
        g_t = 0.5 * (_nb(phi_ext, -dx, dy) - _nb(phi_ext, dx, -dy))
        active = wall_face & (g_t.abs() > tol_grad)
        # the wetting constraint is applied on every wall face: where the
        # tangential gradient vanishes (bulk) it degenerates to the mirror
        # value phi(face) on its own, so no extra guard is needed
        ghost_face = (phi + ghost_acc.new_tensor(cot) * g_t.abs()).clamp(min=lo, max=hi)
        # move the face-wise ghost onto the solid cell one step along (dy, dx)
        ghost_solid = _nb(ghost_face, -dy, -dx)
        face_solid = _nb(wall_face.to(phi.dtype), -dy, -dx) > 0.5
        ghost_acc = torch.where(face_solid, ghost_acc + ghost_solid, ghost_acc)
        count = torch.where(face_solid, count + 1.0, count)
        n_faces += int(wall_face.sum())
        n_active += int(active.sum())
        if bool(active.any()):
            grad_samples.append(g_t[active].detach().abs())

    has = count > 0
    phi_solid = torch.where(has, ghost_acc / count.clamp(min=1.0), torch.zeros_like(phi))
    phi_solid = phi_solid.clamp(min=lo, max=hi)

    if not return_diagnostics:
        return phi_solid

    g_all = torch.cat(grad_samples) if grad_samples else torch.zeros(1, dtype=phi.dtype)
    diag = {
        "theta_deg": float(theta_deg),
        "n_wall_face_nodes": float(n_faces),
        "n_interface_face_nodes": float(n_active),
        "n_ghost_cells": float(int(has.sum())),
        "ghost_mean": float(phi_solid[has].mean()) if bool(has.any()) else 0.0,
        "grad_t_mean": float(g_all.mean()),
        "grad_t_max": float(g_all.max()),
    }
    return phi_solid, diag


def wall_local_contact_angle(
    phi: torch.Tensor,
    phi_solid: torch.Tensor,
    solid: torch.Tensor,
    tol_grad: float = 0.05,
    phi_gate: float = 0.9,
    periodic: Tuple[bool, bool] = (True, False),
    face_subset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Local contact angle implied by the *imposed* wall gradient, per node.

    Reads back the condition the solver will see: the wall-normal difference
    ``phi(f) - phi(s)`` (one lattice step) against the tangential difference
    ``|d(phi)/dt|``, giving ``theta = atan2(|g_t|, g_n)`` in degrees.  This is
    an independent estimator of the contact angle that involves neither a
    circle fit nor a cap-area inversion -- it measures the boundary condition
    itself, so a mismatch with the measured droplet angle isolates
    discretisation / curvature effects from BC-construction errors.

    Returns a 1-D tensor of angles (degrees) for the wall-face nodes that
    carry an interface.
    """
    angles = []
    for dy, dx in WALL_DIRECTIONS:
        wall_face = _wall_face(solid, dy, dx, periodic)
        if not bool(wall_face.any()):
            continue
        phi_ext = _extend_into_solid(phi, solid, dy, dx)
        g_t = 0.5 * (_nb(phi_ext, -dx, dy) - _nb(phi_ext, dx, -dy))
        g_n = _nb(phi_solid, dy, dx) - phi  # d(phi)/d(towards wall), 1 lattice step
        active = wall_face & (g_t.abs() > tol_grad) & (phi.abs() < phi_gate)
        if face_subset is not None:
            active = active & face_subset
        if not bool(active.any()):
            continue
        ang = torch.atan2(g_t[active].abs(), g_n[active]) * (180.0 / math.pi)
        angles.append(ang)
    if not angles:
        return torch.zeros(0)
    return torch.cat(angles)