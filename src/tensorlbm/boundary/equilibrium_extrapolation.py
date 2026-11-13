"""Equilibrium-difference extrapolation (non-equilibrium extrapolation) BC.

Cross-validation target
-----------------------
``hietwll/LBM_Taichi`` (``lbm_solver.py:130``, inventoried in
``docs/oss_inventory_lbm_taichi.md``) prescribes the populations of a boundary
node from the first inner node with

.. math::

    f_i(x_b) = f_i^{eq}(x_b) - f_i^{eq}(x_1) + f_i(x_1)

i.e. the *equilibrium difference* between the wall and the interior node is
added to the interior population.  Algebraically the same statement is

.. math::

    f_i(x_b) = f_i^{eq}(x_b) + f_i^{neq}(x_1), \\qquad
    f_i^{neq} = f_i - f_i^{eq},

which is the *non-equilibrium extrapolation* of Guo, Zheng & Shi (2002,
*Phys. Fluids* 14, 2007): extrapolate :math:`f^{neq}` to the wall, take
:math:`f^{eq}` from the wall's Dirichlet data.  The OSS form is therefore the
**zero-order (single-node) variant** of that family, with the wall density
taken from the inner node; the original paper and the shipped second-order
option here linear-extrapolate :math:`f^{neq}` (and optionally :math:`\\rho`)
from two inner nodes.

What this module adds over the OSS practice
-------------------------------------------
* ``order=1``: bit-for-bit the OSS rule (``f_eq(b) - f_eq(1) + f(1)``), exposed
  through :func:`oss_equilibrium_difference` so the identity can be *tested*,
  not just asserted.
* ``order=2``: :math:`f^{neq}_b = 2 f^{neq}_1 - f^{neq}_2` (Guo 2002) and,
  with ``rho_mode="extrapolate"``, :math:`\\rho_b = 2\\rho_1 - \\rho_2`.
* The wall density choice is explicit (``rho_mode``), because the OSS copy of
  :math:`\\rho_1` is a modelling decision with a measurable cost (see
  ``docs/XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION.md``).

Conventions
-----------
Fields are ``(9, ny, nx)`` for D2Q9 (velocity index first), matching
:mod:`tensorlbm.d2q9`.  A boundary plane is addressed by node *slices*, which
lets the same kernel serve a channel (bottom/top rows), a lid-driven cavity
(four walls) and, by extension, any axis-aligned wall.  Boundary nodes carry
prescribed populations and are **not** collided or streamed -- the caller must
keep them out of both kernels (see :func:`fluid_mask_from_walls`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..d2q9 import equilibrium, macroscopic

__all__ = [
    "EquilibriumExtrapolationSpec",
    "apply_equilibrium_extrapolation",
    "apply_channel_extrapolation",
    "apply_cavity_extrapolation",
    "fluid_mask_from_walls",
    "oss_equilibrium_difference",
    "non_equilibrium_part",
]


@dataclass(frozen=True)
class EquilibriumExtrapolationSpec:
    """Numerical choices of one extrapolation application.

    Attributes:
        order: ``1`` = OSS single-inner-node rule; ``2`` = Guo (2002) linear
            extrapolation of :math:`f^{neq}` from two inner nodes.
        rho_mode: ``"copy"`` takes the wall density from the first inner node
            (what the OSS implementation does); ``"extrapolate"`` uses
            :math:`2\\rho_1-\\rho_2` (needs ``order=2``).
    """

    order: int = 1
    rho_mode: str = "copy"

    def __post_init__(self) -> None:
        if self.order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {self.order!r}")
        if self.rho_mode not in ("copy", "extrapolate"):
            raise ValueError(f"rho_mode must be 'copy' or 'extrapolate', got {self.rho_mode!r}")
        if self.order == 1 and self.rho_mode == "extrapolate":
            raise ValueError("rho_mode='extrapolate' needs two inner nodes (order=2)")


def non_equilibrium_part(f: torch.Tensor) -> torch.Tensor:
    """Return :math:`f - f^{eq}(\\rho, u)` for a ``(9, ny, nx)`` field."""
    rho, ux, uy = macroscopic(f)
    return f - equilibrium(rho, ux, uy)


def oss_equilibrium_difference(f: torch.Tensor, dst, src1, ux_wall, uy_wall, rho_wall=None) -> torch.Tensor:
    """Literal transcription of the OSS rule, for identity testing.

    ``f[:, dst] = f_eq(b) - f_eq(1) + f(1)`` with the boundary density copied
    from node 1 unless ``rho_wall`` is given.  Kept separate from the modular
    implementation so :func:`apply_equilibrium_extrapolation` can be checked
    against it (``tests/test_equilibrium_extrapolation_bc.py``).

    Args:
        f: distribution field before/after collision, whichever the caller feeds
            to the wall (source of ``f(1)``).
        dst: node slices of the boundary plane.
        src1: node slices of the first inner node plane.
        ux_wall, uy_wall: Dirichlet wall velocity, broadcastable to the plane.
        rho_wall: wall density; ``None`` copies the inner-node density.
    """
    rho_s, ux_s, uy_s = macroscopic(f)
    s1 = (src1[1], src1[2])
    feq_src = equilibrium(rho_s[s1], ux_s[s1], uy_s[s1])
    rho_b = rho_s[s1] if rho_wall is None else torch.as_tensor(rho_wall, dtype=f.dtype, device=f.device)
    feq_dst = equilibrium(
        _broadcast(rho_b, f, dst[1], dst[2]),
        _broadcast(ux_wall, f, dst[1], dst[2]),
        _broadcast(uy_wall, f, dst[1], dst[2]),
    )
    return feq_dst - feq_src + f[:, src1[1], src1[2]]


def _broadcast(value, f: torch.Tensor, ys, xs):
    """Broadcast a scalar or ``(ny_plane, nx_plane)`` tensor onto a node plane."""
    ny_plane = len(range(*ys.indices(f.shape[1])))
    nx_plane = len(range(*xs.indices(f.shape[2])))
    if isinstance(value, torch.Tensor):
        return value.to(dtype=f.dtype, device=f.device)
    return torch.full(
        (ny_plane, nx_plane), float(value), dtype=f.dtype, device=f.device
    )


def apply_equilibrium_extrapolation(
    f: torch.Tensor,
    dst,
    src1,
    src2,
    ux_wall,
    uy_wall,
    spec: EquilibriumExtrapolationSpec | None = None,
    *,
    rho_wall=None,
) -> torch.Tensor:
    """Return a copy of ``f`` with the boundary plane set by extrapolation.

    The caller supplies the field whose *post-collision* values should be
    emitted into the fluid: applying this to ``f`` before collision and then
    colliding (and streaming) the whole field would relax the prescribed
    non-equilibrium part away again, so boundary nodes must be excluded from
    the collision kernel.

    Args:
        f: ``(9, ny, nx)`` populations (already collided if the caller uses a
            post-collision-prescription ordering).
        dst: ``(slice, slice, slice)`` for the boundary plane.
        src1: slices of the first inner node (nearest to the wall).
        src2: slices of the second inner node (required for ``order=2``).
        ux_wall, uy_wall: Dirichlet wall velocity (scalar or plane tensor).
        spec: :class:`EquilibriumExtrapolationSpec`; default ``order=1,
            rho_mode='copy'`` = the OSS rule.
        rho_wall: optional explicit wall density plane (overrides ``rho_mode``).

    Returns:
        New tensor; ``f`` itself is not modified.
    """
    spec = spec or EquilibriumExtrapolationSpec()
    out = f.clone()

    rho_s, ux_s, uy_s = macroscopic(f)
    s1 = (src1[1], src1[2])
    s2 = (src2[1], src2[2])
    feq_src1 = equilibrium(rho_s[s1], ux_s[s1], uy_s[s1])
    fneq_1 = f[:, src1[1], src1[2]] - feq_src1
    if spec.order == 2:
        feq_src2 = equilibrium(rho_s[s2], ux_s[s2], uy_s[s2])
        fneq_2 = f[:, src2[1], src2[2]] - feq_src2
        fneq_b = 2.0 * fneq_1 - fneq_2
    else:
        fneq_b = fneq_1

    if rho_wall is not None:
        rho_b = _broadcast(rho_wall, f, dst[1], dst[2])
    elif spec.rho_mode == "extrapolate":
        rho_b = 2.0 * rho_s[s1] - rho_s[s2]
    else:
        rho_b = rho_s[s1]

    feq_dst = equilibrium(
        rho_b if isinstance(rho_b, torch.Tensor) else _broadcast(rho_b, f, dst[1], dst[2]),
        _broadcast(ux_wall, f, dst[1], dst[2]),
        _broadcast(uy_wall, f, dst[1], dst[2]),
    )
    out[:, dst[1], dst[2]] = feq_dst + fneq_b
    return out


def _plane(ny: int, nx: int, axis: str, which: int, offset: int):
    """Node slices of the plane ``which`` (0 = low side, 1 = high side).

    ``offset`` is signed in node units: ``which=0`` maps to index ``offset``,
    ``which=1`` to ``n - 1 + offset``, so ``+1`` is inward on the low side and
    ``-1`` inward on the high side.
    """
    n = ny if axis == "y" else nx
    idx = which * (n - 1) + offset
    if not 0 <= idx < n:
        raise ValueError(f"plane index {idx} outside 0..{n - 1} ({axis}, which={which}, offset={offset})")
    if axis == "y":
        return (slice(None), slice(idx, idx + 1), slice(None))
    return (slice(None), slice(None), slice(idx, idx + 1))


def apply_channel_extrapolation(
    f: torch.Tensor,
    ux_bottom: float,
    ux_top: float,
    spec: EquilibriumExtrapolationSpec | None = None,
) -> torch.Tensor:
    """Apply the BC to the first/last **rows** of a channel field (walls span x).

    Rows ``0`` and ``ny-1`` are the wall nodes; the wall planes coincide with
    those node rows (no half-cell offset), which is why the effective channel
    width in lattice units is ``ny - 1`` -- calibrated in the XVAL bench.
    Normal velocity is zero on both walls.
    """
    ny, nx = f.shape[1], f.shape[2]
    out = f
    for which, uxw in ((0, ux_bottom), (1, ux_top)):
        sign = 1 if which == 0 else -1  # inward direction
        dst = _plane(ny, nx, "y", which, 0)
        src1 = _plane(ny, nx, "y", which, sign)
        src2 = _plane(ny, nx, "y", which, 2 * sign)
        out = apply_equilibrium_extrapolation(out, dst, src1, src2, uxw, 0.0, spec)
    return out


def apply_cavity_extrapolation(
    f: torch.Tensor,
    u_lid: float,
    spec: EquilibriumExtrapolationSpec | None = None,
) -> torch.Tensor:
    """Apply the BC to all four walls of a closed cavity (lid on the top row).

    Walls sit on the boundary nodes (``wall_offset = 0`` in
    :func:`tensorlbm.lid_driven_cavity.centreline_coordinates`).  Corners are
    written twice; the top-row prescription wins (lid velocity), which is the
    same ordering the OSS implementation gets from a single mask loop.
    """
    ny, nx = f.shape[1], f.shape[2]
    out = f
    planes = (
        (_plane(ny, nx, "y", 0, 0), _plane(ny, nx, "y", 0, 1), _plane(ny, nx, "y", 0, 2), 0.0),
        (_plane(ny, nx, "y", 1, 0), _plane(ny, nx, "y", 1, -1), _plane(ny, nx, "y", 1, -2), u_lid),
        (_plane(ny, nx, "x", 0, 0), _plane(ny, nx, "x", 0, 1), _plane(ny, nx, "x", 0, 2), 0.0),
        (_plane(ny, nx, "x", 1, 0), _plane(ny, nx, "x", 1, -1), _plane(ny, nx, "x", 1, -2), 0.0),
    )
    for dst, src1, src2, uxw in planes:
        out = apply_equilibrium_extrapolation(out, dst, src1, src2, uxw, 0.0, spec)
    return out


def fluid_mask_from_walls(ny: int, nx: int, walls: str = "channel") -> torch.Tensor:
    """Boolean ``(ny, nx)`` mask of nodes that are collided/streamed.

    ``walls="channel"`` marks rows ``0`` and ``ny-1`` as prescribed;
    ``"cavity"`` also marks columns ``0`` and ``nx-1``.
    """
    mask = torch.zeros((ny, nx), dtype=torch.bool)
    mask[0, :] = True
    mask[-1, :] = True
    if walls == "cavity":
        mask[:, 0] = True
        mask[:, -1] = True
    elif walls != "channel":
        raise ValueError(f"unknown walls={walls!r}")
    return ~mask