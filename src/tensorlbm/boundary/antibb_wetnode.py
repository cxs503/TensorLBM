"""Anti-bounceback (Inamuro) wet-node Dirichlet-velocity boundary condition.

This module implements the *anti-bounceback* (ABB) rule of Inamuro, Yoshino &
Ogino (1999), the wet-node velocity boundary condition discussed in
Krueger et al., *The Lattice Boltzmann Method: Principles and Practice*,
ch. 8 ("anti-bounceback" / Inamuro boundary condition).  The reference
practice in the textbook code base (``lbm-principles-practice/code``) applies
the rule at boundary nodes that are part of the computational domain ("wet
nodes"): the wall itself carries distribution functions, and populations that
would stream in from outside the domain are replaced by the anti-symmetric
combination of the post-collision populations and an equilibrium-like term
built from the prescribed wall velocity.

Rule (per boundary node ``x_b`` with inward unit normal ``n``)::

    f_i(x_b, t+1) = -f_opp(i)^*(x_b, t) + 2 w_i rho_w S_i(u_w)
                    for every direction i with c_i . n > 0

with the equilibrium-consistent source factor

    S_i(u_w) = 1 + (9/2) (c_i . u_w)^2 - (3/2) |u_w|^2

(``closure="exact"``, the default) or its linearised textbook variant
``S_i = 1 + 3 (c_i . u_w)`` (``closure="linear"``).  Note that ``f_i^eq`` is
exactly anti-symmetric in the ``u`` term and symmetric in the ``u^2`` terms,
i.e. ``f_i^eq + f_opp(i)^eq = 2 w_i rho (1 + (9/2)(c_i.u)^2 - (3/2)|u|^2)``;
only the quadratic form reproduces a prescribed equilibrium at the wet node.
The linearised form leaves an ``O(u)`` wall-velocity bias (measured
``-u_w/3`` -> ``+u_w`` overshoot; see the verification report).

where

* ``f_i`` are the distributions that would arrive from outside the domain
  (unknown after streaming),
* ``f_opp(i)^*`` is the *post-collision* population of the opposite direction
  at the same node (known),
* ``rho_w`` is the density attributed to the wall node (either a constant
  ``rho0`` -- the usual incompressible choice -- or extrapolated from the
  adjacent fluid node),
* ``u_w`` is the prescribed wall velocity, and ``c_s^2 = 1/3``.

Consequences (all verified in ``docs/antibb_wetnode_verification_20261113.md``):

* the no-slip surface coincides with the *boundary node plane*, i.e. the
  effective wall offset is ``0`` lattice units -- unlike halfway bounce-back
  whose effective wall sits ``0.5`` lu outside the boundary node.  A channel
  bounded by wet nodes at rows ``0`` and ``ny-1`` therefore has hydraulic
  width ``H = ny - 1``, whereas the same rows with halfway bounce-back give
  ``H = ny``.
* the Dirichlet velocity is imposed on the boundary node itself, so moving
  walls (Couette, lid-driven, moving bodies) need no momentum-exchange
  bookkeeping: ``u_w`` enters the rule explicitly.
* the scheme is second-order accurate for walls aligned with lattice links and
  the resulting wall-node velocity equals ``u_w`` up to ``O(Ma^2)`` normal
  residual (measured below), which is the usual compressibility artefact of
  the method.

The implementation is backend agnostic (a pure ``torch.where`` expression, no
host/device transfers) and works for any lattice whose velocities/weights are
supplied; D2Q9 defaults come from :mod:`tensorlbm.d2q9`.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorlbm.d2q9 import C as C2Q9
from tensorlbm.d2q9 import OPPOSITE as OPPOSITE2Q9
from tensorlbm.d2q9 import W_EXACT as W2Q9_EXACT

__all__ = [
    "antibb_unknown_mask",
    "apply_antibb_wetnode",
    "adjacent_fluid_values",
    "wall_density_from_fluid",
    "effective_wall_offset",
    "DEFAULT_NORMALS_BOTTOM_TOP",
]


def _lattice_tensors(
    velocities: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    opposite: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
):
    if velocities is None:
        velocities = C2Q9.to(device=device, dtype=dtype)
    else:
        velocities = velocities.to(device=device, dtype=dtype)
    if weights is None:
        # d2q9.W is a float32 table; rebuilding from the exact rational values
        # keeps the anti-symmetric sum rule at machine precision in float64.
        weights = torch.tensor(W2Q9_EXACT, dtype=dtype, device=device)
    else:
        weights = weights.to(device=device, dtype=dtype)
    if opposite is None:
        if velocities.shape[0] != OPPOSITE2Q9.shape[0]:
            raise ValueError(
                "must pass explicit `opposite` for lattices other than D2Q9 "
                f"(got q={velocities.shape[0]})"
            )
        opposite = OPPOSITE2Q9.to(device=device)
    return velocities, weights, opposite


def _direction_normal_dot(
    velocities: torch.Tensor, normals: torch.Tensor
) -> torch.Tensor:
    """``c_i . n`` for every direction ``i`` (shape ``(q, *spatial)``)."""
    dot = velocities[:, 0].view(-1, *([1] * (normals.dim() - 1))) * normals[..., 0]
    for d in range(1, velocities.shape[1]):
        dot = dot + velocities[:, d].view(-1, *([1] * (normals.dim() - 1))) * normals[..., d]
    return dot


def antibb_unknown_mask(
    normals: torch.Tensor,
    wet: Optional[torch.Tensor] = None,
    velocities: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Boolean mask of the populations the ABB rule must overwrite.

    A population ``f_i`` is unknown (it would stream in from outside the
    domain) iff ``c_i . n > 0`` at a wet node, where ``n`` points *into* the
    fluid.  Directions tangential to the wall (``c_i . n == 0``) stream from
    neighbouring boundary nodes and are left untouched.

    Parameters
    ----------
    normals : Tensor, shape ``(*spatial, d)``
        Inward unit normals, defined at least on the wet nodes (ignored
        elsewhere).
    wet : Tensor of bool, shape ``spatial``, optional
        Boundary-node mask.  ``None`` means "every node".
    velocities : Tensor, shape ``(q, d)``, optional
        Lattice velocities; D2Q9 if omitted.
    """
    if velocities is None:
        velocities = C2Q9.to(device=normals.device, dtype=normals.dtype)
    dot = _direction_normal_dot(velocities, normals)
    mask = dot > 0
    if wet is not None:
        mask = mask & wet.unsqueeze(0)
    return mask


def apply_antibb_wetnode(
    f_post: torch.Tensor,
    f_streamed: torch.Tensor,
    wet: torch.Tensor,
    normals: torch.Tensor,
    u_wall: torch.Tensor | float,
    rho_w: torch.Tensor | float = 1.0,
    *,
    closure: str = "exact",
    velocities: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    opposite: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the Inamuro anti-bounceback rule on the wet nodes.

    Parameters
    ----------
    f_post : Tensor, shape ``(q, *spatial)``
        Post-collision populations at time ``t`` (before streaming).
    f_streamed : Tensor, shape ``(q, *spatial)``
        Populations after streaming at time ``t+1``.  Values in the unknown
        directions are irrelevant (they are overwritten); all other values are
        preserved.
    wet : Tensor of bool, shape ``spatial``
        Mask of wet boundary nodes.
    normals : Tensor, shape ``(*spatial, d)``
        Inward unit normal per node (only wet nodes are used).  Axis-aligned
        normals give the textbook plane-wall rule; arbitrary (curved-wall)
        normals work as long as the classification ``c_i . n > 0`` is
        unambiguous.
    u_wall : float or Tensor, shape ``spatial`` or ``(*spatial, d)``
        Prescribed wall velocity.  A scalar is interpreted as the ``x``
        component with zero ``y``/``z`` components (a wall moving along the
        lattice ``x`` axis).
    rho_w : float or Tensor, shape ``spatial``
        Density used in the equilibrium-like term (default ``1.0``).  See
        :func:`wall_density_from_fluid` for the extrapolated variant.
    closure : {"exact", "linear", "none"}
        Source factor ``S_i(u_w)``:

        * ``"exact"`` (default) -- ``1 + (9/2)(c_i.u_w)^2 - (3/2)|u_w|^2``:
          reproduces ``f_i^eq(u_w)`` exactly on a wall node that is already at
          the prescribed velocity, so the Dirichlet value is imposed without
          bias.
        * ``"linear"`` -- ``1 + 3 (c_i.u_w)``: the widely quoted linearised
          textbook form.  Retained for comparison / regression; it biases the
          wall velocity by ``O(u_w)`` (measured overshoot ``u_w/3``).
        * ``"none"`` -- ``1``: pure anti-bounceback (zero-velocity wall);
          negative control for moving walls.

    Returns
    -------
    Tensor, shape ``(q, *spatial)``
        Populations at ``t+1`` with the unknown boundary populations set.
    """
    if f_post.shape != f_streamed.shape:
        raise ValueError(
            f"f_post and f_streamed shapes differ: {tuple(f_post.shape)} vs "
            f"{tuple(f_streamed.shape)}"
        )
    if f_post.dim() < 2:
        raise ValueError("f must have shape (q, *spatial)")

    q = f_post.shape[0]
    device = f_post.device
    dtype = f_post.dtype
    velocities, weights, opposite = _lattice_tensors(velocities, weights, opposite, device, dtype)
    if velocities.shape[0] != q:
        raise ValueError(f"lattice has q={velocities.shape[0]} directions, f has {q}")

    spatial = f_post.shape[1:]
    if tuple(wet.shape) != tuple(spatial):
        raise ValueError(
            f"wet mask shape {tuple(wet.shape)} != population spatial shape {tuple(spatial)}"
        )
    if tuple(normals.shape) != tuple(spatial) + (velocities.shape[1],):
        raise ValueError(
            f"normals shape {tuple(normals.shape)} != {tuple(spatial) + (velocities.shape[1],)}"
        )

    # ---- prescribed wall velocity -> (…, d) tensor ----
    view_shape = (-1,) + (1,) * (len(spatial))
    if not torch.is_tensor(u_wall):
        uw = torch.zeros(tuple(spatial) + (velocities.shape[1],), device=device, dtype=dtype)
        uw[..., 0] = float(u_wall)
    else:
        uw = u_wall.to(device=device, dtype=dtype)
        if uw.dim() == len(spatial):  # per-node x-component only
            zero_pad = torch.zeros(tuple(spatial) + (velocities.shape[1] - 1,),
                                   device=device, dtype=dtype)
            uw = torch.cat([uw.unsqueeze(-1), zero_pad], dim=-1)
        elif tuple(uw.shape) != tuple(spatial) + (velocities.shape[1],):
            raise ValueError(
                f"u_wall shape {tuple(uw.shape)} is neither {tuple(spatial)} nor "
                f"{tuple(spatial) + (velocities.shape[1],)}"
            )

    if torch.is_tensor(rho_w):
        rho = rho_w.to(device=device, dtype=dtype)
        if tuple(rho.shape) != tuple(spatial):
            raise ValueError(
                f"rho_w shape {tuple(rho.shape)} != spatial shape {tuple(spatial)}"
            )
        rho_term = 2.0 * weights.view(view_shape) * rho.unsqueeze(0)
    else:
        rho_term = 2.0 * float(rho_w) * weights.view(view_shape)

    # c_i . u_w  (shape (q, *spatial))
    cu = torch.zeros((q,) + tuple(spatial), device=device, dtype=dtype)
    for d in range(velocities.shape[1]):
        cu = cu + velocities[:, d].view(view_shape) * uw[..., d].unsqueeze(0)

    if closure == "exact":
        uw_sq = torch.zeros(tuple(spatial), device=device, dtype=dtype)
        for d in range(velocities.shape[1]):
            uw_sq = uw_sq + uw[..., d] ** 2
        source = 1.0 + 4.5 * cu * cu - 1.5 * uw_sq.unsqueeze(0)
    elif closure == "linear":
        source = 1.0 + 3.0 * cu
    elif closure == "none":
        source = torch.ones_like(cu)
    else:
        raise ValueError(f"unknown closure {closure!r}")
    injected = -f_post[opposite] + rho_term * source
    unknown = antibb_unknown_mask(normals, wet, velocities)
    return torch.where(unknown, injected, f_streamed)


def effective_wall_offset() -> float:
    """Distance (lattice units) between the boundary-node plane and the wall.

    The anti-bounceback rule places the no-slip surface *on* the wet node,
    hence ``0.0``.  Halfway bounce-back instead places it ``0.5`` lu outside
    the boundary node.  This is the quantity that converts a node count into
    a hydraulic width ``H``.
    """
    return 0.0


def adjacent_fluid_values(
    field: torch.Tensor,
    wet: torch.Tensor,
    normals: torch.Tensor,
) -> torch.Tensor:
    """Sample ``field`` one node along ``+n`` (into the fluid) from wet nodes.

    Used to build the extrapolated wall density ``rho_w`` of Inamuro et al.
    Normals are rounded to the nearest lattice vector (axis-aligned normals
    are exact); indices are clamped so the gather is always in bounds.
    """
    if field.dim() != 2:
        raise ValueError("adjacent_fluid_values currently supports 2-D fields only")
    ny, nx = field.shape
    yy = torch.arange(ny, device=field.device, dtype=field.dtype).view(-1, 1)
    xx = torch.arange(nx, device=field.device, dtype=field.dtype).view(1, -1)
    ty = (yy + normals[..., 1].round()).clamp(0, ny - 1).long().expand(ny, nx)
    tx = (xx + normals[..., 0].round()).clamp(0, nx - 1).long().expand(ny, nx)
    sampled = field[ty, tx]
    return torch.where(wet, sampled, field)


def wall_density_from_fluid(
    f: torch.Tensor,
    wet: torch.Tensor,
    normals: torch.Tensor,
) -> torch.Tensor:
    """Density at wet nodes extrapolated from the adjacent fluid node.

    ``rho(x_b) <- rho(x_b + n)``.  This is the zeroth-order extrapolation of
    Inamuro et al. (1999) and is the recommended ``rho_w`` for pressure-driven
    flows where the wall density is not known a priori; for incompressible
    flow the constant ``rho_w = 1`` choice gives the same accuracy within
    ``O(Ma^2)``.
    """
    rho = f.sum(dim=0)
    return adjacent_fluid_values(rho, wet, normals)


def bottom_top_normals(ny: int, nx: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Normals for a channel bounded by wet-node rows ``0`` and ``ny-1``.

    Row ``0`` gets ``n = (0, +1)`` (fluid above); row ``ny-1`` gets
    ``n = (0, -1)``.  Interior nodes get a zero normal (unused).
    """
    normals = torch.zeros((ny, nx, 2), device=device, dtype=dtype)
    normals[0, :, 1] = 1.0
    normals[-1, :, 1] = -1.0
    return normals


DEFAULT_NORMALS_BOTTOM_TOP = bottom_top_normals