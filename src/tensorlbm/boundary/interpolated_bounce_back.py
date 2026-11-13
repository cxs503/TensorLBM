"""First-order interpolated bounce-back (2-point Bouzidi closure) from a sub-grid SDF.

Why this module exists
----------------------
The production boundary operators (:func:`tensorlbm.boundaries.bounce_back_cells`,
:func:`tensorlbm.boundaries3d.bounce_back_cells_3d`) implement *half-way*
bounce-back: the no-slip plane is placed half a lattice unit away from the last
fluid node, i.e. on the face between the fluid node and the solid node.  For a
wall that is not lattice aligned (inclined or curved wall) the solid set is a
*staircase* whose face is not the physical wall, so the effective no-slip
surface is displaced by a geometry-dependent amount ``delta_eff`` of order one
lattice unit.  Report ``docs/POISEUILLE_3D_ELLIPSE_REFERENCE_GEOMETRY_AUDIT.md``
measured ``delta_eff = +0.35 / +0.31`` cells on a staircase ellipse (a = 20 / 40)
and showed that a *single global* scale correction (``s^Q``) cannot absorb it
because the displacement is anisotropic along the staircase.

This module supplies the missing *local* correction: the wall location is taken
from the sub-grid signed distance function (``sdf > 0`` in fluid, ``< 0`` in
solid) instead of from the staircase face, and the reflected population is
reconstructed by linear interpolation between the two nodes of the cut link
(Bouzidi, Firdaouss & Lallemand, *Phys. Fluids* 13 (2001) 3452, 2-point form).
The operator is a **drop-in replacement** for the half-way swap; at ``q = 1/2``
it collapses exactly onto half-way bounce-back (this is asserted by a test).

Calling convention (post-streaming, two time levels)
----------------------------------------------------
``f_pre``  = post-collision (pre-streaming) populations at time ``n``.
``f_new``  = streamed populations (pure periodic shift) at time ``n+1/2``.
The returned array is ``f_new`` with the solid nodes' outgoing directions
rewritten, so that the next streaming step delivers the interpolated reflection
to the adjacent fluid node.  For a cut link ``x_f (fluid) -- x_s = x_f + c_i
(solid)`` with ``q in (0,1)`` measured from ``x_f`` towards the wall:

``q < 1/2``::

    f(x_s, c_opp(i)) = 2q f_pre(x_f, i) + (1 - 2q) f_new(x_f, i)

``q >= 1/2``::

    f(x_s, c_opp(i)) = (1/2q) f_pre(x_f, i) + ((2q - 1)/2q) f_pre(x_f, opp(i))

At ``q = 1/2`` both branches reduce to ``f_new(x_f, i)``, which is exactly the
value the half-way swap would have delivered.  This mirrors the single
geometric statement already used by
:mod:`tensorlbm.curved_boundary_calibration` (P48-N01, "the population arriving
from the wall equals the post-collision population at the specular mirror of the
solid node"), restricted to the 2-point (linear) stencil so that it can be
applied as a fixed-size stencil update to an arbitrary production state.

The wall distance is obtained from the level set by first-order link
interpolation::

    q = s_f / (s_f - s_s)

with ``s_f``/``s_s`` the signed distances at the fluid/solid end of the link.
No geometry-specific code lives here: the analytic (or voxelised) SDF is supplied
by the caller, so the operator is domain agnostic.

Degenerate links (``q < 1/2`` where the node behind the fluid node is itself
solid, i.e. a fluid layer thinner than one cell) fall back to ``q = 1/2``; the
number of such links is returned for disclosure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tensorlbm.d2q9 import C as _C2
from tensorlbm.d2q9 import OPPOSITE as _OPP2
from tensorlbm.d3q19 import C as _C3
from tensorlbm.d3q19 import OPPOSITE as _OPP3

__all__ = [
    "InterpolatedBounceBackReport",
    "alpha_bfl",
    "alpha_cli",
    "alpha_yli",
    "interpolated_bounce_back",
    "interpolated_bounce_back_2d",
    "interpolated_bounce_back_3d",
]


@dataclass(frozen=True)
class InterpolatedBounceBackReport:
    """Bookkeeping for one application of the operator (for evidence tables)."""

    cut_links: int
    interpolated_links: int
    fallback_links: int
    q_min: float
    q_max: float
    q_mean: float


def alpha_bfl(q: torch.Tensor) -> torch.Tensor:
    """Bouzidi-Firdaouss-Lallemand rule: ``alpha = 2`` for ``q <= 1/2``, ``1/q`` above.

    This is the rule pylbm encodes as ``BFLBounceBack`` / ``_alpha_bfl``
    (``~/oss-cache/pylbm/pylbm/boundary.py``, ``alpha = 2`` for ``delta <= 0.5``
    and ``1/delta`` otherwise); with it the general ``alpha`` closure below
    collapses onto the two hard-coded branches of the historical operator.
    """
    return torch.where(q <= 0.5, torch.full_like(q, 2.0), 1.0 / q.clamp_min(0.5))


def alpha_cli(q: torch.Tensor) -> torch.Tensor:
    """Central linear interpolation ``alpha = 4/(1 + 2q)`` (pylbm ``CLIBounceBack``).

    Substituting into ``alpha_hat = -1 + alpha (1/2 + q)`` gives ``alpha_hat = 1``
    for every ``q``, i.e. this member of the family needs no post-collision
    correction term (pylbm: "the scheme is parametrized without correction").
    """
    return 4.0 / (1.0 + 2.0 * q)


def alpha_yli(q: torch.Tensor) -> torch.Tensor:
    """Yu-Mei-Luo-Shyy rule ``alpha = 2/(1 + q)`` (pylbm ``YLIBounceBack``)."""
    return 2.0 / (1.0 + q)


def _resolve_alpha(
    alpha: float | Callable[[torch.Tensor], torch.Tensor] | None,
    q: torch.Tensor,
) -> torch.Tensor:
    """Materialise the per-link ``alpha`` coefficient of the interpolated closure."""
    if alpha is None:
        return alpha_bfl(q)
    if callable(alpha):
        out = alpha(q)
        if not isinstance(out, torch.Tensor):
            raise TypeError("alpha callable must return a torch.Tensor")
        return out.to(device=q.device, dtype=q.dtype)
    return torch.full_like(q, float(alpha))


def _spatial_dims(field: torch.Tensor) -> tuple[int, ...]:
    """Roll dims for a field of shape ``spatial`` (populations are indexed first)."""
    return tuple(range(field.ndim))


def _roll(t: torch.Tensor, shift: tuple[int, ...], dims: tuple[int, ...]) -> torch.Tensor:
    return torch.roll(t, shifts=shift, dims=dims)


def interpolated_bounce_back(
    f_new: torch.Tensor,
    f_pre: torch.Tensor,
    sdf: torch.Tensor,
    solid: torch.Tensor,
    c: torch.Tensor,
    opp: torch.Tensor,
    *,
    q_min: float = 0.05,
    q_max: float = 0.95,
    alpha: float | Callable[[torch.Tensor], torch.Tensor] | None = None,
    post_collision_correction: bool = False,
    f_old: torch.Tensor | None = None,
    return_report: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, InterpolatedBounceBackReport]:
    """Apply 2-point interpolated bounce-back on every cut link of an SDF geometry.

    Args:
        f_new: streamed populations, shape ``(Q, *spatial)``.
        f_pre: post-collision populations at the previous time level, same shape.
        sdf: signed distance in lattice units, ``> 0`` fluid, ``< 0`` solid,
            shape ``spatial`` (the analytic wall sits at ``sdf = 0``).
        solid: boolean solid mask, shape ``spatial``.
        c: ``(Q, dim)`` lattice velocity table (``d2q9.C`` / ``d3q19.C``).
        opp: ``(Q,)`` opposite-direction table.
        q_min, q_max: clipping of the interpolated wall distance.  The clip keeps
            the linear stencil bounded (weights in ``[0, 1]``); links whose
            geometric ``q`` falls outside are reported as clipped.
        alpha: family parameter of the interpolated closure (pylbm
            ``LinearBounceBack``): ``None`` keeps the Bouzidi-Firdaouss-Lallemand
            rule (:func:`alpha_bfl`, the historical two-branch operator),
            :func:`alpha_cli` / :func:`alpha_yli` select the central / YLI
            members, and a float or callable ``alpha(q)`` gives a custom member.
            The reflected value is

                ``alpha_hat f_pre(i) + beta_hat f_pre(opp i) + beta f_new(i)``

            with ``alpha_hat = -1 + alpha (1/2 + q)``, ``beta = 1 - alpha q``,
            ``beta_hat = 1 - alpha/2`` (zero-velocity wall, so the antisymmetric
            equilibrium term ``-alpha e^-_q(x_w)`` of the pylbm form vanishes).
        post_collision_correction: add the ``K_hat (g^-_q + S^-_q)`` term with
            ``K_hat = 1 - alpha_hat``, i.e. the antisymmetric part of the change
            the collision and the source produced at the fluid node last step.
            This is what makes the family *parametrized* (pylbm: steady state
            then depends on the relaxation rates only through ``Lambda+ Lambda-``).
            Requires ``f_old``.
        f_old: pre-collision populations at the same time level as ``f_pre``
            (``f_pre = collision(f_old) + source``); only used when
            ``post_collision_correction`` is set.  Difference ``f_pre - f_old``
            is the collision+source change, so a body-force source must be
            included in ``f_pre`` for its antisymmetric part to enter the term.
        return_report: also return an :class:`InterpolatedBounceBackReport`.

    Returns:
        Populations with the solid-side reflected values rewritten (new tensor;
        the inputs are not modified).
    """
    if f_new.shape != f_pre.shape:
        raise ValueError(f"f_new/f_pre shape mismatch: {tuple(f_new.shape)} vs {tuple(f_pre.shape)}")
    spatial = tuple(f_new.shape[1:])
    if tuple(sdf.shape) != spatial or tuple(solid.shape) != spatial:
        raise ValueError(
            f"sdf/solid must have shape {spatial}, got {tuple(sdf.shape)} / {tuple(solid.shape)}"
        )
    if bool(torch.isnan(sdf).any()):
        raise ValueError("sdf contains NaN")
    if bool((sdf[~solid] <= 0).any()):
        raise ValueError("sdf must be positive on fluid cells")
    if bool((sdf[solid] >= 0).any()):
        raise ValueError("sdf must be negative on solid cells")
    if post_collision_correction:
        if f_old is None:
            raise ValueError("post_collision_correction=True requires f_old")
        if f_old.shape != f_pre.shape:
            raise ValueError(
                f"f_old/f_pre shape mismatch: {tuple(f_old.shape)} vs {tuple(f_pre.shape)}"
            )

    dim = len(spatial)
    device = f_new.device
    c = c.to(device=device)
    opp = opp.to(device=device)
    fluid = ~solid
    dims = _spatial_dims(sdf)

    out = f_new.clone()
    n_cut = 0
    n_interp = 0
    n_fallback = 0
    q_used: list[torch.Tensor] = []

    for i in range(1, c.shape[0]):
        d = int(opp[i].item())
        # ``c`` is stored (cx, cy[, cz]) but the population array is (Q, ...,
        # ny, nx) / (Q, nz, ny, nx), so the index-space shift is reversed.
        shift = tuple(int(v) for v in reversed(c[i].tolist()))
        if all(v == 0 for v in shift):
            continue

        nb_solid = _roll(solid, tuple(-v for v in shift), dims)  # solid at x + c_i
        link = fluid & nb_solid
        if not bool(link.any()):
            continue

        s_f = sdf
        s_s = _roll(sdf, tuple(-v for v in shift), dims)
        q = s_f / (s_f - s_s)

        # A q < 1/2 stencil reaches the node behind x_f along -c_i; if that node is
        # solid the needed population does not exist -> degrade to half-way.
        behind_fluid = _roll(fluid, shift, dims)  # fluid at x - c_i
        needs_low = q < 0.5
        bad = link & needs_low & ~behind_fluid
        n_fallback += int(bad.sum().item())

        q_link = torch.where(link, q, torch.full_like(q, 0.5))
        q_link = torch.where(bad, torch.full_like(q_link, 0.5), q_link)
        clipped = q_link.clamp(q_min, q_max)
        n_cut += int(link.sum().item())
        n_interp += int(link.sum().item()) - int(bad.sum().item())
        q_used.append(clipped[link])

        f_i_pre = f_pre[i]
        f_d_pre = f_pre[d]
        f_i_new = f_new[i]

        alpha_q = _resolve_alpha(alpha, clipped)
        alpha_hat = -1.0 + alpha_q * (0.5 + clipped)
        beta = 1.0 - alpha_q * clipped
        beta_hat = 1.0 - 0.5 * alpha_q
        val = alpha_hat * f_i_pre + beta_hat * f_d_pre + beta * f_i_new
        if post_collision_correction:
            # antisymmetric part of the collision+source change at the fluid node
            assert f_old is not None  # guaranteed by validation above
            g_i = f_i_pre - f_old[i]
            g_d = f_d_pre - f_old[d]
            val = val + (1.0 - alpha_hat) * 0.5 * (g_i - g_d)

        target = _roll(link, shift, dims)  # solid cell x_s = x_f + c_i
        out[d] = torch.where(target, _roll(val, shift, dims), out[d])

    if not return_report:
        return out

    if q_used:
        qall = torch.cat(q_used)
        rep = InterpolatedBounceBackReport(
            cut_links=n_cut,
            interpolated_links=n_interp,
            fallback_links=n_fallback,
            q_min=float(qall.min().item()),
            q_max=float(qall.max().item()),
            q_mean=float(qall.mean().item()),
        )
    else:
        rep = InterpolatedBounceBackReport(0, 0, 0, 0.5, 0.5, 0.5)
    return out, rep


def interpolated_bounce_back_2d(
    f_new: torch.Tensor,
    f_pre: torch.Tensor,
    sdf: torch.Tensor,
    solid: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor | tuple[torch.Tensor, InterpolatedBounceBackReport]:
    """D2Q9 wrapper; see :func:`interpolated_bounce_back`."""
    return interpolated_bounce_back(f_new, f_pre, sdf, solid, _C2, _OPP2, **kwargs)  # type: ignore[arg-type]


def interpolated_bounce_back_3d(
    f_new: torch.Tensor,
    f_pre: torch.Tensor,
    sdf: torch.Tensor,
    solid: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor | tuple[torch.Tensor, InterpolatedBounceBackReport]:
    """D3Q19 wrapper; see :func:`interpolated_bounce_back`."""
    return interpolated_bounce_back(f_new, f_pre, sdf, solid, _C3, _OPP3, **kwargs)  # type: ignore[arg-type]