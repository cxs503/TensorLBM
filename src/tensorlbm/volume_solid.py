"""Volumetric (soft-mask) solid representation for D2Q9 — solid-limit closures.

Context
-------
A *differentiable* / soft-mask LBM cannot use a binary solid mask (a step
function has zero gradient everywhere), so the body is carried as a
**volume-of-fluid field** ``m(x) in [0, 1]`` (solid fraction per cell).  The
question this module answers is *how that field is turned into a wall rule*,
because the answer fixes whether the model has a well defined **solid limit**:

``partial_bounce_back`` — amplitude closure (Walsh, Burwinkle & Saar 2009)
    The fraction is read as a *reflectivity*: per node the reflection
    coefficient is ``beta = clip(2 m, 0, 1)`` and each opposite population
    pair is mixed with weight ``beta``.  This is the classical
    "penalty proportional to solid fraction" closure.  It has one property
    that matters here: **any smoothing of the mask lowers beta in the
    interface band**, i.e. a diffuse volume representation leaks slip and the
    measured drag drifts with the mask scale (the defect this module is meant
    to remove).

``interface_bounce_back`` — geometry closure (this work)
    The same field is read as *geometry*.  The interface is reconstructed from
    the volume fraction with the standard first-order VOF relation
    ``phi = (m - 1/2) / |grad m|`` (positive inside the solid; Parker & Youngs
    1992, Scardovelli & Zaleski 2000) and the no-slip condition is imposed on
    that reconstructed interface with Bouzidi–Firdaouss–Lallemand interpolated
    bounce-back (Bouzidi, Firdaouss & Lallemand 2001, Phys. Fluids 13:3452):

        q >= 1/2 :  f_j(x) = (1/(2q)) f_i^post(x) + (1 - 1/(2q)) f_j^post(x)
        q <  1/2 :  f_j(x) = 2q f_i^post(x) + (1 - 2q) f_i^post(x - c_i)

    with ``j = opposite(i)``, ``q`` the sub-grid wall distance along the link
    measured from ``x``.  Smoothing the mask changes ``|grad m|`` and ``m``
    together, so the reconstructed interface — and therefore the drag — is
    invariant under the smoothing (up to the reconstruction error).

Both closures are **exactly** the repository half-way rule in the binary
limit: with ``m in {0, 1}`` the reconstructed distance is ``q = 1/2`` and both
closures reproduce :func:`tensorlbm.boundaries.bounce_back_cells`
bit-for-bit (``tests/test_volume_solid.py``).

Force bookkeeping
-----------------
``partial_bounce_back_force`` is the momentum-exchange (Ladd 1994) sum of the
closure written in pair form: the momentum of the pair ``(i, j)`` before the
mixing is ``(f_i - f_j) c_i`` and after it is ``(1 - 2 beta)(f_i - f_j) c_i``,
so the momentum handed to the solid is ``2 beta (f_i - f_j) c_i``.  With
``beta = 1`` on the solid cells this is *identically* the repository's
``compute_obstacle_forces`` sum ``2 sum_i c_i f_i`` over those cells — the
regression test pins that identity, which is what makes the volumetric drag a
continuation of the hard-wall momentum-exchange drag rather than a new
estimator with its own bias.

Everything here is pure ``torch`` on the ``(9, ny, nx)`` population layout used
by :mod:`tensorlbm.boundaries` / :mod:`tensorlbm.solver`; no hand-written
collision or streaming.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .boundaries import C, OPPOSITE, W

__all__ = [
    "PAIRS",
    "blur_mask",
    "interface_bounce_back",
    "mask_scale_to_sigma",
    "partial_bounce_back",
    "partial_bounce_back_force",
    "signed_distance_from_fraction",
    "vof_circle",
    "vof_plane_wall",
]

#: Opposite-direction pairs (rest direction 0 is its own opposite and is never
#: mixed: its momentum is zero).
PAIRS: tuple[tuple[int, int], ...] = ((1, 3), (2, 4), (5, 7), (6, 8))


def _shift(field: torch.Tensor, d: tuple[int, int]) -> torch.Tensor:
    """Return ``field`` sampled at ``x + d`` (``field`` is indexed ``[y, x]``).

    ``d`` is a ``(dy, dx)`` offset, so callers must swap the ``C`` row
    ``(cx, cy)`` before calling.
    """
    return torch.roll(field, (-int(d[0]), -int(d[1])), (0, 1))


def vof_circle(
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    radius: float,
    *,
    sub: int = 16,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Volume-of-fluid solid fraction of a disc on a ``(ny, nx)`` cell grid.

    Cell centres sit at integer ``(y, x)`` indices; the disc is centred at
    ``(cy, cx)`` and every cell is supersampled ``sub x sub`` times, so the
    returned field is the exact area fraction to ``O(1/sub^2)`` (the total
    solid volume is ``pi R^2`` to better than 1e-3 for ``sub = 16``).
    """
    if sub < 1:
        raise ValueError("sub must be >= 1")
    offs = (torch.arange(sub, dtype=dtype) + 0.5) / sub - 0.5
    ys = torch.arange(ny, dtype=dtype).view(ny, 1, 1) + offs.view(1, sub, 1)
    xs = torch.arange(nx, dtype=dtype).view(1, nx, 1) + offs.view(1, 1, sub)
    dy = (ys - cy) ** 2  # (ny, sub, 1)
    dx = (xs - cx) ** 2  # (1, nx, sub)
    inside = (dy[:, :, :, None] + dx[None, :, :, :]) <= radius * radius
    return inside.to(dtype).mean(dim=(1, 3))


def vof_plane_wall(
    ny: int,
    nx: int,
    *,
    axis: str = "y",
    lower: float | None = None,
    upper: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Volume fraction of a slab solid: ``lower <= coord < upper`` along *axis*.

    Used by the plane-Poiseuille analytic anchor: a wall that is only partly
    filled (``lower`` / ``upper`` fractional) is representable, which is
    exactly the soft-wall situation a volumetric solid has to handle.
    """
    if axis == "y":
        coord = torch.arange(ny, dtype=dtype).view(ny, 1).expand(ny, nx)
    elif axis == "x":
        coord = torch.arange(nx, dtype=dtype).view(1, nx).expand(ny, nx)
    else:  # pragma: no cover - guarded
        raise ValueError("axis must be 'y' or 'x'")
    lo = -1.0e9 if lower is None else float(lower)
    hi = 1.0e9 if upper is None else float(upper)
    return ((coord >= lo) & (coord < hi)).to(dtype)


def mask_scale_to_sigma(scale: float, sigma_max: float = 1.0) -> float:
    """Map the *mask scale* ``s in [0, 1]`` to an interface blur width.

    ``s = 1`` is the solid limit (no smoothing, the volume fractions are the
    geometry), ``s < 1`` smears the same solid over ``sigma = (1 - s) *
    sigma_max`` cells.  The blur is volume preserving, so the solid volume and
    (to ``O(sigma^2/R)``) the interface position are unchanged: the *only*
    thing that changes between mask scales is how sharp the volume
    representation is.  That is what makes the solid-limit criterion a
    controlled comparison rather than a geometry change.
    """
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"mask scale must lie in (0, 1], got {scale}")
    return (1.0 - float(scale)) * float(sigma_max)


def blur_mask(
    m: torch.Tensor,
    sigma: float,
    *,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Volume-preserving separable Gaussian blur of a fraction field.

    Neumann (``replicate``) padding means no flux leaves the domain, so the
    blurred field has the same integral as the input up to round-off — the
    solid volume is preserved by construction.
    """
    if sigma <= 0.0:
        return m.clone()
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=m.dtype, device=m.device)
    kernel = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    out = m.view(1, 1, *m.shape)
    out = F.conv2d(F.pad(out, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return out.view(*m.shape)


def signed_distance_from_fraction(m: torch.Tensor, *, clamp: float = 1.5) -> torch.Tensor:
    """First-order VOF interface reconstruction: ``phi = (m - 1/2) / |grad m|``.

    Positive inside the solid, in lattice units, clamped to ``+-clamp`` where
    the field is flat (both flat phases and the far field).  ``m = 1/2`` is the
    reconstructed interface by definition, and the relation is the standard
    first-order VOF estimate of the interface distance (Parker & Youngs 1992;
    Scardovelli & Zaleski 2000, Annu. Rev. Fluid Mech. 31:567).
    """
    # one-sided (max-modulus) differences: for a sharp volume fraction the
    # field changes by O(1) *within one cell*, so a two-cell centred stencil
    # halves the slope and doubles the recovered distance.  Taking the larger
    # of the forward/backward differences reproduces the exact linear relation
    # ``m - 1/2 = slope * d`` in both the sharp and the blurred limit.
    gx = torch.maximum((torch.roll(m, -1, 1) - m).abs(), (m - torch.roll(m, 1, 1)).abs())
    gy = torch.maximum((torch.roll(m, -1, 0) - m).abs(), (m - torch.roll(m, 1, 0)).abs())
    g = torch.sqrt(gx * gx + gy * gy)
    flat = g <= 1e-6
    phi = torch.where(flat, torch.full_like(m, 0.0), (m - 0.5) / g.clamp_min(1e-12))
    big = torch.where(m >= 0.5, torch.full_like(m, float(clamp)), torch.full_like(m, -float(clamp)))
    phi = torch.where(flat, big, phi)
    return phi.clamp(-float(clamp), float(clamp))


def partial_bounce_back(f: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Amplitude closure: mix opposite populations with weight ``2 m``.

    ``f`` is the **post-stream** population array (the same place where
    :func:`tensorlbm.boundaries.bounce_back_cells` acts) and ``m`` the solid
    fraction field.  ``beta = clip(2 m, 0, 1)``: ``beta = 0`` leaves the fluid
    untouched, ``beta = 1`` is the full half-way swap, intermediate values are
    the usual partial bounce-back of a partially filled cell.
    """
    beta = (2.0 * m).clamp(0.0, 1.0)
    out = f.clone()
    active = beta > 0.0
    for i, j in PAIRS:
        gi = f[i]
        gj = f[j]
        out[i] = torch.where(active, (1.0 - beta) * gi + beta * gj, gi)
        out[j] = torch.where(active, (1.0 - beta) * gj + beta * gi, gj)
    return out


def partial_bounce_back_force(f: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Force on the solid handed over by :func:`partial_bounce_back`.

    Pair form of the momentum-exchange method (Ladd 1994): the pair momentum
    ``(f_i - f_j) c_i`` shrinks to ``(1 - 2 beta)(f_i - f_j) c_i``, so the
    solid receives ``2 beta (f_i - f_j) c_i``.  For ``beta = 1`` on the solid
    cells on the post-stream array this is *identically*
    :func:`tensorlbm.boundaries.compute_obstacle_forces`
    (``2 sum_i c_i f_i`` over those cells).
    """
    beta = (2.0 * m).clamp(0.0, 1.0)
    c = C.to(torch.float64)
    fx = torch.zeros((), dtype=torch.float64)
    fy = torch.zeros((), dtype=torch.float64)
    for i, j in PAIRS:
        d = (beta * (f[i].to(torch.float64) - f[j].to(torch.float64))).sum()
        fx = fx + 2.0 * d * c[i, 0]
        fy = fy + 2.0 * d * c[i, 1]
    return fx, fy


def interface_bounce_back(
    f: torch.Tensor,
    f_post: torch.Tensor,
    phi: torch.Tensor,
    *,
    q_floor: float = 1.0e-3,
) -> torch.Tensor:
    """Geometry closure: Bouzidi interpolated bounce-back on the VOF interface.

    Args:
        f: post-stream populations, modified in place on a copy.  A fluid node
            ``x`` whose link in direction ``i`` crosses the interface
            (``phi(x) < 0 < phi(x + c_i)``) receives in the opposite direction
            ``j`` the interpolated value instead of whatever streams out of
            the solid cell.
        f_post: post-collision populations of the *same* step (``f`` before
            streaming) — the interpolated bounce-back is written in terms of
            post-collision values.
        phi: signed distance field from :func:`signed_distance_from_fraction`
            (positive inside the solid), lattice units.

    Returns:
        New population array.  Cells with ``phi >= 0`` are reset to the rest
        equilibrium of a motionless unit-density solid, so nothing that leaks
        out of the body interior can reach a fluid node (every fluid link that
        crosses the interface is overwritten here).
    """
    out = f.clone()
    for i in range(1, 9):
        j = int(OPPOSITE[i])
        ci = (int(C[i, 1]), int(C[i, 0]))  # (dy, dx); C rows are (cx, cy)
        phi_b = _shift(phi, ci)
        crossing = (phi < 0.0) & (phi_b > 0.0)
        if not bool(crossing.any()):
            continue
        q = (-phi) / (phi_b - phi).clamp_min(1e-12)
        q = q.clamp(float(q_floor), 1.0)
        f_i = f_post[i]
        f_j = f_post[j]
        value_hi = (1.0 / (2.0 * q)) * f_i + (1.0 - 1.0 / (2.0 * q)) * f_j
        f_i_up = _shift(f_i, (-ci[0], -ci[1]))
        value_lo = 2.0 * q * f_i + (1.0 - 2.0 * q) * f_i_up
        value = torch.where(q >= 0.5, value_hi, value_lo)
        out[j] = torch.where(crossing, value, out[j])
    # frozen solid: rest equilibrium, zero momentum, no memory of the interior
    solid = phi >= 0.0
    if bool(solid.any()):
        for i in range(9):
            out[i] = torch.where(solid, torch.full_like(out[i], float(W[i])), out[i])
    return out