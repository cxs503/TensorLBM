"""Conservative TVB / minmod slope limiter in the Legendre modal basis.

The limiter follows Cockburn & Shu (1989): an element is flagged *troubled*
when the deviation of its endpoint values from the cell mean exceeds
``M * dx^2`` (the TVB threshold).  A flagged element is replaced by the linear
profile

    u(xi) = ubar + 0.5 (dr + dl) + 0.5 (dr - dl) xi ,

whose endpoint deviations ``dl``, ``dr`` are the minmod-limited forward /
backward differences of the neighbouring cell means.  Because only the
``k >= 1`` modes are touched, the cell mean ``m_0`` -- and therefore the
discrete mass ledger -- is *exactly* preserved (this is what makes the square
wave TVD test and the mass-conservation test compatible).

``M = 0`` gives the pure minmod limiter (TVD); a finite ``M`` relaxes the
limiter where the local slope is already ``O(dx^2)`` (smooth flow).
"""

from __future__ import annotations

import numpy as np

from .element import RefElement

__all__ = ["minmod", "minmod3", "nodal_to_modal", "modal_to_nodal", "tvb_limit"]


def minmod(*args: np.ndarray) -> np.ndarray:
    """Minmod of an arbitrary number of arrays (elementwise)."""
    if len(args) == 0:
        raise ValueError("minmod needs at least one argument")
    out = np.asarray(args[0], dtype=np.float64)
    out = np.zeros_like(out) + out
    for a in args[1:]:
        a = np.asarray(a, dtype=np.float64)
        out = np.where(
            (out * a) > 0.0,
            np.sign(out) * np.minimum(np.abs(out), np.abs(a)),
            0.0,
        )
    return out


def minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return minmod(a, minmod(b, c))


def nodal_to_modal(u: np.ndarray, elem: RefElement) -> np.ndarray:
    """Nodal values ``(..., n)`` -> Legendre modal coefficients ``(..., n)``."""
    return u @ elem.Ven_inv.T


def modal_to_nodal(m: np.ndarray, elem: RefElement) -> np.ndarray:
    """Legendre modal coefficients ``(..., n)`` -> nodal values ``(..., n)``."""
    n = elem.n_node
    Ven = np.stack(
        [np.polynomial.legendre.Legendre.basis(k)(elem.nodes) for k in range(n)],
        axis=1,
    )
    return m @ Ven.T


def cell_means(u: np.ndarray, elem: RefElement) -> np.ndarray:
    """Cell averages of nodal data ``(..., N, n)`` -> ``(..., N)``.

    The mean is ``m_0`` of the Legendre expansion, i.e. the ``L2`` projection on
    the constants; it equals ``(1/dx) int u`` and is preserved by the limiter.
    """
    return nodal_to_modal(u, elem)[..., 0]


def tvb_limit(
    u: np.ndarray,
    elem: RefElement,
    dx: float,
    M: float = 0.0,
    axis: int = -2,
) -> np.ndarray:
    """Apply the Cockburn-Shu minmod/TVB limiter along cell axis ``axis``.

    Args:
        u: nodal data ``(..., N, n_node)`` with the node axis last and the cell
            axis at position ``axis`` (``-2`` by default).
        elem: reference element.
        dx: cell width (uniform grid).
        M: TVB threshold parameter (units of ``dx^2``); ``M = 0`` == pure
            minmod (TVD), which is what the square-wave test uses.
        axis: axis holding the ``N`` cells.

    Returns a new array with the same cell means.  Periodic in the cell axis.
    """
    u = np.asarray(u, dtype=np.float64)
    if axis < 0:
        axis = u.ndim + axis
    assert axis == u.ndim - 2, "limiter expects (..., N, n_node) with node axis last"

    means = cell_means(u, elem)  # (..., N)
    m_prev = np.roll(means, 1, axis=-1)
    m_next = np.roll(means, -1, axis=-1)
    d_minus = means - m_prev
    d_plus = m_next - means

    u_l = u[..., 0]  # left endpoint value (xi = -1)
    u_r = u[..., -1]  # right endpoint value (xi = +1)
    dev_l = u_l - means
    dev_r = u_r - means

    thresh = M * dx * dx
    troubled = (np.abs(dev_l) > thresh) | (np.abs(dev_r) > thresh)
    if not np.any(troubled):
        return u.copy()

    # ONE symmetric limited slope (Cockburn-Shu).  Limiting the two endpoints
    # independently gives a profile whose mean is no longer ubar, i.e. it breaks
    # conservation -- the symmetric minmod keeps the cell average exact:
    #     u(xi) = ubar + slope * xi        (xi averages to zero)
    # All candidates must be *slopes* (same units!): the endpoint deviation of a
    # linear profile is slope*dx/2, so divide by dx/2 -- mixing the two units
    # silently halves the slope and turns the scheme first-order.
    slope = minmod(
        2.0 * dev_r / dx,
        -2.0 * dev_l / dx,
        d_plus / dx,
        d_minus / dx,
    )

    xi = elem.nodes  # (n,)
    # u(xi) = ubar + slope * (x - x_c) = ubar + (dx/2) slope xi   (mean exact)
    prof = means[..., None] + 0.5 * dx * slope[..., None] * xi[None, :]

    mask = troubled[..., None]
    return np.where(mask, prof, u)