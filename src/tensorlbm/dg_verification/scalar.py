"""2-D scalar advection + diffusion on a periodic tensor-product DG grid.

Field layout (numpy, float64)

    u[iy, ix, jy, jx]      cell indices ``(iy, ix)``, node indices ``(jy, jx)``

with the *last* node axis paired with the *last* cell axis (x).  Element
operators come from :mod:`element`; multi-dimensional derivatives are applied
axis by axis as tensor products (exact for the tensor-product nodal basis).

Fluxes
------
* ``upwind`` (default): the face flux is taken from the upwind side of the
  face-normal velocity, evaluated at the face point.  For a divergence-free
  rotating field this is the least dissipative choice and is what reproduces
  the textbook ``p + 1`` L2 convergence of nodal DG.
* ``rusanov``: local Lax-Friedrichs, ``0.5(fL + fR) - 0.5 alpha (uR - uL)``,
  kept for dissipation comparisons.

Diffusion uses LDG (local discontinuous Galerkin) with the alternating flux
pair ``(u_hat = u^-, q_hat = q^+)``, which is compact, stable on periodic boxes
and yields the optimal order for the heat operator.
"""

from __future__ import annotations

import numpy as np

from .element import RefElement

__all__ = [
    "apply_axis",
    "advection_rhs_2d",
    "diffusion_rhs_2d",
    "convection_diffusion_rhs_2d",
    "ssp_rk3",
    "integral_2d",
    "cell_mean_2d",
    "mass_ratio_drift",
    "node_coords_2d",
]


# ---------------------------------------------------------------------------
# small linear-algebra helpers (last-axis matrix application)
# ---------------------------------------------------------------------------


def _apply_last(a: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """``a[..., j] -> sum_j mat[k, j] a[..., j]`` (reduce last axis)."""
    shape = a.shape
    return (a.reshape(-1, shape[-1]) @ mat.T).reshape(shape[:-1] + (mat.shape[0],))


def apply_axis(a: np.ndarray, mat: np.ndarray, axis: int) -> np.ndarray:
    """Apply ``mat`` along ``axis`` (last axis uses the fast contiguous path)."""
    if axis == a.ndim - 1:
        return _apply_last(a, mat)
    return np.moveaxis(_apply_last(np.moveaxis(a, axis, -1), mat), -1, axis)


def face_term(f_l: np.ndarray, f_r: np.ndarray, elem: RefElement, node_axis: int, ndim: int) -> np.ndarray:
    """Assemble ``S[:, 0] f_l + S[:, 1] f_r`` on ``node_axis`` of an ``ndim`` array.

    ``f_l``/``f_r`` are indexed like the cell lattice (the node axis removed);
    the result has the full ``ndim`` shape with the node axis restored.  This is
    the only place where the face-selection matrix is placed, so the x and y
    contributions share one (axis-correct) code path.
    """
    ax = node_axis % ndim
    shape = [1] * ndim
    shape[ax] = elem.n_node
    s0 = elem.S[:, 0].reshape(shape)
    s1 = elem.S[:, 1].reshape(shape)
    return np.expand_dims(f_l, ax) * s0 + np.expand_dims(f_r, ax) * s1


def _face_flux(u_l, u_r, a_n, scheme: str = "upwind"):
    """Scalar face flux from the left/right states and the face-normal speed."""
    f_l = a_n * u_l
    f_r = a_n * u_r
    if scheme == "upwind":
        return np.where(a_n >= 0.0, f_l, f_r)
    if scheme == "rusanov":
        return 0.5 * (f_l + f_r) - 0.5 * np.abs(a_n) * (u_r - u_l)
    raise ValueError(f"unknown flux scheme {scheme!r}")


def _directional_rhs(u, f, a_n_face, elem, h, cell_axis, node_axis, scheme):
    """DG weak-form RHS contribution of ``-d/dx (a u)`` along one axis.

    Weak form of ``u_t + f_x = 0`` used here (the standard one):

        M du/dt = +int_K f phi_i'  -  [ f*_r phi_i(x_r) - f*_l phi_i(x_l) ]

    so the surface term enters with a *minus* sign.  Getting this wrong turns
    the upwind flux into a downwind one and the scheme grows exponentially.
    """
    dvol = apply_axis(f, elem.Gr, node_axis)
    u_r_cell = u.take(-1, axis=node_axis)  # right trace of each cell
    u_l_cell = u.take(0, axis=node_axis)  # left trace
    # global face i lies between cell i (left state) and cell i+1 (right state)
    f_star = _face_flux(u_r_cell, np.roll(u_l_cell, -1, axis=cell_axis), a_n_face, scheme)
    f_star_left = np.roll(f_star, 1, axis=cell_axis)
    surf = face_term(f_star_left, f_star, elem, node_axis, u.ndim)
    return apply_axis(dvol - surf, elem.Minv, node_axis) * (2.0 / h)


def advection_rhs_2d(
    u: np.ndarray,
    ax,
    ay,
    elem: RefElement,
    dx: float,
    dy: float,
    scheme: str = "upwind",
) -> np.ndarray:
    """RHS of ``u_t + (a_x u)_x + (a_y u)_y = 0`` (velocity may vary in space).

    ``ax``/``ay`` may be scalars or nodal fields broadcastable to ``u``; the
    face-normal velocity is the nodal trace at the face point.
    """
    if np.ndim(ax) == 0:
        ax = np.full_like(u, float(ax), dtype=np.float64)
    else:
        ax = np.broadcast_to(np.asarray(ax, dtype=np.float64), u.shape)
    if np.ndim(ay) == 0:
        ay = np.full_like(u, float(ay), dtype=np.float64)
    else:
        ay = np.broadcast_to(np.asarray(ay, dtype=np.float64), u.shape)

    ax_face = ax.take(-1, axis=-1)  # (Ny, Nx, n_y)
    rhs = _directional_rhs(u, ax * u, ax_face, elem, dx, -2, -1, scheme)

    ay_face = ay.take(-1, axis=-2)  # (Ny, Nx, n_x)
    rhs = rhs + _directional_rhs(u, ay * u, ay_face, elem, dy, -3, -2, scheme)
    return rhs


# ---------------------------------------------------------------------------
# LDG diffusion
# ---------------------------------------------------------------------------


def diffusion_rhs_2d(u: np.ndarray, nu: float, elem: RefElement, dx: float, dy: float) -> np.ndarray:
    """RHS of ``u_t = nu * laplacian(u)`` via LDG on the periodic grid."""
    rhs = np.zeros_like(u)
    for dim in (1, 0):  # 1 -> x (cell -2, node -1), 0 -> y (cell -3, node -2)
        if dim == 1:
            cell_axis, node_axis, h = -2, -1, dx
        else:
            cell_axis, node_axis, h = -3, -2, dy

        # (1) q = d_a u  with  u_hat = u^-  (trace of the cell on the low side).
        #     int q phi = -int u phi' + [u_hat phi]  ->  -Gr u + S [u_hat_l, u_hat_r]
        u_hat = u.take(-1, axis=node_axis)
        u_hat_l = np.roll(u_hat, 1, axis=cell_axis)
        q = -apply_axis(u, elem.Gr, node_axis)
        q = q + face_term(u_hat_l, u_hat, elem, node_axis, u.ndim)
        q = apply_axis(q, elem.Minv, node_axis) * (2.0 / h)

        # (2) u_t = nu * d_a q  with  q_hat = q^+  (trace of the neighbour);
        #     this is a conservation law, so its surface term enters with '-' 
        q_left = q.take(0, axis=node_axis)
        q_hat = np.roll(q_left, -1, axis=cell_axis)  # value of cell i+1 at face i
        q_hat_l = np.roll(q_hat, 1, axis=cell_axis)
        # u_t = +nu q_x  (no minus, unlike the advection flux!)  so the surface
        # term keeps its '+' sign here -- this sign split is what makes the LDG
        # operator elliptic (nu u_xx) rather than the negative-definite -nu u_xx.
        du = -apply_axis(q, elem.Gr, node_axis)
        du = du + face_term(q_hat_l, q_hat, elem, node_axis, u.ndim)
        rhs += nu * apply_axis(du, elem.Minv, node_axis) * (2.0 / h)
    return rhs


def convection_diffusion_rhs_2d(
    u: np.ndarray,
    ax,
    ay,
    nu: float,
    elem: RefElement,
    dx: float,
    dy: float,
    scheme: str = "upwind",
) -> np.ndarray:
    """RHS of ``u_t + (a u)_x + (b u)_y = nu Laplacian(u)``."""
    rhs = advection_rhs_2d(u, ax, ay, elem, dx, dy, scheme)
    if nu:
        rhs = rhs + diffusion_rhs_2d(u, nu, elem, dx, dy)
    return rhs


# ---------------------------------------------------------------------------
# time integration, quadrature, diagnostics
# ---------------------------------------------------------------------------


def ssp_rk3(u: np.ndarray, dt: float, rhs, *args, **kwargs) -> np.ndarray:
    """One SSP-RK3 (Shu-Osher) step of ``u_t = rhs(u)``."""
    k1 = u + dt * rhs(u, *args, **kwargs)
    k2 = 0.75 * u + 0.25 * (k1 + dt * rhs(k1, *args, **kwargs))
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (k2 + dt * rhs(k2, *args, **kwargs))


def integral_2d(u: np.ndarray, elem: RefElement, dx: float, dy: float) -> float:
    """Cell-quadrature integral of a nodal field over the whole box."""
    iy = elem.integral(u, dy)  # (..., n_x)
    return float(elem.integral(iy, dx).sum())


def cell_mean_2d(u: np.ndarray, elem: RefElement) -> np.ndarray:
    """``(Ny, Nx)`` cell-mean field of a nodal DG field."""
    return u.mean(axis=(-1, -2))


def mass_ratio_drift(u0, u, elem: RefElement, dx: float, dy: float) -> float:
    """Relative drift ``|int u - int u0| / |int u0|`` of the DG mass ledger."""
    m0 = integral_2d(u0, elem, dx, dy)
    m1 = integral_2d(u, elem, dx, dy)
    return abs(m1 - m0) / max(abs(m0), 1e-300)


def node_coords_2d(
    ny: int, nx: int, elem: RefElement, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Physical coordinates of every node, shape ``(ny, nx, n, n)``."""
    xi = elem.nodes
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    x = xc[:, None] + 0.5 * dx * xi[None, :]  # (nx, n)
    y = yc[:, None] + 0.5 * dy * xi[None, :]  # (ny, n)
    X = np.broadcast_to(x[None, :, None, :], (ny, nx, elem.n_node, elem.n_node))
    Y = np.broadcast_to(y[:, None, :, None], (ny, nx, elem.n_node, elem.n_node))
    return np.array(X), np.array(Y)


def gauss_coords_2d(
    ny: int, nx: int, elem: RefElement, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-point coordinates ``(ny, nx, nq, nq)`` and the 2-D weight field."""
    nq = elem.wq.size
    xq = np.polynomial.legendre.leggauss(nq)[0]
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    x = xc[:, None] + 0.5 * dx * xq[None, :]
    y = yc[:, None] + 0.5 * dy * xq[None, :]
    X = np.broadcast_to(x[None, :, None, :], (ny, nx, nq, nq))
    Y = np.broadcast_to(y[:, None, :, None], (ny, nx, nq, nq))
    W = np.broadcast_to(
        (0.25 * dx * dy * elem.wq[None, :] * elem.wq[:, None])[None, None],
        (ny, nx, nq, nq),
    )
    return np.array(X), np.array(Y), np.array(W)


def l2_error_vs(u: np.ndarray, exact, elem: RefElement, dx: float, dy: float) -> dict:
    """L1/L2/Linf error of a nodal DG field against an exact function.

    The error is evaluated at Gauss quadrature points, where the DG field is
    the exact polynomial trace; the L2 norm therefore uses the true quadrature
    weights (``p + 3`` Gauss points over-integrate the mass matrix).
    """
    ny, nx = u.shape[0], u.shape[1]
    X, Y, W = gauss_coords_2d(ny, nx, elem, dx, dy)
    # field at the Gauss points: tensor-product interpolation, axis by axis
    uq = apply_axis(u, elem.V, -1)  # (..., ny_node, nq_x)
    uq = apply_axis(uq, elem.V, -2)  # (..., nq_y, nq_x)
    err = uq - exact(X, Y)
    tot = W.sum()
    return {
        "L1": float((np.abs(err) * W).sum() / tot),
        "L2": float(np.sqrt((err * err * W).sum() / tot)),
        "Linf": float(np.abs(err).max()),
    }