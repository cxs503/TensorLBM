"""Taylor-Hood Q2-Q1 finite-element reference solver (structured quads, Newton).

Element pair
------------
Biquadratic velocity (9 nodes) / bilinear pressure (4 nodes) on a uniform
structured quad mesh -- the classic LBB-stable Taylor-Hood pair.  For smooth
solutions the velocity error is ``O(h^3)`` in the discrete ``L2`` norm
(``O(h^2)`` for the pressure), which is what makes the FEM leg of the
cross-solver benchmark *structurally different* from LBM/FVM (both second order).

Implementation notes
--------------------
* uniform elements => the linear local matrices are built once and reused;
  the convective Jacobian is evaluated per element and vectorised with einsum;
* two mesh topologies are supported: ``bc="walls"`` (Dirichlet velocity, used by
  the lid-driven cavity) and ``bc="periodic"`` (used by the MMS study, where the
  opposite faces are identified by modular node numbering);
* the discrete system is solved with ``scipy.sparse.linalg.splu``; the pressure
  gauge is pinned on one node;
* evaluation at cell centres is *exact*: for Q2 the cell centre is the 9th
  element node, so the cell-centre velocity is a nodal value (no interpolation
  error is introduced when comparing with the cell-centred LBM/FVM fields).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["FEMResult", "solve_taylor_hood"]

_G1D = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GW1D = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


def _l1(t, k):
    if k == 0:
        return t * (t - 1.0) / 2.0
    if k == 1:
        return 1.0 - t * t
    return t * (t + 1.0) / 2.0


def _dl1(t, k):
    if k == 0:
        return t - 0.5
    if k == 1:
        return -2.0 * t
    return t + 0.5


def _basis_tables(h):
    """N, dN/dx, dN/dy (9 gauss x 9 nodes) and the 2-D quadrature weights."""
    gx, gw = _G1D, _GW1D
    xi = np.repeat(gx, 3)
    eta = np.tile(gx, 3)
    wgt = np.outer(gw, gw).ravel()
    n = np.zeros((9, 9))
    dx = np.zeros((9, 9))
    dy = np.zeros((9, 9))
    for a in range(9):
        p, q = a // 3, a % 3
        n[:, a] = _l1(xi, q) * _l1(eta, p)
        dx[:, a] = _dl1(xi, q) * _l1(eta, p) * (2.0 / h)
        dy[:, a] = _l1(xi, q) * _dl1(eta, p) * (2.0 / h)
    # bilinear pressure basis, local node b = 2*r + s
    pn = np.zeros((9, 4))
    dpx = np.zeros((9, 4))
    dpy = np.zeros((9, 4))
    for b in range(4):
        r, s = b // 2, b % 2
        ls = 0.5 * (1.0 + (2 * s - 1) * xi)
        lr = 0.5 * (1.0 + (2 * r - 1) * eta)
        pn[:, b] = ls * lr
        dpx[:, b] = 0.5 * (2 * s - 1) * lr * (2.0 / h)
        dpy[:, b] = ls * 0.5 * (2 * r - 1) * (2.0 / h)
    jw = wgt * (h * h / 4.0)
    return n, dx, dy, pn, dpx, dpy, jw


def _cell_to_nodes(field2d, n):
    """Bilinear interpolation of a cell-centred field onto the Q2 node grid.

    Used for grid sequencing (seed a fine solve with the coarse solution), which
    cuts the number of Newton iterations from ~15 to ~4.
    """
    nc = field2d.shape[0]
    xc = (np.arange(nc) + 0.5) / nc
    xn = np.arange(2 * n + 1) / (2.0 * n)
    tmp = np.empty((nc, 2 * n + 1))
    for j in range(nc):
        tmp[j, :] = np.interp(xn, xc, field2d[j, :])
    out = np.empty((2 * n + 1, 2 * n + 1))
    for i in range(2 * n + 1):
        out[i, :] = np.interp(xn, xc, tmp[:, i])
    return out


def _basis_ho(h, ng=6):
    """High-order (``ng x ng`` Gauss) table used *only* for the source term.

    The manufactured source is trigonometric, so the 3x3 rule that is exact for
    the polynomial (Q2) stiffness/mass integrands is **not** enough: it leaves a
    quadrature defect of order ``(h k)^6`` in the load vector, which sets a
    residual floor around 1e-10 and destroys the measured convergence order.
    A 6x6 rule removes it to well below round-off.
    """
    gx, gw = np.polynomial.legendre.leggauss(ng)
    xi = np.repeat(gx, ng)
    eta = np.tile(gx, ng)
    wgt = np.outer(gw, gw).ravel()
    n = np.zeros((ng * ng, 9))
    for a in range(9):
        p, q = a // 3, a % 3
        n[:, a] = _l1(xi, q) * _l1(eta, p)
    return xi, eta, wgt * (h * h / 4.0), n


@dataclass
class FEMResult:
    u: np.ndarray          # cell-centre velocities, shape (n, n) (axis 0 = y)
    v: np.ndarray
    pressure: np.ndarray   # (n, n) cell averages
    n: int
    newton_iterations: int
    converged: bool
    residual: float
    dof_count: int
    history: list[dict] = field(default_factory=list)

    @property
    def speed(self):
        return np.hypot(self.u, self.v)


def solve_taylor_hood(
    n: int,
    nu: float,
    bc: str = "walls",
    u_lid: float = 0.0,
    mms=None,
    max_newton: int = 25,
    tol: float = 1e-11,
    seed_velocity=None,
) -> FEMResult:
    """Solve steady incompressible Navier-Stokes with Q2-Q1 elements.

    Parameters
    ----------
    bc:
        ``"walls"`` -- Dirichlet velocity (cavity: lid = ``u_lid``, others 0).
        ``"periodic"`` -- all faces identified; ``mms`` supplies the force.
    """
    h = 1.0 / n
    n_tab, dx_tab, dy_tab, pn_tab, dpx_tab, dpy_tab, jw = _basis_tables(h)
    periodic = bc == "periodic"

    nvx = 2 * n if periodic else 2 * n + 1
    npx = n if periodic else n + 1
    n_period_v = 2 * n if periodic else 2 * n + 1

    def vidx(rows, cols):
        if periodic:
            return (rows % n_period_v) * n_period_v + (cols % n_period_v)
        return rows * nvx + cols

    def pidx(rows, cols):
        if periodic:
            return (rows % n) * n + (cols % n)
        return rows * npx + cols

    ex = np.repeat(np.arange(n), n)
    ey = np.tile(np.arange(n), n)
    ne = ex.size

    vdofs = np.empty((ne, 9), dtype=np.int64)
    for a in range(9):
        p, q = a // 3, a % 3
        vdofs[:, a] = vidx(2 * ey + p, 2 * ex + q)
    pdofs = np.empty((ne, 4), dtype=np.int64)
    for b in range(4):
        r, s = b // 2, b % 2
        pdofs[:, b] = pidx(ey + r, ex + s)

    nq = (n_period_v) ** 2 if periodic else nvx * nvx
    npn = n * n if periodic else npx * npx
    nu_dof = 2 * nq
    ndof = nu_dof + npn

    # ---------------------------------------------------------------- constants
    positive = jw[:, None] * n_tab  # (g, a) weighted integrand, w|J|N_a
    kv = np.einsum("g,ga,gb->ab", jw, dx_tab, dx_tab) + np.einsum("g,ga,gb->ab", jw, dy_tab, dy_tab)
    bx = np.einsum("g,gb,ga->ba", jw, pn_tab, dx_tab)  # (4, 9): int q d v /dx
    by = np.einsum("g,gb,ga->ba", jw, pn_tab, dy_tab)

    # forcing at the quadrature points (physical coordinates)
    xc = (ex + 0.5) * h
    yc = (ey + 0.5) * h
    xi = np.repeat(_G1D, 3)
    eta = np.tile(_G1D, 3)
    gx = xc[:, None] + 0.5 * h * xi[None, :]
    gy = yc[:, None] + 0.5 * h * eta[None, :]
    xi_ho, eta_ho, jw_ho, n_ho = _basis_ho(h, 6)
    gx_ho = xc[:, None] + 0.5 * h * xi_ho[None, :]
    gy_ho = yc[:, None] + 0.5 * h * eta_ho[None, :]
    if mms is not None:
        fx_ho, fy_ho = mms.source(gx_ho, gy_ho)
    else:
        fx_ho = np.zeros_like(gx_ho)
        fy_ho = np.zeros_like(gx_ho)
    f_loc_x = np.einsum("g,ga,eg->ea", jw_ho, n_ho, fx_ho)
    f_loc_y = np.einsum("g,ga,eg->ea", jw_ho, n_ho, fy_ho)

    # ------------------------------------------------------------- BC / dof map
    if periodic:
        free = np.ones(ndof, dtype=bool)
        pin_p = 0
        free[nu_dof + pin_p] = False
        gvals = np.zeros(nu_dof)
    else:
        vrows = np.arange(nvx)
        wall_nodes = set()
        for j in range(nvx):
            wall_nodes.add(int(j))                     # y = 0
            wall_nodes.add(int((nvx - 1) * nvx + j))   # y = 1
        for i in range(nvx):
            wall_nodes.add(int(i * nvx))               # x = 0
            wall_nodes.add(int(i * nvx + nvx - 1))     # x = 1
        wall_nodes = np.array(sorted(wall_nodes), dtype=np.int64)
        free = np.ones(ndof, dtype=bool)
        gvals = np.zeros(nu_dof)
        free[wall_nodes] = False
        free[nq + wall_nodes] = False
        free[nu_dof] = False  # pin pressure gauge
        # lid velocity on the top row
        top = np.array([int((nvx - 1) * nvx + j) for j in range(nvx)], dtype=np.int64)
        gvals[top] = u_lid

    u_vec = np.zeros(nu_dof)
    if seed_velocity is not None:
        su = _cell_to_nodes(seed_velocity[0], n)
        sv = _cell_to_nodes(seed_velocity[1], n)
        for a in range(9):
            pp, qq = a // 3, a % 3
            ids = vidx(2 * ey + pp, 2 * ex + qq)
            u_vec[ids] = su[2 * ey + pp, 2 * ex + qq]
            u_vec[nq + ids] = sv[2 * ey + pp, 2 * ex + qq]
    if not periodic:
        u_vec[wall_nodes] = gvals[wall_nodes]
        u_vec[nq + wall_nodes] = 0.0
    p_vec = np.zeros(npn)

    # ------------------------------------------------------------------- Newton
    def _block(row_ids, col_ids, vals):
        """Scatter element blocks: ``row_ids`` (ne,A), ``col_ids`` (ne,B), vals (ne,A,B)."""
        ne_, a_ = row_ids.shape
        b_ = col_ids.shape[1]
        r = np.repeat(row_ids, b_, axis=1)
        c = np.tile(col_ids, (1, a_))
        return r.ravel(), c.ravel(), np.asarray(vals).reshape(ne_, a_ * b_).ravel()

    def assemble(u_vec, p_vec, want_jac=True):
        ue_loc = u_vec[vdofs]          # (ne, 9)
        ve_loc = u_vec[nq + vdofs]
        pe_loc = p_vec[pdofs]
        u_g = np.einsum("ga,ea->eg", n_tab, ue_loc)
        v_g = np.einsum("ga,ea->eg", n_tab, ve_loc)
        ux = np.einsum("ga,ea->eg", dx_tab, ue_loc)
        uy = np.einsum("ga,ea->eg", dy_tab, ue_loc)
        vx = np.einsum("ga,ea->eg", dx_tab, ve_loc)
        vy = np.einsum("ga,ea->eg", dy_tab, ve_loc)

        ru = np.einsum("g,ga,eg->ea", jw, n_tab, u_g * ux + v_g * uy) - f_loc_x
        rv = np.einsum("g,ga,eg->ea", jw, n_tab, u_g * vx + v_g * vy) - f_loc_y
        ru = ru + nu * (ue_loc @ kv.T)
        rv = rv + nu * (ve_loc @ kv.T)
        ru = ru - pe_loc @ bx
        rv = rv - pe_loc @ by
        rp = ue_loc @ bx.T + ve_loc @ by.T

        r = np.zeros(ndof)
        np.add.at(r, vdofs.ravel(), ru.ravel())
        np.add.at(r, (nq + vdofs).ravel(), rv.ravel())
        np.add.at(r, (nu_dof + pdofs).ravel(), rp.ravel())

        if not want_jac:
            return r, None
        # ---- Jacobian blocks (convective linearisation + constants)
        a11 = np.einsum("g,ag,gb,eg->eab", jw, n_tab, n_tab, ux)
        a11 += np.einsum("g,ag,gb,eg->eab", jw, n_tab, dx_tab, u_g)
        a11 += np.einsum("g,ag,gb,eg->eab", jw, n_tab, dy_tab, v_g)
        a12 = np.einsum("g,ag,gb,eg->eab", jw, n_tab, n_tab, uy)
        a21 = np.einsum("g,ag,gb,eg->eab", jw, n_tab, n_tab, vx)
        a22 = np.einsum("g,ag,gb,eg->eab", jw, n_tab, n_tab, vy)
        a22 += np.einsum("g,ag,gb,eg->eab", jw, n_tab, dx_tab, u_g)
        a22 += np.einsum("g,ag,gb,eg->eab", jw, n_tab, dy_tab, v_g)
        a11 = a11 + nu * kv[None, :, :]
        a22 = a22 + nu * kv[None, :, :]

        parts = [
            _block(vdofs, vdofs, a11),
            _block(vdofs + nq, vdofs, a21),
            _block(vdofs, vdofs + nq, a12),
            _block(vdofs + nq, vdofs + nq, a22),
            # d rp / du = bx  (row: pressure dof, col: u dof)
            _block(pdofs + nu_dof, vdofs, np.broadcast_to(bx, (ne, 4, 9))),
            _block(pdofs + nu_dof, vdofs + nq, np.broadcast_to(by, (ne, 4, 9))),
            # d ru / dp = -bx^T  (row: u dof, col: pressure dof)
            _block(vdofs, pdofs + nu_dof, np.broadcast_to(-bx.T, (ne, 9, 4))),
            _block(vdofs + nq, pdofs + nu_dof, np.broadcast_to(-by.T, (ne, 9, 4))),
        ]
        rows = np.concatenate([p_[0] for p_ in parts])
        cols = np.concatenate([p_[1] for p_ in parts])
        data = np.concatenate([p_[2] for p_ in parts])
        jac = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
        return r, jac

    x = np.concatenate([u_vec, p_vec])
    import os as _os

    if _os.environ.get("FEM_DEBUG_JAC"):
        r0, j0 = assemble(x[:nu_dof], x[nu_dof:])
        rng = np.random.default_rng(0)
        d = rng.standard_normal(ndof)
        d[~free] = 0.0
        eps = 1e-7 * max(1.0, float(np.abs(x).max())) / max(1.0, float(np.abs(d).max()))
        xp = x + eps * d
        rp_, _ = assemble(xp[:nu_dof], xp[nu_dof:], want_jac=False)
        fd = (rp_ - r0) / eps
        jd = j0 @ d
        mask = free
        denom = np.abs(jd[mask]).max()
        print("[FEM_DEBUG_JAC] rel err", float(np.abs(fd[mask] - jd[mask]).max() / denom))
        print("[FEM_DEBUG_JAC] fd max", float(np.abs(fd[mask]).max()), "jd max", denom)
    history = []
    converged = False
    residual = float("inf")
    it = 0
    for it in range(1, max_newton + 1):
        r, jac = assemble(x[:nu_dof], x[nu_dof:])
        r = r.copy()
        r[~free] = 0.0
        residual = float(np.abs(r[free]).max())
        history.append({"newton": it, "residual": residual})
        if residual < tol:
            converged = True
            break
        import os as _os2
        if _os2.environ.get("FEM_DEBUG_JAC") and it == 3:
            rng = np.random.default_rng(1)
            d = rng.standard_normal(ndof)
            d[~free] = 0.0
            eps = 1e-7 * max(1.0, float(np.abs(x).max())) / max(1.0, float(np.abs(d).max()))
            xp = x + eps * d
            rp_, _ = assemble(xp[:nu_dof], xp[nu_dof:], want_jac=False)
            rp_[~free] = 0.0
            fd = (rp_ - r) / eps
            jd = jac @ d
            print("[FEM_DEBUG_JAC it3] rel err",
                  float(np.abs(fd[free] - jd[free]).max() / np.abs(jd[free]).max()),
                  "fd", float(np.abs(fd[free]).max()), "jd", float(np.abs(jd[free]).max()))
        jff = jac[free][:, free]
        rhs = -r[free]
        try:
            delta = spla.spsolve(jff.tocsc(), rhs)
        except Exception:  # pragma: no cover
            delta = spla.lsqr(jff, rhs, atol=1e-14, btol=1e-14, iter_lim=5000)[0]
        # backtracking line search on the free-dof residual norm (damped Newton)
        r0 = float(np.linalg.norm(rhs))
        step = 1.0
        accepted = False
        for _ in range(4):
            trial = x.copy()
            trial[free] += step * delta
            if not np.all(np.isfinite(trial)):
                step *= 0.5
                continue
            rt, _ = assemble(trial[:nu_dof], trial[nu_dof:], want_jac=False)
            rt[~free] = 0.0
            rn = float(np.linalg.norm(rt[free]))
            if rn < r0 or rn < tol:
                x = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            x[free] += delta
            if not np.all(np.isfinite(x)):
                raise RuntimeError("Newton diverged (non-finite update)")

    # ------------------------------------------------------------- post-process
    u_vec, p_vec = x[:nu_dof].copy(), x[nu_dof:].copy()
    if not periodic:
        u_vec[wall_nodes] = gvals[wall_nodes]
    cid = vidx(2 * ey + 1, 2 * ex + 1)
    u_c = np.zeros(n * n)
    v_c = np.zeros(n * n)
    u_c[ey * n + ex] = u_vec[cid]
    v_c[ey * n + ex] = u_vec[nq + cid]
    u_c = u_c.reshape(n, n)
    v_c = v_c.reshape(n, n)
    p_c = np.zeros(n * n)
    pc = pidx(ey, ex)
    for b in range(4):
        r_, s_ = b // 2, b % 2
        p_c[ey * n + ex] += 0.25 * p_vec[pidx(ey + r_, ex + s_)]
    return FEMResult(
        u=u_c, v=v_c, pressure=p_c.reshape(n, n), n=n, newton_iterations=it,
        converged=converged, residual=residual, dof_count=ndof, history=history,
    )