#!/usr/bin/env python3
"""R3 hex-permeability FD reference solver (prereg: /nfs/.../hex_ref/prereg.md v1.0).

MAC staggered finite-difference Stokes, periodic cell, body-force driven,
circular cylinders u=0. Discrete structure mathematically mirrors the
controller W4-A lineage (fd_stokes_ctrl.py): solid faces -> Dirichlet rows
(u=0), p of all-faces-solid cells pinned, one guaranteed-fluid cell's
continuity row replaced by the pressure gauge, continuity rows kept for
all other cells (incl. solid-centre ones). Fully vectorised assembly.

Conventions (W9-C locked): U_s superficial = whole-cell mean of u_x
(solid contributes 0); f = G L^2/(nu U_s) with G=nu=W=1 -> f = 1/U_s;
K_s/R^2 = pi/(vf f); R^2 = vf*(cell area)/(n_cyl*pi).
"""

from __future__ import annotations

import json
import math
import resource
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LATTICES = {
    # name: (H/W aspect of the periodic box, [(cx,cy) cylinder offsets], basis a1, a2)
    "sq": (1.0, [(0.5, 0.5)], (1.0, 0.0), (0.0, 1.0)),
    "hexA": (97.0 / 56.0, [(0.0, 0.0), (0.5, 97.0 / 56.0 / 2)], (1.0, 0.0), (0.5, 97.0 / 56.0 / 2)),
    "hexB": (26.0 / 15.0, [(0.0, 0.0), (0.5, 26.0 / 15.0 / 2)], (1.0, 0.0), (0.5, 26.0 / 15.0 / 2)),
}


def dist2_to_cylinders(X, Y, cylinders, a1, a2):
    """Squared distance to the nearest periodic image of any cylinder."""
    B = np.array([[a1[0], a2[0]], [a1[1], a2[1]]])
    Binv = np.linalg.inv(B)
    best = np.full(X.shape, np.inf)
    for cx, cy in cylinders:
        dx0 = X - cx
        dy0 = Y - cy
        fx = Binv[0, 0] * dx0 + Binv[0, 1] * dy0
        fy = Binv[1, 0] * dx0 + Binv[1, 1] * dy0
        fx -= np.round(fx)
        fy -= np.round(fy)
        for m in (-1, 0, 1):
            for n in (-1, 0, 1):
                dx = B[0, 0] * (fx + m) + B[0, 1] * (fy + n)
                dy = B[1, 0] * (fx + m) + B[1, 1] * (fy + n)
                d2 = dx * dx + dy * dy
                np.minimum(best, d2, out=best)
    return best


def lap1d(n, h):
    """Periodic 1D second difference / h^2: (u[i+1]-2u[i]+u[i-1])/h^2."""
    i = np.arange(n)
    rows = np.concatenate([i, (i + 1) % n, (i - 1) % n])
    cols = np.concatenate([i, i, i])
    vals = np.concatenate([np.full(n, -2.0), np.full(n, 1.0), np.full(n, 1.0)])
    return (sp.coo_matrix((vals, (rows, cols)), shape=(n, n)) / h**2).tocsr()


def solve_case(lattice: str, vf: float, nx: int, ny: int, permc: str = "MMD_AT_PLUS_A") -> dict:
    t0 = time.time()
    aspect, cylinders, a1, a2 = LATTICES[lattice]
    W = 1.0
    H = aspect * W
    h = W / nx
    assert abs(ny * h - H) < 1e-12 * H, "grid must tile the box exactly"
    n_cyl = len(cylinders)
    R2 = vf * W * H / (n_cyl * math.pi)
    R = math.sqrt(R2)

    ii = np.arange(nx)
    jj = np.arange(ny)
    # u-faces at (i h, (j+1/2) h); v-faces at ((i+1/2) h, j h); cells at centres.
    Xu, Yu = np.meshgrid(ii * h, (jj + 0.5) * h, indexing="ij")
    Xv, Yv = np.meshgrid((ii + 0.5) * h, jj * h, indexing="ij")
    Xc, Yc = np.meshgrid((ii + 0.5) * h, (jj + 0.5) * h, indexing="ij")
    d2u = dist2_to_cylinders(Xu, Yu, cylinders, a1, a2)
    d2v = dist2_to_cylinders(Xv, Yv, cylinders, a1, a2)
    d2c = dist2_to_cylinders(Xc, Yc, cylinders, a1, a2)
    solid_u = d2u <= R2
    solid_v = d2v <= R2
    solid_c = d2c <= R2

    nu = nx * ny
    idx_u = np.arange(nu)
    idx_v = nu + np.arange(nu)
    idx_p = 2 * nu + np.arange(nu)
    N = 3 * nu

    # Laplacians (index layout i*ny + j -> x-major kron).
    Lx = sp.kron(lap1d(nx, h), sp.eye(ny), format="csr")
    Ly = sp.kron(sp.eye(nx), lap1d(ny, h), format="csr")
    A = Lx + Ly

    # pressure-gradient blocks: Gx u-row = (p[i,j]-p[i-1,j])/h etc. (local nu x nu)
    II, J = np.meshgrid(ii, jj, indexing="ij")  # II=x-index, J=y-index
    flat = (II * ny + J).ravel()
    rng = np.arange(nu)
    rows_gx = np.concatenate([rng, rng])
    cols_gx = np.concatenate([flat, (((II - 1) % nx) * ny + J).ravel()])
    vals_gx = np.concatenate([np.full(nu, 1.0 / h), np.full(nu, -1.0 / h)])
    Gx = sp.coo_matrix((vals_gx, (rows_gx, cols_gx)), shape=(nu, nu)).tocsr()
    rows_gy = np.concatenate([rng, rng])
    cols_gy = np.concatenate([flat, (II * ny + ((J - 1) % ny)).ravel()])
    vals_gy = np.concatenate([np.full(nu, 1.0 / h), np.full(nu, -1.0 / h)])
    Gy = sp.coo_matrix((vals_gy, (rows_gy, cols_gy)), shape=(nu, nu)).tocsr()

    # 3x3 block system: [A 0 -Gx; 0 A -Gy; Gx^T Gy^T 0]
    K = sp.bmat([[A, None, -Gx], [None, A, -Gy], [Gx.T, Gy.T, None]], format="csr")

    # masks / Dirichlet elimination
    keep = np.ones(N, dtype=bool)
    keep[idx_u[solid_u.ravel()]] = False
    keep[idx_v[solid_v.ravel()]] = False
    # all-faces-solid cells: pin p
    afs = solid_u[(II + 1) % nx, J] & solid_u[II, J] & solid_v[II, (J + 1) % ny] & solid_v[II, J]
    keep[idx_p[afs.ravel()]] = False
    # gauge: replace continuity row of the cell farthest from cylinders
    gauge = int(np.argmax(d2c.ravel()))
    keep[idx_p[gauge]] = False

    Dkeep = sp.diags(keep.astype(np.float64))
    K = (Dkeep @ K @ Dkeep + sp.diags((~keep).astype(np.float64))).tocsr()
    K.eliminate_zeros()

    rhs = np.zeros(N)
    rhs[idx_u[~solid_u.ravel()]] = -1.0  # body force G=1 on fluid u-faces

    t1 = time.time()
    lu = spla.splu(K.tocsc(), permc_spec=permc)
    t2 = time.time()
    sol = lu.solve(rhs)
    t3 = time.time()

    res = float(np.abs(K @ sol - rhs).max())
    u = sol[idx_u].reshape(nx, ny)
    v = sol[idx_v].reshape(nx, ny)
    uc = 0.5 * (u + np.roll(u, -1, axis=0))
    U_s = float(uc.mean())
    u_f = float(uc[~solid_c].mean())
    phi_a = float(solid_c.mean())
    f_press = 1.0 / U_s
    K_s_over_R2 = U_s * n_cyl * math.pi / (vf * (W * H))  # = pi/(vf f) for sq
    # discrete divergence on kept continuity rows
    div = ((np.roll(u, -1, axis=0) - u) + (np.roll(v, -1, axis=1) - v)) / h
    div_free = div.ravel()[~afs.ravel()]
    div_max = float(np.abs(div_free).max()) if div_free.size else 0.0

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    return dict(
        lattice=lattice,
        aspect=aspect,
        vf=vf,
        nx=nx,
        ny=ny,
        h=h,
        R2=R2,
        R=R,
        n_cyl=n_cyl,
        cylinders=[list(c) for c in cylinders],
        gauge=gauge,
        phi_actual=phi_a,
        U_s=U_s,
        u_f=u_f,
        f_press=f_press,
        K_s_over_R2=K_s_over_R2,
        K_i_over_r2=K_s_over_R2 / (1.0 - vf),
        res_inf=res,
        div_max=div_max,
        nnz=int(K.nnz),
        maxrss_gb=round(maxrss, 3),
        t_assemble=round(t1 - t0, 1),
        t_factor=round(t2 - t1, 1),
        t_solve=round(t3 - t2, 1),
        permc=permc,
    )


if __name__ == "__main__":
    import sys

    lat, vf, nx, ny = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    out = solve_case(lat, vf, nx, ny)
    print(json.dumps(out, indent=1))
