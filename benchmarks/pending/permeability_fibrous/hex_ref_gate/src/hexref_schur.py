#!/usr/bin/env python3
"""R3 FD reference solver, Schur-complement CG variant (prereg fallback route).

Same discrete MAC system as hexref_fd.solve_case (identical unknowns/masks/
gauge), solved as: A_vel u = f - D p (SPD blocks, sparse LU once), then
S p = D^T A_vel^{-1} f, S = D^T A_vel^{-1} D via CG with an FFT periodic
Poisson preconditioner. A/B-validated against the direct saddle solve.
"""

from __future__ import annotations

import json
import math
import resource
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from hexref_fd import LATTICES, dist2_to_cylinders


def _lap_coo(n, h):
    i = np.arange(n)
    rows = np.concatenate([i, (i + 1) % n, (i - 1) % n])
    cols = np.concatenate([i, i, i])
    vals = np.concatenate([np.full(n, -2.0), np.full(n, 1.0), np.full(n, 1.0)]) / h**2
    return rows, cols, vals


def solve_case_schur(lattice: str, vf: float, nx: int, ny: int, tol: float = 1e-13) -> dict:
    t0 = time.time()
    aspect, cylinders, a1, a2 = LATTICES[lattice]
    W = 1.0
    H = aspect * W
    h = W / nx
    assert abs(ny * h - H) < 1e-12 * H
    n_cyl = len(cylinders)
    R2 = vf * W * H / (n_cyl * math.pi)
    R = math.sqrt(R2)

    ii = np.arange(nx)
    jj = np.arange(ny)
    Xu, Yu = np.meshgrid(ii * h, (jj + 0.5) * h, indexing="ij")
    Xv, Yv = np.meshgrid((ii + 0.5) * h, jj * h, indexing="ij")
    Xc, Yc = np.meshgrid((ii + 0.5) * h, (jj + 0.5) * h, indexing="ij")
    d2u = dist2_to_cylinders(Xu, Yu, cylinders, a1, a2)
    d2v = dist2_to_cylinders(Xv, Yv, cylinders, a1, a2)
    d2c = dist2_to_cylinders(Xc, Yc, cylinders, a1, a2)
    solid_u = (d2u <= R2).ravel()
    solid_v = (d2v <= R2).ravel()
    solid_c = d2c <= R2

    II, J = np.meshgrid(ii, jj, indexing="ij")
    nu = nx * ny
    flat = (II * ny + J).ravel()

    afs = (
        solid_u.ravel()[((II + 1) % nx) * ny + J]
        & solid_u.ravel()[II * ny + J]
        & solid_v.ravel()[II * ny + ((J + 1) % ny)]
        & solid_v.ravel()[II * ny + J]
    )
    gauge = int(np.argmax(d2c.ravel()))
    pfree = ~(afs.ravel().copy())
    pfree[gauge] = False
    cmap = -np.ones(nu, dtype=np.int64)
    cmap[pfree] = np.arange(pfree.sum())
    np_ = int(pfree.sum())

    # L = periodic 5-point Laplacian entries, built by index arithmetic
    i_all = np.arange(nu)
    ix = i_all // ny
    iy = i_all % ny
    ent_r = []
    ent_c = []
    ent_v = []
    for shift, axis in ((1, "x"), (-1, "x"), (1, "y"), (-1, "y")):
        if axis == "x":
            nb = ((ix + shift) % nx) * ny + iy
        else:
            nb = ix * ny + ((iy + shift) % ny)
        ent_r.append(i_all)
        ent_c.append(nb)
        ent_v.append(np.full(nu, 1.0 / h**2))
    ent_r.append(i_all)
    ent_c.append(i_all)
    ent_v.append(np.full(nu, -4.0 / h**2))
    Lr = np.concatenate(ent_r)
    Lc = np.concatenate(ent_c)
    Lv = np.concatenate(ent_v)

    fluid_u = ~solid_u
    fluid_v = ~solid_v
    umap = -np.ones(nu, dtype=np.int64)
    umap[fluid_u] = np.arange(fluid_u.sum())
    vmap = -np.ones(nu, dtype=np.int64)
    vmap[fluid_v] = np.arange(fluid_v.sum())

    # A_u (SPD): -L on fluid_u x fluid_u
    keep = fluid_u[Lr] & fluid_u[Lc]
    Au = sp.coo_matrix(
        (-Lv[keep], (umap[Lr[keep]], umap[Lc[keep]])), shape=(fluid_u.sum(), fluid_u.sum())
    ).tocsr()
    keep = fluid_v[Lr] & fluid_v[Lc]
    Av = sp.coo_matrix(
        (-Lv[keep], (vmap[Lr[keep]], vmap[Lc[keep]])), shape=(fluid_v.sum(), fluid_v.sum())
    ).tocsr()

    # D operators from Gx/Gy patterns: u-face k -> cells (i,j) +1/h, (i-1,j) -1/h
    def grad_entries(flat_cells_plus, flat_cells_minus):
        rows = np.concatenate([np.arange(nu), np.arange(nu)])
        cols = np.concatenate([flat_cells_plus, flat_cells_minus])
        vals = np.concatenate([np.full(nu, 1.0 / h), np.full(nu, -1.0 / h)])
        return rows, cols, vals

    gr, gc, gv = grad_entries(flat, (((II - 1) % nx) * ny + J).ravel())
    keep = fluid_u[gr] & pfree[gc]
    Dx = sp.coo_matrix(
        (gv[keep], (umap[gr[keep]], cmap[gc[keep]])), shape=(fluid_u.sum(), np_)
    ).tocsr()
    gr2, gc2, gv2 = grad_entries(flat, (II * ny + ((J - 1) % ny)).ravel())
    keep = fluid_v[gr2] & pfree[gc2]
    Dy = sp.coo_matrix(
        (gv2[keep], (vmap[gr2[keep]], cmap[gc2[keep]])), shape=(fluid_v.sum(), np_)
    ).tocsr()

    t1 = time.time()
    lu_u = spla.splu(Au.tocsc(), permc_spec="MMD_AT_PLUS_A")
    lu_v = spla.splu(Av.tocsc(), permc_spec="MMD_AT_PLUS_A")
    t2 = time.time()

    fu_ones = np.ones(fluid_u.sum())
    w0 = lu_u.solve(fu_ones)
    b = Dx.T @ w0

    # FFT periodic Poisson preconditioner on the full cell grid
    kx = 2.0 * np.sin(np.pi * np.fft.fftfreq(nx))
    ky = 2.0 * np.sin(np.pi * np.fft.fftfreq(ny))
    lam = (kx[:, None] ** 2 + ky[None, :] ** 2) / h**2
    lam[0, 0] = lam[0, 1] if nx >= 2 and ny >= 2 else lam[1, 0]
    pfree_idx = np.flatnonzero(pfree)
    pf_i = pfree_idx // ny
    pf_j = pfree_idx % ny
    full_r = np.zeros((nx, ny))
    full_r[pf_i, pf_j] = 1.0  # embed indicator (kept for shape checks)

    def precond(z):
        g = np.zeros((nx, ny))
        g[pf_i, pf_j] = z
        G = np.fft.fft2(g) / lam
        s = np.real(np.fft.ifft2(G))
        return s[pf_i, pf_j]

    def s_apply(p):
        wu = lu_u.solve(Dx @ p)
        wv = lu_v.solve(Dy @ p)
        return Dx.T @ wu + Dy.T @ wv

    S = spla.LinearOperator((np_, np_), matvec=s_apply, dtype=np.float64)
    M = spla.LinearOperator((np_, np_), matvec=precond, dtype=np.float64)
    t3 = time.time()
    try:
        p, info = spla.cg(S, b, M=M, rtol=tol, maxiter=10000)
    except TypeError:
        p, info = spla.cg(S, b, M=M, tol=tol, maxiter=10000)
    t4 = time.time()
    cg_res = float(np.abs(S @ p - b).max() / max(np.abs(b).max(), 1e-300))

    u = np.zeros(nu)
    u[fluid_u] = w0 - lu_u.solve(Dx @ p)
    v = np.zeros(nu)
    v[fluid_v] = -lu_v.solve(Dy @ p)
    u2 = u.reshape(nx, ny)
    v2 = v.reshape(nx, ny)
    uc = 0.5 * (u2 + np.roll(u2, -1, axis=0))
    U_s = float(uc.mean())
    u_f = float(uc[~solid_c].mean())
    phi_a = float(solid_c.mean())
    f_press = 1.0 / U_s
    K_s_over_R2 = U_s * n_cyl * math.pi / (vf * (W * H))
    div = ((np.roll(u2, -1, axis=0) - u2) + (np.roll(v2, -1, axis=1) - v2)) / h
    div_free = div.ravel()[pfree]
    div_max = float(np.abs(div_free).max())
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
        gauge=gauge,
        n_afs=int(afs.sum()),
        phi_actual=phi_a,
        U_s=U_s,
        u_f=u_f,
        f_press=f_press,
        K_s_over_R2=K_s_over_R2,
        K_i_over_r2=K_s_over_R2 / (1.0 - vf),
        solver="schur_cg",
        cg_info=info,
        cg_res=cg_res,
        tol=tol,
        div_max=div_max,
        maxrss_gb=round(maxrss, 3),
        t_assemble=round(t1 - t0, 1),
        t_factor=round(t2 - t1, 1),
        t_cg=round(t4 - t3, 1),
    )


if __name__ == "__main__":
    lat, vf, nx, ny = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    out = solve_case_schur(lat, vf, nx, ny)
    print(json.dumps(out, indent=1))
