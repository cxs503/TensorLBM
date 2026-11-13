#!/usr/bin/env python3
"""XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF -- TensorLBM vs haimingz/vivsim.

Anchor
------
``XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF`` (owner FEAT-ANCHOR-FEEDER, round 1).

Decision
--------
Does TensorLBM's immersed-boundary *Peskin tensor-product stencil* + *multi-pass
direct forcing* (``tensorlbm.ibm`` / ``tensorlbm.ibm_force_transfer``) reproduce
the mature OSS implementation (``haimingz/vivsim`` ``vivsim/ib/{kernels,stencil,
mdf,geometry}.py`` @ fb84b12) to better than the owner's 15 % bar, and if not,
which sub-operator is responsible and does the deviation matter in a solver?

What the OSS implementation does (source-read, zero hand-copy)
--------------------------------------------------------------
``vivsim/ib/kernels.py:25``   ``kernel_peskin_4pt`` -- the classic Peskin (2002)
                            4-point delta, support ``|r| <= 2``, two branches.
``vivsim/ib/kernels.py:4``    ``kernel_peskin_3pt`` (2-point/3-point hat family)
``vivsim/ib/kernels.py:46``   ``kernel_cosine_4pt`` -- a *separate* 4-point cosine
                            delta that TensorLBM does not offer.
``vivsim/ib/stencil.py:7``    ``get_ib_stencil`` -- **tensor product**: 1-D kernel
                            evaluated on ``arange(-r+1, r+1)`` offsets around
                            ``floor(x)`` in each direction and multiplied; the
                            resulting ``(M, 4**d)`` weights + flat indices are
                            precomputed once and reused.
``vivsim/ib/stencil.py:54``   ``interpolate`` -- ``einsum('ms,cms->mc')``
``vivsim/ib/stencil.py:81``   ``spread``      -- ``.at[:, idx].add(...)`` scatter
``vivsim/ib/mdf.py:8``        ``multi_direct_forcing`` -- **MDF**: iterate
                            ``step = (U_t - u_m) * (2 ds)``, spread the step,
                            update ``u_m += 0.5 * interp(spread(step))``;
                            converged force = ``2 A^{-1} r0`` where
                            ``A = interp . spread`` (the *round-trip* operator).
``vivsim/ib/geometry.py:21``  ``get_ds`` -- marker spacing weights (preconditioner)

TensorLBM's counterpart (source-read)
-------------------------------------
``ibm.py:83``                 ``ibm_delta_4pt`` -- same Peskin 4-pt formulas
                              (its docstring calls it "Peskin 4-point (cosine)",
                              which is a mislabel: it is the Peskin kernel, and
                              the cosine kernel is a different function).
``ibm_force_transfer.py:152``  ``build_stencil`` -- tensor-product stencil, same
                              offsets ``floor(p) - support/2 + 1 + arange(support)``.
``ibm_force_transfer.py:326``  ``direct_forcing`` -- Guo-corrected *pointwise*
                              multi-pass: ``u_phys = u_int + 0.5 f/rho``,
                              ``f += U_t - u_phys``.  Its fixed point is
                              ``f = 2 rho (U_t - u_int)``: it implicitly assumes
                              ``A = I``, i.e. it does **not** solve the round-trip
                              interface condition that MDF solves.

Levels
------
L1 kernel/stencil identity (exactness, 2-D + 3-D).
L2 MDF operator: convergence rate, converged force, and the *interface residual*
   ``U_t - (u_int + 0.5 A F)`` achieved by each arm; sweep over marker spacing h.
L3 porting: ``direct_forcing_mdf`` added to TensorLBM (round-trip consistent).
L4 end-to-end: cylinder at Re = 20 / 40 in one LBM, arms = {TL baseline,
   TL+MDF, OSS MDF}; Cd vs the DOI-anchored table already in the repo
   (``src/tensorlbm/ibm_openwater.py:130``, Dennis & Chang 1970).

Usage
-----
    PYTHONPATH=src python xval/XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF/run_xval.py \
        --out xval/XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF/artifacts/xval.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys

sys.path.insert(0, str(HERE))

import oss_ref  # noqa: E402  (vendored OSS, numpy-shimmed)
from oss_ref import np_jax_shim as shim  # noqa: E402

from tensorlbm.boundaries import cylinder_mask, far_field_bc_2d  # noqa: E402
from tensorlbm.d2q9 import C as C2D  # noqa: E402
from tensorlbm.d2q9 import W as W2D  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.ibm import ibm_delta_4pt, ibm_delta_hat  # noqa: E402
from tensorlbm.ibm_force_transfer import (  # noqa: E402
    build_cylinder_markers_2d,
    build_stencil,
    direct_forcing,
    direct_forcing_mdf,
    interior_mask_force,
)
from tensorlbm.solver import collide_mrt, stream  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def markers_circle(cx: float, cy: float, radius: float, h: float) -> np.ndarray:
    """Circle markers at arc spacing ~h, ordered (cos, sin) -- OSS convention."""
    n = max(8, int(round(2.0 * math.pi * radius / h)))
    t = np.arange(n) * (2.0 * math.pi / n)
    return np.stack([cx + radius * np.cos(t), cy + radius * np.sin(t)], axis=1)


def rel(a, b) -> float:
    a = float(a)
    b = float(b)
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / scale


def oss_operators(pts: np.ndarray, ny: int, nx: int, ux, uy, n_iter: int, ds_override=None):
    """Run the vendored OSS stencil + MDF; returns grid force (2,nx,ny) and marker force."""
    mx, my = pts[:, 0], pts[:, 1]
    w, idx = oss_ref.get_ib_stencil(shim.asarray(mx), shim.asarray(my), ny)
    grid_u = shim.asarray(np.stack([ux.T, uy.T]))  # OSS layout (component, nx, ny)
    u_target = shim.asarray(np.zeros((pts.shape[0], 2)))
    ds = np.asarray(oss_ref.get_ds(shim.asarray(pts))) if ds_override is None else np.full(len(pts), ds_override)
    grid_force, marker_reaction = oss_ref.multi_direct_forcing(
        grid_u, w, idx, u_target, shim.asarray(ds), n_iter=n_iter
    )
    return (
        np.asarray(grid_force),  # (2, nx, ny)
        -np.asarray(marker_reaction),  # (M, 2) marker force
        np.asarray(w),
        np.asarray(idx),
        np.asarray(ds),
    )


def tl_operators(pts, ux, uy, n_correctors: int, kernel="4pt", scheme="pointwise"):
    ny, nx = ux.shape
    st = build_stencil(torch.tensor(pts, dtype=torch.float64), (ny, nx), kernel)
    u_grid = torch.tensor(np.stack([uy, ux]))  # TensorLBM layout (component, ny, nx)
    u_target = torch.zeros((pts.shape[0], 2), dtype=torch.float64)
    if scheme == "pointwise":
        f_m, f_grid, body_force, _ = direct_forcing(st, u_grid, u_target, None, n_correctors=n_correctors)
    else:
        f_m, f_grid, body_force, _ = direct_forcing_mdf(st, u_grid, u_target, None, n_mdf=n_correctors)
    return f_m.numpy()[:, ::-1], f_grid.numpy(), st


def iface_residual_pointwise(pts, ux, uy, f_marker, kernel="4pt"):
    """Residual ``U_t - (u_int + 0.5 * interp(spread(F)))`` for a *pointwise* (Guo) arm."""
    st = build_stencil(torch.tensor(pts, dtype=torch.float64), ux.shape, kernel)
    ux_t = torch.tensor(ux)
    uy_t = torch.tensor(uy)
    u_int = st.interpolate(torch.stack([uy_t, ux_t])).T.numpy()  # (M, 2) in (x, y)
    fm_xy = np.ascontiguousarray(np.asarray(f_marker, dtype=float))
    f_grid = st.spread(torch.tensor(fm_xy[:, ::-1].copy())).numpy()  # (2, ny, nx) in (y, x)
    u_after = u_int + 0.5 * st.interpolate(torch.tensor(f_grid)).T.numpy()
    return float(np.abs(u_after).max()), u_after


# --------------------------------------------------------------------------- #
# L1 -- kernel + stencil identity
# --------------------------------------------------------------------------- #


def level1_kernel(*_a):
    r = np.linspace(-3.0, 3.0, 4001)
    out = {
        "upstream": {
            "peskin_4pt@r=0": float(oss_ref.kernel_peskin_4pt(shim.asarray(0.0))),
            "peskin_4pt@r=1": float(oss_ref.kernel_peskin_4pt(shim.asarray(1.0))),
        },
        "tensorlbm": {
            "ibm_delta_4pt@r=0": float(ibm_delta_4pt(torch.tensor(0.0))),
            "ibm_delta_4pt@r=1": float(ibm_delta_4pt(torch.tensor(1.0))),
        },
    }
    oss4 = np.asarray(oss_ref.kernel_peskin_4pt(shim.asarray(r)))
    tl4 = ibm_delta_4pt(torch.tensor(r)).numpy()
    tlc = ibm_delta_hat(torch.tensor(r)).numpy()
    oss_cos = np.asarray(oss_ref.kernel_cosine_4pt(shim.asarray(r)))
    oss3 = np.asarray(oss_ref.kernel_peskin_3pt(shim.asarray(r)))
    out["kernel_maxabs_diff_peskin4pt"] = float(np.abs(oss4 - tl4).max())
    out["oss_has_cosine_4pt_not_in_tensorlbm"] = {
        "phi(0)": float(np.asarray(oss_ref.kernel_cosine_4pt(shim.asarray(0.0)))),
        "sum_over_integers": float(sum(float(np.asarray(oss_ref.kernel_cosine_4pt(shim.asarray(float(k))))) for k in range(-3, 4))),
    }
    out["oss_peskin_3pt_sum_over_integers"] = float(sum(float(np.asarray(oss_ref.kernel_peskin_3pt(shim.asarray(float(k))))) for k in range(-3, 4)))
    out["oss_peskin_4pt_sum_over_integers"] = float(sum(float(np.asarray(oss_ref.kernel_peskin_4pt(shim.asarray(float(k))))) for k in range(-3, 4)))
    out["tensorlbm_hat_sum_over_integers"] = float(sum(float(ibm_delta_hat(torch.tensor(float(k)))) for k in range(-3, 4)))
    out["profiles"] = {"r": r.tolist(), "oss_peskin_4pt": oss4.tolist(), "tl_4pt": tl4.tolist(),
                       "tl_hat": tlc.tolist(), "oss_cosine_4pt": oss_cos.tolist(),
                       "oss_peskin_3pt": oss3.tolist()}
    return out


def _stencil_compare(pts, shape, kernel, oss_build, oss_shape=None):
    """Compare TensorLBM (idx, w) with the OSS (idx, w) after axis mapping."""
    # TensorLBM: shape is (ny, nx, ...) with physical x on the LAST axis
    tl_st = build_stencil(torch.tensor(pts, dtype=torch.float64), shape, kernel)
    tl_idx = tl_st.idx.numpy().reshape(tl_st.n_markers, -1)
    tl_w = tl_st.w.numpy().reshape(tl_st.n_markers, -1)
    oss_w, oss_idx = oss_build(pts)
    oss_w = np.asarray(oss_w)
    oss_idx = np.asarray(oss_idx)
    # decode both to physical multi-indices, then key by (x, y[, z])
    d = pts.shape[1]
    strides = [1] * d
    for ax in range(d - 2, -1, -1):
        strides[ax] = strides[ax + 1] * shape[ax + 1]
    # TensorLBM store order: dim 0 of `shape` is axis for pts[:, d-1]
    tlb = {}
    for m in range(tl_st.n_markers):
        acc = {}
        for k in range(tl_idx.shape[1]):
            lin = int(tl_idx[m, k])
            rem = lin
            node = []
            for ax in range(d):
                node.append(rem // strides[ax])
                rem -= node[-1] * strides[ax]
            key = tuple(reversed(node))  # physical (x, y[, z])
            acc[key] = acc.get(key, 0.0) + float(tl_w[m, k])
        tlb[m] = acc
    ossb = {}
    n_stencil = oss_idx.shape[1]
    oss_shape = tuple(oss_shape) if oss_shape is not None else tuple(shape)
    oss_strides = [1] * d
    for ax in range(d - 2, -1, -1):
        oss_strides[ax] = oss_strides[ax + 1] * oss_shape[ax + 1]
    for m in range(oss_idx.shape[0]):
        acc = {}
        for k in range(n_stencil):
            lin = int(oss_idx[m, k])
            rem = lin
            node = []
            for ax in range(d):
                node.append(rem // oss_strides[ax])
                rem -= node[-1] * oss_strides[ax]
            acc[tuple(node)] = acc.get(tuple(node), 0.0) + float(oss_w[m, k])
        ossb[m] = acc
    maxdiff = 0.0
    keymatch = True
    for m in tlb:
        keys = set(k for k in set(tlb[m]) | set(ossb[m]) if abs(tlb[m].get(k, 0.0)) + abs(ossb[m].get(k, 0.0)) > 1e-15)
        if set(k for k in tlb[m] if abs(tlb[m][k]) > 1e-15) != set(k for k in ossb[m] if abs(ossb[m][k]) > 1e-15):
            keymatch = False
        for k in keys:
            maxdiff = max(maxdiff, abs(tlb[m].get(k, 0.0) - ossb[m].get(k, 0.0)))
    return {
        "n_markers": int(tl_st.n_markers),
        "tl_points_per_marker": int(tl_idx.shape[1]),
        "oss_points_per_marker": int(n_stencil),
        "max_abs_weight_diff": float(maxdiff),
        "nonzero_node_keys_identical": bool(keymatch),
        "partition_unity_err_tl": abs(tl_st.peak_normalization() - 1.0),
    }


def ny_of(shape, d, ax):
    return shape[ax]


def level1_stencil():
    out = {}
    # ---- 2-D, Peskin 4-pt
    pts = markers_circle(20.37, 19.13, 6.3, 0.9)
    ny, nx = 48, 40

    def oss2d(p):
        w, idx = oss_ref.get_ib_stencil(shim.asarray(p[:, 0]), shim.asarray(p[:, 1]), ny)
        return w, idx

    out["2d_peskin4pt"] = _stencil_compare(pts, (ny, nx), "4pt", oss2d, oss_shape=(nx, ny))
    # ---- 2-D, TensorLBM hat vs OSS 4-pt geometry (zeros outside support)
    def oss2d_hat(p):
        w, idx = oss_ref.get_ib_stencil(shim.asarray(p[:, 0]), shim.asarray(p[:, 1]), ny,
                                        kernel=oss_ref.kernel_cosine_4pt)
        return w, idx
    out["2d_cosine4pt_vs_tl4pt_MAXDIFF_expected_large"] = _stencil_compare(pts, (ny, nx), "4pt", oss2d_hat, oss_shape=(nx, ny))
    # ---- 3-D sphere
    sys.path.insert(0, str(HERE / "oss_ref"))
    import oss_ref.stencil3d as s3d  # noqa: E402

    n = 60
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    rxy = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    R = 5.0
    pts3 = np.stack([rxy * np.cos(phi), rxy * np.sin(phi), z], axis=1) * R + np.array([16.3, 15.7, 16.1])
    shp = (32, 32, 32)  # TensorLBM order (nz, ny, nx) with x last

    def oss3(p):
        w, idx = s3d.get_ib_stencil(shim.asarray(p), shp)  # upstream default Peskin 4-pt, radius 2
        return w, idx

    out["3d_peskin4pt"] = _stencil_compare(pts3, shp, "4pt", oss3, oss_shape=shp)
    # 1-D sanity: kernel sums
    out["oss_upstream_files"] = {
        "kernels.py": "kernels.py", "stencil.py": "stencil.py", "mdf.py": "mdf.py",
        "geometry.py": "geometry.py", "stencil3d.py": "stencil3d.py",
    }
    return out


# --------------------------------------------------------------------------- #
# L2 -- MDF operator: convergence, converged force, interface residual
# --------------------------------------------------------------------------- #


def _uniform_mdf_case(h: float, n_big: int = 40):
    ny, nx = 64, 64
    pts = markers_circle(32.13, 31.87, 8.0, h)
    ux = np.full((ny, nx), 0.05)
    uy = np.zeros((ny, nx))
    res = {"h": h, "n_markers": int(len(pts)), "oss_n_iter": [], "tl_n_corr": []}

    gf_big, fm_big, w, idx, ds = oss_operators(pts, ny, nx, ux, uy, n_big)
    for n in [1, 2, 3, 5, 10, 20, 40]:
        if n > n_big:
            continue
        gf, fm, *_ = oss_operators(pts, ny, nx, ux, uy, n)
        slip_abs, _ = iface_residual_pointwise(pts, ux, uy, fm)
        res["oss_n_iter"].append({
            "n_iter": n,
            "grid_force_x": float(gf[0].sum()),
            "force_rel_vs_converged": rel(gf[0].sum(), gf_big[0].sum()),
            "iface_slip_rel": slip_abs / 0.05,
        })
    for n in [0, 1, 2, 3, 5, 10, 20, 40]:
        fm, gf, st = tl_operators(pts, ux, uy, n, scheme="pointwise")
        slip_abs, _ = iface_residual_pointwise(pts, ux, uy, fm)
        res["tl_n_corr"].append({
            "n_correctors": n,
            "grid_force_x": float(gf[1].sum()),
            "force_rel_vs_oss_converged": rel(gf[1].sum(), gf_big[0].sum()),
            "iface_slip_rel": slip_abs / 0.05,
        })
    # MDF port (L3) at the same settings
    res["tl_mdf"] = []
    for n in [1, 2, 3, 5, 10, 20]:
        fm, gf, st = tl_operators(pts, ux, uy, n, scheme="mdf")
        slip_abs, _ = iface_residual_pointwise(pts, ux, uy, fm)
        res["tl_mdf"].append({
            "n_mdf": n,
            "grid_force_x": float(gf[1].sum()),
            "force_rel_vs_oss_converged": rel(gf[1].sum(), gf_big[0].sum()),
            "iface_slip_rel": slip_abs / 0.05,
        })
    res["oss_converged"] = {
        "grid_force_x": float(gf_big[0].sum()),
        "marker_force_x_sum": float(fm_big.sum(axis=0)[0]),
        "ds_mean": float(ds.mean()),
    }
    # ds is a preconditioner only: same converged force with ds = 1
    gf_ds1, fm_ds1, *_ = oss_operators(pts, ny, nx, ux, uy, n_big, ds_override=1.0)
    res["ds_preconditioner_only"] = {
        "grid_force_x_ds_true": float(gf_big[0].sum()),
        "grid_force_x_ds_1": float(gf_ds1[0].sum()),
        "rel": rel(gf_ds1[0].sum(), gf_big[0].sum()),
    }
    # extended iteration study: does the OSS fixed point fully converge?
    res["convergence_ext"] = []
    for n in (40, 100, 200, 400):
        gf_x, fm_x, *_ = oss_operators(pts, ny, nx, ux, uy, n)
        slip_x, _ = iface_residual_pointwise(pts, ux, uy, fm_x)
        fm_m, gf_m, _ = tl_operators(pts, ux, uy, min(n, 200), scheme="mdf")
        slip_m, _ = iface_residual_pointwise(pts, ux, uy, fm_m)
        res["convergence_ext"].append({
            "n_iter": n,
            "oss_grid_force_x": float(gf_x[0].sum()),
            "oss_iface_slip_rel": slip_x / 0.05,
            "tl_mdf_grid_force_x": float(gf_m[1].sum()),
            "tl_mdf_iface_slip_rel": slip_m / 0.05,
            "tl_mdf_force_rel_vs_oss": rel(gf_m[1].sum(), gf_x[0].sum()),
        })

    # round-trip operator row sum (explains the amplification)
    st = build_stencil(torch.tensor(pts, dtype=torch.float64), (ny, nx), "4pt")
    a_row = st.interpolate(st.spread(torch.ones((len(pts), 2), dtype=torch.float64)))[0].numpy()
    res["round_trip"] = {
        "A_row_sum_mean": float(a_row.mean()),
        "A_row_sum_min": float(a_row.min()),
        "A_row_sum_max": float(a_row.max()),
        "expected_amplification_1_over_row_sum": float(1.0 / a_row.mean()),
        "measured_amplification": float(gf_big[0].sum() / (2.0 * (-0.05) * len(pts))),
    }
    return res


def level2_mdf():
    return [_uniform_mdf_case(h) for h in (0.5, 0.7, 0.9, 1.2, 2.0, 3.0)]


# --------------------------------------------------------------------------- #
# L4 -- end-to-end cylinder in a free stream
# --------------------------------------------------------------------------- #


def guo_term(f, fx, fy, tau):
    """Guo (2002) forcing source, float64, ``f_i += (1-1/(2 tau)) w_i 3 c_i.F``."""
    w = W2D.to(torch.float64).view(9, 1, 1)
    cx = C2D[:, 0].to(torch.float64).view(9, 1, 1)
    cy = C2D[:, 1].to(torch.float64).view(9, 1, 1)
    g = 1.0 - 1.0 / (2.0 * tau)
    return f + w * 3.0 * g * (cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0))


def run_cylinder(re: float, arm: str, D: int = 8, ny_D: float = 20.0, nx_D: float = 30.0,
                 u_in: float = 0.1, n_steps: int = 6000, h_marker: float = 1.0,
                 n_iter: int = 10, sample_every: int = 20, threads: int = 8):
    torch.set_num_threads(threads)
    dt = torch.float64
    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    nx, ny = int(nx_D * D), int(ny_D * D)
    cx, cy = float(nx) / 3.0, ny / 2.0
    R = D / 2.0
    solid = cylinder_mask(nx, ny, cx, cy, R, torch.device("cpu"))
    pts = markers_circle(cx, cy, R, h_marker)

    st = build_stencil(torch.tensor(pts, dtype=torch.float64), (ny, nx), "4pt")
    w, idx = oss_ref.get_ib_stencil(shim.asarray(pts[:, 0]), shim.asarray(pts[:, 1]), ny)
    ds = np.asarray(oss_ref.get_ds(shim.asarray(pts)))
    u_target_oss = shim.asarray(np.zeros((len(pts), 2)))
    u_target_tl = torch.zeros((len(pts), 2), dtype=torch.float64)

    rho = torch.ones((ny, nx), dtype=dt)
    ux0 = torch.full_like(rho, u_in)
    uy0 = torch.zeros_like(rho)
    for _ in range(3):  # smooth inflow ramp-in
        pass
    f = equilibrium(rho, ux0, uy0)

    F = torch.zeros((2, ny, nx), dtype=dt)  # lagged body force
    hist = {"t": [], "cd": [], "slip": []}
    dyn = 0.5 * u_in**2 * D
    t0 = time.time()
    for step in range(1, n_steps + 1):
        f = collide_mrt(f, tau)
        f = guo_term(f, F[1], F[0], tau)  # F stored (y, x) rows; guo wants (fx, fy)
        f = stream(f)
        f = far_field_bc_2d(f, u_in)
        rho = f.sum(0)
        ux = (f * C2D[:, 0].to(dt).view(9, 1, 1)).sum(0) / rho
        uy = (f * C2D[:, 1].to(dt).view(9, 1, 1)).sum(0) / rho

        # --- surface operator (the ONLY thing that changes between arms) ---
        if arm == "oss_mdf":
            gf, fm, *_ = oss_operators(pts, ny, nx, ux.numpy(), uy.numpy(), n_iter)
            F_surf = torch.tensor(np.stack([gf[1].T, gf[0].T]))  # (nx,ny)->(ny,nx), rows (y, x)
            fm_np = fm
        elif arm == "tl_pointwise":
            fm_t, fg, _body, _tq = direct_forcing(st, torch.stack([uy, ux]), u_target_tl, None, n_correctors=n_iter)
            F_surf = fg  # already (component=(y,x), ny, nx)
            fm_np = fm_t.numpy()[:, ::-1]
        elif arm == "tl_mdf":
            fm_t, fg, _body, _tq = direct_forcing_mdf(st, torch.stack([uy, ux]), u_target_tl, None, n_mdf=n_iter)
            F_surf = fg
            fm_np = fm_t.numpy()[:, ::-1]
        else:
            raise ValueError(arm)
        F_int = torch.zeros((2, ny, nx), dtype=dt)
        F = F_surf + F_int
        # interior penalty (shared by all arms, unchanged): full velocity deficit
        F = F + interior_mask_force(torch.stack([uy, ux]), solid, torch.zeros((2, ny, nx), dtype=dt))

        if step % sample_every == 0:
            drag = -float(fm_np.sum(axis=0)[0])
            slip_abs, _ = iface_residual_pointwise(pts, ux.numpy(), uy.numpy(), fm_np)
            hist["t"].append(step * u_in / D)
            hist["cd"].append(drag / dyn)
            hist["slip"].append(slip_abs / u_in)
    wall = time.time() - t0
    cd = np.asarray(hist["cd"])
    half = cd[len(cd) // 2:]
    return {
        "arm": arm,
        "re": re,
        "setup": {"D": D, "nx": nx, "ny": ny, "u_in": u_in, "nu": nu, "tau": tau,
                  "n_steps": n_steps, "h_marker": h_marker, "n_iter": n_iter,
                  "blockage": D / ny, "dt": "float64", "grid_steps_per_D": nx_D},
        "cd_mean_last_half": float(np.mean(half)),
        "cd_std_last_half": float(np.std(half)),
        "cd_last": float(cd[-1]),
        "slip_mean_last_half": float(np.mean(hist["slip"][len(hist["slip"]) // 2:])),
        "wall_s": wall,
        "hist": hist,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def make_figures(out: dict, figdir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    made = []

    # F1 kernel + stencil
    L1 = out["L1_kernel"]
    r = np.asarray(L1["profiles"]["r"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(r, L1["profiles"]["oss_peskin_4pt"], label="OSS kernel_peskin_4pt", lw=2)
    ax[0].plot(r, L1["profiles"]["tl_4pt"], "--", label="TensorLBM ibm_delta_4pt", lw=1.5)
    ax[0].plot(r, L1["profiles"]["tl_hat"], ":", label="TensorLBM ibm_delta_hat")
    ax[0].plot(r, L1["profiles"]["oss_cosine_4pt"], "-.", label="OSS kernel_cosine_4pt")
    ax[0].set_xlabel("r [lattice units]")
    ax[0].set_ylabel(r"$\phi(r)$")
    ax[0].set_title("1-D kernels (tensor-product factors)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    sub = out["L1_stencil"]["2d_peskin4pt"]
    ax[1].bar(["max|Δw|", "nonzero keys differ"], [sub["max_abs_weight_diff"], float(not sub["nonzero_node_keys_identical"])])
    ax[1].set_yscale("log")
    ax[1].set_title(f"2-D stencil identity ({sub['n_markers']} markers, {sub['tl_points_per_marker']} pts each)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = figdir / "f1_kernel_stencil.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)

    # F2 convergence of the two MDF formulations
    L2 = out["L2_mdf"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for case in L2:
        it = [e["n_iter"] for e in case["oss_n_iter"]]
        ax[0].semilogy(it, [max(e["force_rel_vs_converged"], 1e-16) for e in case["oss_n_iter"]], "o-",
                       label=f"OSS MDF h={case['h']}")
        ic = [e["n_correctors"] for e in case["tl_n_corr"]]
        ax[0].semilogy(ic, [max(e["force_rel_vs_oss_converged"], 1e-16) for e in case["tl_n_corr"]], "x--",
                       label=f"TL pointwise h={case['h']}")
    ax[0].set_xlabel("iterations")
    ax[0].set_ylabel("relative force error")
    ax[0].set_title("iteration convergence")
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].grid(alpha=0.3, which="both")
    hs = [c["h"] for c in L2]
    ax[1].plot(hs, [c["round_trip"]["A_row_sum_mean"] for c in L2], "o-", label=r"mean row sum of $A=\mathcal{I}\mathcal{S}$")
    ax[1].plot(hs, [c["oss_converged"]["grid_force_x"] and c["round_trip"]["measured_amplification"] for c in L2], "s-",
               label="measured force amplification (OSS/TL)")
    ax[1].set_xlabel("marker spacing h [cells]")
    ax[1].set_title("round-trip operator vs marker density")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = figdir / "f2_mdf_convergence.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)

    # F3 interface residual vs h
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, lab, st in (("oss_n_iter", "OSS MDF (n_iter=10)", "o-"),
                         ("tl_n_corr", "TL pointwise (n_correctors=10)", "x--"),
                         ("tl_mdf", "TL MDF port (n_mdf=10)", "^:")):
        ys = []
        for c in L2:
            e = [q for q in c[key] if ("n_iter" in q and q["n_iter"] == 10) or ("n_correctors" in q and q["n_correctors"] == 10) or ("n_mdf" in q and q["n_mdf"] == 10)]
            ys.append(max(e[0]["iface_slip_rel"], 1e-16) if e else np.nan)
        ax.loglog(hs, ys, st, label=lab)
    ax.axhline(0.15, color="r", ls="--", lw=1, label="owner 15% bar")
    ax.set_xlabel("marker spacing h [cells]")
    ax.set_ylabel(r"interface slip $|u_{marker}|/U$")
    ax.set_title("achieved no-slip at markers (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = figdir / "f3_interface_residual.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)

    # F4 end-to-end
    L4 = out.get("L4_cylinder", {})
    if L4:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for re_key, runs in L4.items():
            for arm, r in runs.items():
                ax[0].plot(r["hist"]["t"], r["hist"]["cd"], label=f"Re={re_key} {arm}")
                ax[1].semilogy(r["hist"]["t"], np.maximum(r["hist"]["slip"], 1e-16), label=f"Re={re_key} {arm}")
        ax[0].set_xlabel("t U/D")
        ax[0].set_ylabel("$C_D$")
        ax[0].set_title("cylinder drag history")
        ax[0].legend(fontsize=6)
        ax[0].grid(alpha=0.3)
        ax[1].axhline(0.15, color="r", ls="--", lw=1)
        ax[1].set_xlabel("t U/D")
        ax[1].set_ylabel("interface slip / U")
        ax[1].set_title("marker no-slip during the run")
        ax[1].legend(fontsize=6)
        ax[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = figdir / "f4_cylinder_end_to_end.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(p)
    return made


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #


def verdict(out):
    L1 = out["L1_stencil"]
    s2 = L1["2d_peskin4pt"]
    s3 = L1["3d_peskin4pt"]
    L2 = out["L2_mdf"]
    h09 = [c for c in L2 if abs(c["h"] - 0.9) < 1e-9][0]
    tl10 = [e for e in h09["tl_n_corr"] if e["n_correctors"] == 10][0]
    oss10 = [e for e in h09["oss_n_iter"] if e["n_iter"] == 10][0]
    mdf10 = [e for e in h09["tl_mdf"] if e["n_mdf"] == 10][0]
    v = {
        "L1_stencil_identity": {
            "2d_max_abs_weight_diff": s2["max_abs_weight_diff"],
            "2d_keys_identical": s2["nonzero_node_keys_identical"],
            "3d_max_abs_weight_diff": s3["max_abs_weight_diff"],
            "3d_keys_identical": s3["nonzero_node_keys_identical"],
            "kernel_max_abs_diff": out["L1_kernel"]["kernel_maxabs_diff_peskin4pt"],
            "pass": bool(s2["max_abs_weight_diff"] < 1e-12 and s3["max_abs_weight_diff"] < 1e-12
                         and s2["nonzero_node_keys_identical"] and s3["nonzero_node_keys_identical"]
                         and out["L1_kernel"]["kernel_maxabs_diff_peskin4pt"] < 1e-12),
        },
        "L2_force_agreement_at_h09": {
            "oss_mdf_10": oss10["grid_force_x"],
            "tl_pointwise_10": tl10["grid_force_x"],
            "tl_mdf_port_10": mdf10["grid_force_x"],
            "rel_tl_pointwise_vs_oss": tl10["force_rel_vs_oss_converged"],
            "rel_tl_mdf_vs_oss": mdf10["force_rel_vs_oss_converged"],
            "pass_pointwise": bool(tl10["force_rel_vs_oss_converged"] < 0.15),
            "pass_mdf_port": bool(mdf10["force_rel_vs_oss_converged"] < 0.15),
        },
        "L2_interface_slip_at_h09": {
            "oss_mdf": oss10["iface_slip_rel"],
            "tl_pointwise": tl10["iface_slip_rel"],
            "tl_mdf_port": mdf10["iface_slip_rel"],
            "pass_pointwise": bool(tl10["iface_slip_rel"] < 0.15),
            "pass_oss": bool(oss10["iface_slip_rel"] < 0.15),
        },
    }
    L4 = out.get("L4_cylinder", {})
    if L4:
        per = {}
        for re_key, runs in L4.items():
            cd = {a: r["cd_mean_last_half"] for a, r in runs.items()}
            ref = out["literature_cd"][re_key]
            row = {"cd": cd, "ref_literature": ref}
            if "oss_mdf" in cd and "tl_pointwise" in cd:
                row["rel_tl_pointwise_vs_oss"] = abs(cd["tl_pointwise"] - cd["oss_mdf"]) / abs(cd["oss_mdf"])
                row["pass_pointwise_vs_oss"] = bool(row["rel_tl_pointwise_vs_oss"] < 0.15)
            if "oss_mdf" in cd and "tl_mdf" in cd:
                row["rel_tl_mdf_vs_oss"] = abs(cd["tl_mdf"] - cd["oss_mdf"]) / abs(cd["oss_mdf"])
                row["pass_mdf_vs_oss"] = bool(row["rel_tl_mdf_vs_oss"] < 0.15)
            for a in cd:
                row[f"rel_{a}_vs_lit"] = abs(cd[a] - ref) / ref
            per[re_key] = row
        v["L4_end_to_end"] = per
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "artifacts" / "xval_peskin_tp_mdf.json"))
    ap.add_argument("--quick", action="store_true", help="skip the end-to-end LBM (operator levels only)")
    ap.add_argument("--re", default="20,40")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--figdir", default=str(HERE / "figures"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    from tensorlbm.ibm_openwater import CYLINDER_CD_2D

    out = {"anchor": "XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF",
           "oss": "haimingz/vivsim@fb84b12 vivsim/ib/{kernels,stencil,mdf,geometry}.py + ib3d/stencil.py",
           "literature_cd": {str(k): v for k, v in CYLINDER_CD_2D.items()},
           "literature_source": "src/tensorlbm/ibm_openwater.py:130 CYLINDER_CD_2D (Dennis & Chang 1970 JFM 42:471 for Re<=100)"}
    t0 = time.time()
    out["L1_kernel"] = level1_kernel()
    print(f"[xval] L1 kernel done ({time.time()-t0:.1f}s)", flush=True)
    out["L1_stencil"] = level1_stencil()
    print(f"[xval] L1 stencil: 2d diff={out['L1_stencil']['2d_peskin4pt']['max_abs_weight_diff']:.3e} "
          f"3d diff={out['L1_stencil']['3d_peskin4pt']['max_abs_weight_diff']:.3e}", flush=True)
    out["L2_mdf"] = level2_mdf()
    for c in out["L2_mdf"]:
        print(f"[xval] L2 h={c['h']}: A_row_sum={c['round_trip']['A_row_sum_mean']:.3f} "
              f"amp={c['round_trip']['measured_amplification']:.3f} "
              f"ossF={c['oss_converged']['grid_force_x']:.4f}", flush=True)
    if not args.quick:
        out["L4_cylinder"] = {}
        for re_val in [float(s) for s in args.re.split(",")]:
            key = str(int(re_val))
            out["L4_cylinder"][key] = {}
            for arm in ("oss_mdf", "tl_pointwise", "tl_mdf"):
                r = run_cylinder(re_val, arm, n_steps=args.steps, threads=args.threads)
                out["L4_cylinder"][key][arm] = r
                print(f"[xval] L4 Re={key} {arm}: Cd={r['cd_mean_last_half']:.4f} "
                      f"slip={r['slip_mean_last_half']:.3e} ({r['wall_s']:.0f}s)", flush=True)
    out["verdict"] = verdict(out)
    figs = make_figures(out, Path(args.figdir))
    out["figures"] = [p.name for p in figs]
    out["wall_s"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["verdict"], indent=2))
    print(f"[xval] wrote {args.out} in {out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()