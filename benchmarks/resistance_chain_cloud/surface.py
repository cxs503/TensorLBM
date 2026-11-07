#!/usr/bin/env python3
"""Surface loads from the stored time-averaged fields (exposed-face integration).

The per-cell Ladd momentum exchange stored by ``run.py`` is *not* a local load
(full-way bounce-back over a filled solid: interior cell pairs cancel only in
the global sum).  ``Cp`` and ``Cf`` in this module are therefore reconstructed
from the time-averaged fields on the **fluid side** of every exposed face of the
staircase surface:

* exposed face = (solid cell, axis-aligned fluid 4-neighbour) with unit lattice
  length and face normal ``+/- e_x`` or ``+/- e_y``;
* ``Cp_face = (rho - 1) c_s^2 / (1/2 rho U^2)`` at the adjacent fluid cell
  (half-way bounce-back -> wall value at O(dx));
* ``Cf_face = tau_w / (1/2 rho U^2) = 4 nu u_t / U`` with ``tau_w = 2 mu u_t``
  (wall at ``0.5 dx``, ``u_t = u . t`` normalised by U, ``t = (-n_y, n_x)``);
* ``ds_face = |n_face . n_surface|`` removes the staircase (Manhattan) length
  bias, so ``sum ds -> perimeter``;
* force on the body from pressure is ``-oint p n dA`` (``n`` outward), hence
  ``Cd_p = -sum Cp n_x ds / D`` and ``Cd_f = +sum Cf t_x ds / D``.

The angular profile is binned by the angle of the face midpoint with 10 deg
bins and a circular 3-tap smoothing, mirroring ``run.py``.
"""

from __future__ import annotations

import numpy as np

N_BINS = 36
SMOOTH_PASSES = 2


def _circ_smooth(v: np.ndarray, passes: int = SMOOTH_PASSES) -> np.ndarray:
    out = v.astype(np.float64).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    if not finite.all():
        idx = np.arange(out.size)
        pos = idx[finite]
        for i in idx[~finite]:
            d = (pos - i + out.size // 2) % out.size - out.size // 2
            out[i] = out[pos[np.argmin(np.abs(d))]]
    for _ in range(passes):
        out = 0.25 * np.roll(out, 1) + 0.5 * out + 0.25 * np.roll(out, -1)
    return out


def exposed_faces(mask: np.ndarray) -> dict:
    """Enumerate exposed faces of a Boolean solid mask.

    Returns arrays ``ys, xs`` (solid cell), ``fy, fx`` (adjacent fluid cell),
    ``n`` (unit face normal, (N,2)) and ``mid`` (face midpoint in cell units).
    """
    ny, nx = mask.shape
    ys, xs, fy, fx, ns, mid = [], [], [], [], [], []
    for ddy, ddx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        src = mask
        dst_y = np.broadcast_to(np.arange(ny)[:, None] + ddy, (ny, nx))
        dst_x = np.broadcast_to(np.arange(nx)[None, :] + ddx, (ny, nx))
        ok = (dst_y >= 0) & (dst_y < ny) & (dst_x >= 0) & (dst_x < nx)
        dst_fluid = np.zeros_like(mask)
        dst_fluid[ok] = ~mask[dst_y[ok].ravel(), dst_x[ok].ravel()] if False else \
            ~mask[np.clip(dst_y, 0, ny - 1)[ok], np.clip(dst_x, 0, nx - 1)[ok]]
        sel = src & dst_fluid
        yy, xx = np.nonzero(sel)
        ys.append(yy)
        xs.append(xx)
        fy.append(yy + ddy)
        fx.append(xx + ddx)
        ns.append(np.repeat([[ddx, ddy]], yy.size, axis=0))
        mid.append(np.stack([xx + 0.5 * ddx, yy + 0.5 * ddy], axis=1))
    return {
        "ys": np.concatenate(ys),
        "xs": np.concatenate(xs),
        "fy": np.concatenate(fy),
        "fx": np.concatenate(fx),
        "n": np.concatenate(ns).astype(float),
        "mid": np.concatenate(mid).astype(float),
    }


def surface_loads(npz: dict, geo: str) -> dict:
    """Per-face Cp/Cf, drag decomposition and the binned angular profiles."""
    mask = npz["mask"].astype(bool)
    cp_field = npz["cp"]
    ux = npz["ux"]
    uy = npz["uy"]
    keys = npz.files if hasattr(npz, "files") else list(npz.keys())
    a = float(npz["a"]) if "a" in keys else 0.5 * float(npz["D"])
    b = float(npz["b"]) if "b" in keys else 0.5 * float(npz["D"])
    D = float(npz["D"])
    cx, cy = float(npz["cx"]), float(npz["cy"])
    u_in = float(npz["u_in"])
    re = float(npz["re"])
    nu = u_in * D / re

    f = exposed_faces(mask)
    fxs, fys = f["fx"], f["fy"]
    n = f["n"]
    cp_face = cp_field[fys, fxs]
    # tangential direction (CCW rotation of the face normal)
    tx, ty = -n[:, 1], n[:, 0]
    ut = ux[fys, fxs] * tx + uy[fys, fxs] * ty  # normalised by U_inf
    # u_t is normalised by U_inf -> physical tau_w = mu * (u_t*U_inf)/0.5
    cf_face = 4.0 * nu * ut / u_in  # = tau_w/(0.5 rho U_inf^2)

    # staircase length correction: project the face normal on the true surface
    mx = f["mid"][:, 0] - cx
    my = f["mid"][:, 1] - cy
    gx, gy = mx / a**2, my / b**2  # true outward normal direction
    gn = np.hypot(gx, gy)
    gn[gn == 0] = 1.0
    gx, gy = gx / gn, gy / gn
    ds = np.abs(n[:, 0] * gx + n[:, 1] * gy)

    cd_p = float(-np.sum(cp_face * n[:, 0] * ds) / D)
    cd_f = float(np.sum(cf_face * tx * ds) / D)

    ang = np.arctan2(my / b, mx / a)
    bidx = np.clip(((ang + np.pi) / (2 * np.pi) * N_BINS).astype(int), 0, N_BINS - 1)
    phi = (np.arange(N_BINS) + 0.5) / N_BINS * 2 * np.pi - np.pi
    cp_bin = np.full(N_BINS, np.nan)
    cf_bin = np.full(N_BINS, np.nan)
    cp_std = np.full(N_BINS, np.nan)
    cnt = np.zeros(N_BINS, dtype=int)
    for i in range(N_BINS):
        sel = bidx == i
        if not sel.any():
            continue
        w = ds[sel]
        sw = w.sum()
        cp_bin[i] = float(np.sum(cp_face[sel] * w) / sw) if sw > 0 else float(np.mean(cp_face[sel]))
        cf_bin[i] = float(np.sum(cf_face[sel] * w) / sw) if sw > 0 else float(np.mean(cf_face[sel]))
        cp_std[i] = float(np.std(cp_face[sel])) if sel.sum() > 1 else 0.0
        cnt[i] = int(sel.sum())

    return {
        "phi": phi,
        "cp_bin": _circ_smooth(cp_bin),
        "cf_bin": _circ_smooth(cf_bin),
        "cp_bin_raw": cp_bin,
        "cp_bin_std": cp_std,
        "cf_bin_raw": cf_bin,
        "bin_count": cnt,
        "cd_pressure": cd_p,
        "cd_friction": cd_f,
        "cd_pf_sum": cd_p + cd_f,
        "n_faces": int(f["ys"].size),
        "perimeter_sum_ds": float(ds.sum()),
        "cp_face_min": float(np.min(cp_face)),
        "cp_face_max": float(np.max(cp_face)),
        "cp_face_mean_area_weighted": float(np.sum(cp_face * ds) / ds.sum()),
    }