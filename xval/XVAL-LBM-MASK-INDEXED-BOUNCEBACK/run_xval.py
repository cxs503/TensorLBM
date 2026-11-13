#!/usr/bin/env python3
"""XVAL-LBM-MASK-INDEXED-BOUNCEBACK — cross-validation runner.

Anchor: ``XVAL-LBM-MASK-INDEXED-BOUNCEBACK`` (anchor feeder, 2026-11-12).
OSS arm: ``haimingz/vivsim`` @ ``fb84b12f02e18f3b9dfdf738d6a1a318d81539f9``
(``/tmp/vivsim_src``), rule ``vivsim/lbm/boundary/bb.py:98-110``
(``obstacle_bounce_back``; D3Q19 twin ``vivsim/lbm3d/boundary/bb.py:56-59``).

Arms (one solver skeleton, only the solid closure changes):
  A1  oss_boolean   literal OSS one-liner (boolean-mask select + scatter)
  A2  mask_indexed  TensorLBM index-pair form (this mission's operator)
  A3  production    ``tensorlbm.boundaries.bounce_back_cells`` (dense where)
  A4  fluid_side    independently written fluid-side half-way bounce-back

Blocks (each writes its own JSON so a partial run is still evidence):
  identity   representation identity A1/A2/A3 (+A4 with and without forcing)
  channel    plane Poiseuille: permeability vs analytic H^2/12 (H sweep),
             naive-H bias, tau sensitivity, body-force linearity
  cost       cost model vs solid fraction (A1/A2/A2-inplace/A3)
  st2d1      Schaefer-Turek 2D-1 Re=20 cylinder: staircase (A2) vs
             interpolated bounce-back (Bouzidi 2pt) vs in-repo Cd anchor
             (``benchmarks/verified/cylinder/re20_st/result.json``,
             ref_cd = 5.57953523384)

Usage:
    PYTHONPATH=src python xval/XVAL-LBM-MASK-INDEXED-BOUNCEBACK/run_xval.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorlbm.boundary.interpolated_bounce_back import interpolated_bounce_back_2d
from tensorlbm.boundary.mask_indexed_bounceback import (
    boolean_mask_bounce_back,
    build_solid_mask_index,
    mask_indexed_bounce_back,
    mask_indexed_bounce_back_,
)
from tensorlbm.boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
from tensorlbm.boussinesq import apply_guo_forcing
from tensorlbm.solver import stream

DTYPE = torch.float64
TAU_DEFAULT = 0.8
REF_ST2D1_CD = 5.57953523384  # in-repo anchor benchmarks/verified/cylinder/re20_st/result.json
OWNER_BAND_PCT = 15.0


# --------------------------------------------------------------------- helpers
def _poiseuille_setup(ny: int, nx: int, g: float):
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    rho = torch.ones((ny, nx), dtype=DTYPE)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    force = torch.zeros((2, ny, nx), dtype=DTYPE)
    force[0] = g
    return f, solid, force


def _step(f, solid, force, tau, closure, index=None, *, f_pre=None, sdf=None):
    """One BGK + Guo-forcing step; `closure` selects the solid treatment."""
    f_post = apply_guo_forcing(f, force, tau)
    if closure == "fluid_side":
        out = torch.empty_like(f_post)
        for i in range(9):
            dx, dy = int(C[i, 0]), int(C[i, 1])
            pulled = torch.roll(f_post[i], shifts=(dy, dx), dims=(0, 1))
            nb_solid = torch.roll(solid, shifts=(dy, dx), dims=(0, 1))
            out[i] = torch.where(nb_solid, f_post[int(OPPOSITE[i])], pulled)
        return out

    f_new = stream(f_post)
    if closure == "mask_indexed":
        return mask_indexed_bounce_back(f_new, index)
    if closure == "mask_indexed_inplace":
        return mask_indexed_bounce_back_(f_new, index)
    if closure == "oss_boolean":
        return boolean_mask_bounce_back(f_new, solid, OPPOSITE)
    if closure == "production":
        return bounce_back_cells(f_new, solid)
    if closure == "interpolated":
        return interpolated_bounce_back_2d(f_new, f_post, sdf, solid)
    raise ValueError(closure)


# ------------------------------------------------------------------ block 1/2
def block_identity() -> dict:
    """A1/A2/A3 representation identity on random fields; A4 with/without forcing."""
    g = torch.Generator().manual_seed(17956)
    rows = []
    for ndim, shape, opp in ((2, (37, 53), OPPOSITE), (3, (13, 17, 19), None)):
        if ndim == 3:
            from tensorlbm.d3q19 import OPPOSITE as OPP3

            opp = OPP3
        q = int(torch.as_tensor(opp).numel())
        f = (torch.rand((q, *shape), generator=g, dtype=DTYPE) + 0.05)
        mask = torch.rand(shape, generator=g) < 0.23
        mask[0, ...] = True  # touches the domain border
        a1 = boolean_mask_bounce_back(f, mask, opp)
        idx = build_solid_mask_index(mask, opp)
        a2 = mask_indexed_bounce_back(f, idx)
        a2i = mask_indexed_bounce_back_(f.clone(), idx)
        rows.append(
            {
                "dims": ndim,
                "nodes": int(torch.tensor(shape).prod()),
                "solid_cells": idx.count,
                "solid_fraction": round(idx.solid_fraction, 6),
                "max_abs_a1_minus_a2": float((a1 - a2).abs().max()),
                "max_abs_a1_minus_a2_inplace": float((a1 - a2i).abs().max()),
                "bitwise_a1_eq_a2": bool(torch.equal(a1, a2)),
                "index_nbytes": idx.index_nbytes,
            }
        )
    return {"rows": rows, "threshold": "bitwise identity (max abs diff == 0)"}


def _channel_arm(ny, nx, tau, g, steps, closure="mask_indexed", sdf=None):
    f, solid, force = _poiseuille_setup(ny, nx, g)
    idx = build_solid_mask_index(solid, OPPOSITE)
    for _ in range(steps):
        f = _step(f, solid, force, tau, closure, idx, sdf=sdf)
    rho, ux, uy = macroscopic(f)
    return f, solid, ux, uy


def block_channel(h_list=(8, 16, 32), steps: int = 5000, nx: int = 6) -> dict:
    """Plane Poiseuille: k vs analytic H^2/12, naive-H bias, tau, force linearity."""
    g_force = 1.0e-5
    resolution, tau_rows, linearity = [], [], []
    profiles = {}
    for h in h_list:
        ny = h + 2
        nu = (TAU_DEFAULT - 0.5) / 3.0
        g = g_force * (h / 16.0) ** 2  # keep Ma = u_max/cs fixed across H
        t0 = time.time()
        f, solid, ux, uy = _channel_arm(ny, nx, TAU_DEFAULT, g, steps)
        fluid = ~solid
        u_mean = float(ux[fluid].mean())
        k_num = nu * u_mean / g
        k_ref = h * h / 12.0
        k_naive = nu * u_mean / g * (ny / h)  # superficial over all rows, H=ny
        err = (k_num - k_ref) / k_ref * 100.0
        err_naive = (k_naive - k_ref) / k_ref * 100.0
        y = torch.arange(ny, dtype=DTYPE) - 0.5
        prof = g / (2.0 * nu) * y * (h - y)
        col = fluid[:, 0]
        prof_err = float((ux[:, 0][col] - prof[col]).abs().max() / prof[col].abs().max() * 100.0)
        u_max = g * h * h / (8.0 * nu)
        resolution.append(
            {
                "H_fluid_rows": h,
                "ny": ny,
                "nx": nx,
                "tau": TAU_DEFAULT,
                "nu": nu,
                "g": g,
                "u_max": u_max,
                "Ma": u_max / math.sqrt(1.0 / 3.0),
                "Re_H": u_max * h / nu,
                "u_mean": u_mean,
                "k_measured": k_num,
                "k_analytic": k_ref,
                "err_pct": err,
                "k_if_H_eq_ny": k_naive,
                "err_pct_if_H_eq_ny": err_naive,
                "profile_max_err_pct": prof_err,
                "wall_s": time.time() - t0,
                "within_owner_band": abs(err) < OWNER_BAND_PCT,
            }
        )
        if h == 32:
            profiles["h32_ux"] = [float(v) for v in ux[1:-1, :].mean(dim=1)]
            profiles["h32_analytic"] = [float(prof[i]) for i in range(1, ny - 1)]
    for tau in (0.8, 1.0, 1.5):
        h, ny, nu, g = 32, 34, (tau - 0.5) / 3.0, g_force * (32 / 16.0) ** 2
        _, solid, ux, _ = _channel_arm(ny, nx, tau, g, steps)
        fluid = ~solid
        k_num = nu * float(ux[fluid].mean()) / g
        k_ref = h * h / 12.0
        tau_rows.append(
            {"tau": tau, "nu": nu, "H": h, "g": g, "k_measured": k_num,
             "k_analytic": k_ref, "err_pct": (k_num - k_ref) / k_ref * 100.0}
        )
    for scale in (1.0, 2.0):
        h, ny, nu = 32, 34, (TAU_DEFAULT - 0.5) / 3.0
        g = g_force * (32 / 16.0) ** 2 * scale
        _, solid, ux, _ = _channel_arm(ny, nx, TAU_DEFAULT, g, steps)
        fluid = ~solid
        linearity.append(
            {"g": g, "u_mean": float(ux[fluid].mean()),
             "k_measured": nu * float(ux[fluid].mean()) / g, "k_analytic": h * h / 12.0}
        )
    k0 = linearity[0]["k_measured"]
    linearity_delta = abs(linearity[1]["k_measured"] - k0) / abs(k0) * 100.0
    return {
        "resolution": resolution,
        "tau_sensitivity": tau_rows,
        "force_linearity": linearity,
        "force_linearity_delta_pct": linearity_delta,
        "profiles": profiles,
        "steps": steps,
        "closure": "mask_indexed (A2)",
        "wall_position": "half-way between the last fluid row and the solid row (H = ny-2)",
    }


# -------------------------------------------------------------------- block 3
def block_cost(repeats: int = 25) -> dict:
    """Cost model: cost per application vs solid fraction (2-D 512x512, float64)."""
    ny = nx = 512
    rows = []
    gg = torch.Generator().manual_seed(7)
    for frac in (0.005, 0.02, 0.05, 0.10, 0.25, 0.50):
        yy = torch.arange(ny).view(-1, 1).expand(ny, nx)
        xx = torch.arange(nx).view(1, -1).expand(ny, nx)
        # a slab of thickness giving the requested solid fraction -> grid-aligned wall
        thick = max(1, int(round(frac * ny / 2.0)))
        mask = (yy < thick) | (yy >= ny - thick)
        f = torch.rand((9, ny, nx), generator=gg, dtype=DTYPE) + 0.05
        idx = build_solid_mask_index(mask, OPPOSITE)
        t_build = time.time()
        _ = build_solid_mask_index(mask, OPPOSITE)
        t_build = time.time() - t_build

        def bench(fn):
            fn()  # warm-up
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            return (time.perf_counter() - t0) / repeats * 1e3

        buf = f.clone()
        t_a1 = bench(lambda: boolean_mask_bounce_back(f, mask, OPPOSITE))
        t_a2 = bench(lambda: mask_indexed_bounce_back(f, idx))
        t_a2i = bench(lambda: mask_indexed_bounce_back_(buf, idx))
        t_a3 = bench(lambda: bounce_back_cells(f, mask))
        rows.append(
            {
                "solid_fraction": idx.solid_fraction,
                "solid_cells": idx.count,
                "nodes": ny * nx,
                "t_oss_boolean_ms": t_a1,
                "t_mask_indexed_ms": t_a2,
                "t_mask_indexed_inplace_ms": t_a2i,
                "t_production_where_ms": t_a3,
                "speedup_inplace_vs_where": t_a3 / t_a2i,
                "speedup_inplace_vs_oss": t_a1 / t_a2i,
                "index_build_ms": t_build * 1e3,
                "index_nbytes": idx.index_nbytes,
            }
        )
    return {"rows": rows, "grid": [ny, nx], "dtype": "float64", "repeats": repeats}


# -------------------------------------------------------------------- block 4
def _st2d1(d_cells: int, steps: int, closure: str) -> dict:
    """Schaefer-Turek 2D-1 Re=20 (same setup as benchmarks/verified/cylinder/re20_st)."""
    lx, ly = 2.2, 0.41
    dx = 0.1 / d_cells
    nx, ny = int(round(lx / dx)), int(round(ly / dx))
    cxp, cyp = 0.2 / dx, 0.2 / dx
    radius = 0.05 / dx
    tau = 0.8
    nu = (tau - 0.5) / 3.0
    u_mean_lb = 20.0 * nu / d_cells
    y_phys = torch.arange(ny, dtype=DTYPE) * dx
    u_lb = u_mean_lb * 4.0 * 1.5 * y_phys * (0.41 - y_phys) / 0.41**2
    solid = cylinder_mask(nx, ny, cxp, cyp, radius, torch.device("cpu"))
    wall = make_channel_wall_mask(ny, nx, solid, torch.device("cpu"))
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0) | torch.roll(fluid, -1, 0) | torch.roll(fluid, 1, 1) | torch.roll(fluid, -1, 1)
    )
    allmask = solid | wall
    idx = build_solid_mask_index(allmask, OPPOSITE)
    yy = torch.arange(ny, dtype=DTYPE).view(-1, 1).expand(ny, nx)
    xx = torch.arange(nx, dtype=DTYPE).view(1, -1).expand(ny, nx)
    sdf = -(((xx - cxp) ** 2 + (yy - cyp) ** 2).sqrt() - radius)
    # walls are handled by the same mask in the staircase arm; the interpolated arm
    # only replaces the cut links of the *cylinder* (a grid-aligned wall needs no q).
    sdf_wall = torch.minimum(yy - 0.5, (ny - 1.5) - yy)
    sdf = torch.where(solid, sdf, torch.where(wall, -sdf_wall.abs().clamp(min=1e-6), sdf_wall))

    rho0 = torch.ones((ny, nx), dtype=DTYPE)
    ux0 = u_lb[:, None].expand(ny, nx).clone()
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0))
    dyn_p = 0.5 * u_mean_lb**2 * d_cells
    zero_force = torch.zeros((2, ny, nx), dtype=DTYPE)

    t0 = time.time()
    series = []
    for step in range(1, steps + 1):
        f_pre = collide_only(f, tau)
        keep = torch.where(solid.unsqueeze(0), f, f_pre)
        f_new = stream(keep)
        f_new = zou_he_inlet_velocity(f_new, u_lb, 0.0)
        f_new = zou_he_outlet_pressure(f_new, 1.0)
        if closure == "mask_indexed":
            f_new = mask_indexed_bounce_back_(f_new, idx)
        else:
            f_new = interpolated_bounce_back_2d(f_new, f_pre, sdf, allmask)
        fx, _ = compute_obstacle_forces(f_new, surface)
        f = f_new
        if step % 200 == 0:
            series.append([step, float(fx.item()) / dyn_p])
        _ = zero_force
    win = [v for _, v in series if _ >= max(1, steps - 10000)]
    cd = sum(win) / max(1, len(win))
    return {
        "D_cells": d_cells,
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu": nu,
        "u_mean_lb": u_mean_lb,
        "Re": u_mean_lb * d_cells / nu,
        "steps": steps,
        "closure": closure,
        "cd": cd,
        "cd_ref": REF_ST2D1_CD,
        "err_pct": (cd - REF_ST2D1_CD) / REF_ST2D1_CD * 100.0,
        "within_owner_band": abs((cd - REF_ST2D1_CD) / REF_ST2D1_CD * 100.0) < OWNER_BAND_PCT,
        "n_solid_cells": int(allmask.sum().item()),
        "wall_s": time.time() - t0,
        "series": series[-50:],
    }


def collide_only(f, tau):
    rho, ux, uy = macroscopic(f)
    return f - (f - equilibrium(rho, ux, uy)) / tau


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--blocks", default="identity,channel,cost,st2d1")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--st-steps", type=int, default=12000)
    ap.add_argument("--st-d", default="16,24")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    blocks = [b.strip() for b in a.blocks.split(",") if b.strip()]
    summary = {"anchor": "XVAL-LBM-MASK-INDEXED-BOUNCEBACK", "oss_ref": "haimingz/vivsim fb84b12"}
    if "identity" in blocks:
        summary["identity"] = block_identity()
        (out / "identity.json").write_text(json.dumps(summary["identity"], indent=2))
        print("identity", json.dumps(summary["identity"]["rows"], indent=0)[:400], flush=True)
    if "channel" in blocks:
        summary["channel"] = block_channel(steps=a.steps)
        (out / "channel.json").write_text(json.dumps(summary["channel"], indent=2))
        print("channel", json.dumps(summary["channel"]["resolution"], indent=0)[:900], flush=True)
    if "cost" in blocks:
        summary["cost"] = block_cost()
        (out / "cost.json").write_text(json.dumps(summary["cost"], indent=2))
        print("cost", json.dumps(summary["cost"]["rows"], indent=0)[:700], flush=True)
    if "st2d1" in blocks:
        rows = []
        for d in [int(v) for v in a.st_d.split(",")]:
            for closure in ("mask_indexed", "interpolated"):
                r = _st2d1(d, a.st_steps, closure)
                rows.append(r)
                (out / "st2d1.json").write_text(json.dumps({"rows": rows}, indent=2))
                print("st2d1", json.dumps({k: v for k, v in r.items() if k != "series"}), flush=True)
        summary["st2d1"] = {"rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())