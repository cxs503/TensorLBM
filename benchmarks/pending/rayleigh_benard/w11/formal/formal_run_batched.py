#!/usr/bin/env python
"""Batched formal runner (contingency for contended GPU6): all arms of a level
as one batch, layout (B, Q, ny, nx). Eager torch, mirrors library arithmetic
op-for-op; validated BITWISE against the serial library runner on completed
ny64 arms before any formal use.

Deviations from serial (documented):
- correct_mass: rescale f *= m0/f.sum() per arm WITHOUT the <1e-30 guard
  (provably inactive: mass >= ny*nx >= 4096); removes a per-step host sync.
- Arms finalize at their own N (segment loop over sorted unique N); kept arms
  are re-stacked by value-preserving copies -> per-arm values bitwise identical.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard")
sys.path.insert(0, str(BASE / "ref"))
sys.path.insert(0, str(BASE / "probe"))
sys.path.insert(0, "/nfs/wangxi/worktrees/bm_w11/src")

import rb_lbm as R

from tensorlbm.d2q9 import C as C9i
from tensorlbm.d2q9 import W as W9f32
from tensorlbm.thermal import C5 as C5i
from tensorlbm.thermal import W5 as W5f32

REF = json.loads((BASE / "ref" / "ref_table.json").read_text())
WINDOWS = ((0.15, 0.90), (0.05, 0.75), (0.30, 1.00))
AMP = 1e-4
ALL_ARMS = (0.85, 0.95, 1.0, 1.05, 1.15, 1.3)
OUTDIR = BASE / "formal" / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)

# lattice constants (fp64 mirrors of library fp32 values -> exact same fp64
# doubles as the library's .to(float64) casts)
# EXACT library doubles: fp32 tensors cast to fp64 (bitwise same values the
# serial library path uses; correct_mass absorbs the 7.45e-9 W9 sum offset
# identically in both runners).
C9 = C9i.double()
W9 = W9f32.double()
C5 = C5i.double()
W5 = W5f32.double()
OPP9 = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6])
TAU, TAU_T = R.TAU, R.TAU_T
T_HOT, T_COLD = R.T_HOT, R.T_COLD


def equilibrium9(rho, ux, uy):
    # rho,ux,uy: (B,ny,nx) -> (B,9,ny,nx); mirrors d2q9.equilibrium exactly,
    # INCLUDING library dtypes: w fp32 (W raw), cx/cy int64 (C raw) so that
    # mixed-dtype calls (fp32 seed velocities at init) round cu/u_sq in fp32
    # exactly as the library does (bitwise-critical at init).
    w = W9f32.to(rho.device).view(1, 9, 1, 1)
    cx = C9i.to(rho.device)[:, 0].view(1, 9, 1, 1)
    cy = C9i.to(rho.device)[:, 1].view(1, 9, 1, 1)
    u_sq = ux * ux + uy * uy
    cu = cx * ux.unsqueeze(1) + cy * uy.unsqueeze(1)
    return w * rho.unsqueeze(1) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(1))


def macroscopic9(f):
    cx = C9i.to(f.device)[:, 0].view(1, 9, 1, 1)
    cy = C9i.to(f.device)[:, 1].view(1, 9, 1, 1)
    rho = f.sum(dim=1)
    rho_safe = rho.clamp(min=1e-12)
    ux = (f * cx).sum(dim=1) / rho_safe
    uy = (f * cy).sum(dim=1) / rho_safe
    return rho, ux, uy


def equilibrium5(T, ux, uy):
    # library dtypes verbatim: w fp32 (W5), cx/cy int64 (C5) -- fp32 cu at
    # init exactly as thermal.temperature_equilibrium
    w = W5f32.to(T.device).view(1, 5, 1, 1)
    cx = C5i.to(T.device)[:, 0].view(1, 5, 1, 1)
    cy = C5i.to(T.device)[:, 1].view(1, 5, 1, 1)
    cu = cx * ux.unsqueeze(1) + cy * uy.unsqueeze(1)
    return w * T.unsqueeze(1) * (1.0 + 3.0 * cu)


def streamq(f, C):
    # f: (B,Q,ny,nx); mirrors solver.stream/temperature_stream index math
    B, Q, ny, nx = f.shape
    dev = f.device
    c = C.to(dev)
    y_src = (
        torch.arange(ny, device=dev, dtype=torch.long).view(1, ny, 1) - c[:, 1].long().view(Q, 1, 1)
    ) % ny
    x_src = (
        torch.arange(nx, device=dev, dtype=torch.long).view(1, 1, nx) - c[:, 0].long().view(Q, 1, 1)
    ) % nx
    b_idx = torch.arange(B, device=dev, dtype=torch.long).view(B, 1, 1, 1)
    q_idx = torch.arange(Q, device=dev, dtype=torch.long).view(1, Q, 1, 1).expand(B, Q, ny, nx)
    y_idx = y_src.view(1, Q, ny, 1).expand(B, Q, ny, nx)
    x_idx = x_src.view(1, Q, 1, nx).expand(B, Q, ny, nx)
    return f[b_idx, q_idx, y_idx, x_idx]


def collide_force9(f, tau, F):
    # mirrors thermal.collide_bgk_force
    rho, ux, uy = macroscopic9(f)
    rho_safe = rho.clamp(min=1e-12)
    fx, fy = F[:, 0], F[:, 1]
    uxs = ux + 0.5 * fx / rho_safe
    uys = uy + 0.5 * fy / rho_safe
    feq = equilibrium9(rho, uxs, uys)
    f_new = f - (f - feq) / tau
    w = W9f32.to(f.device).view(1, 9, 1, 1)
    cx = C9i.to(f.device)[:, 0].view(1, 9, 1, 1)
    cy = C9i.to(f.device)[:, 1].view(1, 9, 1, 1)
    cF = cx * fx.unsqueeze(1) + cy * fy.unsqueeze(1)
    uF = uxs * fx + uys * fy
    cu = cx * uxs.unsqueeze(1) + cy * uys.unsqueeze(1)
    Fi = w * (1.0 - 0.5 / tau) * (3.0 * (cF - uF.unsqueeze(1)) + 9.0 * cu * cF)
    return f_new + Fi


def temp_bc_half(g):
    # mirrors rb_lbm.apply_rb_temp_bc construction="half" on (B,5,ny,nx)
    dev = g.device
    w = W5f32.to(dev).view(1, 5, 1)
    w3 = float(W5f32[3])
    w4 = float(W5f32[4])
    t1 = g[:, :, 1, :].sum(dim=1)  # (B,nx)
    tn2 = g[:, :, -2, :].sum(dim=1)
    t_bot = 2.0 * T_HOT - t1
    t_top = 2.0 * T_COLD - tn2
    neqb = g[:, 3, 1, :] - w3 * t1
    neqt = g[:, 4, -2, :] - w4 * tn2
    g = g.clone()
    g[:, :, 0, :] = w * t_bot.unsqueeze(1)
    g[:, 3, 0, :] = g[:, 3, 0, :] + neqb
    g[:, 0, 0, :] = g[:, 0, 0, :] - neqb
    g[:, :, -1, :] = w * t_top.unsqueeze(1)
    g[:, 4, -1, :] = g[:, 4, -1, :] + neqt
    g[:, 0, -1, :] = g[:, 0, -1, :] - neqt
    return g


def run_level(ny, fracs, out_prefix="f"):
    lev = next(x for x in REF["levels"] if x["ny"] == ny)
    nx, d, k = lev["nx"], lev["d"], lev["k"]
    spt = lev["steps_per_tstar"]
    arms = []
    for frac in fracs:
        a = next(x for x in lev["arms"] if abs(x["frac"] - frac) < 1e-9)
        gref = a["gamma_ref"]
        n = (
            int(6 * spt)
            if gref == 0.0
            else int(min(max(math.ceil(3.0 * spt / abs(gref)), 4000), 6 * spt))
        )
        arms.append(
            {
                "frac": frac,
                "Ra": a["Ra"],
                "gref": gref,
                "N": n,
                "every": max(1, n // 600),
                "gb": a["Ra"] * R.NU * R.ALPHA / d**3,
            }
        )
    npz = dict(np.load(BASE / "ref" / f"eig_ny{ny}.npz"))
    device = torch.device("cuda")
    torch.set_default_dtype(torch.float64)
    B = len(arms)

    # base state + seeds (per arm, stacked)
    Tbar = R.base_state(ny, nx, device, "half").double()  # (ny,nx)
    ths, uxs, uys = [], [], []
    for a in arms:
        tag = f"ny{ny}/arm{a['frac']}"
        shape = {
            kk: npz[f"{tag}/{kk}"].tolist() for kk in ("Theta_node", "Phi_node", "Phider_node")
        }
        th_s, ux_s, uy_s = R.seed_fields(ny, nx, device, AMP, "eigen", shape, "half")
        ths.append(th_s)
        uxs.append(ux_s)
        uys.append(uy_s)
    th_s = torch.stack(ths)
    ux_s = torch.stack(uxs)
    uy_s = torch.stack(uys)
    f = equilibrium9(torch.ones((B, ny, nx), device=device), ux_s, uy_s)
    g = equilibrium5(Tbar.unsqueeze(0) + th_s, ux_s, uy_s)
    wall = torch.zeros((1, ny, nx), dtype=torch.bool, device=device)
    wall[:, 0, :] = True
    wall[:, -1, :] = True
    m0 = torch.tensor(
        [float(fi.sum()) for fi in f], device=device
    )  # per-arm 3D sum = serial kernel
    gb_b = torch.tensor([a["gb"] for a in arms], device=device).view(B, 1, 1)
    Ths = [
        torch.tensor(npz[f"ny{ny}/arm{a['frac']}/Theta_node"], dtype=torch.float64, device=device)
        for a in arms
    ]
    xs = torch.arange(nx, device=device, dtype=torch.float64)
    cosk = torch.cos(k * xs).view(1, 1, -1)
    # EXACT serial projection arithmetic (rb_lbm.project_amplitudes):
    # nT = float((Th**2 * cos**2).sum()) -- same elementwise+reduction order.
    nrm = [max(float((t.view(ny, 1) ** 2 * cosk**2).sum()), 1e-300) for t in Ths]

    for a in arms:
        a["steps"], a["amp"] = [], []
    t0 = time.time()
    step = 0
    active = list(range(B))
    order = sorted(range(B), key=lambda i: arms[i]["N"])
    seg_ends = sorted({arms[i]["N"] for i in order})
    prev = 0
    for seg in seg_ends:
        while step < seg:
            th_now = g.sum(dim=1) - Tbar.unsqueeze(0)
            if any(step % arms[i]["every"] == 0 for i in active):
                for p, i in enumerate(active):
                    if step % arms[i]["every"] == 0:
                        aT = float((th_now[p] * Ths[i].view(ny, 1) * cosk).sum()) / nrm[i]
                        arms[i]["steps"].append(step)
                        arms[i]["amp"].append(aT)
            rho, ux, uy = macroscopic9(f)
            F = torch.zeros((f.shape[0], 2, ny, nx), device=device)
            F[:, 1] = gb_b * th_now
            geq = equilibrium5(g.sum(dim=1), ux, uy)
            g = g - (g - geq) / TAU_T
            g = streamq(g, C5)
            g = temp_bc_half(g)
            f_pre = f
            f = collide_force9(f, TAU, F)
            f = torch.where(wall.unsqueeze(1), f_pre[:, OPP9.to(device)], f)
            f = streamq(f, C9)
            s_mass = torch.stack([fi.sum() for fi in f])  # per-arm 3D sum = serial kernel
            f = f * (m0.view(-1, 1, 1, 1) / s_mass.view(-1, 1, 1, 1))
            step += 1
        th_now = g.sum(dim=1) - Tbar.unsqueeze(0)
        rho_fin, ux_fin, uy_fin = macroscopic9(f)
        s_fin = torch.stack([fi.sum() for fi in f])
        th_fin = g.sum(dim=1) - Tbar.unsqueeze(0)
        wall_now = time.time() - t0
        for i in [i for i in active if arms[i]["N"] == seg]:
            p = active.index(i)
            dm = float(s_fin[p] - m0[p])
            if seg % arms[i]["every"] == 0:  # serial samples only at multiples of every
                aT = float((th_now[p] * Ths[i].view(ny, 1) * cosk).sum()) / nrm[i]
                arms[i]["steps"].append(step)
                arms[i]["amp"].append(aT)
            arms[i]["health_extra"] = {
                "mass_drift_abs": dm,
                "mass_drift_rel": dm / float(m0[p]),
                "u_max": float(ux_fin[p].abs().max()),
                "v_max": float(uy_fin[p].abs().max()),
                "theta_min": float(th_fin[p].min()),
                "theta_max": float(th_fin[p].max()),
            }
            finalize(arms[i], ny, d, out_prefix, wall_s=wall_now)
        keep_pos = [p for p, i in enumerate(active) if arms[i]["N"] > seg]
        if len(keep_pos) != len(active):
            active = [active[p] for p in keep_pos]
            if keep_pos:
                idx = torch.tensor(keep_pos, device=device, dtype=torch.long)
                f = f[idx].contiguous()
                g = g[idx].contiguous()
                m0 = m0[idx].clone()
                gb_b = gb_b[idx].clone()
    print(f"BATCH_LEVEL_ny{ny}_DONE wall={time.time() - t0:.0f}s", flush=True)


def finalize(a, ny, d, out_prefix, wall_s):
    st = np.array(a["steps"], dtype=float)
    z = np.log(np.abs(np.array(a["amp"])))
    slopes, gammas = [], []
    n = len(st)
    for lo, hi in WINDOWS:
        i0, i1 = int(lo * n), max(int(hi * n), int(lo * n) + 5)
        sl = float(np.polyfit(st[i0:i1], z[i0:i1], 1)[0])
        slopes.append(sl)
        gammas.append(sl * d * d / R.ALPHA)
    spread = (max(slopes) - min(slopes)) / abs(np.mean(slopes)) if np.mean(slopes) else 0.0
    gamma_sim = float(np.mean(gammas))
    af, al = float(a["amp"][0]), float(a["amp"][-1])
    e_folds = abs(math.log(abs(al / af))) if af and al else float("inf")
    out = {
        "ny": ny,
        "frac": a["frac"],
        "Ra": a["Ra"],
        "g_beta": a["gb"],
        "amp": AMP,
        "dtype": "float64",
        "mass_correct_every_step": True,
        "runner": "batched",
        "steps": a["N"],
        "sample_every": a["every"],
        "windows": [list(w) for w in WINDOWS],
        "slopes_per_step": slopes,
        "gammas_windows": gammas,
        "slope_spread_rel": spread,
        "gamma_sim": gamma_sim,
        "gamma_ref": a["gref"],
        "rel_err": (gamma_sim / a["gref"] - 1.0) if a["gref"] else None,
        "report_only": a["frac"] == 1.0,
        "health": {"a_first": af, "a_last": al, "e_folds": e_folds, **a["health_extra"]},
        "wall_seconds": wall_s,
        "abort": None,
        "seed_rows": "npz row 0 = lower ghost mirror; rows 1..d = fluid; last = upper ghost (E3a)",
    }
    (OUTDIR / f"{out_prefix}_ny{ny}_arm{a['frac']}.json").write_text(json.dumps(out, indent=1))
    rel = out["rel_err"]
    print(
        f"[batched] ny={ny} arm={a['frac']} N={a['N']} g_sim={gamma_sim:+.6f} "
        f"rel={(100 * rel if rel is not None else float('nan')):+.4f}% spread={spread:.4f} efolds={e_folds:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    import os

    levels = [int(x) for x in sys.argv[1:]] or [128, 256]
    out_prefix = os.environ.get("OUT_PREFIX", "f")
    for ny in levels:
        run_level(ny, ALL_ARMS, out_prefix)
    print("BATCHED_DONE", flush=True)
