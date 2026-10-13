#!/usr/bin/env python3
"""Phase-0 couette design-selection arm 2: dtype ladder.

E. fixed-init fp32 H=64 -> confirm return to fp32 floor (closes the
   anti-convergence loop: bug was the H-scaling driver)
F. fixed-init fp64 ladder H in {16,32,64,128} -> TRUE scheme-error H-scaling
   (does the discrete half-way-BB Couette attractor decrease with H once
   fp32 quantization is removed?)  Steady detection per arm: fixed 8*t_D
   schedule + last-4-probe stability check.

Same kernels as diag arms A-D (verbatim moving_wall fn, provenance md5).
"""

from __future__ import annotations

import hashlib
import json
import sys

import torch

WT = "/nfs/wangxi/worktrees/bm_w11"
sys.path.insert(0, f"{WT}/src")

from tensorlbm.d3q19 import OPPOSITE, W_EXACT64, C, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

OUT = "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_dtype.json"
DEVICE = torch.device("cpu")
CS2 = 1.0 / 3.0


def moving_wall_bounce_back3d(f_pre, f, wall, u_wall):
    opp = OPPOSITE.to(DEVICE)
    c = C.to(DEVICE)
    w = W_EXACT64.to(device=DEVICE, dtype=f.dtype)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(19, 1, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(19, 1, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


def run(H, dtype, tau, U0, steps, probe_every, tag, nx=None, nz=None):
    ny = H + 2
    if nx is None:
        nx, nz = 8, 8
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=DEVICE)
    wall[:, 0, :] = True
    wall[:, -1, :] = True
    u_wall = torch.zeros((nz, ny, nx), dtype=dtype, device=DEVICE)
    u_wall[:, -1, :] = U0
    yy = torch.arange(ny, device=DEVICE, dtype=dtype)
    u0 = (
        torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0)
        .view(1, ny, 1)
        .expand(nz, ny, nx)
        .clone()
    )
    ones = torch.ones((nz, ny, nx), dtype=dtype, device=DEVICE)
    zeros = torch.zeros((nz, ny, nx), dtype=dtype, device=DEVICE)
    f = equilibrium3d(ones, u0, zeros, zeros)
    m0 = float(f.sum().item())
    ymid = ny // 2

    def prof_stats(f):
        rho, ux, uy, uz = macroscopic3d(f)
        ana = (U0 * (yy[1 : ny - 1].to(torch.float64) - 0.5) / H_eff).to(dtype)
        u_fl = ux[:, 1 : ny - 1, :]
        u_line = u_fl.to(torch.float64).mean(dim=(0, 2))
        dev = u_line - ana.to(torch.float64)
        return {
            "l2_rel": float(
                torch.sqrt((dev**2).mean()) / torch.sqrt((ana.to(torch.float64) ** 2).mean())
            ),
            "dev_mid": float(dev[ymid - 1]),
            "dev_bot": float(dev[0]),
            "dev_top": float(dev[-1]),
            "mass_drift_pct": (float(f.sum().item()) - m0) / m0 * 100.0,
        }

    rec = {
        "tag": tag,
        "H": H,
        "nx": nx,
        "nz": nz,
        "dtype": str(dtype).replace("torch.", ""),
        "tau": tau,
        "U0": U0,
        "steps": steps,
        "t0": prof_stats(f),
        "probes": [],
    }
    for t in range(1, steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau)
        f = moving_wall_bounce_back3d(f_pre, f, wall, u_wall)
        f = stream3d(f)
        if t % probe_every == 0 or t == steps:
            s = prof_stats(f)
            s["step"] = t
            rec["probes"].append(s)
    return rec


def main() -> None:
    src = f"{WT}/benchmarks/pending/couette_3d/run.py"
    with open(src, "rb") as fh:
        src_md5 = hashlib.md5(fh.read()).hexdigest()
    rep = {"provenance_run_py_md5": src_md5}
    # E: fp32 H64, fixed init -- expect ~1e-5 floor (vs buggy 3.4e-3)
    # t_D = H^2/nu scaling; H64: nu=0.1 -> t_D ~ 4100; 24k steps = ~6 t_D
    rep["E_fp32_H64"] = run(64, torch.float32, 0.8, 0.05, 24000, 4000, "E")
    # F: fp64 ladder. steps ~ 6 t_D each.
    for h, st in ((16, 4000), (32, 10000), (64, 24000), (128, 60000)):
        rep[f"F_fp64_H{h}"] = run(h, torch.float64, 0.8, 0.05, st, max(500, st // 6), f"F{h}")
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=2)
    for k, r in rep.items():
        if not isinstance(r, dict) or "probes" not in r:
            continue
        print(f"== {k} ({r['dtype']}, H={r['H']}) t0_l2={r['t0']['l2_rel']:.3e}")
        for p in r["probes"]:
            print(
                f"   t={p['step']:6d} l2={p['l2_rel']:.4e} mid={p['dev_mid']:+.3e} "
                f"bot={p['dev_bot']:+.3e} top={p['dev_top']:+.3e} mass={p['mass_drift_pct']:+.5f}%"
            )


if __name__ == "__main__":
    main()
