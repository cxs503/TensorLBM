#!/usr/bin/env python3
"""Phase-0 couette diagnosis: decompose the anti-convergent pilot error.

Arms (all CPU, tiny):
  A. as-is skeleton path (z-plane-zero init bug present) at H=16 -- reproduce
  B. fixed init (pure clamped ramp; no u0[0/-1] plane zeroing) at H=16
  C. B + record deviation at t=0 (equilibrium-at-ramp vs analytic) and at
     t=2000/8000/20000 -- transient contamination vs secular drift split
  D. B at H=32 with mass / mid-u / top-u / mean-u tracked every 2000 steps
     to 30000 -- secular drift rate measurement
Also: spanwise std over FLUID rows only (excl wall rows), and nx=nz=16 vs 8
variant at H=16 (arm B) to kill the spanstd mystery.

moving_wall_bounce_back3d copied VERBATIM from
worktree benchmarks/pending/couette_3d/run.py (provenance: git blob of that
file, md5 recorded). Library kernels: d3q19/solver3d only.
"""

from __future__ import annotations

import hashlib
import json
import sys

import torch

WT = "/nfs/wangxi/worktrees/bm_w11"
sys.path.insert(0, f"{WT}/src")
sys.path.insert(0, f"{WT}/benchmarks")

from tensorlbm.d3q19 import OPPOSITE, W_EXACT64, C, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

OUT = "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_arms.json"
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


def run(H, nx, nz, tau, U0, steps, fixed_init, probe_every, tag):
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=DEVICE)
    wall[:, 0, :] = True
    wall[:, -1, :] = True
    u_wall = torch.zeros((nz, ny, nx), device=DEVICE)
    u_wall[:, -1, :] = U0
    yy = torch.arange(ny, device=DEVICE, dtype=torch.float32)
    u0 = (
        torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0)
        .view(1, ny, 1)
        .expand(nz, ny, nx)
        .clone()
    )
    if not fixed_init:
        u0[0, :, :] = 0.0  # skeleton bug: z-planes zeroed (2D indexing lifted)
        u0[-1, :, :] = 0.0
    f = equilibrium3d(
        torch.ones((nz, ny, nx), device=DEVICE),
        u0,
        torch.zeros((nz, ny, nx), device=DEVICE),
        torch.zeros((nz, ny, nx), device=DEVICE),
    )
    m0 = float(f.sum().item())
    ymid = ny // 2

    def step(f):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau)
        f = moving_wall_bounce_back3d(f_pre, f, wall, u_wall)
        return stream3d(f)

    def prof_stats(f):
        rho, ux, uy, uz = macroscopic3d(f)
        ana = U0 * (yy[1 : ny - 1] - 0.5) / H_eff
        u_fl = ux[:, 1 : ny - 1, :]  # (nz, H, nx)
        u_line = u_fl.mean(dim=(0, 2))  # plane mean over z,x
        dev = u_line - ana
        span_fluid = float(u_fl.std(dim=0).mean().item())
        mom = float((rho * ux).sum().item())
        mass = float(f.sum().item())
        return {
            "l2_rel": float(torch.sqrt((dev**2).mean()) / torch.sqrt((ana**2).mean())),
            "max_abs": float(dev.abs().max()),
            "dev_mid": float(dev[ymid - 1]),
            "dev_bot": float(dev[0]),
            "dev_top": float(dev[-1]),
            "span_fluid_std": span_fluid,
            "mass_drift_pct": (mass - m0) / m0 * 100.0,
            "mom_x": mom,
            "uz_max": float(uz[:, 1 : ny - 1, :].abs().max()),
        }

    rec = {
        "tag": tag,
        "H": H,
        "nx": nx,
        "nz": nz,
        "tau": tau,
        "U0": U0,
        "steps": steps,
        "fixed_init": fixed_init,
        "nu": nu,
        "t0": prof_stats(f),
        "probes": [],
    }
    for t in range(1, steps + 1):
        f = step(f)
        if t % probe_every == 0 or t == steps:
            s = prof_stats(f)
            s["step"] = t
            rec["probes"].append(s)
    return rec


def main() -> None:
    src = f"{WT}/benchmarks/pending/couette_3d/run.py"
    with open(src, "rb") as fh:
        src_md5 = hashlib.md5(fh.read()).hexdigest()
    rep = {
        "provenance_run_py_md5": src_md5,
        "moving_wall_fn": "copied verbatim from pending/couette_3d/run.py",
    }
    rep["A_asis_H16"] = run(16, 8, 8, 0.8, 0.05, 6000, False, 1000, "A")
    rep["B_fixed_H16"] = run(16, 8, 8, 0.8, 0.05, 6000, True, 1000, "B")
    rep["B_fixed_H16_nx16"] = run(16, 16, 16, 0.8, 0.05, 6000, True, 1000, "B16")
    rep["C_fixed_H16_long"] = run(16, 8, 8, 0.8, 0.05, 30000, True, 3000, "C")
    rep["D_fixed_H32_long"] = run(32, 8, 8, 0.8, 0.05, 30000, True, 3000, "D")
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=2)
    for k in (
        "A_asis_H16",
        "B_fixed_H16",
        "B_fixed_H16_nx16",
        "C_fixed_H16_long",
        "D_fixed_H32_long",
    ):
        r = rep[k]
        print(f"== {k} (init_fixed={r['fixed_init']}, nx={r['nx']}) ==")
        print("   t=0   :", {kk: f"{vv:.3e}" for kk, vv in r["t0"].items()})
        for p in r["probes"][:: max(1, len(r["proves"] if False else r["probes"]) // 4)]:
            print(
                f"   t={p['step']:6d} l2={p['l2_rel']:.3e} dev_mid={p['dev_mid']:+.3e} "
                f"mass={p['mass_drift_pct']:+.4f}% span={p['span_fluid_std']:.2e} mom={p['mom_x']:.2f}"
            )


if __name__ == "__main__":
    main()
