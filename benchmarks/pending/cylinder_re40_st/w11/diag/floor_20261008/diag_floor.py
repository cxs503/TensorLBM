"""W11-A floor diagnosis 2026-10-08 — DIAGNOSTIC ONLY, never touches formal/.

Exact clone of src/run_formal.py (frozen prereg protocol §2/§4/§6: area-lock
mask, tau = 0.5+3*nu with nu = u_lb*D_eff/Re, plateau criterion from 60 t_c,
Cd = mean last 20 t_c, cap 150 t_c) with ONLY the §3 domain (LX_D/UP_D/H_D)
and optionally §5 u_lb parameterized, to measure residual domain/Ma
sensitivity of the frozen family configuration against the formal tier.

Baseline for comparison: formal/re40_D40.json (LX=40D UP=10D H=20D
u=0.05, cd=1.5852098888020207, err +5.82%).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/nfs/wangxi/worktrees/bm_w11/src")
from tensorlbm.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402

BASE = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady"
SPONGE_D, SPONGE_ALPHA = 10.0, 10.0  # §2 frozen sponge
PLATEAU_REL = 0.0015  # §6 frozen
CAP_TC = 150.0  # §6 frozen


def area_lock_mask(D: int, nx: int, ny: int, cy: float, cx: float, dev):
    """§4: bisection so R_area = sqrt(N_solid/pi) is closest to D/2.
    Verbatim import-equivalent of run_formal.area_lock_mask (copied to keep
    the diagnostic self-contained; identical math)."""
    from tensorlbm.boundaries import cylinder_mask

    R = D / 2.0
    lo, hi = R - 1.0, R + 1.0
    best = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        n = int(cylinder_mask(nx, ny, cx, cy, mid, dev).sum())
        ra = math.sqrt(n / math.pi)
        if best is None or abs(ra - R) < best[2]:
            best = (mid, n, abs(ra - R), ra)
        if ra > R:
            hi = mid
        else:
            lo = mid
    r_mask = best[0]
    solid = cylinder_mask(nx, ny, cx, cy, r_mask, dev)
    n = int(solid.sum())
    rows = torch.any(solid, dim=1).nonzero().flatten()
    r_row = (int(rows.max()) - int(rows.min()) + 1) / 2.0
    r_area = math.sqrt(n / math.pi)
    return solid, {
        "r_mask": float(r_mask),
        "n_solid": n,
        "R_row": float(r_row),
        "R_area": float(r_area),
        "R_eff": float(r_area),
        "D_eff_cells": 2.0 * float(r_area),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=40.0)
    ap.add_argument("--D", type=int, default=40)
    ap.add_argument("--lx-d", type=float, default=40.0)
    ap.add_argument("--up-d", type=float, default=10.0)
    ap.add_argument("--h-d", type=float, default=20.0)
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--tag", type=str, required=True)
    a = ap.parse_args()
    name = a.tag
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(0)

    nx, ny = int(round(a.lx_d * a.D)), int(round(2 * a.h_d * a.D))
    cx, cy = round(a.up_d * a.D), ny // 2
    solid, ginfo = area_lock_mask(a.D, nx, ny, float(cy), float(cx), dev)
    d_eff = ginfo["D_eff_cells"]
    nu = a.u * d_eff / a.re  # §4 relock
    tau = 0.5 + 3.0 * nu  # §2 family formula
    sigma = make_sponge_strength(
        ny, nx, int(nx - SPONGE_D * a.D), int(SPONGE_D * a.D), power=2.0, device=dev
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)
    dyn_p = 0.5 * a.u**2 * d_eff  # §4 denominator

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    f = equilibrium(rho0, torch.full_like(rho0, a.u), torch.zeros_like(rho0))
    tc = int(round(d_eff / a.u))  # §6
    win = 10 * tc
    max_steps = int(CAP_TC * tc)
    cd_series: list[float] = []
    win_means = []
    plateau_at = None
    stop_reason = "cap_150tc"
    for step in range(1, max_steps + 1):
        f = stream(collide_mrt(f, tau, tau_field=tau_field))
        fx, fy = compute_obstacle_forces(f, solid)
        f = far_field_bc_2d(f, a.u, solid)
        cd = float(fx.item()) / dyn_p
        if not math.isfinite(cd):
            stop_reason = "nan_abort"
            break
        cd_series.append(cd)
        if step % win == 0:
            win_means.append(sum(cd_series[-win:]) / win)
            print(
                f"[{name}] step={step} ({step / tc:.0f} t_c) "
                f"cd_win={win_means[-1]:.6f} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
            if step >= 6 * win:  # §6: check from 60 t_c
                m1, m2 = win_means[-2], win_means[-1]
                if abs(m2 - m1) / abs(m1) < PLATEAU_REL:
                    plateau_at = step
                    stop_reason = "plateau"
                    break

    tail = min(len(cd_series), 20 * tc)
    cd = sum(cd_series[-tail:]) / tail
    ref = json.load(open(f"{BASE}/ref/ref_lock.json"))["locked"]
    cd_ref = ref[f"re{a.re:g}"]["primary"]["value"]
    wall = time.time() - t0
    out = {
        "diagnostic_only": True,
        "tag": name,
        "purpose": "floor diagnosis 2026-10-08: residual domain/Ma sensitivity "
        "of frozen W11-A config; NOT a formal run, never enters gates",
        "re": a.re,
        "D": a.D,
        "u_lb": a.u,
        "ma": a.u * math.sqrt(3.0),
        "tau": tau,
        "nu_lb": nu,
        "nx": nx,
        "ny": ny,
        "lx_d": a.lx_d,
        "up_d": a.up_d,
        "h_d": a.h_d,
        "sponge_d": SPONGE_D,
        "sponge_alpha": SPONGE_ALPHA,
        "cx": cx,
        "cy": cy,
        **ginfo,
        "t_c_steps": tc,
        "steps": len(cd_series),
        "plateau_at": plateau_at,
        "cap_steps": max_steps,
        "stop_reason": stop_reason,
        "cd": cd,
        "cd_last10tc": win_means[-1] if win_means else None,
        "cd_ref_primary": cd_ref,
        "err_pct": (cd / cd_ref - 1.0) * 100.0,
        "wall_s": wall,
        "device": "cuda",
        "protocol": "run_formal.py clone, frozen §2/§4/§6; only §3 domain/u_lb varies",
    }
    outdir = f"{BASE}/diag/floor_20261008"
    with open(f"{outdir}/{name}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez(
        f"{outdir}/trace_{name}.npz",
        cd=np.asarray(cd_series, dtype=np.float64),
        win_means=np.asarray(win_means, dtype=np.float64),
        t_c_steps=tc,
        plateau_at=plateau_at,
    )
    print(json.dumps(out), flush=True)
    sys.exit(2 if stop_reason.startswith("nan") else 0)


if __name__ == "__main__":
    main()
