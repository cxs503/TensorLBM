"""W11-A formal runs: free-stream steady cylinder, FROZEN prereg protocol.

prereg/prereg.md (frozen body md5 26695108eb7ff58690cbd44a0fee4dd7 + errata
E1-E5) — implemented literally:
  §2 solver: collide_mrt(tau, tau_field=sponge) -> stream -> forces
     (post-stream, pre-BB) -> far_field_bc_2d; tau = 0.5 + 3*nu; sponge
     W_sp = 10D power 2, tau_eff = tau*(1 + 10*sigma); free-stream init.
  §3 domain: LX = 40D, UP = 10D (cylinder centre from inlet), H = 20D.
  §4 geometry: area-equivalent mask (bisection on r_mask so
     R_area = sqrt(N_solid/pi) closest to D/2); nu = u_lb*D_eff/Re;
     Cd denominator 0.5*u_lb^2*D_eff.
  §5 ladder: u_lb = 0.05 fixed; tau follows (E2-corrected values).
  §6 steady protocol: t_c = D_eff/u_lb steps; adjacent 10-t_c windows,
     |dm|/m < 0.15%, checked from 60 t_c; cap 150 t_c; Cd = mean last
     20 t_c; full trace archived (npz).
  §7 stop-loss (E4): D160 single-run GPU wall > 17 h -> stop, report;
     any NaN -> stop. No corrections of any kind.

Single case per invocation:  python run_formal.py --re 40 --D 40
Outputs: formal/re{re}_D{D}.json + formal/trace_re{re}_D{D}.npz
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
    cylinder_mask,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402

BASE = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady"
LX_D, UP_D, H_D = 40.0, 10.0, 20.0  # §3 frozen domain
SPONGE_D, SPONGE_ALPHA = 10.0, 10.0  # §2 frozen sponge
U_LB = 0.05  # §5 frozen
PLATEAU_REL = 0.0015  # §6 frozen
CAP_TC = 150.0  # §6 frozen
STOP_LOSS_S = 17.0 * 3600.0  # §7 + E4 (D160 only)


def area_lock_mask(D: int, nx: int, ny: int, cy: float, cx: float, dev):
    """§4: bisection so R_area = sqrt(N_solid/pi) is closest to D/2."""
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
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--D", type=int, required=True)
    a = ap.parse_args()
    name = f"re{a.re:g}_D{a.D}"
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(0)

    nx, ny = int(round(LX_D * a.D)), int(round(2 * H_D * a.D))
    cx, cy = round(UP_D * a.D), ny // 2
    solid, ginfo = area_lock_mask(a.D, nx, ny, float(cy), float(cx), dev)
    d_eff = ginfo["D_eff_cells"]
    nu = U_LB * d_eff / a.re  # §4 relock
    tau = 0.5 + 3.0 * nu  # §2 family formula
    sigma = make_sponge_strength(
        ny, nx, int(nx - SPONGE_D * a.D), int(SPONGE_D * a.D), power=2.0, device=dev
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)
    dyn_p = 0.5 * U_LB**2 * d_eff  # §4 denominator

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    f = equilibrium(rho0, torch.full_like(rho0, U_LB), torch.zeros_like(rho0))
    tc = int(round(d_eff / U_LB))  # §6
    win = 10 * tc
    max_steps = int(CAP_TC * tc)
    cd_series: list[float] = []
    win_means = []
    plateau_at = None
    stop_reason = "cap_150tc"
    for step in range(1, max_steps + 1):
        f = stream(collide_mrt(f, tau, tau_field=tau_field))
        fx, fy = compute_obstacle_forces(f, solid)
        f = far_field_bc_2d(f, U_LB, solid)
        cd = float(fx.item()) / dyn_p
        if not math.isfinite(cd):  # §7/E4 NaN clause
            stop_reason = "nan_abort"
            break
        cd_series.append(cd)
        if step % win == 0:
            win_means.append(sum(cd_series[-win:]) / win)
            if not bool(torch.isfinite(f).all()):
                stop_reason = "nan_abort_field"
                break
            elapsed = time.time() - t0
            print(
                f"[{name}] step={step} ({step / tc:.0f} t_c) "
                f"cd_win={win_means[-1]:.6f} elapsed={elapsed / 3600:.2f}h",
                flush=True,
            )
            if a.D == 160 and elapsed > STOP_LOSS_S:  # §7 + E4
                stop_reason = "stop_loss_17h"
                break
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
        "name": name,
        "re": a.re,
        "D": a.D,
        "u_lb": U_LB,
        "ma": U_LB * math.sqrt(3.0),
        "tau": tau,
        "nu_lb": nu,
        "nx": nx,
        "ny": ny,
        "lx_d": LX_D,
        "up_d": UP_D,
        "h_d": H_D,
        "sponge_d": SPONGE_D,
        "sponge_alpha": SPONGE_ALPHA,
        "cx": cx,
        "cy": cy,
        **ginfo,
        "g3_relock": {
            "nu_from_D_eff": nu,
            "Re_check": U_LB * d_eff / nu,
            "D_eff_source": "R_area = sqrt(N_solid/pi), machine chain",
        },
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
        "cell_steps_per_s": nx * ny * len(cd_series) / wall,
        "prereg": "frozen body md5 26695108eb7ff58690cbd44a0fee4dd7 + E1-E5",
    }
    with open(f"{BASE}/formal/{name}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez(
        f"{BASE}/formal/trace_{name}.npz",
        cd=np.asarray(cd_series, dtype=np.float64),
        win_means=np.asarray(win_means, dtype=np.float64),
        t_c_steps=tc,
        plateau_at=plateau_at,
    )
    print(json.dumps(out), flush=True)
    sys.exit(2 if stop_reason.startswith("nan") else 0)


if __name__ == "__main__":
    main()
