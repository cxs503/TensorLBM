"""W11-A2 formal runs: free-stream steady cylinder, upstream expanded to 40D.

Based on W11-A formal (run_formal.py) with ONLY §3 domain changed:
  W11-A:  LX=40D, UP=10D, H=20D  -> all 6 cases fail (5.43-6.06% error)
  W11-A2: LX=70D, UP=40D, H=20D  -> diagnosis E-D shows +2.236% (passes!)

prereg protocol frozen (same as W11-A):
  §2 solver: collide_mrt(tau, tau_field=sponge) -> stream -> forces
     (post-stream, pre-BB) -> far_field_bc_2d; tau = 0.5 + 3*nu; sponge
     W_sp = 10D power 2, tau_eff = tau*(1 + 10*sigma); free-stream init.
  §3 domain: LX = 70D, UP = 40D, H = 20D  (EXPANDED UPSTREAM)
  §4 geometry: area-equivalent mask (bisection on r_mask so
     R_area = sqrt(N_solid/pi) closest to D/2); nu = u_lb*D_eff/Re;
     Cd denominator 0.5*u_lb^2*D_eff.
  §5 ladder: u_lb = 0.05 fixed; tau follows.
  §6 steady protocol: t_c = D_eff/u_lb steps; adjacent 10-t_c windows,
     |dm|/m < 0.15%, checked from 60 t_c; cap 150 t_c; Cd = mean last
     20 t_c; full trace archived (npz).
  §7 stop-loss: D160 single-run GPU wall > 17 h -> stop, report;
     any NaN -> stop. No corrections of any kind.

Single case per invocation:  python run_formal_w11a2.py --re 40 --D 40
Outputs: formal_w11a2/re{re}_D{D}.json + formal_w11a2/trace_re{re}_D{D}.npz
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

BASE = "/nfs/wangxi/runs/bm_close/cylinder_re40_st/w11a2"
LX_D, UP_D, H_D = 70.0, 40.0, 20.0  # §3 EXPANDED: UP 10D -> 40D, LX 40D -> 70D
SPONGE_D, SPONGE_ALPHA = 10.0, 10.0  # §2 frozen sponge
U_LB = 0.05  # §5 frozen
PLATEAU_REL = 0.0015  # §6 frozen
CAP_TC = 150.0  # §6 frozen
STOP_LOSS_S = 17.0 * 3600.0  # §7 (D160 only)


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
    tau = 0.5 + 3.0 * nu  # §2 family form
    sigma = make_sponge_strength(
        ny, nx, int(nx - SPONGE_D * a.D), int(SPONGE_D * a.D), power=2.0, device=dev
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)
    dyn_p = 0.5 * U_LB**2 * d_eff  # §4 denominator

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full((ny, nx), U_LB, dtype=torch.float32, device=dev)
    uy0 = torch.zeros((ny, nx), dtype=torch.float32, device=dev)
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, uy0)
    del rho0, ux0, uy0

    t_c = max(1, int(round(d_eff / U_LB)))
    cap_steps = int(CAP_TC * t_c)
    check_from = int(60 * t_c)
    check_every = int(10 * t_c)
    last_n = int(20 * t_c)

    cd_hist: list[float] = []
    trace_cd: list[float] = []
    trace_t: list[int] = []
    plateau_at: int | None = None
    stop_reason = "cap"

    for step in range(1, cap_steps + 1):
        if step > STOP_LOSS_S / 0.1:  # rough guard
            break
        # Match original run_formal.py: collide → stream → forces → far_field
        f = stream(collide_mrt(f, tau, tau_field=tau_field))
        fx = compute_obstacle_forces(f, solid)[0].item()
        cd = fx / dyn_p
        cd_hist.append(cd)
        f = far_field_bc_2d(f, U_LB, solid)
        
        # §6 plateau check
        if len(cd_hist) >= last_n:
            mean_cd = float(np.mean(cd_hist[-last_n:]))
            trace_cd.append(mean_cd)
            trace_t.append(step)
            
            if step >= check_from and step % check_every == 0:
                if len(trace_cd) >= 2:
                    m_prev = trace_cd[-2]
                    m_curr = trace_cd[-1]
                    rel = abs(m_curr - m_prev) / max(abs(m_prev), 1e-12)
                    if rel < PLATEAU_REL:
                        plateau_at = step
                        stop_reason = "plateau"
                        break

    if plateau_at is None:
        plateau_at = cap_steps
    cd_final = float(np.mean(cd_hist[-last_n:])) if cd_hist else float("nan")
    wall_s = time.time() - t0

    # Reference values (Fornberg 1980/1985)
    ref_cd = {20.0: 2.0001, 40.0: 1.498}
    cd_ref = ref_cd.get(a.re, float("nan"))
    err_pct = 100.0 * (cd_final - cd_ref) / cd_ref if cd_ref else float("nan")

    out = {
        "case": "cylinder_re40_st_w11a2",
        "re": a.re,
        "D": a.D,
        "u_lb": U_LB,
        "ma": U_LB / math.sqrt(3.0),
        "tau": float(tau),
        "nu_lb": float(nu),
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
        "t_c_steps": t_c,
        "steps": plateau_at,
        "plateau_at": plateau_at,
        "cap_steps": cap_steps,
        "stop_reason": stop_reason,
        "cd": cd_final,
        "cd_last10tc": trace_cd[-1] if trace_cd else float("nan"),
        "cd_ref_primary": cd_ref,
        "err_pct": err_pct,
        "wall_s": wall_s,
        "device": "cuda",
    }

    out_dir = f"{BASE}/formal_w11a2"
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{name}.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=1)

    trace_path = f"{out_dir}/trace_{name}.npz"
    np.savez_compressed(trace_path, t=np.array(trace_t), cd=np.array(trace_cd))

    print(f"[{name}] cd={cd_final:.6f} ref={cd_ref:.4f} err={err_pct:+.4f}% "
          f"steps={plateau_at} t={wall_s:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
