#!/usr/bin/env python
"""Mission 409: Run startup_poiseuille on SDAA, collect device evidence + criteria."""
from __future__ import annotations
import sys, time, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch_sdaa

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.turbulent_channel import _apply_body_force_2d

DEVICE = torch.device("sdaa")
TSTAR_MARKS = [0.05, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5]
TSTAR_CENTER_MIN = 0.02
CENTER_EVERY = 20
N_ODD = np.arange(1, 1000, 2)

def startup_profile(y_phys, H, nu, a, t):
    u_max = a * H * H / (8.0 * nu)
    yh = y_phys / H
    n = N_ODD.astype(np.float64)
    lam = np.minimum((n * np.pi / H) ** 2 * nu * t, 700.0)
    fac = (1.0 / n**3)[:, None] * np.sin(np.outer(n, yh) * np.pi) * np.exp(-lam)[:, None]
    s = fac.sum(axis=0)
    steady = 4.0 * yh * (1.0 - yh)
    return u_max * (steady - (32.0 / np.pi**3) * s)

def run_case_sdaa(H, tau=0.8, u_max_target=0.03, nx=8, seed=0):
    torch.manual_seed(seed)
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    H_eff = float(ny - 2)
    a = 8.0 * nu * u_max_target / (H_eff * H_eff)
    t_end = int(math.ceil(TSTAR_MARKS[-1] * H_eff * H_eff / nu)) + CENTER_EVERY
    Ma = u_max_target / math.sqrt(1.0 / 3.0)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall[0, :] = True; wall[-1, :] = True
    rho0 = torch.ones((ny, nx), device=DEVICE)
    u0 = torch.zeros((ny, nx), device=DEVICE)
    f = equilibrium(rho0, u0, u0)
    initial_mass = float(f.sum().item())

    a_t = torch.tensor(a, device=DEVICE)
    rows = np.arange(1, ny - 1, dtype=np.float64)
    y_phys = rows - 0.5
    center_row = ny // 2
    mark_steps = {int(round(ts * H_eff * H_eff / nu)): ts for ts in TSTAR_MARKS}
    tstar_center_min_step = int(round(TSTAR_CENTER_MIN * H_eff * H_eff / nu))

    center_hist = []
    snaps = {}
    t0 = time.time()
    for step in range(1, t_end + 1):
        f_pre = f
        f = collide_bgk(f, tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(f.device)], f)
        f = _apply_body_force_2d(f, a_t)
        f = stream(f)
        if step % CENTER_EVERY == 0 and step >= tstar_center_min_step:
            _, ux, _ = macroscopic(f)
            center_hist.append((step, float(ux[center_row, nx // 2].item())))
        if step in mark_steps:
            _, ux, _ = macroscopic(f)
            snaps[step] = ux.mean(dim=1).cpu().numpy().astype(np.float64)[1:ny-1]
    elapsed = time.time() - t0

    # Analysis
    c_err_max = 0.0
    c_rows = []
    for step, u_c in center_hist:
        u_ref = float(startup_profile(np.array([H_eff / 2.0]), H_eff, nu, a, float(step))[0])
        err = abs(u_c - u_ref) / u_max_target
        c_err_max = max(c_err_max, err)
        c_rows.append({"step": step, "tstar": step * nu / (H_eff**2),
                        "u_c_num": u_c, "u_c_ana": u_ref, "err_over_umax_pct": err * 100.0})

    prof_rows = []
    l2_list, ptp_list = [], []
    for step, ts in sorted(mark_steps.items()):
        u_num = snaps[step]
        u_ref = startup_profile(y_phys, H_eff, nu, a, float(step))
        l2 = float(np.linalg.norm(u_num - u_ref) / np.linalg.norm(u_ref))
        ptp = float(np.max(np.abs(u_num - u_ref)) / u_max_target)
        l2_list.append(l2); ptp_list.append(ptp)
        prof_rows.append({"step": step, "tstar": ts, "l2_rel": l2, "max_abs_over_umax": ptp})

    return {
        "H": H, "ny": ny, "nx": nx, "tau": tau, "nu_lb": nu, "H_eff": H_eff,
        "a": a, "u_max_final": u_max_target, "Ma": Ma, "t_end": t_end,
        "centerline_err_max_over_umax_pct": c_err_max * 100.0,
        "profile_l2_max_pct": float(np.max(l2_list)) * 100.0,
        "profile_pointwise_max_over_umax_pct": float(np.max(ptp_list)) * 100.0,
        "centerline_table": c_rows,
        "profile_table": prof_rows,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "y_phys": y_phys.tolist(),
    }

if __name__ == "__main__":
    # Device evidence
    dev_info = {
        "torch_version": torch.__version__,
        "torch_sdaa_version": torch_sdaa.__version__,
        "device_name": torch.sdaa.get_device_name(0),
        "device_count": torch.sdaa.device_count(),
        "current_device": torch.sdaa.current_device(),
    }
    print(json.dumps(dev_info, indent=2))

    # Run H=29 (fast) and H=59 on SDAA
    results = {}
    for H in [29, 59]:
        print(f"\n=== Running H={H} on SDAA ===", flush=True)
        t0 = time.time()
        r = run_case_sdaa(H)
        wall_clock = time.time() - t0
        r["wall_clock_s"] = round(wall_clock, 1)
        r["device_evidence"] = dev_info
        results[f"H{H}"] = r
        print(f"H={H}: center_err={r['centerline_err_max_over_umax_pct']:.3f}% "
              f"l2={r['profile_l2_max_pct']:.3f}% ptp={r['profile_pointwise_max_over_umax_pct']:.3f}% "
              f"steps={r['t_end']} elapsed={r['elapsed_s']}s wall={wall_clock:.1f}s")

    # Save
    out = {"device_evidence": dev_info, "cases": results}
    Path("artifacts/m409").mkdir(parents=True, exist_ok=True)
    Path("artifacts/m409/sdaa_run.json").write_text(json.dumps(out, indent=2))
    print("\nSaved to artifacts/m409/sdaa_run.json")
