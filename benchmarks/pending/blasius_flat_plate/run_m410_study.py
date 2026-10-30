#!/usr/bin/env python
"""M410: Blasius flat-plate domain-height & BC study.

Tests:
  (a) Domain height sweep: y1600 → y3200 → y4800 (mid-plate, mirror top/bottom)
  (b) Top BC change: mirror → extrapolation (zero-gradient)
  (c) Bottom-wall configuration: plate at y=0, mirror only at top

Reference: Blasius Cf = 0.6641146724 / sqrt(Re_x)
Target: Cf error ≤ 3%, u_edge/U = 1.00 ± 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import tensorlbm
from tensorlbm.boundaries import zou_he_outlet_pressure
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, collide_trt, stream

SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)

CF_COEFF = 2.0 * 0.3320573362151963

# Grid configs for m410 study
GRIDS = {
    # Baseline: mid-plate, mirror top/bottom
    "plate400_y1600": dict(nx=440, ny=1600, le=20, plate_len=400, probe=200, wall="mid"),
    "plate400_y3200": dict(nx=440, ny=3200, le=20, plate_len=400, probe=200, wall="mid"),
    "plate400_y4800": dict(nx=440, ny=4800, le=20, plate_len=400, probe=200, wall="mid"),
    # Bottom-wall config: plate at y=0, mirror at top only
    "plate400_bot_y800": dict(nx=440, ny=800, le=20, plate_len=400, probe=200, wall="bottom"),
    "plate400_bot_y1600": dict(nx=440, ny=1600, le=20, plate_len=400, probe=200, wall="bottom"),
    "plate400_bot_y3200": dict(nx=440, ny=3200, le=20, plate_len=400, probe=200, wall="bottom"),
}

STATIONS = [30, 60, 90, 120, 150, 180, 240, 300, 340, 380]


def blasius_table(eta_max=10.0, h=0.005):
    def rhs(s):
        f, fp, fpp = s
        return np.array([fp, fpp, -0.5 * f * fpp])

    def rk4(s, dt):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        return s + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    n = int(round(eta_max / h))

    def shoot(fpp0):
        s = np.array([0.0, 0.0, fpp0])
        for _ in range(n):
            s = rk4(s, h)
        return s[1]

    lo, hi = 0.30, 0.37
    flo = shoot(lo) - 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = shoot(mid) - 1.0
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    fpp0 = 0.5 * (lo + hi)

    etas = np.zeros(n + 1)
    fprimes = np.zeros(n + 1)
    s = np.array([0.0, 0.0, fpp0])
    for i in range(1, n + 1):
        s = rk4(s, h)
        etas[i] = i * h
        fprimes[i] = s[1]
    return etas, fprimes, fpp0


def plate_hwbb_mid(f, py, mask):
    f = f.clone()
    f[2, py, :] = torch.where(mask, f[4, py + 1, :], f[2, py, :])
    f[5, py, :-1] = torch.where(mask[:-1], f[7, py + 1, 1:], f[5, py, :-1])
    f[6, py, 1:] = torch.where(mask[1:], f[8, py + 1, :-1], f[6, py, 1:])
    f[4, py, :] = torch.where(mask, f[2, py - 1, :], f[4, py, :])
    f[7, py, 1:] = torch.where(mask[1:], f[5, py - 1, :-1], f[7, py, 1:])
    f[8, py, :-1] = torch.where(mask[:-1], f[6, py - 1, 1:], f[8, py, :-1])
    return f


def plate_hwbb_bottom(f, mask):
    """Half-way BB for bottom wall (y=0)."""
    f = f.clone()
    f[2, 0, :] = torch.where(mask, f[4, 1, :], f[2, 0, :])
    f[5, 0, :-1] = torch.where(mask[:-1], f[7, 1, 1:], f[5, 0, :-1])
    f[6, 0, 1:] = torch.where(mask[1:], f[8, 1, :-1], f[6, 0, 1:])
    return f


def run_case(grid_name, collision, U, nu, n_steps, device_str="cpu",
             avg_steps=200, top_bc="mirror", n_threads=32):
    """Run a single case with configurable domain height and top BC."""
    g = GRIDS[grid_name]
    nx, ny = g["nx"], g["ny"]
    le, plate_len, probe = g["le"], g["plate_len"], g["probe"]
    wall = g["wall"]
    plate_end = le + plate_len

    if wall == "mid":
        py = ny // 2
    else:
        py = 0

    tau = 3.0 * nu + 0.5
    device = torch.device(device_str)
    if device_str == "cpu":
        torch.set_num_threads(n_threads)
    torch.manual_seed(0)

    lam_trt = 3.0 / 16.0
    tau_minus = 0.5 + lam_trt / (tau - 0.5)

    station_cols = [le + xe for xe in STATIONS]

    plate = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    if wall == "mid":
        plate[py, le:plate_end] = True
    else:
        plate[0, le:plate_end] = True
    plate_row = plate[py]

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), U, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0, device=device)
    initial_mass = float(f.sum().item())

    spec = SPEC.to(device)
    feq_in = equilibrium(
        torch.ones((ny, 1), device=device),
        torch.full((ny, 1), U, device=device),
        torch.zeros((ny, 1), device=device),
    )[:, :, 0].contiguous()

    me_cap = {"up": None, "lo": None}

    def apply_collision(f_):
        if collision == "bgk":
            return collide_bgk(f_, tau)
        return collide_trt(f_, tau, lam_trt)

    def step_fn(f_, capture_me=False):
        f_ = apply_collision(f_)
        if capture_me:
            if wall == "mid":
                me_cap["up"] = 2.0 * (
                    torch.roll(f_[8, py + 1, :], 1) - torch.roll(f_[7, py + 1, :], -1)
                )
                me_cap["lo"] = 2.0 * (
                    torch.roll(f_[5, py - 1, :], 1) - torch.roll(f_[6, py - 1, :], -1)
                )
            else:
                me_cap["up"] = 2.0 * (
                    torch.roll(f_[8, 1, :], 1) - torch.roll(f_[7, 1, :], -1)
                )
                me_cap["lo"] = me_cap["up"]  # same for bottom wall
        if wall == "mid":
            f_ = plate_hwbb_mid(f_, py, plate_row)
        else:
            f_ = plate_hwbb_bottom(f_, plate_row)
        f_ = f_.clone()
        # Left BC: mirror (specular)
        f_[:, 0, :] = f_[:, 1, :][spec]
        # Right BC: mirror (specular) - will be overwritten by outlet
        f_[:, -1, :] = f_[:, -2, :][spec]
        f_ = stream(f_)
        f_ = f_.clone()
        # Inlet: equilibrium free-stream
        f_[:, :, 0] = feq_in
        # Outlet: Zou-He pressure
        f_ = zou_he_outlet_pressure(f_, 1.0)
        # Top BC
        if top_bc == "mirror":
            f_[:, -1, :] = f_[:, -2, :][spec]
        elif top_bc == "extrapolation":
            f_[:, -1, :] = f_[:, -2, :]  # zero-gradient
        # Bottom BC for mid-wall: mirror
        if wall == "mid":
            f_[:, 0, :] = f_[:, 1, :][spec]
        # For bottom-wall, y=0 is the plate (handled by BB), no extra BC needed
        return f_

    t0 = time.time()
    umax_hist = []
    for step in range(1, n_steps + 1):
        f = step_fn(f)
        if step % 200 == 0:
            _, ux, _ = macroscopic(f)
            umax_hist.append(float(ux.max().item()))
    elapsed = time.time() - t0

    umax_arr = np.array(umax_hist)
    tail = umax_arr[-10:] if len(umax_arr) >= 10 else umax_arr
    umax_drift = (float(tail.max()) - float(tail.min())) / max(abs(float(tail.mean())), 1e-12)

    # Averaging
    st_t = torch.tensor(station_cols, dtype=torch.int64, device=device)
    prof_acc = torch.zeros((len(STATIONS), ny), device=device)
    me_up_acc = torch.zeros(nx, device=device)
    me_lo_acc = torch.zeros(nx, device=device)
    n_avg = 0
    for _ in range(avg_steps):
        f = step_fn(f, capture_me=True)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, st_t].T
        me_up_acc += me_cap["up"]
        me_lo_acc += me_cap["lo"]
        n_avg += 1
    prof = (prof_acc / n_avg).cpu().numpy()
    me_up = (me_up_acc / n_avg).cpu().numpy()
    me_lo = (me_lo_acc / n_avg).cpu().numpy()

    rho, ux_f, uy_f = macroscopic(f)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0
    finite = bool(torch.isfinite(f).all().item())

    etas, fprimes, fpp0 = blasius_table()

    # Compute diagnostics at probe station
    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)

    iprobe = STATIONS.index(x_eff)
    u_col = prof[iprobe]

    if wall == "mid":
        y_w = py + 0.5
        u_hi = u_col[py + 1:]
        y_hi = np.arange(py + 1, ny, dtype=np.float64) - y_w
    else:
        y_w = 0.5
        u_hi = u_col[1:]
        y_hi = np.arange(1, ny, dtype=np.float64) - y_w

    eta_hi = y_hi / scale
    fp_ref_hi = np.interp(eta_hi, etas, fprimes)
    u_ref_hi = U * fp_ref_hi

    m = (eta_hi > 0.05) & (eta_hi < 5.0) & (fp_ref_hi > 0.02) & (fp_ref_hi < 0.995)
    rel_err = np.abs(u_hi - u_ref_hi) / np.maximum(u_ref_hi, 1e-12)
    l2_rel = float(np.linalg.norm(rel_err[m]) / np.linalg.norm(np.ones(m.sum()))) if m.any() else float("nan")

    # Edge velocity: max in far field
    u_col_far = u_hi[int(40 / 1):] if wall == "mid" else u_hi[40:]
    u_e_local = float(u_col_far.max())

    # Cf at probe
    s_dist = np.array([0.5, 1.5, 2.5])
    A = np.vstack([np.ones(3), s_dist, s_dist**2]).T
    w_der = np.linalg.solve(A.T, np.array([0.0, 1.0, 0.0]))
    u123 = np.array([u_col[py + 1 if wall == "mid" else 1],
                     u_col[py + 2 if wall == "mid" else 2],
                     u_col[py + 3 if wall == "mid" else 3]])
    dudy = float(w_der @ u123)
    Cf_fd = 2.0 * nu * dudy / (U * U)
    Cf_ref = CF_COEFF / math.sqrt(Rex)
    Cf_err_pct = (Cf_fd - Cf_ref) / Cf_ref * 100.0

    # Displacement thickness
    u_edge_for_dstar = max(u_e_local, 1e-12)
    delta_star_meas = float(np.trapezoid(1.0 - u_hi / u_edge_for_dstar, y_hi))
    delta_star_ref = 1.7208 * x_eff / math.sqrt(Rex)

    # u_edge/U at domain top
    if wall == "mid":
        u_edge_top = float(ux_f[ny - 2, probe].item())
    else:
        u_edge_top = float(ux_f[ny - 2, probe].item())

    # Per-station summary
    station_summary = []
    for xe, xcol in zip(STATIONS, station_cols):
        sc = math.sqrt(nu * xe / U)
        Rex_s = U * xe / nu
        ucol = prof[STATIONS.index(xe)]
        u123_s = np.array([ucol[py + 1 if wall == "mid" else 1],
                           ucol[py + 2 if wall == "mid" else 2],
                           ucol[py + 3 if wall == "mid" else 3]])
        dudy_s = float(w_der @ u123_s)
        Cf_fd_s = 2.0 * nu * dudy_s / (U * U)
        Cf_ref_s = CF_COEFF / math.sqrt(Rex_s)
        # local edge
        u_hi_s = ucol[py + 1:] if wall == "mid" else ucol[1:]
        u_e_s = float(u_hi_s[40:].max()) if len(u_hi_s) > 40 else float(u_hi_s.max())
        station_summary.append({
            "x_eff": xe,
            "Rex": round(Rex_s, 1),
            "Cf_fd": round(Cf_fd_s, 6),
            "Cf_ref": round(Cf_ref_s, 6),
            "Cf_err_pct": round((Cf_fd_s - Cf_ref_s) / Cf_ref_s * 100.0, 2),
            "u_edge_local_over_U": round(u_e_s / U, 5),
        })

    result = {
        "case": "M410_blasius_study",
        "grid": grid_name,
        "wall": wall,
        "top_bc": top_bc,
        "collision": collision,
        "lambda_trt": lam_trt if collision == "trt" else None,
        "nx": nx,
        "ny": ny,
        "half_height": ny // 2 if wall == "mid" else ny,
        "U": U,
        "nu": nu,
        "tau": tau,
        "x_eff": x_eff,
        "Rex": Rex,
        "n_steps": n_steps,
        "avg_steps": n_avg,
        "umax_drift": round(umax_drift, 6),
        "Cf_sim": round(Cf_fd, 6),
        "Cf_ref": round(Cf_ref, 6),
        "Cf_err_pct": round(Cf_err_pct, 2),
        "u_edge_local_over_U": round(u_e_local / U, 5),
        "u_edge_top_over_U": round(u_edge_top / U, 5),
        "delta_star_meas": round(delta_star_meas, 4),
        "delta_star_ref": round(delta_star_ref, 4),
        "delta_star_over_H": round(delta_star_meas / (ny // 2 if wall == "mid" else ny), 6),
        "l2_rel_err": round(l2_rel, 4),
        "mass_drift_pct": round(mass_drift_pct, 6),
        "finite": finite,
        "elapsed_s": round(elapsed, 1),
        "stations": station_summary,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=str, default="plate400_y1600",
                    help="Comma-separated grid names")
    ap.add_argument("--collision", default="trt", choices=["bgk", "trt"])
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--avg-steps", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-bc", default="mirror", choices=["mirror", "extrapolation"])
    ap.add_argument("--out", default="/tmp/m410_study.json")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    case_names = [c.strip() for c in args.cases.split(",")]
    results = []
    for cn in case_names:
        if cn not in GRIDS:
            print(f"SKIP: {cn} not in GRIDS", flush=True)
            continue
        print(f"\n=== Running {cn} (top_bc={args.top_bc}) ===", flush=True)
        r = run_case(cn, args.collision, args.U, args.nu, args.steps,
                     device_str=args.device, avg_steps=args.avg_steps,
                     top_bc=args.top_bc, n_threads=args.threads)
        results.append(r)
        print(f"  Cf_sim={r['Cf_sim']:.6f} Cf_ref={r['Cf_ref']:.6f} "
              f"err={r['Cf_err_pct']:+.2f}% u_edge/U={r['u_edge_local_over_U']:.5f} "
              f"δ*/H={r['delta_star_over_H']:.6f} L2={r['l2_rel_err']:.4f} "
              f"elapsed={r['elapsed_s']:.1f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
