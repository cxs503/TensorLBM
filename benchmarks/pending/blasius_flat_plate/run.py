#!/usr/bin/env python
"""W7-D: Blasius flat-plate boundary layer revival — BGK baseline vs TRT Lambda=3/16.

Old record (benchmarks/pending/blasius_flat_plate, 2026-08-19): C_f error
+102% (plate200) / +76% (plate400) at tau=0.53 BGK; tau-scan 0.53->0.80
monotonically worsened (-> +127%). Suspected root cause: BGK bounce-back
wall-position tau coupling. Fix under test: TRT collision with magic
parameter Lambda = 3/16 (library collide_trt, solver.py, Ginzburg 2008),
tau_minus = 0.5 + Lambda/(tau_plus-0.5) = 6.75 at tau_plus=0.53.

Geometry/protocol: EXACT replica of the old run.py (mid-domain one-row thin
plate, fluid-side-reflection half-way BB on both surfaces, feq free-stream
inlet, library zou_he_outlet_pressure outlet, mirror ghost rows top/bottom,
U=0.05, nu=0.01, 30k steps + 200-step averaging).

Additions vs old run.py (measurement only, no physics change):
  * --collision bgk|trt switch (both via library collide_* entries)
  * multi-station C_f(x): FD 3-point one-sided gradient (old primary
    method) AND momentum-exchange from the bounce populations (cross-check)
  * linear-extrapolation slip diagnostic u(0) at main probe
  * eta-profile L2 / first-cell behaviour as in old record

Reference (pre-registered, locked before formal runs):
  Blasius ODE self-solved (RK4 shooting, h=0.005), f''(0)=0.3320573362,
  C_f_ref = 2 f''(0)/sqrt(Re_x) = 0.6641146724/sqrt(Re_x)
  (old-record convention 0.664 differs by 0.017%).

Usage:
  python run.py <out.json> --grid plate200 \
      --collision trt --U 0.05 --nu 0.01 --steps 30000 --device cpu
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # <repo>/src

import tensorlbm
from tensorlbm.boundaries import zou_he_outlet_pressure
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, collide_trt, stream

# Mirror (specular) direction map for symmetry BC: cy -> -cy.
# D2Q9: 0:(0,0) 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1) 5:(1,1) 6:(-1,1) 7:(-1,-1) 8:(1,-1)
SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)

GRIDS = {
    "plate200": dict(nx=240, ny=1400, le=20, plate_len=200, probe=200),
    "plate400": dict(nx=440, ny=1400, le=20, plate_len=400, probe=200),
    "plate400_y1600": dict(nx=440, ny=1600, le=20, plate_len=400, probe=200),
}

# C_f stations in x_eff (= x - le): shared core + plate400-only tail.
STATIONS_CORE = [30, 60, 90, 120, 150, 180]
STATIONS_TAIL = [240, 300, 340, 380]

CF_COEFF = 2.0 * 0.3320573362151963  # 0.6641146724303926, locked reference
CF_COEFF_OLD = 0.664  # old-record convention (disclosure only)


def blasius_table(eta_max: float = 10.0, h: float = 0.005):
    """Solve f''' + 0.5*f*f'' = 0, f(0)=f'(0)=0, f'(inf)=1 (RK4 + bisection)."""

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


def plate_hwbb_mid(f: torch.Tensor, py: int, mask: torch.Tensor) -> torch.Tensor:
    """Fluid-side-reflection half-way BB for the thin plate row y=py.

    Verbatim from the old record run.py (upper + lower surfaces; the solid
    row is built from adjacent fluid rows post-collision with the correct
    opposite mapping including x-shifts, then streamed).
    """
    f = f.clone()
    # upper surface (fluid above, wall at py+0.5)
    f[2, py, :] = torch.where(mask, f[4, py + 1, :], f[2, py, :])
    f[5, py, :-1] = torch.where(mask[:-1], f[7, py + 1, 1:], f[5, py, :-1])
    f[6, py, 1:] = torch.where(mask[1:], f[8, py + 1, :-1], f[6, py, 1:])
    # lower surface (fluid below, wall at py-0.5)
    f[4, py, :] = torch.where(mask, f[2, py - 1, :], f[4, py, :])
    f[7, py, 1:] = torch.where(mask[1:], f[5, py - 1, :-1], f[7, py, 1:])
    f[8, py, :-1] = torch.where(mask[:-1], f[6, py - 1, 1:], f[8, py, :-1])
    return f


def run_case(
    grid_name, collision, U, nu, n_steps, out_path, device_str="cpu", avg_steps=200, n_threads=32
):
    g = GRIDS[grid_name]
    nx, ny = g["nx"], g["ny"]
    le, plate_len, probe = g["le"], g["plate_len"], g["probe"]
    plate_end = le + plate_len
    py = ny // 2
    tau = 3.0 * nu + 0.5  # tau_plus for TRT (nu set by symmetric part)
    device = torch.device(device_str)
    if device_str == "cpu":
        torch.set_num_threads(n_threads)
    torch.manual_seed(0)

    lam_trt = 3.0 / 16.0
    tau_minus = 0.5 + lam_trt / (tau - 0.5)

    stations = list(STATIONS_CORE)
    if grid_name.startswith("plate400"):
        stations += STATIONS_TAIL
    station_cols = [le + xe for xe in stations]

    plate = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    plate[py, le:plate_end] = True
    plate_row = plate[py]  # (nx,)

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
    row_far1, row_far2 = py + 200, py + 600  # far-field overshoot rows

    def apply_collision(f_):
        if collision == "bgk":
            return collide_bgk(f_, tau)
        return collide_trt(f_, tau, lam_trt)

    def step_fn(f_, capture_me=False):
        f_ = apply_collision(f_)
        if capture_me:
            # ME wall shear = x-momentum exchange of the bounced links (raw,
            # no equilibrium subtraction: still fluid gives f8 == f7 -> 0):
            #   up(x) = 2*[f8(py+1,x-1) - f7(py+1,x+1)]
            #   lo(x) = 2*[f5(py-1,x-1) - f6(py-1,x+1)]
            # roll(a,-1)[x] = a[x+1]; roll(a,1)[x] = a[x-1] (wrap unread).
            me_cap["up"] = 2.0 * (
                torch.roll(f_[8, py + 1, :], 1) - torch.roll(f_[7, py + 1, :], -1)
            )
            me_cap["lo"] = 2.0 * (
                torch.roll(f_[5, py - 1, :], 1) - torch.roll(f_[6, py - 1, :], -1)
            )
        f_ = plate_hwbb_mid(f_, py, plate_row)
        f_ = f_.clone()
        f_[:, 0, :] = f_[:, 1, :][spec]
        f_[:, -1, :] = f_[:, -2, :][spec]
        f_ = stream(f_)
        f_ = f_.clone()
        f_[:, :, 0] = feq_in
        f_ = zou_he_outlet_pressure(f_, 1.0)
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

    # averaging window: station columns of ux + per-column ME + far rows
    st_t = torch.tensor(station_cols, dtype=torch.int64, device=device)
    prof_acc = torch.zeros((len(station_cols), ny), device=device)
    me_up_acc = torch.zeros(nx, device=device)
    me_lo_acc = torch.zeros(nx, device=device)
    row_far_acc = torch.zeros((2, nx), device=device)
    n_avg = 0
    for _ in range(avg_steps):
        f = step_fn(f, capture_me=True)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, st_t].T
        me_up_acc += me_cap["up"]
        me_lo_acc += me_cap["lo"]
        row_far_acc[0] += ux[row_far1]
        row_far_acc[1] += ux[row_far2]
        n_avg += 1
    prof = (prof_acc / n_avg).cpu().numpy()  # (n_station, ny)
    me_up = (me_up_acc / n_avg).cpu().numpy()  # (nx,)
    me_lo = (me_lo_acc / n_avg).cpu().numpy()
    row_far = (row_far_acc / n_avg).cpu().numpy()  # (2, nx) u/U at rows py+200, py+600

    rho, ux_f, uy_f = macroscopic(f)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0
    finite = bool(torch.isfinite(f).all().item())

    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    etas, fprimes, fpp0 = blasius_table()

    iprobe = stations.index(x_eff)
    u_col = prof[iprobe]
    y_w = py + 0.5
    y_hi = np.arange(py + 1, ny, dtype=np.float64) - y_w
    eta_hi = y_hi / scale
    fp_ref_hi = np.interp(eta_hi, etas, fprimes)
    u_ref_hi = U * fp_ref_hi
    u_hi = u_col[py + 1 :]

    m = (eta_hi > 0.05) & (eta_hi < 5.0) & (fp_ref_hi > 0.02) & (fp_ref_hi < 0.995)
    rel_err = np.abs(u_hi - u_ref_hi) / np.maximum(u_ref_hi, 1e-12)
    l2_rel = (
        float(np.linalg.norm(rel_err[m]) / np.linalg.norm(np.ones(m.sum())))
        if m.any()
        else float("nan")
    )
    max_rel_pct = float(rel_err[m].max() * 100.0) if m.any() else float("nan")

    u_lo = u_col[:py]
    sym_err_pct = float(
        np.max(np.abs(u_lo[: py - 1][::-1] - u_hi[: py - 1]) / max(U, 1e-12)) * 100.0
    )
    u_edge_probe = float(ux_f[ny - 2, probe].item())

    tab = []
    for eta_t in [1.0, 2.0, 3.0, 4.0, 5.0]:
        y_t = eta_t * scale + y_w
        u_interp = float(np.interp(y_t, np.arange(py + 1, ny, dtype=np.float64), u_hi))
        fp_ref = float(np.interp(eta_t, etas, fprimes))
        tab.append(
            {
                "eta": eta_t,
                "y_row": round(y_t, 3),
                "u_sim_over_U": round(u_interp / U, 5),
                "fprime_ref": round(fp_ref, 5),
                "rel_err_pct": round(abs(u_interp / U - fp_ref) / fp_ref * 100.0, 3),
            }
        )

    # ---- per-station C_f (FD 3-point one-sided) + ME ----
    # Measurement-formula fix (2026-09-22, see NOTES): the old record (and the
    # baseline reproduction R1) used np.linalg.solve(A, [0,1,0]) which solves
    # A.w = e2; the derivative functional requires A^T.w = e2. For s =
    # (0.5, 1.5, 2.5): correct weights (-2, 3, -1), buggy (-1.25, 3, -1);
    # buggy dudy = correct + 0.75*u(0.5). Both reported.
    s_dist = np.array([0.5, 1.5, 2.5])
    A = np.vstack([np.ones(3), s_dist, s_dist**2]).T
    w_der = np.linalg.solve(A.T, np.array([0.0, 1.0, 0.0]))
    w_der_old = np.linalg.solve(A, np.array([0.0, 1.0, 0.0]))
    station_rows = []
    for xe, xcol in zip(stations, station_cols):
        sc = math.sqrt(nu * xe / U)
        Rex_s = U * xe / nu
        ucol = prof[stations.index(xe)]
        u123 = np.array([ucol[py + 1], ucol[py + 2], ucol[py + 3]])
        dudy = float(w_der @ u123)
        Cf_fd = 2.0 * nu * dudy / (U * U)
        Cf_ref_s = CF_COEFF / math.sqrt(Rex_s)
        me_x = 0.5 * (me_up[xcol] + me_lo[xcol])  # mean of both surfaces
        Cf_me = 2.0 * float(me_x) / (U * U)
        # slip diagnostic (main probe only, from linear fit through s=1.5..5.5)
        slip_u0 = None
        wall_shift = None
        if xe == x_eff:
            sf = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
            uf = np.array([ucol[py + 2], ucol[py + 3], ucol[py + 4], ucol[py + 5], ucol[py + 6]])
            pf = np.polyfit(sf, uf, 1)
            slip_u0 = float(pf[1])
            wall_shift = float(slip_u0 / max(dudy, 1e-30))
        first_cell_fp_ref = float(np.interp(0.5 / sc, etas, fprimes))
        # edge velocity used by the local BL: max of column outside BL rows
        u_col_far = ucol[py + 40 :]
        u_e_local = float(u_col_far.max())
        dudy_old = float(w_der_old @ u123)
        station_rows.append(
            {
                "x_eff": xe,
                "x_col": xcol,
                "Rex": round(Rex_s, 1),
                "scale": round(sc, 4),
                "Cf_fd": Cf_fd,
                "Cf_me": Cf_me,
                "Cf_ref": Cf_ref_s,
                "Cf_fd_err_pct": (Cf_fd - Cf_ref_s) / Cf_ref_s * 100.0,
                "Cf_fd_err_pct_old_buggy_weights": (2.0 * nu * dudy_old / (U * U) - Cf_ref_s)
                / Cf_ref_s
                * 100.0,
                "Cf_me_err_pct": (Cf_me - Cf_ref_s) / Cf_ref_s * 100.0,
                "dudy_fd": dudy,
                "dudy_fd_old_buggy": dudy_old,
                "me_taux": float(me_x),
                "u_edge_local_over_U": round(u_e_local / U, 5),
                "Cf_fd_norm_ue_err_pct": (
                    (2.0 * nu * dudy / (u_e_local**2)) - CF_COEFF / math.sqrt(Rex_s * u_e_local / U)
                )
                / (CF_COEFF / math.sqrt(Rex_s * u_e_local / U))
                * 100.0,
                "first_rows_u_over_U": [round(float(ucol[py + 1 + i]) / U, 5) for i in range(12)],
                "first_cell_u_over_U": float(ucol[py + 1] / U),
                "first_cell_eta": round(0.5 / sc, 4),
                "first_cell_rel_err_pct": abs(float(ucol[py + 1]) / U - first_cell_fp_ref)
                / first_cell_fp_ref
                * 100.0,
                "slip_u0": slip_u0,
                "eff_wall_shift_cells": wall_shift,
            }
        )

    main_st = station_rows[iprobe]
    Cf_sim = main_st["Cf_fd"]
    Cf_ref = main_st["Cf_ref"]
    Cf_err_pct = main_st["Cf_fd_err_pct"]

    delta_star_meas = float(np.trapezoid(1.0 - u_hi / max(u_edge_probe, 1e-12), y_hi))
    delta_star_ref = 1.7208 * x_eff / math.sqrt(Rex)

    profile_rows = []
    for i, eta_v in enumerate(eta_hi):
        if i % 5 == 0:  # subsample: every 5th row (dy=5 -> eta step 5/6*...)
            profile_rows.append(
                {
                    "y": int(py + 1 + i),
                    "eta": round(float(eta_v), 4),
                    "u_sim": round(float(u_hi[i]), 7),
                    "u_blasius": round(float(u_ref_hi[i]), 7),
                    "rel_err_pct": round(float(rel_err[i]) * 100.0, 3),
                }
            )

    result = {
        "case": "W7D_blasius_flat_plate_trt_revival",
        "grid": grid_name,
        "lattice": "D2Q9",
        "collision": collision,
        "lambda_trt": lam_trt if collision == "trt" else None,
        "tau_minus": tau_minus if collision == "trt" else None,
        "boundary": "equilibrium free-stream inlet + Zou-He pressure outlet (rho=1) + mirror(slip) top/bottom + mid-domain thin plate (fluid-side-reflection half-way BB, both surfaces)",
        "extrap": "none",
        "tensorlbm_file": tensorlbm.__file__,
        "nx": nx,
        "ny": ny,
        "plate_y": py,
        "plate_len": plate_len,
        "U": U,
        "nu": nu,
        "tau": tau,
        "x0_leading_edge": le,
        "x_probe": probe,
        "x_eff": x_eff,
        "Rex": Rex,
        "Ma": U / math.sqrt(1.0 / 3.0),
        "n_steps": n_steps,
        "avg_steps": n_avg,
        "umax_drift_last_2000": umax_drift,
        "umax_history": [round(float(v), 6) for v in umax_arr[::5]],
        "blasius_fpp0_shooting": fpp0,
        "blasius_fpp0_ref": 0.3320573362151963,
        "cf_coeff_locked": CF_COEFF,
        "cf_coeff_old_convention": CF_COEFF_OLD,
        "l2_rel_err_profile": l2_rel,
        "max_rel_err_profile_pct": max_rel_pct,
        "eta_tab_comparison": tab,
        "u_edge_over_U_probe": round(u_edge_probe / U, 5),
        "symmetry_max_dev_over_U_pct": sym_err_pct,
        "delta_star_meas": round(delta_star_meas, 4),
        "delta_star_ref_1p7208_x_over_sqrtRex": round(delta_star_ref, 4),
        "Cf_sim": Cf_sim,
        "Cf_ref_locked": Cf_ref,
        "Cf_err_pct": Cf_err_pct,
        "Cf_err_pct_old_convention": (Cf_sim - CF_COEFF_OLD / math.sqrt(Rex))
        / (CF_COEFF_OLD / math.sqrt(Rex))
        * 100.0,
        "stations": station_rows,
        "far_rows_u_over_U": {
            "rows": [row_far1, row_far2],
            "x_cols": list(range(0, nx, 4)),
            "u_over_U_row1": [round(float(v) / U, 6) for v in row_far[0][::4]],
            "u_over_U_row2": [round(float(v) / U, 6) for v in row_far[1][::4]],
        },
        "mass_drift_pct": mass_drift_pct,
        "finite": finite,
        "elapsed_s": round(elapsed, 1),
        "profile_subsampled": profile_rows,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="W7-D Blasius revival: BGK baseline vs TRT 3/16")
    ap.add_argument("out_json", type=str)
    ap.add_argument("--grid", type=str, default="plate200", choices=list(GRIDS))
    ap.add_argument("--collision", type=str, default="trt", choices=["bgk", "trt"])
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--avg-steps", type=int, default=200)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    repo_src = str(Path(__file__).resolve().parents[3] / "src")
    assert tensorlbm.__file__.startswith(repo_src), f"library hijack: {tensorlbm.__file__}"
    steps = args.smoke if args.smoke > 0 else args.steps
    r = run_case(
        args.grid,
        args.collision,
        args.U,
        args.nu,
        steps,
        args.out_json,
        args.device,
        avg_steps=args.avg_steps,
        n_threads=args.threads,
    )
    keys = [
        "grid",
        "collision",
        "tau",
        "tau_minus",
        "x_eff",
        "Rex",
        "n_steps",
        "umax_drift_last_2000",
        "l2_rel_err_profile",
        "max_rel_err_profile_pct",
        "u_edge_over_U_probe",
        "delta_star_meas",
        "delta_star_ref_1p7208_x_over_sqrtRex",
        "Cf_sim",
        "Cf_ref_locked",
        "Cf_err_pct",
        "Cf_err_pct_old_convention",
        "mass_drift_pct",
        "finite",
        "elapsed_s",
    ]
    print(json.dumps({k: r[k] for k in keys}, indent=2))
    print("stations: (err = corrected FD | old-buggy weights | ME | norm-by-ue)")
    for s in r["stations"]:
        print(
            f"  x_eff={s['x_eff']:4d} Rex={s['Rex']:7.1f} Cf_err={s['Cf_fd_err_pct']:+7.2f}% "
            f"old={s['Cf_fd_err_pct_old_buggy_weights']:+7.2f}% me={s['Cf_me_err_pct']:+7.2f}% "
            f"ue={s['u_edge_local_over_U']:.4f} cf_ue={s['Cf_fd_norm_ue_err_pct']:+7.2f}% "
            f"fc={s['first_cell_rel_err_pct']:+6.2f}% wallshift={s['eff_wall_shift_cells']}"
        )


if __name__ == "__main__":
    main()
