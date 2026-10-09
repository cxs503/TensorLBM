#!/usr/bin/env python
"""启动 Poiseuille 流（2D 通道, D2Q9）—— 瞬态层流解析解 benchmark.

Physics
-------
Fluid at rest in a plane channel (walls at y=0, H); at t=0 a constant
streamwise body force / pressure gradient is switched on
(a = const, dp/dx = -rho*a).  The transient solution is the classical
Fourier series (e.g. Pedley; Ethier & Critchley; Fachinotti & Le Bot):

    u(y,t) = u_max * [ 4*ŷ*(1-ŷ) - (32/pi^3) * sum_{n odd} (1/n^3) sin(n*pi*ŷ)
                       * exp(-n^2 pi^2 nu t / H^2) ],   ŷ = y/H
    u_max = a H^2 / (8 nu)

which starts at u=0 for 0 < y < H (Fourier identity sum_{odd}
sin(n pi ŷ)/n^3 = pi^3 ŷ(1-ŷ)/8) and converges to the parabola
u_max*4*ŷ*(1-ŷ) as t -> infinity; at the centreline it reduces to the
often-quoted Pedley form u_c = u_max[1 - (32/pi^3) sum (1/n^3)(-1)^((n-1)/2)
exp(...)].  Derived and cross-checked against the PDE residual (see
README).  Series evaluated with numpy float64 (odd n up to 999; at the
earliest recorded time t* = nu t/H^2 = 0.05 terms beyond n~10 are < 1e-6,
no Gibbs contamination).

Setup (true simulation, no extrapolation / no correction factors)
----------------------------------------------------------------
Identical channel/driver machinery as the Womersley benchmark (same
library entries, same step order):
- x-periodic (solver.stream), walls rows 0/ny-1 via the pre-streaming
  half-way bounce-back (the repo-validated analytic-channel wall pattern,
  identical to verified/poiseuille_2d / couette_2d: f <- f_pre[OPPOSITE]
  at wall rows before streaming; no-slip planes at y = 0.5 / ny-1.5, so
  H_eff = ny-2 and fluid rows are measured at y_phys = row-0.5).
  GAP NOTE: tensorlbm.boundaries.bounce_back_cells (the library wall
  function) implements the post-streaming full-way variant (walls ON the
  nodes, H_eff = ny-1) - it converges to the startup analytic too but
  with ~4x larger, slower-converging error; the repo's verified pattern
  is used instead (see womersley README for the full evidence);
- constant body force via tensorlbm.turbulent_channel._apply_body_force_2d
  (library's own force-driven-channel path; injects exactly rho*a per
  step).  GAP NOTE: solver.py has no public D2Q9 force entry (see
  womersley README) - recorded, not bypassed.
- Start from REST at t=0, force on from the first step.  The transient
  IS the object of the benchmark (no skip window); comparisons use the
  full series solution at the same lattice time.

Acceptance (fixed a priori, identical on every grid)
----------------------------------------------------
- centerline u_c(t) sampled every 20 steps for nu t/H^2 >= 0.02:
      max |u_c_num - u_c_ana| / u_max <= 3 %
- profiles at t* = nu t / H^2 in {0.05, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5}:
      L2 relative error <= 3 % and max pointwise error / u_max <= 3 %
- error decreases from H to ~2H (grid convergence).

Usage
-----
    run.py single H out.json [--tau 0.8] [--umax 0.03]
    run.py scan out_dir [--H 59 119] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import numpy as np
import torch
from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.turbulent_channel import _apply_body_force_2d

DEVICE = torch.device("cpu")

TSTAR_MARKS = [0.05, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5]
TSTAR_CENTER_MIN = 0.02
CENTER_EVERY = 20
N_ODD = np.arange(1, 1000, 2)  # odd harmonics for the series


def startup_profile(y_phys: np.ndarray, H: float, nu: float, a: float, t: float) -> np.ndarray:
    """Series solution u(y,t) for suddenly-applied constant force (float64)."""
    u_max = a * H * H / (8.0 * nu)
    yh = y_phys / H
    n = N_ODD.astype(np.float64)
    # vectorised: eigenvalue form  (n pi / H)^2 nu t  so the exponent is
    # n^2 pi^2 nu t / H^2 exactly (mode sin(n pi y / H) of the heat operator)
    lam = np.minimum((n * np.pi / H) ** 2 * nu * t, 700.0)  # clip against denormals
    fac = (1.0 / n**3)[:, None] * np.sin(np.outer(n, yh) * np.pi) * np.exp(-lam)[:, None]
    s = fac.sum(axis=0)
    steady = 4.0 * yh * (1.0 - yh)  # = u/u_max parabola
    return u_max * (steady - (32.0 / np.pi**3) * s)


def run_case(
    H: int,
    tau: float,
    u_max_target: float,
    nx: int,
    out_path: str | None,
    seed: int = 0,
    compile_mode: str | None = "default",
) -> dict:
    torch.manual_seed(seed)
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    H_eff = float(ny - 2)  # half-way BB walls at 0.5 / ny-1.5
    a = 8.0 * nu * u_max_target / (H_eff * H_eff)
    u_max = u_max_target  # analytic final centerline speed
    t_end = int(math.ceil(TSTAR_MARKS[-1] * H_eff * H_eff / nu)) + CENTER_EVERY
    Ma = u_max / math.sqrt(1.0 / 3.0)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall[0, :] = True
    wall[-1, :] = True

    rho0 = torch.ones((ny, nx), device=DEVICE)
    u0 = torch.zeros((ny, nx), device=DEVICE)
    f = equilibrium(rho0, u0, u0)
    initial_mass = float(f.sum().item())

    def _step(f: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        # pre-streaming half-way bounce-back on the wall rows (the
        # repo-validated analytic-channel wall pattern of verified/
        # poiseuille_2d + couette_2d: replace the wall rows' post-collision
        # state with the reflection of their PRE-collision state, built from
        # the library OPPOSITE table), then the library body force, then the
        # library periodic stream.
        f_pre = f
        f = collide_bgk(f, tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(f.device)], f)
        f = _apply_body_force_2d(f, a_t)
        return stream(f)

    step_fn = route_step(_step, compile_mode, name=f"startup_poiseuille[H{H}]")

    rows = np.arange(1, ny - 1, dtype=np.float64)
    y_phys = rows - 0.5  # half-way walls -> fluid rows at row - 0.5
    center_row = ny // 2  # H odd -> row ny//2 sits exactly at y = H_eff/2
    mark_steps = {int(round(ts * H_eff * H_eff / nu)): ts for ts in TSTAR_MARKS}
    tstar_center_min_step = int(round(TSTAR_CENTER_MIN * H_eff * H_eff / nu))

    center_hist: list[tuple[int, float]] = []
    snaps: dict[int, np.ndarray] = {}
    t0 = time.time()
    a_t = torch.tensor(a, device=DEVICE)
    for step in range(1, t_end + 1):
        f = step_fn(f, a_t)
        if step % CENTER_EVERY == 0 and step >= tstar_center_min_step:
            _, ux, _ = macroscopic(f)
            center_hist.append((step, float(ux[center_row, nx // 2].item())))
        if step in mark_steps:
            _, ux, _ = macroscopic(f)
            snaps[step] = ux.mean(dim=1).cpu().numpy().astype(np.float64)[1 : ny - 1]
    elapsed = time.time() - t0

    # ---- analysis (float64) -------------------------------------------------
    c_err_max = 0.0
    c_rows = []
    for step, u_c in center_hist:
        u_ref = float(startup_profile(np.array([H_eff / 2.0]), H_eff, nu, a, float(step))[0])
        err = abs(u_c - u_ref) / u_max
        c_err_max = max(c_err_max, err)
        c_rows.append(
            {
                "step": step,
                "tstar": step * nu / (H_eff * H_eff),
                "u_c_num": u_c,
                "u_c_ana": u_ref,
                "err_over_umax_pct": err * 100.0,
            }
        )
    prof_rows = []
    l2_list, ptp_list = [], []
    for step, ts in sorted(mark_steps.items()):
        u_num = snaps[step]
        u_ref = startup_profile(y_phys, H_eff, nu, a, float(step))
        l2 = float(np.linalg.norm(u_num - u_ref) / np.linalg.norm(u_ref))
        ptp = float(np.max(np.abs(u_num - u_ref)) / u_max)
        l2_list.append(l2)
        ptp_list.append(ptp)
        prof_rows.append({"step": step, "tstar": ts, "l2_rel": l2, "max_abs_over_umax": ptp})

    result = {
        "case": "startup_poiseuille",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "pre-streaming half-way bounce-back (verified/poiseuille_2d pattern, f_pre[OPPOSITE] at wall rows; walls at 0.5 / ny-1.5, H_eff = ny-2), x-periodic",
        "driving": "constant body force a=8*nu*u_max/H^2 via library _apply_body_force_2d, composed collide->halfway-BB->force->stream",
        "extrap": "none",
        "entries": {
            "collide": "tensorlbm.solver.collide_bgk",
            "stream": "tensorlbm.solver.stream",
            "force": "tensorlbm.turbulent_channel._apply_body_force_2d",
            "wall": "pre-streaming half-way BB via library OPPOSITE table (verified/poiseuille_2d pattern; see gap note)",
            "ic_measure": "tensorlbm.d2q9.equilibrium / macroscopic",
        },
        "H": H,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu_lb": nu,
        "H_eff": H_eff,
        "a": a,
        "u_max_final": u_max,
        "Ma": Ma,
        "t_end": t_end,
        "tstar_marks": TSTAR_MARKS,
        "center_every": CENTER_EVERY,
        "tstar_center_min": TSTAR_CENTER_MIN,
        "compile_mode": compile_mode,
        "seed": seed,
        "centerline_err_max_over_umax_pct": c_err_max * 100.0,
        "profile_l2_max_pct": float(np.max(l2_list)) * 100.0,
        "profile_pointwise_max_over_umax_pct": float(np.max(ptp_list)) * 100.0,
        "centerline_table_every10": c_rows[::10],
        "profile_table": prof_rows,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "y_profile": [round(float(v), 6) for v in y_phys],
        "u_final_num": [round(float(v), 8) for v in snaps[sorted(mark_steps)[-1]]],
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def scan(
    H_list: list[int],
    tau: float,
    u_max: float,
    nx: int,
    out_dir: str,
    tol_pct: float = 3.0,
    compile_mode: str | None = "default",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for H in H_list:
        p = out_dir / f"case_H{H}.json"
        r = run_case(H, tau, u_max, nx, str(p), compile_mode=compile_mode)
        cases.append(r)
        print(
            f"H={H:4d}: center={r['centerline_err_max_over_umax_pct']:.3f}% "
            f"l2={r['profile_l2_max_pct']:.3f}% ptp={r['profile_pointwise_max_over_umax_pct']:.3f}% "
            f"steps={r['t_end']} elapsed={r['elapsed_s']}s",
            flush=True,
        )
    rs = sorted(cases, key=lambda r: r["H"])
    e_lo = rs[0]["centerline_err_max_over_umax_pct"]
    e_hi = rs[-1]["centerline_err_max_over_umax_pct"]
    err_decreased = bool(e_hi < e_lo)
    all_pass_tol = all(
        r["centerline_err_max_over_umax_pct"] <= tol_pct
        and r["profile_l2_max_pct"] <= tol_pct
        and r["profile_pointwise_max_over_umax_pct"] <= tol_pct
        for r in cases
    )
    summary = {
        "case": "startup_poiseuille_convergence",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "pre-streaming half-way bounce-back (verified/poiseuille_2d pattern), x-periodic",
        "driving": "constant body force a (library _apply_body_force_2d)",
        "extrap": "none",
        "H_list": H_list,
        "tau": tau,
        "u_max_target": u_max,
        "tol_pct": tol_pct,
        "per_grid": [
            {
                k: r[k]
                for k in r
                if k not in ("y_profile", "u_final_num", "centerline_table_every10")
            }
            for r in cases
        ],
        "convergence": {
            "metric": "centerline_err_max_over_umax_pct",
            "values": [r["centerline_err_max_over_umax_pct"] for r in rs],
            "err_decreased": err_decreased,
        },
        "passed": bool(all_pass_tol and err_decreased),
        "status": "VERIFIED" if (all_pass_tol and err_decreased) else "NOT_PASSED",
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    print(f"\nstatus={summary['status']} tol_pass={all_pass_tol} err_decreased={err_decreased}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Startup Poiseuille flow (D2Q9)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("H", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--umax", type=float, default=0.03)
    p1.add_argument("--nx", type=int, default=16)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--device", default="cuda")
    add_compile_mode_arg(p1)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--H", type=int, nargs="+", default=[59, 119])
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--umax", type=float, default=0.03)
    p2.add_argument("--nx", type=int, default=16)
    p2.add_argument("--device", default="cuda")
    add_compile_mode_arg(p2)

    args = ap.parse_args()
    global DEVICE
    DEVICE = torch.device(args.device)
    compile_mode = compile_mode_from_args(args)
    if args.mode == "single":
        r = run_case(args.H, args.tau, args.umax, args.nx, args.out_json, compile_mode=compile_mode)
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in [
                        "H",
                        "tau",
                        "nu_lb",
                        "a",
                        "u_max_final",
                        "t_end",
                        "centerline_err_max_over_umax_pct",
                        "profile_l2_max_pct",
                        "profile_pointwise_max_over_umax_pct",
                        "mass_drift_pct",
                        "finite",
                        "elapsed_s",
                    ]
                },
                indent=2,
            )
        )
    else:
        scan(args.H, args.tau, args.umax, args.nx, args.out_dir, compile_mode=compile_mode)


if __name__ == "__main__":
    main()
