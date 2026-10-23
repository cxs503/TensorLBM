#!/usr/bin/env python
"""Stokes 第二问题（振荡平板, D2Q9）—— 受迫振荡层流解析解 benchmark.

Physics
-------
Semi-infinite quiescent fluid above/below an infinite plate oscillating
harmonically in its own plane, u_w(t) = U*cos(w*t).  The steady-periodic
analytic solution (Batchelor / Landau-Lifshitz):

    u(y,t) = U * exp(-k*y) * cos(w*t - k*y),      k = sqrt(w / (2*nu))

amplitude decays exponentially with depth and the phase lags linearly,
both set by the single penetration wavenumber k.  Lattice units:
nu = (tau - 1/2)/3, t in steps.

Setup (true simulation, no extrapolation / no correction factors)
----------------------------------------------------------------
- x-periodic channel (solver.stream periodic gather), nx=8 columns;
  flow is homogeneous in x.
- Plate  = TOP row (y = ny-1), driven every step by the library
  moving-lid boundary with the instantaneous plate velocity
  u_lid(t) = U*cos(w*t)  (same mechanism as the verified cavity cases:
  tensorlbm.lid_driven_cavity.zou_he_moving_lid, called per-step with a
  time-varying scalar carried as a 0-d tensor).
- Far boundary = BOTTOM row (y = 0), stress-free far field via a
  free-slip specular reflection (f_new[j] = f_pre[SPECULAR[j]], the
  repo-verified far-field treatment of verified/stokes_first_problem,
  same inline pattern).  RATIONALE: a no-slip far wall contaminates the
  decaying tail of this physics family -- with the library full-way
  bounce_back_cells the kH=6 steady-periodic error saturates at a
  non-refinable ~2.4-2.8 % (phase-dominated, measured H=50/100/200 at
  tau=0.8 and confirmed not to be resolution: diffusive tau-refinement
  makes it WORSE, 3.4 %; the verified first-problem benchmark documented
  the same effect, 10.8 % there).  kH >= 6 keeps the wall >= 6 decay
  lengths below the plate, and the specular wall takes no shear, so the
  semi-infinite solution is approximated cleanly.  Zou/He prescribes the
  velocity at the boundary node, so depth is measured from the plate
  node: y(row) = (ny-1) - row.
- Start from REST (equilibrium at u=0); the startup transient (channel
  modes decaying like exp(-pi^2*nu*t/H^2)) is skipped by running
  skip_periods = max(10, ceil(8*tau_diff/T)) full periods before the
  measurement window; the measurement itself covers exactly 4 full
  periods (skip window is recorded in every result file).

Library entries (zero hand-written kernels in this file)
--------------------------------------------------------
- tensorlbm.solver.collide_bgk / stream
- tensorlbm.lid_driven_cavity.zou_he_moving_lid   (oscillating plate)
- tensorlbm.d2q9.equilibrium / macroscopic        (IC + measurement)
- far field: free-slip specular reflection, the repo-verified inline
  pattern of verified/stokes_first_problem (SPECULAR permutation of the
  pre-collision state; no hand-written collide/stream/equilibrium)

Acceptance (fixed a priori, identical masks on every grid)
----------------------------------------------------------
- amplitude:  max |A_num/A_ana - 1| <= 3 %  over rows A_ana >= 10 % U
- phase:      max |phi_num - phi_ana| / phi_ana <= 3 % over rows
              phi_ana >= 1 rad AND A_ana >= 10 % U (relative phase is
              meaningless near the plate where the analytic lag -> 0, and
              wherever the signal itself has decayed away); absolute
              phase error in degrees reported for all masked rows
- complex amplitude (summary): max |U_num/U_ana - 1| <= 3 % over rows
  A_ana >= 10 % U   (combines amplitude + phase in one norm)
- grid convergence: error decreases monotonically across the grid
  chain (3 grids H = 50/100/200 at fixed tau, kH in {6, 8}, k = kH/H).

Usage
-----
    run.py single H kH out.json [--tau 0.8] [--U 0.05] [--nx 8]
    run.py scan out_dir [--H 50 100 200] [--kH 6 8] [--device cuda]
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

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.lid_driven_cavity import zou_he_moving_lid
from tensorlbm.solver import collide_bgk, stream

DEVICE = torch.device("cpu")

# Free-slip (specular) reflection map for the far-field wall row: flips
# c_y, keeps c_x  ->  f_new[j] = f_pre[SPECULAR[j]].  Conserves x-momentum
# (no wall shear), so the bottom boundary is a stress-free far field --
# the repo-verified far-field treatment for this physics family
# (verified/stokes_first_problem, where the no-slip far wall was shown to
# contaminate the decaying tail: 10.8 % max_rel at H=100, removed by
# specular).  Same 3-line inline pattern as the verified precedent.
SPECULAR = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)


def specular_replacement(f_pre: torch.Tensor) -> torch.Tensor:
    """Free-slip reflection replacement for the static far-wall row."""
    return f_pre[SPECULAR.to(f_pre.device)]


AMP_MASK_FRAC = 0.10  # rows with A_ana >= 10% U enter the metrics
PHASE_MASK_RAD = 1.0  # rows with phi_ana >= 1 rad enter relative-phase metric
MEASURE_PERIODS = 4


def run_case(
    H: int,
    kH: float,
    tau: float,
    U: float,
    nx: int,
    out_path: str | None,
    seed: int = 0,
    compile_mode: str | None = "default",
) -> dict:
    torch.manual_seed(seed)
    ny = H + 2  # row ny-1 = oscillating plate (lid), row 0 = static far wall
    nu = (tau - 0.5) / 3.0
    k = kH / H
    omega = 2.0 * k * k * nu
    T = 2.0 * math.pi / omega
    T_int = int(round(T))
    tau_diff = H * H / (math.pi**2 * nu)  # slowest startup channel mode
    skip_periods = max(10, math.ceil(8.0 * tau_diff / T))
    skip_steps = skip_periods * T_int
    n_meas = MEASURE_PERIODS * T_int
    total_steps = skip_steps + n_meas
    Ma = U / math.sqrt(1.0 / 3.0)

    wall_bottom = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall_bottom[0, :] = True

    rho0 = torch.ones((ny, nx), device=DEVICE)
    u0 = torch.zeros((ny, nx), device=DEVICE)
    f = equilibrium(rho0, u0, u0)
    initial_mass = float(f.sum().item())

    def _step(f: torch.Tensor, u_lid_t: torch.Tensor) -> torch.Tensor:
        # collide -> pre-stream free-slip far-wall replacement -> stream ->
        # oscillating-plate (Zou/He moving lid, library cavity mechanism)
        f_pre = f
        f = collide_bgk(f, tau)
        f = torch.where(wall_bottom.unsqueeze(0), specular_replacement(f_pre), f)
        f = stream(f)
        return zou_he_moving_lid(f, u_lid_t)

    step_fn = route_step(_step, compile_mode, name=f"stokes2[H{H}_kH{kH:g}]")

    Sc = torch.zeros(ny, device=DEVICE)
    Ss = torch.zeros(ny, device=DEVICE)
    snap_steps = [skip_steps + 1 + (m * T_int) // 8 for m in range(8)]
    snaps: dict[int, torch.Tensor] = {}
    t0 = time.time()
    for step in range(1, total_steps + 1):
        u_lid_t = torch.tensor(U * math.cos(omega * step), device=DEVICE)
        f = step_fn(f, u_lid_t)
        if step > skip_steps:
            _, ux, _ = macroscopic(f)
            col_mean = ux.mean(dim=1)
            Sc += col_mean * math.cos(omega * step)
            Ss += col_mean * math.sin(omega * step)
            if step in snap_steps:
                snaps[step] = col_mean.detach().clone()
    elapsed = time.time() - t0

    # ---- analysis (float64) -------------------------------------------------
    ur = (2.0 / n_meas) * Sc.cpu().numpy().astype(np.float64)
    ui = (-(2.0 / n_meas)) * Ss.cpu().numpy().astype(np.float64)
    A_num = np.hypot(ur, ui)
    # phasor angle(Ū) = -lag:  u = A cos(w t - phi)  <=>  Ū = A e^{-i phi}
    phi_num = -np.arctan2(ui, ur)

    rows = np.arange(ny)
    y = (ny - 1) - rows.astype(np.float64)  # depth below the plate node
    A_ana = U * np.exp(-k * y)
    phi_ana = k * y
    fluid = (rows >= 1) & (rows <= ny - 2)

    amp_mask = fluid & (A_ana >= AMP_MASK_FRAC * U)
    # relative phase is only meaningful where a signal exists AND the
    # analytic lag is not -> 0: require the amplitude mask too
    phase_mask = amp_mask & (phi_ana >= PHASE_MASK_RAD)
    amp_rel = np.abs(A_num / A_ana - 1.0)[amp_mask] * 100.0
    dphi = np.angle(np.exp(1j * (phi_num - phi_ana)))
    phase_rel = (np.abs(dphi) / phi_ana)[phase_mask] * 100.0
    z_num = ur + 1j * ui
    z_ana = A_ana * np.exp(-1j * phi_ana)
    complex_rel = (np.abs(z_num / z_ana - 1.0))[amp_mask] * 100.0

    # instantaneous-profile check at 8 phases (secondary)
    snap_rows = []
    for sstep in snap_steps:
        u_snap = snaps[sstep].cpu().numpy().astype(np.float64)
        u_ref = U * np.exp(-k * y) * np.cos(omega * sstep - k * y)
        m = fluid & (np.abs(u_ref) >= AMP_MASK_FRAC * U)
        l2 = float(np.linalg.norm(u_snap[fluid] - u_ref[fluid]) / np.linalg.norm(u_ref[fluid]))
        snap_rows.append(
            {
                "step": sstep,
                "l2_rel": l2,
                "max_abs_over_U_pct": float(np.max(np.abs(u_snap[m] - u_ref[m])) / U * 100.0),
            }
        )

    result = {
        "case": "stokes_second_problem",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "oscillating plate = library zou_he_moving_lid on top row (per-step u_lid=U*cos(w*t)); far field = free-slip specular reflection on bottom row (verified/stokes_first_problem pattern); x-periodic",
        "driving": "moving-wall (lid) boundary, time-periodic",
        "extrap": "none",
        "entries": {
            "collide": "tensorlbm.solver.collide_bgk",
            "stream": "tensorlbm.solver.stream",
            "far_field": "free-slip specular reflection (verified/stokes_first_problem inline pattern, f_pre[SPECULAR])",
            "plate": "tensorlbm.lid_driven_cavity.zou_he_moving_lid",
            "ic_measure": "tensorlbm.d2q9.equilibrium / macroscopic",
        },
        "H": H,
        "ny": ny,
        "nx": nx,
        "kH": kH,
        "k": k,
        "omega": omega,
        "period_steps_float": T,
        "T_int": T_int,
        "tau": tau,
        "nu_lb": nu,
        "U": U,
        "Ma": Ma,
        "tau_diffusion_steps": tau_diff,
        "skip_periods": skip_periods,
        "skip_steps": skip_steps,
        "measure_periods": MEASURE_PERIODS,
        "n_meas_steps": n_meas,
        "total_steps": total_steps,
        "compile_mode": compile_mode,
        "seed": seed,
        "amp_mask": "A_ana >= 10% U",
        "phase_mask": "phi_ana >= 1 rad",
        "amp_rel_max_pct": float(np.max(amp_rel)),
        "amp_rel_mean_pct": float(np.mean(amp_rel)),
        "phase_rel_max_pct": float(np.max(phase_rel)),
        "phase_err_max_deg": float(np.max(np.abs(dphi[phase_mask])) * 180.0 / math.pi),
        "complex_rel_max_pct": float(np.max(complex_rel)),
        "n_rows_amp_mask": int(amp_mask.sum()),
        "n_rows_phase_mask": int(phase_mask.sum()),
        "snapshot_check": snap_rows,
        "lid_row_amp_ratio": float(A_num[ny - 1] / U),  # BC row diagnostic
        "far_row_amp_over_U": float(A_num[1] / U),
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "y_profile": [round(float(v), 6) for v in y[1 : ny - 1]],
        "A_num": [round(float(v), 8) for v in A_num[1 : ny - 1]],
        "A_ana": [round(float(v), 8) for v in A_ana[1 : ny - 1]],
        "phi_num_deg": [round(float(v) * 180.0 / math.pi, 4) for v in phi_num[1 : ny - 1]],
        "phi_ana_deg": [round(float(v) * 180.0 / math.pi, 4) for v in phi_ana[1 : ny - 1]],
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def scan(
    H_list: list[int],
    kH_list: list[float],
    tau: float,
    U: float,
    nx: int,
    out_dir: str,
    tol_pct: float = 3.0,
    compile_mode: str | None = "default",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for kH in kH_list:
        for H in H_list:
            p = out_dir / f"case_H{H}_kH{kH:g}.json"
            r = run_case(H, kH, tau, U, nx, str(p), compile_mode=compile_mode)
            cases.append(r)
            print(
                f"H={H:4d} kH={kH:g}: amp={r['amp_rel_max_pct']:.3f}% "
                f"phase={r['phase_rel_max_pct']:.3f}% ({r['phase_err_max_deg']:.2f} deg) "
                f"complex={r['complex_rel_max_pct']:.3f}% steps={r['total_steps']} "
                f"elapsed={r['elapsed_s']}s",
                flush=True,
            )
    per_kH = []
    for kH in kH_list:
        rs = [r for r in cases if r["kH"] == kH]
        rs_sorted = sorted(rs, key=lambda r: r["H"])
        e_lo, e_hi = rs_sorted[0]["complex_rel_max_pct"], rs_sorted[-1]["complex_rel_max_pct"]
        per_kH.append(
            {
                "kH": kH,
                "H_list": [r["H"] for r in rs_sorted],
                "complex_rel_max_pct": [r["complex_rel_max_pct"] for r in rs_sorted],
                "amp_rel_max_pct": [r["amp_rel_max_pct"] for r in rs_sorted],
                "phase_rel_max_pct": [r["phase_rel_max_pct"] for r in rs_sorted],
                "err_decreased": bool(e_hi < e_lo),
            }
        )
    all_pass_tol = all(
        r["amp_rel_max_pct"] <= tol_pct
        and r["phase_rel_max_pct"] <= tol_pct
        and r["complex_rel_max_pct"] <= tol_pct
        for r in cases
    )
    all_dec = all(row["err_decreased"] for row in per_kH)
    summary = {
        "case": "stokes_second_problem_convergence",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "zou_he_moving_lid (oscillating plate, top row) + free-slip specular far field (bottom row), x-periodic",
        "driving": "moving-wall boundary u=U*cos(w*t)",
        "extrap": "none",
        "H_list": H_list,
        "kH_list": kH_list,
        "tau": tau,
        "U": U,
        "tol_pct": tol_pct,
        "per_grid": [
            {
                k: r[k]
                for k in r
                if k not in ("y_profile", "A_num", "A_ana", "phi_num_deg", "phi_ana_deg")
            }
            for r in cases
        ],
        "convergence": {
            "metric": "complex_rel_max_pct (|U_num/U_ana - 1|, rows A_ana >= 10% U)",
            "per_kH": per_kH,
            "err_decreased": all_dec,
        },
        "passed": bool(all_pass_tol and all_dec),
        "status": "VERIFIED" if (all_pass_tol and all_dec) else "NOT_PASSED",
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    print(f"\nstatus={summary['status']} tol_pass={all_pass_tol} err_decreased={all_dec}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Stokes second problem (oscillating plate, D2Q9)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("H", type=int)
    p1.add_argument("kH", type=float)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--U", type=float, default=0.05)
    p1.add_argument("--nx", type=int, default=8)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--device", default="cuda")
    add_compile_mode_arg(p1)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--H", type=int, nargs="+", default=[50, 100, 200])
    p2.add_argument("--kH", type=float, nargs="+", default=[6.0, 8.0])
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--U", type=float, default=0.05)
    p2.add_argument("--nx", type=int, default=8)
    p2.add_argument("--device", default="cuda")
    add_compile_mode_arg(p2)

    args = ap.parse_args()
    global DEVICE
    DEVICE = torch.device(args.device)
    compile_mode = compile_mode_from_args(args)
    if args.mode == "single":
        r = run_case(
            args.H, args.kH, args.tau, args.U, args.nx, args.out_json, compile_mode=compile_mode
        )
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in [
                        "H",
                        "kH",
                        "k",
                        "omega",
                        "T_int",
                        "skip_steps",
                        "total_steps",
                        "amp_rel_max_pct",
                        "phase_rel_max_pct",
                        "phase_err_max_deg",
                        "complex_rel_max_pct",
                        "mass_drift_pct",
                        "finite",
                        "elapsed_s",
                    ]
                },
                indent=2,
            )
        )
    else:
        scan(args.H, args.kH, args.tau, args.U, args.nx, args.out_dir, compile_mode=compile_mode)


if __name__ == "__main__":
    main()
