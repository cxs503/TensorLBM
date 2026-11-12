#!/usr/bin/env python
"""Stokes first problem (impulsively started plate / Rayleigh problem) — pointwise
erfc-profile validation, acceptance target ``u/U = erfc(eta)`` for eta in [0, 2] at
t* = 1.0 with **pointwise relative error < 2%** (case pool: LBM-STOKES-FIRST-PROBLEM).

Why a second benchmark next to ``run.py`` (B24) — B24 measured ``max_rel`` only in the
region ``u_ana > 5% U``.  The case-pool criterion is stricter: it demands pointwise
relative error out to ``eta = 2`` where ``u/U = erfc(2) = 4.68e-3``, i.e. the error is
graded against 0.47% of the plate speed.  That tail is where every error source of
LBM shows up amplified (see docs/stokes_first_problem_tstar1_20261112.md):

  * compressibility: LBM solves weakly compressible NS, velocity defect ~ O(Ma^2) * U
    -> at U = 0.05 that alone is >> 2% of u(eta=2);
  * wall position: half-way bounce-back puts the wall at y = 0.5 lattice; a 0.5-lattice
    misplacement is ~4% of u(eta=2) at delta = 25;
  * float32 round-off: 1e-7 * U level -> ~1% of u(eta=2).

So this script uses float64, an Ma scan to separate the compressibility error, and the
exact half-way wall offset in the analytic profile.

Non-dimensionalisation (documented, since the case pool fixes only "t* = 1.0")
---------------------------------------------------------------------------
Reference length ``delta_ref = 25`` coarse-lattice units, reference time
``t_ref = delta_ref^2 / nu``.  With ``t* = t / t_ref = nu * t / delta_ref^2``:
``t* = 1`` <=> boundary-layer thickness ``delta = sqrt(nu t) = delta_ref``, and the
measurement window ``eta in [0, 2]`` <=> ``y in [0, 4 delta]`` sits strictly inside the
domain (top wall at ``eta = 4`` for H = 200), so the erfc half-space solution is
applicable.  Under grid refinement (dx/2, dt/2) the lattice reference length doubles
(delta_ref' = 50) so t* = 1 is the same physical state.

Usage
-----
    python run_tstar1.py single --H 200 --tau 0.65 --U 1e-3 --tstar 1.0 --out case.json
    python run_tstar1.py matrix --out-dir scan_out
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

from tensorlbm.d2q9 import C, OPPOSITE, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_trt, stream  # noqa: E402

torch.set_num_threads(1)  # tiny grids: multi-thread overhead dominates

CS2 = 1.0 / 3.0
DELTA_REF = 25.0  # coarse-lattice boundary-layer reference thickness at t* = 1
# specular (free-slip) reflection: flip c_y, keep c_x -> stress-free far field
SPECULAR = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)


def erfc(x: np.ndarray) -> np.ndarray:
    """Vectorised erfc in float64 (scipy if available, else math.erf)."""
    x = np.asarray(x, dtype=np.float64)
    try:
        from scipy.special import erfc as _erfc

        return _erfc(x)
    except ImportError:
        flat = np.array([math.erf(v) for v in x.ravel()], dtype=np.float64)
        return (1.0 - flat).reshape(x.shape)


def _wall_specular(f_pre: torch.Tensor) -> torch.Tensor:
    return f_pre[SPECULAR.to(f_pre.device)]


def _wall_moving(f_pre: torch.Tensor, U: float) -> torch.Tensor:
    """Pre-streaming half-way bounce-back replacement for a wall moving at (U, 0).

    f_new[opp(i)] = f_pre[i] - 2 w_i rho_w (c_i . u_w) / cs^2    (Ladd correction)
    """
    rho_w = f_pre.sum(dim=0)
    cx = C[:, 0].to(f_pre.device).to(f_pre.dtype)
    w = W.to(f_pre.device).to(f_pre.dtype)
    mom = 2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * (cx.view(9, 1, 1) * U) / CS2
    opp = OPPOSITE.to(f_pre.device)
    return f_pre[opp] - mom[opp]


def analytic_profile(y_from_wall: np.ndarray, U: float, nu: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    eta = y_from_wall / (2.0 * np.sqrt(nu * t))
    return eta, U * erfc(eta)


def run_case(
    H: int,
    tau: float,
    U: float,
    tstar: float,
    nx: int = 4,
    delta_ref: float = DELTA_REF,
    top: str = "specular",
    collision: str = "bgk",
    lambda_trt: float = 3.0 / 16.0,
    eta_max: float = 2.0,
    n_eta: int = 41,
    record_tstars: tuple[float, ...] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """One Stokes-first-problem run; returns the eta-resolved error table."""
    nu = (tau - 0.5) / 3.0
    nsteps = int(round(tstar * delta_ref**2 / nu))
    ny = H + 2  # rows 0 and ny-1 are wall rows; fluid rows 1..H
    device = torch.device("cpu")

    wall_bottom = torch.zeros((ny, nx), dtype=torch.bool)
    wall_bottom[0, :] = True
    wall_top = torch.zeros((ny, nx), dtype=torch.bool)
    wall_top[-1, :] = True

    rho0 = torch.ones((ny, nx), dtype=dtype)
    zero = torch.zeros((ny, nx), dtype=dtype)
    f = equilibrium(rho0, zero, zero)
    mass0 = float(f.sum())

    if record_tstars is None:
        record_tstars = (tstar,)
    record_steps = {
        int(round(ts * delta_ref**2 / nu)): float(ts) for ts in record_tstars if int(round(ts * delta_ref**2 / nu)) > 0
    }
    # row index -> y measured from the half-way wall plane (y_wall = 0.5 lattice)
    y_from_wall = np.arange(1, ny - 1, dtype=np.float64) - 0.5

    profiles: dict[int, np.ndarray] = {}
    t0 = time.time()
    for step in range(1, nsteps + 1):
        f_pre = f
        f = collide_bgk(f, tau) if collision == "bgk" else collide_trt(f, tau, lambda_trt)
        # pre-streaming boundary treatment (repo convention, see run.py)
        f = torch.where(wall_bottom.unsqueeze(0), _wall_moving(f_pre, U), f)
        if top == "specular":
            f = torch.where(wall_top.unsqueeze(0), _wall_specular(f_pre), f)
        elif top == "noslip":
            f = torch.where(wall_top.unsqueeze(0), f_pre[OPPOSITE.to(f_pre.device)], f)
        else:
            raise ValueError(f"unknown top BC {top!r}")
        f = stream(f)
        if step in record_steps:
            rho, ux, _ = macroscopic(f)
            profiles[step] = ux[1 : ny - 1, 0].detach().numpy().astype(np.float64)
            rho_dev = float((rho[1 : ny - 1] - 1.0).abs().max())
    elapsed = time.time() - t0

    rows = []
    for step, ts in sorted(record_steps.items()):
        u_num = profiles[step]
        eta_native, u_ana_native = analytic_profile(y_from_wall, U, nu, float(step))
        # pointwise comparison on a fixed eta grid (linear interp of the LBM profile)
        eta_grid = np.linspace(0.0, eta_max, n_eta)
        u_num_g = np.interp(eta_grid, eta_native, u_num)
        u_ana_g = U * erfc(eta_grid)
        rel = np.abs(u_num_g - u_ana_g) / u_ana_g * 100.0
        # native-node error inside the window (denser, no interpolation)
        m = eta_native <= eta_max + 1e-12
        rel_nat = np.abs(u_num[m] - u_ana_native[m]) / u_ana_native[m] * 100.0
        m_int = eta_grid >= 0.1
        rows.append(
            {
                "tstar": ts,
                "rel_err_pct_at_eta2": float(rel[-1]),
                "max_rel_err_pct_eta0.1_2": float(np.max(rel[m_int])),
                "step": int(step),
                "delta_lb": float(np.sqrt(nu * step)),
                "max_rel_err_pct_eta0_2": float(np.max(rel)),
                "max_rel_err_pct_native_nodes": float(np.max(rel_nat)),
                "rel_err_pct_at_eta": [round(float(v), 5) for v in rel],
                "eta_grid": [round(float(v), 5) for v in eta_grid],
                "u_over_U_num": [round(float(v / U), 8) for v in u_num_g],
                "u_over_U_ana": [round(float(v), 8) for v in erfc(eta_grid)],
                "eta_native": [round(float(v), 6) for v in eta_native[m]],
                "u_over_U_native": [round(float(v / U), 8) for v in u_num[m]],
                "u_over_U_ana_native": [round(float(v), 8) for v in u_ana_native[m] / U],
            }
        )

    return {
        "case": "LBM-STOKES-FIRST-PROBLEM",
        "H": H,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu_lb": nu,
        "U": U,
        "Ma": float(U / math.sqrt(CS2)),
        "top_bc": top,
        "collision": collision if collision == "bgk" else f"trt(lambda={lambda_trt})",
        "delta_ref": delta_ref,
        "nsteps": nsteps,
        "tstar": tstar,
        "wall_plane_lattice": 0.5,
        "dtype": str(dtype).replace("torch.", ""),
        "rho_max_dev_from_1": rho_dev,
        "mass_drift_pct": (float(f.sum()) - mass0) / mass0 * 100.0,
        "finite": bool(torch.isfinite(f).all()),
        "elapsed_s": round(elapsed, 2),
        "profiles": rows,
    }


def _fmt(cfg: dict) -> str:
    p = cfg["profiles"][0]
    return (
        f"H={cfg['H']:4d} tau={cfg['tau']:.2f} nu={cfg['nu_lb']:.4f} U={cfg['U']:<7g} "
        f"Ma={cfg['Ma']:.2e} {cfg['collision']:<8s} top={cfg['top_bc']:<8s} "
        f"t*={p['tstar']:.2f} delta={p['delta_lb']:5.1f} "
        f"max_rel(eta0-2)={p['max_rel_err_pct_eta0_2']:7.3f}%  "
        f"max_rel(nodes)={p['max_rel_err_pct_native_nodes']:7.3f}% "
        f"[eta.1-2: {p['max_rel_err_pct_eta0.1_2']:.4f}%  eta2: {p['rel_err_pct_at_eta2']:.4f}%]"
    )


MATRIX = [
    # (tag, kwargs) — compressibility scan / resolution pair / collision pair / t* pair
    ("U0.05_H200_bgk", dict(H=200, tau=0.65, U=0.05, tstar=1.0)),
    ("U0.01_H200_bgk", dict(H=200, tau=0.65, U=0.01, tstar=1.0)),
    ("U1e-3_H200_bgk", dict(H=200, tau=0.65, U=1e-3, tstar=1.0)),
    ("U1e-3_H200_trt", dict(H=200, tau=0.65, U=1e-3, tstar=1.0, collision="trt")),
    ("U1e-3_H200_bgk_topnoslip", dict(H=200, tau=0.65, U=1e-3, tstar=1.0, top="noslip")),
    ("U1e-3_H200_bgk_tstar0.64", dict(H=200, tau=0.65, U=1e-3, tstar=0.64, delta_ref=25.0)),
    # refined physical resolution: dx/2, dt/2 -> nu'=2nu, t'=2t, delta_ref'=2*25
    ("U1e-3_H400_bgk_fine", dict(H=400, tau=0.8, U=1e-3, tstar=1.0, delta_ref=50.0)),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("single")
    p1.add_argument("--H", type=int, default=200)
    p1.add_argument("--tau", type=float, default=0.65)
    p1.add_argument("--U", type=float, default=1e-3)
    p1.add_argument("--tstar", type=float, default=1.0)
    p1.add_argument("--delta-ref", type=float, default=DELTA_REF)
    p1.add_argument("--top", default="specular", choices=["specular", "noslip"])
    p1.add_argument("--collision", default="bgk", choices=["bgk", "trt"])
    p1.add_argument("--out", type=Path, required=True)
    p2 = sub.add_parser("matrix")
    p2.add_argument("--out-dir", type=Path, required=True)
    p2.add_argument("--only", default="", help="comma separated tags (default: all)")
    args = ap.parse_args()

    if args.mode == "single":
        r = run_case(
            args.H, args.tau, args.U, args.tstar, delta_ref=args.delta_ref,
            top=args.top, collision=args.collision,
            record_tstars=(args.tstar,),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2))
        print(_fmt(r))
        p = r["profiles"][0]
        for eta, rel in zip(p["eta_grid"], p["rel_err_pct_at_eta"]):
            print(f"  eta={eta:5.2f}  u/U num={np.interp(eta, p['eta_grid'], p['u_over_U_num']):.6f} "
                  f"ana={np.interp(eta, p['eta_grid'], p['u_over_U_ana']):.6f}  rel={rel:8.3f}%")
        return

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    only = {t.strip() for t in args.only.split(",") if t.strip()}
    for tag, kw in MATRIX:
        if only and tag not in only:
            continue
        r = run_case(record_tstars=(kw["tstar"],), **kw)
        r["tag"] = tag
        (out / f"{tag}.json").write_text(json.dumps(r, indent=2))
        summary.append(r)
        print(_fmt(r), flush=True)

    tol = 2.0
    have = {r["tag"] for r in summary}
    best = [r for r in summary if r["tag"] in have and r["tag"] in
            ("U1e-3_H200_bgk", "U1e-3_H200_trt", "U1e-3_H200_bgk_topnoslip",
             "U1e-3_H200_bgk_tstar0.64", "U1e-3_H400_bgk_fine")]
    passed = all(r["profiles"][0]["max_rel_err_pct_eta0_2"] <= tol for r in best)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "case": "LBM-STOKES-FIRST-PROBLEM",
                "criterion": "pointwise rel err of u/U vs erfc(eta), eta in [0,2], t*=1.0",
                "tol_pct": tol,
                "delta_ref_coarse": DELTA_REF,
                "runs": [
                    {
                        k: r[k]
                        for k in ("tag", "H", "tau", "nu_lb", "U", "Ma", "collision", "top_bc",
                                  "delta_ref", "nsteps", "tstar", "mass_drift_pct", "finite", "elapsed_s")
                    }
                    | {"max_rel_err_pct_eta0_2": r["profiles"][0]["max_rel_err_pct_eta0_2"],
                       "max_rel_err_pct_native_nodes": r["profiles"][0]["max_rel_err_pct_native_nodes"]}
                    for r in summary
                ],
                "all_lt_2pct": passed,
                "status": "VERIFIED" if passed else "NOT_PASSED",
            },
            indent=2,
        )
    )
    print(f"\nall < 2% (acceptance group): {passed}")


if __name__ == "__main__":
    main()