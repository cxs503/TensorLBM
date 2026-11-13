#!/usr/bin/env python3
"""CASE-LBM-KOVASZNAY-EXACT -- breaking the Ma^2 floor with TRT: a real order-of-accuracy study.

Why this arm exists
-------------------
The case pool entry (``cases/kovasznay_2d``, missions m17704 / m16861 / m16974) settled two
things about the anchor ``CASE-LBM-KOVASZNAY-EXACT``:

1. the acceptance gate (velocity-field L2 relative error < 3 % at Re = 40, ny = 128) passes
   with 2.56 % -- but 99.6 %-99.9 % of that number is the zero-gradient outlet boundary layer,
   not discretisation error;
2. the *interior* (boundary-free) error is **grid independent** (u: 0.1102 -> 0.1023 -> 0.1017 %
   for ny = 32 -> 64 -> 128), i.e. the case, as shipped, cannot report an observed order of
   accuracy.  The cause was located quantitatively: the interior error is an O(Ma^2)
   compressibility floor (halving U0 at fixed Re drops the interior error ~3.5x, Ma^2 predicts 4x).

The stated follow-up ("换碰撞模型（TRT / regularized）") was left open: lowering Ma at fixed Re
means lowering tau+ = 1/2 + 3 nu, and BGK + Zou-He inlet + any outlet closure diverges at
tau ~ 0.57 (measured NaN).  This module does that follow-up:

* collision: library ``tensorlbm.solver.collide_trt`` (TRT, magic parameter Lambda = 3/16 by
  default) instead of ``collide_bgk`` -- TRT damps the anti-symmetric non-equilibrium moment
  independently and is linearly stable much closer to tau = 1/2;
* outlet: **analytic Dirichlet** on the last column (the "assignment arm" of m16861, used here
  only to remove the boundary layer so the truncation error becomes observable -- it is NOT
  proposed as an engineering BC);
* measured: interior L2 error vs. the exact solution on a grid ladder, observed order p and
  Richardson-extrapolated error, plus a Ma ladder to show the floor dropping as Ma^2.

All lattice operators are public library functions (``collide_bgk`` / ``collide_trt`` /
``stream`` / ``macroscopic`` / ``equilibrium`` / ``zou_he_inlet_velocity``); the analytic
Dirichlet outlet writes the exact equilibrium into the last column and is labelled as such.

Usage
-----
::

    python cases/kovasznay_2d/trt_convergence.py check
    python cases/kovasznay_2d/trt_convergence.py stability --out /tmp/stab.json
    python cases/kovasznay_2d/trt_convergence.py ladder --grids 32 64 128 --u0 0.0075 --out /tmp/l.json
    python cases/kovasznay_2d/trt_convergence.py all --out artifacts/m18406/trt_convergence.json

Reference
---------
Kovasznay, L. I. G. (1948). Laminar flow behind a two-dimensional grid.
*Proc. Camb. Phil. Soc.* 44, 58-62.  DOI:10.1017/S0305004100023999
Ginzburg, I. (2008). Two-relaxation-time lattice Boltzmann scheme ... *Commun. Comput. Phys.*
3(2), 427-478 (magic parameter Lambda = 3/16).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.boundaries import zou_he_inlet_velocity  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_trt, stream  # noqa: E402

from analytic import kovasznay_lambda, make_field  # noqa: E402

RE_DEFAULT = 40.0


# ---------------------------------------------------------------------------
# one steady-state run
# ---------------------------------------------------------------------------


def _collide(f: torch.Tensor, tau: float, collision: str, lam_trt: float) -> torch.Tensor:
    if collision == "bgk":
        return collide_bgk(f, tau)
    if collision == "trt":
        return collide_trt(f, tau, lam_trt)
    raise ValueError(f"collision must be 'bgk' or 'trt', got {collision!r}")


def run_case(
    ny: int,
    *,
    re: float = RE_DEFAULT,
    u0: float = 0.03,
    xmax: float = 3.0,
    collision: str = "trt",
    lam_trt: float = 3.0 / 16.0,
    outlet: str = "dirichlet",
    min_steps: int = 2000,
    max_steps: int = 30000,
    check_every: int = 500,
    drift_tol: float = 1e-9,
    threads: int = 8,
    dtype: torch.dtype = torch.float64,
    field_out: Path | None = None,
) -> dict:
    """Advance one Kovasznay configuration to steady state and measure the error.

    Returns a dict of scalars only (JSON-safe), so the caller can assemble tables
    programmatically.  ``finite=False`` marks a diverged arm instead of raising: the
    stability wall is a *result* here, not an error.
    """
    torch.set_num_threads(threads)
    field = make_field(ny, re=re, u0=u0, xmax=xmax)
    nx = field.nx
    tau = field.tau
    if tau <= 0.5:
        raise ValueError(f"tau+ must exceed 0.5, got {tau}")

    u_ana_np, v_ana_np = field.u(), field.v()
    u_ana = torch.as_tensor(u_ana_np, dtype=dtype)
    v_ana = torch.as_tensor(v_ana_np, dtype=dtype)
    rho0 = torch.ones((ny, nx), dtype=dtype)
    f = equilibrium(rho0, u_ana, v_ana)

    u_in = u_ana[:, 0].contiguous()
    v_in = v_ana[:, 0].contiguous()
    # analytic Dirichlet outlet: exact equilibrium in the last column
    f_out_col = equilibrium(
        torch.ones((ny, 1), dtype=dtype),
        u_ana[:, -1].reshape(ny, 1).contiguous(),
        v_ana[:, -1].reshape(ny, 1).contiguous(),
    )

    mass0 = float(f.sum())

    n = 0
    steady = False
    finite = True
    drift = float("nan")
    u_prev: float | None = None
    t0 = time.time()
    while n < max_steps:
        f = _collide(f, tau, collision, lam_trt)
        f = stream(f)
        if outlet == "dirichlet":
            f = f.clone()
            f[:, :, -1] = f_out_col[:, :, -1]
        elif outlet == "zerograd":
            f = f.clone()
            f[:, :, -1] = f[:, :, -2]
        else:
            raise ValueError(f"outlet must be 'dirichlet' or 'zerograd', got {outlet!r}")
        f = zou_he_inlet_velocity(f, u_in, v_in)
        n += 1
        if n % check_every == 0:
            if not bool(torch.isfinite(f).all()):
                finite = False
                break
            _, ux, _ = macroscopic(f)
            norm = float(torch.linalg.vector_norm(ux))
            if u_prev is not None and u_prev > 0:
                drift = (norm - u_prev) / u_prev
                if n >= min_steps and abs(drift) < drift_tol:
                    steady = True
                    break
            u_prev = norm
    elapsed = time.time() - t0

    out = {
        "ny": ny,
        "nx": nx,
        "xmax": xmax,
        "re": re,
        "u0": u0,
        "tau_plus": tau,
        "nu": field.nu,
        "ma_max": field.ma_max,
        "ma2": field.ma_max**2,
        "collision": collision,
        "lambda_trt": lam_trt if collision == "trt" else None,
        "tau_minus": 0.5 + lam_trt / (tau - 0.5) if collision == "trt" else None,
        "outlet": outlet,
        "steps": n,
        "steady": steady,
        "finite": finite,
        "drift_last": drift if math.isfinite(drift) else None,
        "elapsed_s": round(elapsed, 2),
    }
    if not finite:
        return out

    _, ux, uy = macroscopic(f)
    u_num = ux.detach().to(torch.float64).numpy()
    v_num = uy.detach().to(torch.float64).numpy()
    rho = macroscopic(f)[0].detach().to(torch.float64).numpy()

    # interior window: drop the inlet column and ONE full y-period at the outlet
    n_bl = min(ny, max(1, nx // 2))
    lo, hi = 1, max(1, nx - n_bl)
    out["interior_cols"] = [lo, hi]
    out["interior_n_cols"] = hi - lo
    out["full"] = _err_block(u_num, v_num, u_ana_np, v_ana_np, u0)
    out["interior"] = _err_block(
        u_num[:, lo:hi], v_num[:, lo:hi], u_ana_np[:, lo:hi], v_ana_np[:, lo:hi], u0
    )
    out["mass_drift_pct"] = 100.0 * (float(rho.mean()) - 1.0)
    if field_out is not None:
        path = Path(field_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            u_num=u_num, v_num=v_num, u_ana=u_ana_np, v_ana=v_ana_np,
            rho=rho, ny=ny, nx=nx, u0=u0, tau=tau, lam_trt=lam_trt,
        )
    out["u_l2_rel_pct"] = out["full"]["u_l2_rel_pct"]
    out["v_l2_rel_pct"] = out["full"]["v_l2_rel_pct"]
    return out


def _err_block(u_num, v_num, u_a, v_a, u0: float) -> dict:
    du = u_num - u_a
    dv = v_num - v_a

    def l2(a):
        return float(np.sqrt(np.sum(a * a)))

    n = du.size
    return {
        "u_l2_rel_pct": 100.0 * l2(du) / l2(u_a),
        "v_l2_rel_pct": 100.0 * l2(dv) / l2(v_a),
        "u_l2_abs_over_u0_pct": 100.0 * l2(du) / math.sqrt(n) / u0,
        "v_l2_abs_over_u0_pct": 100.0 * l2(dv) / math.sqrt(n) / u0,
        "u_max_abs_over_u0_pct": 100.0 * float(np.max(np.abs(du))) / u0,
        "v_max_abs_over_u0_pct": 100.0 * float(np.max(np.abs(dv))) / u0,
    }


# ---------------------------------------------------------------------------
# analysis: observed order of accuracy / Richardson
# ---------------------------------------------------------------------------


def observed_order(h: list[float], err: list[float]) -> list[dict]:
    """Pairwise observed order ``p = ln(e1/e2) / ln(h1/h2)`` for a refinement sequence."""
    rows = []
    for i in range(len(h) - 1):
        if err[i] <= 0 or err[i + 1] <= 0:
            continue
        rows.append(
            {
                "from_ny": int(round(1.0 / h[i])),
                "to_ny": int(round(1.0 / h[i + 1])),
                "p": math.log(err[i] / err[i + 1]) / math.log(h[i] / h[i + 1]),
            }
        )
    return rows


def richardson(h: list[float], err: list[float], p: float) -> float:
    """Extrapolated h -> 0 error from the finest two points at fixed order ``p``."""
    if len(err) < 2:
        return float("nan")
    e1, e2 = err[-2], err[-1]
    r = h[-2] / h[-1]
    if abs(r**p - 1.0) < 1e-14:
        return float("nan")
    # e(h) = e0 + C h^p with e0 = e(h2) + (e(h2)-e(h1)) / (r^p - 1), r = h1/h2
    return float(e2 + (e2 - e1) / (r**p - 1.0))


# ---------------------------------------------------------------------------
# experiment arms
# ---------------------------------------------------------------------------


def arm_stability(
    *,
    ny: int = 32,
    u0s=(0.03, 0.015, 0.0075, 0.00375, 0.001875),
    steps: int = 3000,
    threads: int = 8,
    lambdas=(3.0 / 16.0,),
) -> dict:
    """Ma ladder at fixed ny/Re: where does BGK die, does TRT survive?"""
    rows = []
    for u0 in u0s:
        for collision in ("bgk", "trt"):
            lams = (3.0 / 16.0,) if collision == "bgk" else tuple(lambdas)
            for lam in lams:
                r = run_case(
                    ny,
                    u0=u0,
                    collision=collision,
                    lam_trt=lam,
                    outlet="dirichlet",
                    min_steps=steps,
                    max_steps=steps,
                    check_every=250,
                    drift_tol=0.0,  # run the full arm; measure stability, not convergence
                    threads=threads,
                )
                rows.append(
                    {
                        k: r[k]
                        for k in (
                            "u0", "ma_max", "ma2", "tau_plus", "collision", "lambda_trt",
                            "tau_minus", "steps", "finite", "elapsed_s",
                        )
                    }
                    | (
                        {
                            "interior_u_l2_rel_pct": r["interior"]["u_l2_rel_pct"],
                            "interior_v_l2_rel_pct": r["interior"]["v_l2_rel_pct"],
                            "full_u_l2_rel_pct": r["full"]["u_l2_rel_pct"],
                            "full_v_l2_rel_pct": r["full"]["v_l2_rel_pct"],
                        }
                        if r["finite"]
                        else {"interior_u_l2_rel_pct": None, "interior_v_l2_rel_pct": None,
                              "full_u_l2_rel_pct": None, "full_v_l2_rel_pct": None}
                    )
                )
    return {"arm": "stability", "ny": ny, "re": RE_DEFAULT, "rows": rows}


def arm_ladder(
    *,
    grids=(32, 64, 128),
    u0: float = 0.0075,
    collision: str = "trt",
    lam_trt: float = 3.0 / 16.0,
    max_steps: int = 30000,
    min_steps: int = 2000,
    threads: int = 8,
) -> dict:
    """Grid refinement at fixed (low) Ma: observed order from the interior window."""
    rows = []
    for ny in grids:
        rows.append(
            run_case(
                ny,
                u0=u0,
                collision=collision,
                lam_trt=lam_trt,
                outlet="dirichlet",
                min_steps=min_steps,
                max_steps=max_steps,
                threads=threads,
            )
        )
    ok = [r for r in rows if r["finite"]]
    h = [1.0 / r["ny"] for r in ok]
    eu = [r["interior"]["u_l2_abs_over_u0_pct"] / 100.0 for r in ok]
    ev = [r["interior"]["v_l2_abs_over_u0_pct"] / 100.0 for r in ok]
    fu = [r["full"]["u_l2_rel_pct"] / 100.0 for r in ok]
    fv = [r["full"]["v_l2_rel_pct"] / 100.0 for r in ok]
    order_u = observed_order(h, eu)
    order_v = observed_order(h, ev)
    p_u = order_u[-1]["p"] if order_u else float("nan")
    p_v = order_v[-1]["p"] if order_v else float("nan")
    return {
        "arm": "ladder",
        "u0": u0,
        "collision": collision,
        "lambda_trt": lam_trt if collision == "trt" else None,
        "rows": rows,
        "interior_order_u": order_u,
        "interior_order_v": order_v,
        "full_order_u_relL2": observed_order(h, fu),
        "full_order_v_relL2": observed_order(h, fv),
        "richardson_u_pct": 100.0 * richardson(h, eu, p_u) if math.isfinite(p_u) else float("nan"),
        "richardson_v_pct": 100.0 * richardson(h, ev, p_v) if math.isfinite(p_v) else float("nan"),
    }


def arm_check(ny: int = 16, steps: int = 400, u0: float = 0.03) -> dict:
    """Cheap self-check: reference certificate + TRT/BGK equivalence + invariants."""
    field = make_field(ny, re=RE_DEFAULT, u0=u0)
    cert = field.ns_residual()
    bgk = run_case(ny, u0=u0, collision="bgk", min_steps=steps, max_steps=steps,
                   check_every=steps, drift_tol=0.0, threads=2)
    trt = run_case(ny, u0=u0, collision="trt", min_steps=steps, max_steps=steps,
                   check_every=steps, drift_tol=0.0, threads=2)
    ref = {
        "u_l2_rel_pct": bgk["full"]["u_l2_rel_pct"],
        "v_l2_rel_pct": bgk["full"]["v_l2_rel_pct"],
    }
    return {
        "ny": ny,
        "steps": steps,
        "reference_certificate": cert,
        "bgk": ref,
        "trt": {"u_l2_rel_pct": trt["full"]["u_l2_rel_pct"],
                "v_l2_rel_pct": trt["full"]["v_l2_rel_pct"]},
        # TRT must relax to (almost) the same steady solution as BGK: same tau+
        "trt_minus_bgk_abs_pct_u": abs(trt["full"]["u_l2_rel_pct"] - bgk["full"]["u_l2_rel_pct"]),
        "trt_minus_bgk_abs_pct_v": abs(trt["full"]["v_l2_rel_pct"] - bgk["full"]["v_l2_rel_pct"]),
    }


def arm_pressure_identity(ny: int = 16) -> dict:
    """collide_trt must conserve mass and momentum exactly (both parts are differences)."""
    torch.set_num_threads(2)
    field = make_field(ny, re=RE_DEFAULT, u0=0.03)
    u = torch.as_tensor(field.u(), dtype=torch.float64)
    v = torch.as_tensor(field.v(), dtype=torch.float64)
    f = equilibrium(torch.ones((ny, field.nx), dtype=torch.float64), u, v)
    tau = field.tau
    post = collide_trt(f, tau)
    rho_b, ux_b, uy_b = macroscopic(f)
    rho_a, ux_a, uy_a = macroscopic(post)
    return {
        "mass_rel_change": float((post.sum() - f.sum()) / f.sum()),
        "momentum_x_abs": float((ux_a - ux_b).abs().max()),
        "momentum_y_abs": float((uy_a - uy_b).abs().max()),
        "rho_max_abs_change": float((rho_a - rho_b).abs().max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kovasznay TRT convergence arm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--out", default=None)

    p_stab = sub.add_parser("stability")
    p_stab.add_argument("--ny", type=int, default=32)
    p_stab.add_argument("--u0s", type=float, nargs="+",
                        default=[0.03, 0.015, 0.0075, 0.00375, 0.001875])
    p_stab.add_argument("--steps", type=int, default=3000)
    p_stab.add_argument("--threads", type=int, default=8)
    p_stab.add_argument("--out", default=None)

    p_lad = sub.add_parser("ladder")
    p_lad.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    p_lad.add_argument("--u0", type=float, default=0.0075)
    p_lad.add_argument("--collision", default="trt", choices=["trt", "bgk"])
    p_lad.add_argument("--lambda-trt", type=float, default=3.0 / 16.0)
    p_lad.add_argument("--max-steps", type=int, default=30000)
    p_lad.add_argument("--min-steps", type=int, default=2000)
    p_lad.add_argument("--threads", type=int, default=8)
    p_lad.add_argument("--out", default=None)

    p_fld = sub.add_parser("field")
    p_fld.add_argument("--ny", type=int, default=64)
    p_fld.add_argument("--u0", type=float, default=0.0075)
    p_fld.add_argument("--max-steps", type=int, default=20000)
    p_fld.add_argument("--min-steps", type=int, default=3000)
    p_fld.add_argument("--threads", type=int, default=4)
    p_fld.add_argument("--out", default="artifacts/kovasznay_trt/field_ny64.json")

    p_all = sub.add_parser("all")
    p_all.add_argument("--out", default="artifacts/kovasznay_trt/trt_convergence.json")
    p_all.add_argument("--threads", type=int, default=16)
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "check":
        res = {"check": arm_check(), "pressure_identity": arm_pressure_identity()}
    elif args.cmd == "stability":
        res = arm_stability(ny=args.ny, u0s=tuple(args.u0s), steps=args.steps,
                            threads=args.threads)
    elif args.cmd == "ladder":
        res = arm_ladder(grids=tuple(args.grids), u0=args.u0, collision=args.collision,
                         lam_trt=args.lambda_trt, max_steps=args.max_steps,
                         min_steps=args.min_steps, threads=args.threads)
    elif args.cmd == "field":
        out_path = Path(args.out)
        npz = out_path.with_suffix(".npz")
        res = run_case(args.ny, u0=args.u0, collision="trt", min_steps=args.min_steps,
                       max_steps=args.max_steps, threads=args.threads, field_out=npz)
        res["field_npz"] = str(npz)
    elif args.cmd == "all":
        res = {
            "check": arm_check(),
            "pressure_identity": arm_pressure_identity(),
            "stability_bgk_vs_trt": arm_stability(steps=3000, threads=args.threads),
            "ladder_trt_u0_0p0075": arm_ladder(u0=0.0075, threads=args.threads),
            "ladder_trt_u0_0p00375": arm_ladder(u0=0.00375, threads=args.threads),
            "ladder_bgk_u0_0p03_control": arm_ladder(u0=0.03, collision="bgk",
                                                     threads=args.threads),
        }
    else:  # pragma: no cover
        raise SystemExit(2)

    out = getattr(args, "out", None)
    if out is None and args.cmd != "all":
        pass
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())