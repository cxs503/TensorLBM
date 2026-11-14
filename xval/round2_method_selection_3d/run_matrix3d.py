#!/usr/bin/env python3
"""3-D method-selection matrix: D3Q19 collision x IC protocol on the exact ABC flow.

Anchor ``XVAL-LBM-ROUND2-METHOD-SELECTION-3D``.  Predeclaration:
``predeclared_3d.json`` (written before any run).  Reference: the closed-form
Navier-Stokes solution of the Beltrami (ABC) flow, ``u(t) = u(0) exp(-nu k^2 t)``;
the analytic field helpers are imported from the repo's existing case
(``benchmarks/case_taylor_green_3d/run_case.py``) instead of duplicated.

Every emitted number is produced by this script from the measured arrays; the
closed-form reference is recomputed inside each run (no hand-copied tables).

Usage::

    python3 xval/round2_method_selection_3d/run_matrix3d.py --smoke   # 1 arm, 1 grid
    python3 xval/round2_method_selection_3d/run_matrix3d.py           # full matrix
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "case_taylor_green_3d"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.d3q19 import C, W_EXACT64, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import (  # noqa: E402
    collide_bgk3d,
    collide_mrt3d,
    collide_rlbm3d,
    collide_trt3d,
    stream3d,
)

import run_case as tg3d  # noqa: E402  (existing case: analytic ABC fields + l2_rel)

HERE = REPO / "xval" / "round2_method_selection_3d"
RUNS = HERE / "runs"
DT = torch.float64
CS2 = 1.0 / 3.0
AMP = 0.02


def collide(arm: str, f: torch.Tensor, tau: float) -> torch.Tensor:
    if arm == "bgk":
        return collide_bgk3d(f, tau)
    if arm == "trt":
        return collide_trt3d(f, tau, 3.0 / 16.0)
    if arm == "mrt":
        return collide_mrt3d(f, tau)
    if arm == "rlbm":
        return collide_rlbm3d(f, tau)
    raise ValueError(arm)


def analytic_strain_rate(n: int, amp: float):
    """Symmetric strain-rate tensor S_ab of the ABC field (analytic derivatives)."""
    k = 2.0 * math.pi / n
    x, y, z = tg3d._coords(n)
    A = amp
    zero = torch.zeros_like(x)
    grad = [
        [zero, -A * k * torch.sin(k * y), A * k * torch.cos(k * z)],   # u_x
        [A * k * torch.cos(k * x), zero, -A * k * torch.sin(k * z)],   # u_y
        [-A * k * torch.sin(k * x), A * k * torch.cos(k * y), zero],   # u_z
    ]
    S = [[0.5 * (grad[a][b] + grad[b][a]) for b in range(3)] for a in range(3)]
    return torch.stack([torch.stack(row) for row in S])


def init_distribution(n: int, amp: float, tau: float, ic: str):
    u = tg3d.init_fields("abc", n, amp)
    rho = torch.ones_like(u[0])
    f = equilibrium3d(rho, *u)
    if ic == "eq":
        return f, u
    if ic != "tau":
        raise ValueError(ic)
    S = analytic_strain_rate(n, amp)
    e = C.to(DT)
    Q = torch.einsum("ia,ib->iab", e, e) - CS2 * torch.eye(3, dtype=DT)
    QS = torch.einsum("iab,ab...->i...", Q, S)
    K = -tau / CS2          # theta = 0 ("tau" variant of the existing case)
    return f + K * W_EXACT64[:, None, None, None] * QS, u


def run_arm(arm: str, n: int, tau: float, ic: str, steps: int, cadence: int = 10) -> dict:
    t0 = time.time()
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    k2 = k * k
    f, u0 = init_distribution(n, AMP, tau, ic)
    norm0 = sum(float((c * c).sum()) for c in u0)

    ts, amps = [], []
    for step in range(1, steps + 1):
        f = collide(arm, f, tau)
        f = stream3d(f)
        if step % cadence == 0 or step == steps:
            if not torch.isfinite(f).all():
                return {"arm": arm, "N": n, "tau": tau, "ic": ic, "status": "diverged",
                        "step": step, "nu": nu, "k2": k2, "wall_s": time.time() - t0}
            _, ux, uy, uz = macroscopic3d(f)
            proj = sum(float((a * b).sum()) for a, b in zip((ux, uy, uz), u0)) / norm0
            ts.append(step)
            amps.append(proj)

    t = np.asarray(ts, dtype=float)
    a = np.asarray(amps, dtype=float)
    lo = int(0.2 * len(t))
    slope = float(np.polyfit(t[lo:], np.log(a[lo:]), 1)[0])
    nu_fit = -slope / k2
    err = 100.0 * (nu_fit - nu) / nu
    decay = math.exp(-nu * k2 * steps)
    ex = [decay * c for c in u0]
    return {
        "arm": arm, "N": n, "tau": tau, "ic": ic, "status": "ok", "steps": steps,
        "nu": nu, "k2": k2, "nu_fit": nu_fit, "err_nu_pct": err,
        "l2_rel_final": float(tg3d.l2_rel([ux, uy, uz], ex)),
        "amp_final": float(a[-1]),
        "max_abs_u": float(max(abs(v).max() for v in (ux, uy, uz))),
        "fit_points": int(len(t) - lo), "wall_s": time.time() - t0,
        "cadence": cadence, "amp0": AMP,
    }


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    pre = json.loads((HERE / "predeclared_3d.json").read_text())
    runs: list[dict] = []

    if args.smoke:
        runs.append(run_arm("bgk", 16, 0.56, "tau", 200))
    else:
        for n in pre["protocol"]["N"]:
            steps = pre["protocol"]["steps_main"][str(n)]
            for arm in pre["protocol"]["collide"]:
                key = arm.split("(")[0]
                for ic in pre["protocol"]["ic"]:
                    runs.append(run_arm(key, n, pre["protocol"]["tau_main"], ic, steps))
                    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                      for k, v in runs[-1].items()}), flush=True)
        for arm in pre["protocol"]["collide"]:
            key = arm.split("(")[0]
            runs.append(run_arm(key, pre["protocol"]["steps_low_N"], pre["protocol"]["tau_low"],
                                "tau", pre["protocol"]["steps_low"]))
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in runs[-1].items()}), flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    keys = ["arm", "ic", "N", "tau", "status", "nu", "nu_fit", "err_nu_pct",
            "l2_rel_final", "max_abs_u", "fit_points", "wall_s"]
    with open(RUNS / "table_3d.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k) for k in keys})

    main_runs = [r for r in runs if r["status"] == "ok" and r["tau"] == pre["protocol"]["tau_main"]]

    def get(arm, ic, n):
        for r in main_runs:
            if r["arm"] == arm and r["ic"] == ic and r["N"] == n:
                return r
        return None

    spread, ic_effect = {}, {}
    for n in pre["protocol"]["N"]:
        vals = [get(a, "tau", n)["err_nu_pct"] for a in ("bgk", "trt", "mrt", "rlbm")
                if get(a, "tau", n) and get(a, "tau", n)["status"] == "ok"]
        if vals:
            spread[f"operator_spread_pct_N{n}"] = max(vals) - min(vals)
        for a in ("bgk", "trt", "mrt", "rlbm"):
            r0, r1 = get(a, "eq", n), get(a, "tau", n)
            if r0 and r1 and r0["status"] == "ok" and r1["status"] == "ok":
                ic_effect[f"{a}_N{n}_ic_shift_pct"] = r1["err_nu_pct"] - r0["err_nu_pct"]

    summary = {
        "schema": "xval-round2-method-selection-3d/summary/v1",
        "anchor": pre["anchor"],
        "predeclared": "xval/round2_method_selection_3d/predeclared_3d.json",
        "case": pre["case"],
        "band_pct": 100 * pre["acceptance"]["band"],
        "runs": len(runs),
        "runs_ok": len([r for r in runs if r["status"] == "ok"]),
        "runs_failed": [{"arm": r["arm"], "N": r["N"], "tau": r["tau"], "ic": r["ic"],
                        "status": r["status"], "step": r.get("step")}
                       for r in runs if r["status"] != "ok"],
        "main_matrix": {"N": pre["protocol"]["N"], "tau": pre["protocol"]["tau_main"],
                        "arms": pre["protocol"]["collide"], "ic": pre["protocol"]["ic"]},
        "runs_all": runs,
        "stability_rung": [r for r in runs if r["tau"] == pre["protocol"]["tau_low"]],
        "operator_spread": spread,
        "ic_effect": ic_effect,
        "worst_abs_err_pct": max(abs(r["err_nu_pct"]) for r in main_runs) if main_runs else None,
        "within_band": bool(main_runs) and all(abs(r["err_nu_pct"]) <= 100 * pre["acceptance"]["band"]
                                              for r in main_runs),
    }
    (RUNS / "summary_3d.json").write_text(json.dumps(summary, indent=1, default=float))
    print("wrote", RUNS / "summary_3d.json", "worst |err_nu| =", summary["worst_abs_err_pct"],
          "within band:", summary["within_band"])
    return summary


if __name__ == "__main__":
    main()