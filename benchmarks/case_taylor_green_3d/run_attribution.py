#!/usr/bin/env python3
"""CASE-TAYLOR-GREEN-3D round 2 — attribution of the metric floor + a consistent initial condition.

Round 1 (``run_case.py``, mission 15374) reported

* M2 (energy decay-exponent error, least squares over the *whole* history) = 0.346–0.366 %
  at every grid — suspiciously grid independent, and
* consecutive L2 convergence orders of 0.84–1.19 (only the 24→48 pair reached 2.045),
  explained in round 1 as "a grid-independent error floor".

This script tests the two protocol-level explanations predeclared in
``prereg_round2.json``:

* **H1** — M2 is inflated by the startup transient of the *equilibrium* initial
  condition, so that the fit sees a fast initial settling instead of the asymptotic rate.
  Measured by refitting the same trace over later windows and by fitting the slope of
  ``ln(E_num/E_exact)`` (the asymptotic rate error, IC independent).
* **H2/H3** — the L2 floor is a *state* error of the initial condition:
  ``f = f^eq(u0)`` carries no viscous stress, whereas the exact solution has
  ``sigma = 2 rho nu S`` at t = 0.  Adding the first-order Chapman–Enskog part

      f^(1)_i = K w_i Q_iab S_ab,      Q_iab = e_ia e_ib - c_s^2 delta_ab,
      K = -rho (tau_l - theta) / c_s^2,   theta in {0, 1/2}

  (the two conventions found in the literature; ``theta = 0`` is used by default and
  the choice is *measured*, not assumed) removes it.

Nothing here is tuned after the fact: every number is written to
``benchmarks/case_taylor_green_3d/artifacts/attribution_*.json`` and the criteria are
in ``prereg_round2.json``.

Usage
-----
    python benchmarks/case_taylor_green_3d/run_attribution.py --arm all
    python benchmarks/case_taylor_green_3d/run_attribution.py --arm traces --n 24
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_spec = importlib.util.spec_from_file_location("tg3d_run_case", os.path.join(HERE, "run_case.py"))
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

from tensorlbm.d3q27 import C, W_EXACT64, collide_bgk27, equilibrium27, macroscopic27  # noqa: E402
from tensorlbm.lbm_step_d3q27 import lbm_step_correct_27  # noqa: E402

CS2 = 1.0 / 3.0
FIT_WINDOWS = (0, 5, 10, 20, 50, 100)


# --------------------------------------------------------------------------- #
# initial conditions
# --------------------------------------------------------------------------- #
def consistent_init(case: str, n: int, amp: float, tau: float, ic: str = "tau") -> torch.Tensor:
    """Thin alias of the canonical builder in ``run_case.init_distribution``."""
    return rc.init_distribution(case, n, amp, tau, ic)


# --------------------------------------------------------------------------- #
# analysis helpers
# --------------------------------------------------------------------------- #
def _slope(ts, ys) -> float:
    n = len(ts)
    tm, ym = sum(ts) / n, sum(ys) / n
    num = sum((t - tm) * (y - ym) for t, y in zip(ts, ys))
    den = sum((t - tm) ** 2 for t in ts)
    return num / den


def analyse(hist: list[dict], lam_exact: float) -> dict:
    """Windowed ln E fits (round-1 M2 reproduces at t_start=0) + asymptotic rate error."""
    windows = {}
    for t0 in FIT_WINDOWS:
        sel = [h for h in hist if h["t"] >= t0 and h["E"] > 0.0]
        if len(sel) < 3:
            continue
        lam = -_slope([h["t"] for h in sel], [math.log(h["E"]) for h in sel])
        windows[str(t0)] = {
            "lambda_fit": lam,
            "decay_exponent_rel_err_pct": 100.0 * (lam - lam_exact) / lam_exact,
            "n_points": len(sel),
        }
    # asymptotic rate: ln(E_num / E_exact) ~ -(lambda_num - lambda_exact) t
    half = [h for h in hist if h["t"] >= hist[len(hist) // 2]["t"] and h["E_exact"] > 0.0]
    slope = _slope([h["t"] for h in half], [math.log(h["E"] / h["E_exact"]) for h in half])
    delta = -slope
    # startup transient: first sampled t at which the energy offset stops moving
    offs = [(h["t"], abs(h["E_rel_err_pct"] - hist[-1]["E_rel_err_pct"])) for h in hist]
    settle = None
    for (t, d) in offs:
        if d < 0.1 * abs(hist[-1]["E_rel_err_pct"]):
            settle = t
            break
    return {
        "fit_windows": windows,
        "M2_round1_pct": windows.get("0", {}).get("decay_exponent_rel_err_pct"),
        "asym_rate_err_pct": 100.0 * delta / lam_exact,
        "asym_lambda_num": lam_exact + delta,
        "offset_final_pct": hist[-1]["E_rel_err_pct"],
        "offset_at_settle_pct": settle,
        "transient_settle_steps": settle,
        "l2_final": hist[-1]["l2_rel"],
    }


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def run_trace(case: str, n: int, tau: float, amp: float, steps: int, sample: int, ic: str = "tau") -> dict:
    nu = (tau - 0.5) / 3.0
    k2 = rc.k2_of(case, n)
    lam_exact = 2.0 * nu * k2
    u0 = rc.init_fields(case, n, amp)
    f = consistent_init(case, n, amp, tau, ic)
    _, ux0, uy0, uz0 = macroscopic27(f)
    ic_velocity_err = rc.l2_rel((ux0, uy0, uz0), u0)
    solid = torch.zeros(u0[0].shape, dtype=torch.bool)
    e0 = rc.kinetic_energy(*u0)
    m0 = float(f.sum())
    mass_drift = 0.0
    finite = True
    hist = [{"t": 0, "E": e0, "E_exact": e0, "E_rel_err_pct": 0.0, "l2_rel": 0.0}]
    t0 = time.time()
    with torch.no_grad():
        for step in range(1, steps + 1):
            f = lbm_step_correct_27(f, collide_bgk27, tau, solid, 0.0, far_field_bc_fn=None)
            if step % sample == 0 or step == steps:
                rho, ux, uy, uz = macroscopic27(f)
                finite = finite and bool(torch.isfinite(f).all())
                mass_drift = max(mass_drift, abs(float(f.sum()) - m0) / m0)
                e = rc.kinetic_energy(ux, uy, uz)
                ex = rc.exact_fields(case, n, amp, nu, step)
                ee = rc.kinetic_energy(*ex)
                hist.append(
                    {
                        "t": step,
                        "E": e,
                        "E_exact": ee,
                        "E_rel_err_pct": 100.0 * (e - ee) / ee,
                        "l2_rel": rc.l2_rel((ux, uy, uz), ex),
                    }
                )
    rec = {
        "case": case,
        "ic": ic,
        "n": n,
        "tau": tau,
        "nu": nu,
        "k2": k2,
        "lambda_exact": lam_exact,
        "amp": amp,
        "ma": amp * math.sqrt(3.0),
        "steps": steps,
        "sample_every": sample,
        "finite": finite,
        "mass_drift_rel": mass_drift,
        "ic_velocity_rel_err": ic_velocity_err,
        "wall_s": time.time() - t0,
        "history": hist,
    }
    rec.update(analyse(hist, lam_exact))
    return rec


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def arm_traces(grids=(24, 32), taus_tau=0.6, amp=2e-3, steps=400, sample=5, ics=("eq", "half", "tau")):
    out = []
    for n in grids:
        for ic in ics:
            r = run_trace("tg", n, taus_tau, amp, steps, sample, ic)
            out.append(r)
            print(
                f"trace N={n} ic={ic}: M2(round1)={r['M2_round1_pct']:+.4f}%  "
                f"asym={r['asym_rate_err_pct']:+.4f}%  offset_end={r['offset_final_pct']:+.4f}%  "
                f"l2={r['l2_final']:.3e}  wall={r['wall_s']:.0f}s"
            )
    return out


def arm_grid(grids=(24, 32, 48, 64), tau=0.6, amp=2e-3, steps=500, ic="tau"):
    out = []
    for n in grids:
        r = run_trace("tg", n, tau, amp, steps, max(steps // 10, 10), ic)
        out.append(r)
        print(
            f"grid N={n} ic={ic}: l2={r['l2_final']:.4e}  asym={r['asym_rate_err_pct']:+.4f}%  "
            f"offset_end={r['offset_final_pct']:+.4f}%  wall={r['wall_s']:.0f}s"
        )
    l2 = {str(r["n"]): r["l2_final"] for r in out}
    orders = {}
    keys = sorted(l2, key=int)
    for a, b in zip(keys, keys[1:]):
        if l2[a] > 0 and l2[b] > 0:
            orders[f"{a}_to_{b}"] = math.log2(l2[a] / l2[b])
    order_24_48 = (
        math.log2(l2["24"] / l2["48"]) if "24" in l2 and "48" in l2 and l2["48"] > 0 else None
    )
    return {"runs": out, "l2_by_grid": l2, "orders": orders, "order_24_48": order_24_48}


def arm_rate(n=32, tau=0.6, amp=2e-3, steps=400, sample=10, ic="tau", amps=(2.5e-4, 2e-3, 2e-2), taus=(0.55, 0.6, 1.0)):
    amp_runs, tau_runs = [], []
    for a in amps:
        r = run_trace("tg", n, tau, a, steps, sample, ic)
        amp_runs.append(r)
        print(f"rate amp={a:.1e} Ma={r['ma']:.4f}: asym={r['asym_rate_err_pct']:+.4f}%  l2={r['l2_final']:.3e}")
    for t in taus:
        r = run_trace("tg", n, t, amp, steps, sample, ic)
        tau_runs.append(r)
        print(f"rate tau={t}: lambda={r['lambda_exact']:.4e}: asym={r['asym_rate_err_pct']:+.4f}%  l2={r['l2_final']:.3e}")
    return {"amp_runs": amp_runs, "tau_runs": tau_runs}


def _loglog_slope(xs, ys):
    xs2 = [math.log(abs(x)) for x in xs]
    ys2 = [math.log(abs(y)) for y in ys]
    return _slope(xs2, ys2)


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def evaluate(traces, grid, rate, eq_grid_baseline) -> dict:
    by = {(r["n"], r["ic"]): r for r in traces}

    # R1: whole-history M2 vs asymptotic rate error, N=32, eq IC
    r32 = by[(32, "eq")]
    ratio1 = abs(r32["M2_round1_pct"]) / abs(r32["asym_rate_err_pct"])
    r1 = {"value": ratio1, "pass": ratio1 >= 5.0, "M2_pct": r32["M2_round1_pct"], "asym_pct": r32["asym_rate_err_pct"]}

    # R2: IC repair factor on the L2 field error, CE-IC grid arm vs round-1 eq-IC baseline
    repair = {}
    for k in sorted(grid["l2_by_grid"], key=int):
        if k in eq_grid_baseline and grid["l2_by_grid"][k] > 0:
            repair[k] = eq_grid_baseline[k] / grid["l2_by_grid"][k]
    r2 = {"repair_factor_by_grid": repair, "pass": all(v >= 5.0 for v in repair.values()) and len(repair) >= 3}

    # R3: order recovery
    min_pair = min(grid["orders"].values()) if grid["orders"] else float("nan")
    r3 = {
        "order_24_to_48": grid["order_24_48"],
        "min_consecutive_order": min_pair,
        "pass": bool(grid["order_24_48"] is not None and grid["order_24_48"] >= 1.8 and min_pair >= 1.7),
    }

    # R4: rate accuracy everywhere
    all_asym = (
        [r["asym_rate_err_pct"] for r in traces]
        + [r["asym_rate_err_pct"] for r in grid["runs"]]
        + [r["asym_rate_err_pct"] for r in rate["amp_runs"]]
        + [r["asym_rate_err_pct"] for r in rate["tau_runs"]]
    )
    worst = max(abs(v) for v in all_asym)
    r4 = {"worst_abs_asym_pct": worst, "pass": worst < 0.1, "n_points": len(all_asym)}

    # R5: Ma scaling of the rate error
    slope = _loglog_slope([r["amp"] for r in rate["amp_runs"]], [r["asym_rate_err_pct"] for r in rate["amp_runs"]])
    r5 = {"loglog_slope_vs_amp": slope, "ma2_dominated": bool(slope > 1.0)}

    return {
        "R1_M2_inflation": r1,
        "R2_ic_repair": r2,
        "R3_order_recovery": r3,
        "R4_rate_accuracy": r4,
        "R5_ma_scaling": r5,
        "eq_grid_baseline_l2": eq_grid_baseline,
        "grid_l2": grid["l2_by_grid"],
        "grid_orders": grid["orders"],
    }


def _save(out: str, name: str, payload) -> str:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["all", "traces", "grid", "rate", "report"])
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--steps", type=int, default=0, help="override every arms' step budget (debug)")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("LBM_THREADS", "4")))

    written = []
    if args.arm in ("all", "traces"):
        steps = args.steps or 400
        traces = arm_traces(grids=(24, 32) if steps > 100 else (24,), steps=steps, sample=5)
        written.append(_save(args.out, "attribution_traces.json", {"arm": "traces", "runs": traces}))

    if args.arm in ("all", "grid"):
        steps = args.steps or 500
        grid = arm_grid(steps=steps)
        written.append(_save(args.out, "attribution_grid.json", {"arm": "grid", **grid}))

    if args.arm in ("all", "rate"):
        steps = args.steps or 400
        rate = arm_rate(steps=steps)
        written.append(_save(args.out, "attribution_rate.json", {"arm": "rate", **rate}))

    if args.arm != "report":
        print("\nwrote:")
        for w in written:
            print(" ", w)
        return 0

    traces = json.load(open(os.path.join(args.out, "attribution_traces.json")))["runs"]
    d = json.load(open(os.path.join(args.out, "attribution_grid.json")))
    grid = {k: d[k] for k in ("runs", "l2_by_grid", "orders", "order_24_48")}
    d = json.load(open(os.path.join(args.out, "attribution_rate.json")))
    rate = {"amp_runs": d["amp_runs"], "tau_runs": d["tau_runs"]}

    base = json.load(open(os.path.join(HERE, "artifacts", "convergence.json")))
    eq_grid_baseline = {str(r["n"]): r["M3_l2_field_rel_err_pct"] / 100.0 for r in base["runs"]}
    verdict = evaluate(traces, grid, rate, eq_grid_baseline)
    verdict["round1_reported"] = {
        "M2_pct": {str(r["n"]): r["M2_decay_exponent_rel_err_pct"] for r in base["runs"]},
        "M4_order_24_48": base["M4_observed_order"],
        "consecutive_orders": {"24_32": base.get("M4_order_24_32"), "48_64": base.get("M4_order_48_64")},
        "source": "benchmarks/case_taylor_green_3d/artifacts/convergence.json",
    }
    written.append(_save(args.out, "attribution_summary.json", verdict))

    print("\n=== criteria ===")
    print(json.dumps({k: v for k, v in verdict.items() if k.startswith("R")}, indent=1))
    print("\nwrote:")
    for w in written:
        print(" ", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())