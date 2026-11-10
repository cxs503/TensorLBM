#!/usr/bin/env python3
"""Root-cause isolation of the Re_tau = 590 WMLES blow-up (P372W2-N01).

The wall-modelled channel in :mod:`tensorlbm.wall_model_les.channel` converges
at ``Re_tau = 180`` and fails at ``Re_tau = 590`` (recorded in
``docs/wall_model_les_m8872_report.md`` 5).  A failure is only actionable once
it is attributed, so this script runs a **bisection-style scan on the five axes
that can plausibly own it**:

===============  ==========================================================
axis             what a stable/unstable switch would mean
===============  ==========================================================
``re_tau`` (nu)  the margin ``tau - 1/2 = 3 (nu + nu_t)`` shrinks as the
                 molecular viscosity drops; a hard ``re_tau`` boundary at fixed
                 ``ny`` points at the relaxation margin, not at the wall model
``cs`` (SGS)     raising the Smagorinsky constant re-opens a run => the deficit
                 was sub-grid dissipation, not the coupling
``exchange``     moving the wall-model sampling row decides whether the
                 exchange/first-cell feedback is the trigger
``width`` (box)  box size controls how much resolved turbulence can exist; a
                 width threshold means the run was under-resolved, not unstable
``coupling``     ``none`` (pure bounce-back, no wall model) isolates the wall
                 model from the collision/streaming operators
===============  ==========================================================

Every case is one ``run_channel_wmles`` call in a short box; the per-case JSON
keeps the whole log history (with cell-level stability margins from
:func:`tensorlbm.wall_model_les.channel.stability_diagnostics`), the first
non-finite step and the exception text, so the attribution can be re-derived
from the artifacts instead of from prose.

Usage
-----
    python benchmarks/wall_model_les/re590_stability.py --sweep all --steps 600
    python benchmarks/wall_model_les/re590_stability.py --sweep re --bisect
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.channel import (  # noqa: E402
    ChannelWMLESConfig,
    run_channel_wmles,
)

DEFAULT_OUT = REPO_ROOT / "artifacts" / "wall_model_les_peak"


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def run_case(
    *,
    re_tau: float = 590.0,
    cs: float = 0.1,
    ny: int = 49,
    nx: int = 8,
    nz: int = 8,
    exchange_offset: int = 1,
    wall_coupling: str = "assimilate",
    wall_relaxation: float = 1.0,
    wall_model: str = "equilibrium",
    u_tau: float = 0.02,
    steps: int = 600,
    positivity_limiter: bool = True,
    label: str = "",
    out_dir: Path | None = None,
) -> dict:
    """Run one case and fold it into a stability record."""
    config = ChannelWMLESConfig(
        nx=nx,
        nz=nz,
        ny=ny,
        re_tau=float(re_tau),
        u_tau=float(u_tau),
        smagorinsky_cs=float(cs),
        steps=int(steps),
        averaging_start=int(0.8 * steps),
        stat_interval=10,
        log_interval=max(steps // 12, 1),
        wall_model=wall_model,
        wall_coupling=wall_coupling,
        wall_relaxation=float(wall_relaxation),
        exchange_offset=int(exchange_offset),
        positivity_limiter=positivity_limiter,
        diagnostics=True,
        run_name=label or f"re{re_tau:g}_cs{cs:g}_ny{ny}_ex{exchange_offset}_{wall_coupling}",
    )
    started = time.time()
    error: str | None = None
    stats: dict | None = None
    try:
        stats = run_channel_wmles(config)
    except Exception as exc:  # noqa: BLE001 - a crash is a result here
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    history = (stats or {}).get("history") or []
    first_nonfinite: int | None = None
    last_finite: dict | None = None
    for row in history:
        values = [v for k, v in row.items() if isinstance(v, (int, float)) and k != "step"]
        bad = [v for v in values if isinstance(v, float) and not math.isfinite(v)]
        if bad and first_nonfinite is None:
            first_nonfinite = row.get("step")
        if not bad:
            last_finite = row
    converged = bool(
        stats is not None
        and error is None
        and _finite((stats or {}).get("u_tau_model_mean"))
        and first_nonfinite is None
    )
    record = {
        "label": config.run_name,
        "params": {
            "re_tau": float(re_tau),
            "cs": float(cs),
            "ny": int(ny),
            "nx": int(nx),
            "nz": int(nz),
            "exchange_offset": int(exchange_offset),
            "wall_coupling": wall_coupling,
            "wall_relaxation": float(wall_relaxation),
            "wall_model": wall_model,
            "u_tau": float(u_tau),
            "steps": int(steps),
            "positivity_limiter": bool(positivity_limiter),
        },
        "derived": {
            "nu": config.nu,
            "tau_molecular": config.tau,
            "tau_minus_half_molecular": config.tau - 0.5,
            "delta_y_plus": config.delta_y_plus,
            "y_plus_exchange": config.y_of_node(config.j_exchange) * config.u_tau / config.nu,
            "H": config.H,
            "h": config.h,
            "accel": config.accel,
        },
        "converged": converged,
        "first_nonfinite_step": first_nonfinite,
        "error": error,
        "elapsed_s": round(elapsed, 2),
        "last_finite_log": last_finite,
        "u_tau_model_mean": (stats or {}).get("u_tau_model_mean"),
        "u_tau_model_over_balance": (stats or {}).get("u_tau_model_over_balance"),
        "re_tau_measured_last": (history[-1].get("re_tau_measured") if history else None),
        "history": history,
    }
    if out_dir is not None:
        out = Path(out_dir) / "stability_cases"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{config.run_name}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def _describe(rec: dict) -> str:
    status = "STABLE" if rec["converged"] else "BLOWUP"
    last = rec["last_finite_log"] or {}
    return (
        f"{rec['label']:<52s} {status:<6s} "
        f"dy+={rec['derived']['delta_y_plus']:6.2f} "
        f"tau-1/2={rec['derived']['tau_minus_half_molecular']:.3e} "
        f"ub={last.get('u_bulk', float('nan')):.4f} "
        f"tau_margin={last.get('tau_minus_half_min', float('nan')):.3e} "
        f"{'first_nan=' + str(rec['first_nonfinite_step']) if rec['first_nonfinite_step'] else ''}"
        f"{' err=' + (rec['error'] or '')[:40] if rec['error'] else ''}"
    )


def bisect_boundary(predicate, lo: float, hi: float, *, tol: float = 5.0, log=print) -> dict:
    """Bisect the largest stable value of a scalar knob.

    ``predicate(x) -> bool`` must be monotone (True = stable) on ``[lo, hi]``.
    The returned dict records every probe, so the boundary can be re-derived.
    """
    probes: list[dict] = []
    p_lo = predicate(lo)
    probes.append({"x": lo, "stable": p_lo})
    if not p_lo:
        return {"lo": lo, "hi": lo, "boundary": None, "probes": probes, "note": "lower endpoint unstable"}
    p_hi = predicate(hi)
    probes.append({"x": hi, "stable": p_hi})
    if p_hi:
        return {"lo": hi, "hi": hi, "boundary": None, "probes": probes, "note": "upper endpoint stable"}
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        p_mid = predicate(mid)
        probes.append({"x": mid, "stable": p_mid})
        log(f"  bisect x={mid:.1f} stable={p_mid}")
        if p_mid:
            lo = mid
        else:
            hi = mid
    return {"lo": lo, "hi": hi, "boundary": lo, "probes": probes, "note": "bracketed"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="all",
                    choices=["all", "re", "cs", "exchange", "width", "coupling", "dyplus", "limiter", "long"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--ny", type=int, default=49)
    ap.add_argument("--bisect", action="store_true", help="bisect the Re_tau stability boundary at fixed ny")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--single", action="store_true",
                    help="run one case from the explicit flags (for parallel batches)")
    ap.add_argument("--re-tau", type=float, default=590.0)
    ap.add_argument("--cs", type=float, default=0.1)
    ap.add_argument("--exchange-offset", type=int, default=1)
    ap.add_argument("--wall-coupling", default="assimilate")
    ap.add_argument("--wall-relaxation", type=float, default=1.0)
    ap.add_argument("--u-tau", type=float, default=0.02)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(steps=args.steps, nx=args.nx, nz=args.nz, out_dir=out_dir)
    cases: list[dict] = []

    if args.single:
        rec = run_case(
            re_tau=args.re_tau,
            cs=args.cs,
            ny=args.ny,
            nx=args.nx,
            nz=args.nz,
            exchange_offset=args.exchange_offset,
            wall_coupling=args.wall_coupling,
            wall_relaxation=args.wall_relaxation,
            u_tau=args.u_tau,
            steps=args.steps,
            label=args.label or None,
            out_dir=out_dir,
        )
        print(_describe(rec), flush=True)
        path = out_dir / f"stability_scan_single_{rec['label']}.json"
        path.write_text(json.dumps({"sweep": "single", "steps": args.steps, "cases": [
            {k: rec[k] for k in ("label", "params", "derived", "converged", "first_nonfinite_step",
                                 "error", "last_finite_log")}]}, indent=2, default=str))
        print(f"wrote {path}", flush=True)
        return 0

    def add(**kw):
        rec = run_case(**{**common, **kw})
        print(_describe(rec), flush=True)
        cases.append(rec)
        return rec

    if args.sweep in {"all", "re"}:
        print("== axis 1: Re_tau at fixed ny (nu -> 0 / relaxation margin) ==", flush=True)
        for re_tau in (180.0, 300.0, 395.0, 590.0, 790.0):
            add(re_tau=re_tau, ny=args.ny)

    if args.bisect:
        print("== bisection of the Re_tau boundary at fixed ny ==", flush=True)
        caching: dict[float, bool] = {}

        def stable(re_tau: float) -> bool:
            key = round(re_tau, 3)
            if key not in caching:
                rec = run_case(re_tau=re_tau, ny=args.ny, steps=max(args.steps, 400), **{k: v for k, v in common.items() if k != "steps"})
                print(_describe(rec), flush=True)
                cases.append(rec)
                caching[key] = rec["converged"]
            return caching[key]

        bisect = bisect_boundary(stable, 150.0, 900.0, tol=25.0)
        (out_dir / f"bisect_re_tau_ny{args.ny}.json").write_text(json.dumps(bisect, indent=2))
        print("bisection:", json.dumps({k: v for k, v in bisect.items() if k != "probes"}), flush=True)

    if args.sweep in {"all", "cs"}:
        print("== axis 2: sub-grid dissipation (Cs) at Re_tau = 590 ==", flush=True)
        for cs in (0.0, 0.05, 0.1, 0.2, 0.3):
            add(re_tau=590.0, cs=cs, ny=args.ny)

    if args.sweep in {"all", "exchange"}:
        print("== axis 3: wall-model exchange row at Re_tau = 590 ==", flush=True)
        for off in (1, 2, 3):
            add(re_tau=590.0, exchange_offset=off, ny=args.ny)

    if args.sweep in {"all", "width"}:
        print("== axis 4: box width at Re_tau = 590 ==", flush=True)
        for n in (8, 16, 32):
            add(re_tau=590.0, nx=n, nz=n, ny=args.ny)

    if args.sweep in {"all", "coupling"}:
        print("== axis 5: wall coupling (none = pure bounce-back control) ==", flush=True)
        for coupling in ("none", "stress", "assimilate"):
            add(re_tau=590.0, wall_coupling=coupling, ny=args.ny)

    if args.sweep in {"all", "dyplus"}:
        print("== axis 6: grid (dy+) at fixed Re_tau = 590 ==", flush=True)
        for ny in (40, 79, 158):
            add(re_tau=590.0, ny=ny, steps=max(args.steps, 400))

    if args.sweep == "long":
        # Long, box-resolved cases: the short-box scan above cannot develop
        # enough resolved turbulence to break a run, so the instability search
        # has to be repeated at production box widths and run lengths.
        print("== long cases (production box widths / run lengths) ==", flush=True)
        plan = [
            dict(label="L1_re590_ny49_n32", re_tau=590.0, cs=0.1, ny=49, nx=32, nz=32, steps=3000),
            dict(label="L2_re590_ny49_n32_nosgs", re_tau=590.0, cs=0.0, ny=49, nx=32, nz=32, steps=2000),
            dict(label="L3_re590_ny158_n16", re_tau=590.0, cs=0.1, ny=158, nx=16, nz=16, steps=1500),
            dict(label="L4_re590_ny79_n16", re_tau=590.0, cs=0.1, ny=79, nx=16, nz=16, steps=2000),
            dict(label="L5_re790_ny49_n32", re_tau=790.0, cs=0.1, ny=49, nx=32, nz=32, steps=3000),
        ]
        for case in plan:
            label = case.pop("label")
            add(label=label, **case)

    if args.sweep == "limiter":
        print("== positivity limiter on/off at Re_tau = 590 ==", flush=True)
        for flag in (True, False):
            add(re_tau=590.0, positivity_limiter=flag, ny=args.ny)

    summary = [
        {
            "label": c["label"],
            "params": c["params"],
            "derived": c["derived"],
            "converged": c["converged"],
            "first_nonfinite_step": c["first_nonfinite_step"],
            "error": c["error"],
            "last_finite_log": c["last_finite_log"],
        }
        for c in cases
    ]
    path = out_dir / f"stability_scan_{args.sweep}.json"
    path.write_text(json.dumps({"sweep": args.sweep, "steps": args.steps, "cases": summary}, indent=2, default=str))
    print(f"wrote {path} ({len(summary)} cases)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())