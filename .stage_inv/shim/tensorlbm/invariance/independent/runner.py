"""End-to-end driver for the invariance face.

Two entry points are provided:

* ``python -m tensorlbm.invariance.runner plan``      list the job table
* ``python -m tensorlbm.invariance.runner shard I N --outdir results``
  run job ``I``, ``I+N``, ``I+2N``, ... (the unit of work handed to the compute
  cluster: every job is an independent, single-threaded process that writes one
  JSON file, so a job can be re-run in isolation for audit)
* ``python -m tensorlbm.invariance.runner aggregate --indir results --out <json>``
  assemble the job results into the report JSON (tolerances, convergence orders,
  invariance scatter, verdicts).

No number in the report is typed by hand: the aggregate step only computes from
the per-job JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Callable

import numpy as np

from . import cases
from .galilean import fit_power_law

# --- experiment matrix -------------------------------------------------------
# The requested tilts are 0/15/30/45 deg.  A doubly periodic tilted channel array
# requires a rational slope, so 15 deg -> atan(1/4)=14.036 deg and
# 30 deg -> atan(1/2)=26.565 deg are used and the deviation is declared.
CHANNEL_ANGLES = (0.0, 14.036, 26.565, 45.0)
CHANNEL_LEVELS = ((64, 12.0), (128, 24.0), (192, 36.0))
CHANNEL_STEPS = {64: 12000, 128: 20000, 192: 20000}
CHANNEL_NU = 0.10
CYLINDER_ANGLES = (0.0, 14.036, 26.565, 45.0)
CYLINDER_RE = (100.0, 200.0)
CYLINDER_D = 16
TGV_KW = dict(
    n=64,
    nu=0.02,
    amplitude=0.05,
    mach_list=(0.0, 0.025, 0.05, 0.075, 0.10),
    steps=5000,
    sample_every=25,
    discard=500,
)

# acceptance tolerances (declared *before* the runs, see docs)
TOL_CHANNEL_INVARIANCE = 0.03      # peak velocity spread across tilts, finest grid
TOL_CHANNEL_EXTRAPOLATED = 0.01    # Richardson-extrapolated value across tilts
TOL_CYLINDER_CD = 0.03             # Cd spread across tilts
TOL_CYLINDER_ST = 0.02             # St spread across tilts
TOL_TGV_DRIFT = 0.01               # drift error for the largest boost
TOL_TGV_MA2_R2 = 0.99              # quality of the Ma^2 fit


def _channel_name(theta: float, level: int) -> str:
    return f"case01_channel_theta{theta:06.3f}_L{level}"


def _cylinder_name(theta: float, re: float) -> str:
    return f"case02_cylinder_theta{theta:06.3f}_Re{int(re)}"


def plan() -> list[tuple[str, str, dict[str, Any]]]:
    """The full job table: ``(name, case, kwargs)``."""
    jobs: list[tuple[str, str, dict[str, Any]]] = []
    for L, width in CHANNEL_LEVELS:
        for theta in CHANNEL_ANGLES:
            jobs.append(
                (
                    _channel_name(theta, L),
                    "channel",
                    dict(theta_deg=theta, L=L, width=width, nu=CHANNEL_NU,
                         u_peak_target=0.02, max_steps=CHANNEL_STEPS[L], tol=1e-10),
                )
            )
    for re in CYLINDER_RE:
        for theta in CYLINDER_ANGLES:
            jobs.append(
                (
                    _cylinder_name(theta, re),
                    "cylinder",
                    dict(theta_deg=theta, Re=re, diameter=CYLINDER_D,
                         box_over_d=20.0, u_target=0.06, transient=8000, sampling=18000),
                )
            )
    jobs.append(("case03_tgv", "tgv", dict(TGV_KW)))
    jobs.append(("case04_tgv_crosscheck", "crosscheck_tgv", dict(n=32, nu=0.02, u0=0.05,
                                                                mas=(0.0, 0.05), steps=800)))
    jobs.append(("case04_poiseuille_crosscheck", "crosscheck_poiseuille",
                 dict(width=16, theta_deg=0.0, nu=CHANNEL_NU)))
    return jobs


def _run_crosscheck_tgv(**kw) -> dict:
    from .crosscheck import tgv_crosscheck

    return tgv_crosscheck(**kw)


def _run_crosscheck_poiseuille(**kw) -> dict:
    from .crosscheck import poiseuille_crosscheck

    return poiseuille_crosscheck(**kw)


_CASE_FN: dict[str, Callable[..., dict]] = {
    "channel": cases.inclined_channel,
    "cylinder": cases.cylinder_flow_periodic,
    "tgv": cases.taylor_green_boost,
    "crosscheck_tgv": _run_crosscheck_tgv,
    "crosscheck_poiseuille": _run_crosscheck_poiseuille,
}


def run_job(name: str, case: str, kwargs: dict) -> dict:
    result = _CASE_FN[case](**kwargs)
    result["_job"] = name
    result["_case"] = case
    return result


# --- aggregation -------------------------------------------------------------
def _channel_invariance(runs: list[dict]) -> dict:
    by_level: dict[int, list[dict]] = {}
    for r in runs:
        by_level.setdefault(int(r["L"]), []).append(r)
    levels = {}
    for L, group in sorted(by_level.items()):
        peaks = np.array([g["u_peak"] for g in group])
        fluxes = np.array([g["flux"] for g in group])
        errs = np.array([abs(g["u_peak_error"]) for g in group])
        levels[L] = {
            "width": group[0]["width"],
            "u_peak_mean": float(peaks.mean()),
            "u_peak_spread_rel": float((peaks.max() - peaks.min()) / peaks.mean()),
            "flux_spread_rel": float((fluxes.max() - fluxes.min()) / fluxes.mean()),
            "max_abs_peak_error": float(errs.max()),
            "per_angle": [
                {
                    "theta_requested_deg": g["theta_requested_deg"],
                    "theta_used_deg": g["theta_used_deg"],
                    "u_peak": g["u_peak"],
                    "u_peak_error": g["u_peak_error"],
                    "flux_error": g["flux_error"],
                    "residual_velocity_change": g["residual_velocity_change"],
                    "converged": g["converged"],
                    "steps": g["steps"],
                }
                for g in sorted(group, key=lambda g: g["theta_requested_deg"])
            ],
        }
    # convergence order per angle (error vs channel width in lattice units)
    orders = {}
    for theta in sorted({r["theta_requested_deg"] for r in runs}):
        seq = sorted(
            [r for r in runs if r["theta_requested_deg"] == theta], key=lambda r: r["L"]
        )
        hs = np.array([r["width"] for r in seq])
        es = np.array([abs(r["u_peak_error"]) for r in seq])
        fit = fit_power_law(1.0 / hs, es)
        # Richardson extrapolation of u_peak from the two finest grids
        p = fit["exponent"]
        if seq[-1]["width"] > 0 and math.isfinite(p):
            u_fine = seq[-1]["u_peak"]
            u_coarse = seq[-2]["u_peak"]
            ratio = seq[-1]["width"] / seq[-2]["width"]
            if ratio > 1 and ratio**p > 1:
                u_extrap = u_fine + (u_fine - u_coarse) / (ratio**p - 1.0)
            else:
                u_extrap = u_fine
        else:
            u_extrap = seq[-1]["u_peak"]
        orders[str(theta)] = {
            "observed_order": fit["exponent"],
            "fit_r2": fit["r2"],
            "widths": hs.tolist(),
            "abs_peak_errors": es.tolist(),
            "richardson_u_peak": float(u_extrap),
        }
    ext = np.array([v["richardson_u_peak"] for v in orders.values()])
    finest = max(levels)
    return {
        "levels": levels,
        "orders": orders,
        "finest_level": finest,
        "finest_spread_rel": levels[finest]["u_peak_spread_rel"],
        "richardson_spread_rel": float((ext.max() - ext.min()) / ext.mean()),
        "mean_observed_order": float(
            np.mean([abs(v["observed_order"]) for v in orders.values()])
        ),
    }


def _cylinder_invariance(runs: list[dict]) -> dict:
    out = {}
    for re in sorted({r["Re"] for r in runs}):
        group = sorted([r for r in runs if r["Re"] == re], key=lambda r: r["theta_deg"])
        cd = np.array([g["Cd"] for g in group])
        st = np.array([g["St"] for g in group])
        out[str(int(re))] = {
            "Cd_mean": float(cd.mean()),
            "Cd_spread_rel": float((cd.max() - cd.min()) / cd.mean()),
            "Cd_per_angle": cd.tolist(),
            "St_mean": float(np.nanmean(st)),
            "St_spread_rel": float((np.nanmax(st) - np.nanmin(st)) / np.nanmean(st)),
            "St_per_angle": st.tolist(),
            "angles": [g["theta_deg"] for g in group],
            "blockage": group[0]["blockage"],
            "per_angle": [
                {
                    "theta_deg": g["theta_deg"],
                    "Cd": g["Cd"],
                    "St": g["St"],
                    "Cl_rms": g["Cl_rms"],
                    "max_mach": g["max_mach"],
                    "n_wall_links": g["n_wall_links"],
                }
                for g in group
            ],
        }
    return out


def _tgv_summary(run: dict) -> dict:
    law = run["law"]
    cases_list = run["cases"]
    drift_err = max(r["drift_error_rel"] for r in cases_list)
    return {
        "law": law,
        "max_drift_error_rel": float(drift_err),
        "uniform_flow_fixed_point": run["uniform_flow_fixed_point"],
        "table": [
            {
                "mach": r["mach"],
                "decay_exponent": r["decay_exponent"],
                "decay_exponent_exact": r["decay_exponent_exact"],
                "decay_defect": r["decay_defect"],
                "drift_x": r["drift_x"],
                "drift_error_rel": r["drift_error_rel"],
            }
            for r in cases_list
        ],
    }


def _json_free(obj):
    """Strip the private ``_job``/``_case`` tags and numpy scalars."""
    if isinstance(obj, dict):
        return {k: _json_free(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_free(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def aggregate(results: list[dict], meta: dict | None = None) -> dict:
    channel = _channel_invariance([r for r in results if r["_case"] == "channel"])
    cylinder = _cylinder_invariance([r for r in results if r["_case"] == "cylinder"])
    cross = {
        "tgv": [_json_free(r) for r in results if r["_case"] == "crosscheck_tgv"],
        "poiseuille": [_json_free(r) for r in results if r["_case"] == "crosscheck_poiseuille"],
    }
    tgv_runs = [r for r in results if r["_case"] == "tgv"]
    tgv = _tgv_summary(tgv_runs[0]) if tgv_runs else None

    criteria = {
        "case01_channel": {
            "finest_spread_rel": channel["finest_spread_rel"],
            "tolerance": TOL_CHANNEL_INVARIANCE,
            "pass": bool(channel["finest_spread_rel"] <= TOL_CHANNEL_INVARIANCE),
            "richardson_spread_rel": channel["richardson_spread_rel"],
            "richardson_tolerance": TOL_CHANNEL_EXTRAPOLATED,
            "richardson_pass": bool(
                channel["richardson_spread_rel"] <= TOL_CHANNEL_EXTRAPOLATED
            ),
            "mean_observed_order": channel["mean_observed_order"],
            "max_abs_peak_error_finest": channel["levels"][channel["finest_level"]][
                "max_abs_peak_error"
            ],
        },
        "case02_cylinder": {},
        "case03_tgv": {},
    }
    for re, block in cylinder.items():
        criteria["case02_cylinder"][re] = {
            "Cd_mean": block["Cd_mean"],
            "Cd_spread_rel": block["Cd_spread_rel"],
            "Cd_tolerance": TOL_CYLINDER_CD,
            "Cd_pass": bool(block["Cd_spread_rel"] <= TOL_CYLINDER_CD),
            "St_mean": block["St_mean"],
            "St_spread_rel": block["St_spread_rel"],
            "St_tolerance": TOL_CYLINDER_ST,
            "St_pass": bool(block["St_spread_rel"] <= TOL_CYLINDER_ST),
        }
    if tgv is not None:
        law = tgv["law"]
        criteria["case03_tgv"] = {
            "free_exponent": law["free_exponent"],
            "free_r2": law["free_r2"],
            "c_ma2": law["c_ma2"],
            "ma2_residual": law["residual_ma2"],
            "exponent_pass": bool(
                abs(law["free_exponent"] - 2.0) <= 0.15 and law["free_r2"] >= TOL_TGV_MA2_R2
            ),
            "max_drift_error_rel": tgv["max_drift_error_rel"],
            "drift_tolerance": TOL_TGV_DRIFT,
            "drift_pass": bool(tgv["max_drift_error_rel"] <= TOL_TGV_DRIFT),
            "uniform_flow_velocity_drift": tgv["uniform_flow_fixed_point"][
                "velocity_drift_relative"
            ],
        }
    all_pass = bool(
        criteria["case01_channel"]["pass"]
        and all(v["Cd_pass"] and v["St_pass"] for v in criteria["case02_cylinder"].values())
        and criteria["case03_tgv"].get("exponent_pass", False)
        and criteria["case03_tgv"].get("drift_pass", False)
    )
    return {
        "meta": meta or {},
        "case01_channel": channel,
        "case02_cylinder": cylinder,
        "case03_tgv": tgv,
        "case04_crosscheck": cross,
        "criteria": criteria,
        "all_criteria_pass": all_pass,
        "runs": results,
    }


# --- CLI ---------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("plan")

    shard = sub.add_parser("shard")
    shard.add_argument("index", type=int)
    shard.add_argument("total", type=int)
    shard.add_argument("--outdir", default="results")
    shard.add_argument("--only", default=None, help="restrict to case name substring")

    agg = sub.add_parser("aggregate")
    agg.add_argument("--indir", default="results")
    agg.add_argument("--out", default="invariance_report.json")
    agg.add_argument("--meta", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "plan":
        for i, (name, case, kw) in enumerate(plan()):
            print(f"{i:02d} {name} {case} {kw}")
        return 0
    if args.cmd == "shard":
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        jobs = plan()
        started = 0
        for i in range(args.index, len(jobs), args.total):
            name, case, kwargs = jobs[i]
            if args.only and args.only not in name:
                continue
            target = outdir / f"{name}.json"
            if target.exists():
                continue
            res = run_job(name, case, kwargs)
            target.write_text(json.dumps({name: res}))
            print(f"[shard {args.index}/{args.total}] {name} done", flush=True)
            started += 1
        print(f"[shard {args.index}/{args.total}] {started} jobs", flush=True)
        return 0
    if args.cmd == "aggregate":
        indir = pathlib.Path(args.indir)
        results = []
        for path in sorted(indir.glob("*.json")):
            results.extend(json.loads(path.read_text()).values())
        meta = json.loads(pathlib.Path(args.meta).read_text()) if args.meta else {}
        report = aggregate(results, meta)
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
        crit = report["criteria"]
        print(json.dumps(crit, indent=1))
        print("ALL CRITERIA PASS:", report["all_criteria_pass"])
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())