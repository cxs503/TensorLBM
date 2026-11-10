#!/usr/bin/env python3
"""p83w2-n01: compressibility-error law ``eps(Ma)`` and Mach envelope.

Mach-number sweep of the isothermal-cylinder forced-convection case
(``Pr = 0.7``, ``Re in {40, 100, 200}``) at a fixed grid and a fixed physical
integration time.  Each ``(Re, Ma)`` case is one JSON record; the ``aggregate``
task turns the records into the error law, the Mach envelope, the hold-out
validation and the acceptance criteria of the campaign node.

Tasks
-----
``grid``
    Grid study at one ``(Re, Ma)``: ``Nu`` versus cylinder diameter, used to pick
    the production resolution and to quantify the (Ma-independent) grid error.
``chain``
    One Reynolds-number ladder of Mach numbers, integrated from the highest Mach
    number down with warm starts (the low-Mach end is ``1/Ma`` times more
    expensive per case, see the module docstring).
``aggregate``
    Fit ``Nu0`` (Richardson in ``Ma^2``), fit ``eps = C Ma^n`` for every ``Re``,
    build the envelope, run the pre-registered hold-out prediction and evaluate
    criteria (a)-(f).

Examples
--------
::

    python scripts/run_p83w2_n01_ma_error_law.py --task chain --re 100 \
        --machs 0.4,0.3,0.25,0.2,0.16,0.12,0.08,0.05,0.02 --diameter 24 \
        --out artifacts/p83w2_n01/raw
    python scripts/run_p83w2_n01_ma_error_law.py --task aggregate \
        --out artifacts/p83w2_n01
"""

from __future__ import annotations

import os

# Pin BLAS to one thread before numpy is imported: the kernels are small-array
# reductions and multi-threaded BLAS on a 128-core node makes them slower.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import math

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.compressibility_error_law import (  # noqa: E402
    ThermalCylinderSetup,
    churchill_bernstein_mean_nu,
    fit_ma_error_law,
    mach_envelope,
    mach_floor_from_stiffness,
    predict_holdout_envelope,
    richardson_nu0,
    run_thermal_cylinder,
)

DEFAULT_MACHS = (0.4, 0.3, 0.25, 0.2, 0.16, 0.12, 0.08, 0.05)
TOLERANCE = 0.03


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _record_name(re: float, mach: float, diameter: float, tag: str = "") -> str:
    base = f"re{int(round(re)):04d}_ma{mach:.4f}_d{int(round(diameter)):03d}"
    return base + (f"_{tag}" if tag else "") + ".json"


def _setup(re: float, mach: float, diameter: float, args) -> ThermalCylinderSetup:
    return ThermalCylinderSetup(
        mach=mach,
        re=re,
        pr=args.pr,
        diameter=diameter,
        transient_tu=args.transient_tu,
        sample_tu=args.sample_tu,
        max_tu=args.max_tu,
        collision=args.collision,
        thermal_collision=args.thermal_collision,
    )


def _log(msg: str) -> None:
    print(msg, flush=True)


def task_grid(args) -> None:
    out = Path(args.out)
    for d in args.diameters:
        setup = _setup(args.re, args.mach, d, args)
        t0 = time.time()
        rec = run_thermal_cylinder(setup, keep_fields=args.fields)
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["task"] = "grid"
        rec["cb_reference"] = churchill_bernstein_mean_nu(args.re, args.pr)
        rec["nu_rel_error_vs_cb"] = abs(rec["nu_mean"] - rec["cb_reference"]) / rec["cb_reference"]
        _write(out / _record_name(args.re, args.mach, d, "grid"), rec)
        _log(
            f"grid D={d:.0f} Re={args.re:g} Ma={args.mach:g} "
            f"Nu={rec['nu_mean']:.3f}+-{rec['nu_stderr']:.4f} "
            f"balance={rec['nu_global_balance']:.3f} CB={rec['cb_reference']:.3f} "
            f"({100 * rec['nu_rel_error_vs_cb']:+.1f}%) CD={rec['cd_mean']:.3f} "
            f"steps={rec['steps']} {rec['elapsed_s']:.0f}s"
        )


def task_chain(args) -> None:
    out = Path(args.out)
    # Start the chain at the Mach number closest to --chain-start so that every
    # case except the first is warm started: the cold start is the expensive one
    # and the temperature field is Mach independent, so a converged neighbour is
    # an almost-converged initial condition at the next Mach number.
    machs = sorted(args.machs, key=lambda m: abs(math.log(m / args.chain_start)))
    warm = None
    for mach in machs:
        setup = _setup(args.re, mach, args.diameter, args)
        if warm is None:
            setup = ThermalCylinderSetup(**{**setup.__dict__,
                                            "transient_tu": args.transient_tu * args.cold_factor})
        t0 = time.time()
        rec = run_thermal_cylinder(setup, warm_start=warm, keep_fields=True)
        fields = rec.pop("fields", None)
        if args.save_fields:
            path = Path(str(args.save_fields).format(
                re=int(round(args.re)), mach=f"{mach:.4f}", d=int(round(args.diameter))))
            path.parent.mkdir(parents=True, exist_ok=True)
            rec["fields_saved"] = str(path)
            np.savez_compressed(path, **fields)
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["task"] = "chain"
        rec["warm_started"] = warm is not None
        rec["cb_reference"] = churchill_bernstein_mean_nu(args.re, args.pr)
        rec["nu_rel_error_vs_cb"] = (
            abs(rec["nu_mean"] - rec["cb_reference"]) / rec["cb_reference"]
        )
        _write(out / _record_name(args.re, mach, args.diameter), rec)
        _log(
            f"Re={args.re:g} Ma={mach:g} D={args.diameter:g} "
            f"nu={setup.nu:.5f} tau={setup.tau:.4f} tau_t={setup.tau_t:.4f} "
            f"Nu={rec['nu_mean']:.4f}+-{rec['nu_stderr']:.4f} "
            f"bal={rec['nu_global_balance']:.3f} CD={rec['cd_mean']:.3f} "
            f"St={rec['st']:.3f} div={rec['diverged']} "
            f"steps={rec['steps']} {rec['elapsed_s']:.0f}s"
        )
        if rec["diverged"]:
            _log(f"  Ma={mach:g} DIVERGED at step {rec['diverge_step']}")
            warm = None
            continue
        warm = fields


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _load_records(raw: Path) -> list[dict]:
    recs = []
    for p in sorted(raw.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "case" not in rec:
            continue
        rec["_file"] = p.name
        recs.append(rec)
    return recs


def _fit_re(recs: list[dict], re: float, *, fit_ma_min: float = 0.0,
            fit_ma_max: float = 1.0) -> dict:
    """Richardson Nu0 + error-law fit for one Reynolds number."""
    sel = []
    for rec in recs:
        c = rec["case"]
        if abs(c["re"] - re) > 1e-9 or rec.get("diverged"):
            continue
        if not np.isfinite(rec["nu_mean"]):
            continue
        sel.append(rec)
    sel.sort(key=lambda r: r["case"]["mach"])
    machs = [r["case"]["mach"] for r in sel]
    nus = [r["nu_mean"] for r in sel]
    # measurement uncertainty = max(block scatter, half the residual drift of the
    # averaging window): a case that has not fully settled must not be given a
    # small weight just because its blocks look consistent.
    sig = []
    for r in sel:
        se = r["nu_stderr"] if np.isfinite(r["nu_stderr"]) else 0.0
        drift = abs(r.get("nu_drift_rel", 0.0) or 0.0) * abs(r["nu_mean"]) / 2.0
        sig.append(max(se, drift, max(abs(r["nu_mean"]) * 1e-5, 1e-9)))
    rich = richardson_nu0(machs, nus, sig, max_power=2)
    fit_all = fit_ma_error_law(machs, nus, rich["nu0"], sig,
                               nu0_stderr=rich.get("nu0_stderr", 0.0))
    keep = [(m, v, s) for m, v, s in zip(machs, nus, sig)
            if fit_ma_min <= m <= fit_ma_max]
    if len(keep) >= 3:
        fm, fv, fs = zip(*keep)
        fit = fit_ma_error_law(fm, fv, rich["nu0"], fs,
                               nu0_stderr=rich.get("nu0_stderr", 0.0))
    else:
        fit = fit_all
    return {
        "re": float(re),
        "richardson": rich,
        "error_law_all": fit_all,
        "error_law": fit,
        "points": [
            {
                "mach": r["case"]["mach"],
                "nu": r["nu_mean"],
                "nu_stderr": r["nu_stderr"],
                "nu_drift_rel": r["nu_drift_rel"],
                "cd": r["cd_mean"],
                "st": r["st"],
                "tau": r["case"]["tau"],
                "steps": r["steps"],
                "diverged": r.get("diverged", False),
            }
            for r in sel
        ],
        "ma_stable_max": max(
            [r["case"]["mach"] for r in sel if not r.get("diverged")], default=float("nan")
        ),
        "diameter_eff": sel[0]["case"]["diameter_eff"] if sel else float("nan"),
        "cb_reference": churchill_bernstein_mean_nu(re),
        "nu_rel_error_vs_cb": [
            {
                "mach": r["case"]["mach"],
                "rel_error": abs(r["nu_mean"] - churchill_bernstein_mean_nu(re))
                / churchill_bernstein_mean_nu(re),
            }
            for r in sel
        ],
    }


def task_aggregate(args) -> None:
    out = Path(args.out)
    recs = _load_records(out / "raw")
    if not recs:
        raise SystemExit(f"no records found under {out / 'raw'}")
    res = [r for r in recs if r.get("task") in ("chain", "grid")]
    chain = [r for r in res if r.get("task") == "chain"]
    re_list = sorted({round(r["case"]["re"]) for r in chain})
    per_re = {}
    for re in re_list:
        per_re[re] = _fit_re(chain, re)
    holdout_re = args.holdout_re
    train = [per_re[re] for re in re_list if re != holdout_re]
    holdout = per_re.get(holdout_re)

    # pre-registered hold-out prediction from the *training* Reynolds numbers only
    pred = None
    if train and holdout is not None:
        pred = predict_holdout_envelope(
            [{"re": t["re"], "c": t["error_law"]["c"], "n": t["error_law"]["n"]}
             for t in train],
            holdout_re,
            tol=args.tolerance,
        )
        crit = holdout["error_law"]
        measured_critical_ma = (args.tolerance / crit["c"]) ** (1.0 / crit["n"]) \
            if crit["c"] > 0 and crit["n"] > 0 else float("nan")
        pred["measured_critical_ma"] = measured_critical_ma
        pred["ratio_pred_over_measured"] = (
            pred["ma_pred"] / measured_critical_ma if measured_critical_ma else float("nan")
        )

    env_in = {}
    for re, d in per_re.items():
        law_d = d["error_law"]
        env_in[re] = {
            "nu0": d["richardson"]["nu0"],
            "nu0_stderr": d["richardson"].get("nu0_stderr", float("nan")),
            "c": law_d["c"],
            "n": law_d["n"],
            "n_ci95": law_d.get("n_ci95", [float("nan"), float("nan")]),
            "r2_loglog": law_d["r2_loglog"],
            "ma_stable_max": d["ma_stable_max"],
            "diameter_eff": d["diameter_eff"],
        }
    env = mach_envelope(env_in, tol=args.tolerance)

    # ---------------- criteria (a)-(f) ----------------
    crit: dict = {}
    law = {re: per_re[re]["error_law"] for re in re_list}
    crit["a"] = {
        "id": "a",
        "requirement": "n in [1.8, 2.2] and R2 > 0.98 for Re in {40, 100, 200}",
        "per_re": {
            str(re): {
                "n": law[re]["n"],
                "n_ci95": law[re].get("n_ci95"),
                "r2_loglog": law[re]["r2_loglog"],
                "pass": bool(1.8 <= law[re]["n"] <= 2.2 and law[re]["r2_loglog"] > 0.98),
            }
            for re in re_list
        },
        "pass": bool(all(
            1.8 <= law[re]["n"] <= 2.2 and law[re]["r2_loglog"] > 0.98 for re in re_list
        )),
    }
    if pred is not None:
        ratio = pred["ratio_pred_over_measured"]
        crit["b"] = {
            "id": "b",
            "requirement": "hold-out predicted / measured critical Ma in [0.67, 1.5]",
            "holdout_re": holdout_re,
            "ma_pred": pred["ma_pred"],
            "measured_critical_ma": pred["measured_critical_ma"],
            "ratio": ratio,
            "pass": bool(np.isfinite(ratio) and 0.67 <= ratio <= 1.5),
        }
    else:
        crit["b"] = {"id": "b", "pass": False, "reason": "hold-out Re not available"}
    # (c) every measured case inside its envelope must be within tolerance
    env_rows = {row["re"]: row for row in env["rows"]}
    inside = []
    for re in re_list:
        row = env_rows.get(re)
        nu0 = per_re[re]["richardson"]["nu0"]
        for p in per_re[re]["points"]:
            if not np.isfinite(p["nu"]):
                continue
            in_band = row and row["ma_band"][0] <= p["mach"] <= row["ma_band"][1]
            if not in_band:
                continue
            err = abs(p["nu"] / nu0 - 1.0)
            inside.append({"re": re, "mach": p["mach"], "err": err})
    crit["c"] = {
        "id": "c",
        "requirement": "every measured case inside the envelope is within 3 % of Nu0",
        "cases": inside,
        "max_err": max([c["err"] for c in inside], default=float("nan")),
        "pass": bool(inside) and all(c["err"] < args.tolerance for c in inside),
    }
    # (d) external anchor at Ma <= 0.02
    anchor_rows = [r for r in chain if r["case"]["mach"] <= 0.021 and not r.get("diverged")]
    crit["d"] = {
        "id": "d",
        "requirement": "at Ma <= 0.02, |Nu - Churchill-Bernstein|/CB < 8 %",
        "cases": [
            {
                "re": r["case"]["re"],
                "mach": r["case"]["mach"],
                "nu": r["nu_mean"],
                "cb": r["cb_reference"],
                "rel_error": r["nu_rel_error_vs_cb"],
                "pass": bool(r["nu_rel_error_vs_cb"] < 0.08),
            }
            for r in anchor_rows
        ],
        "pass": bool(anchor_rows) and all(r["nu_rel_error_vs_cb"] < 0.08 for r in anchor_rows),
    }
    # (e) negative control: the unscaled reference Mach number must be unusable
    neg = []
    for r in chain:
        c = r["case"]
        if c["mach"] < 0.35:
            continue
        nu0 = per_re[round(c["re"])]["richardson"]["nu0"]
        err = float("inf") if r.get("diverged") else abs(r["nu_mean"] / nu0 - 1.0)
        neg.append(
            {
                "re": c["re"],
                "mach": c["mach"],
                "rel_error": err,
                "diverged": bool(r.get("diverged")),
                "pass": bool(r.get("diverged") or err > 0.15),
            }
        )
    crit["e"] = {
        "id": "e",
        "requirement": "negative control at Ma ~ 0.4: error > 15 % or divergence",
        "cases": neg,
        "pass": bool(neg) and all(n["pass"] for n in neg),
    }
    overall = all(crit[k]["pass"] for k in ("a", "b", "c", "d", "e"))
    crit["f"] = {
        "id": "f",
        "requirement": "honest FAIL if any criterion is out of bounds",
        "overall_pass": bool(overall),
        "failing": [k for k in ("a", "b", "c", "d", "e") if not crit[k]["pass"]],
    }

    # ---------------- grid extrapolation (Ma-independent discretisation bias) ----
    grid_extrap = None
    grid_dir = out / "grid"
    if grid_dir.exists():
        rows = []
        for p in sorted(grid_dir.glob("*_grid.json")):
            g = json.loads(p.read_text())
            if g.get("diverged"):
                continue
            rows.append({
                "diameter_eff": g["case"]["diameter_eff"],
                "nu": g["nu_mean"],
                "sigma": g["nu_stderr"] if g["nu_stderr"] else 1e-3,
                "mach": g["case"]["mach"],
                "re": g["case"]["re"],
                "cb": g["cb_reference"],
            })
        if len(rows) >= 3:
            d = np.array([r["diameter_eff"] for r in rows])
            nu = np.array([r["nu"] for r in rows])
            sg = np.array([max(r["sigma"], 1e-4) for r in rows])
            A = np.column_stack([np.ones_like(d), 1.0 / d])
            Aw = A / sg[:, None] ** 2 * 1.0
            coef, *_ = np.linalg.lstsq(Aw.T @ A, Aw.T @ nu, rcond=None)
            nu_inf, slope = float(coef[0]), float(coef[1])
            re0 = rows[0]["re"]
            # remove the (small, known) finite-Mach offset of the grid cases
            law0 = per_re.get(round(re0), {}).get("error_law", {})
            eps_grid = (law0.get("c", 0.0) * rows[0]["mach"] ** law0.get("n", 2.0)
                        if law0.get("c") else 0.0)
            nu_inf_ma0 = nu_inf / (1.0 + eps_grid)
            cb0 = rows[0]["cb"]
            grid_extrap = {
                "re": re0,
                "mach_of_grid_cases": rows[0]["mach"],
                "points": rows,
                "nu_at_1_over_d_zero": nu_inf,
                "slope_1_over_d": slope,
                "finite_mach_correction": eps_grid,
                "nu_converged_ma0_grid_inf": nu_inf_ma0,
                "cb": cb0,
                "rel_error_vs_cb": abs(nu_inf_ma0 - cb0) / cb0,
            }

    payload = {
        "schema": "tensorlbm.ma_error_law_record/v1",
        "node": "p83w2-n01",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": sys.version.split()[0], "numpy": np.__version__},
        "protocol": {
            "tolerance": args.tolerance,
            "fit_ma_range": [args.fit_ma_min, args.fit_ma_max],
            "holdout_re": holdout_re,
            "diameter": args.diameter,
            "pr": args.pr,
            "n_records": len(chain),
        },
        "per_re": {str(k): v for k, v in per_re.items()},
        "envelope": env,
        "holdout": pred,
        "grid_extrapolation": grid_extrap,
        "criteria": crit,
    }
    _write(out / "verification.json", payload)
    _log(json.dumps(_json_safe({
        "criteria": {k: crit[k]["pass"] for k in ("a", "b", "c", "d", "e")},
        "overall": crit["f"],
        "envelope": [
            {"re": r["re"], "band": r["ma_band"], "n": r["error_law"]["n"],
             "r2": r["error_law"]["r2_loglog"], "status": r["status"]}
            for r in env["rows"]
        ],
        "holdout": pred,
    }), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=("grid", "chain", "aggregate"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--mach", type=float, default=0.1)
    ap.add_argument("--machs", type=lambda s: tuple(float(x) for x in s.split(",")),
                    default=DEFAULT_MACHS)
    ap.add_argument("--diameter", type=float, default=24.0)
    ap.add_argument("--diameters", type=lambda s: tuple(float(x) for x in s.split(",")),
                    default=(10.0, 16.0, 24.0))
    ap.add_argument("--pr", type=float, default=0.7)
    ap.add_argument("--transient-tu", type=float, default=12.0)
    ap.add_argument("--sample-tu", type=float, default=10.0)
    ap.add_argument("--max-tu", type=float, default=45.0)
    ap.add_argument("--collision", default="reg")
    ap.add_argument("--thermal-collision", default="reg")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--fit-ma-min", type=float, default=0.0)
    ap.add_argument("--fit-ma-max", type=float, default=0.35)
    ap.add_argument("--holdout-re", type=float, default=200.0)
    ap.add_argument("--chain-start", type=float, default=0.1,
                    help="Mach number the chain starts from (cold start)")
    ap.add_argument("--cold-factor", type=float, default=1.8,
                    help="multiplier on --transient-tu for the cold start")
    ap.add_argument("--fields", action="store_true",
                    help="keep the field snapshot in the record")
    ap.add_argument("--save-fields", default=None,
                    help="write the final field snapshot to this .npz path")
    args = ap.parse_args()
    if args.task == "grid":
        task_grid(args)
    elif args.task == "chain":
        args.fields = True
        task_chain(args)
    else:
        task_aggregate(args)


if __name__ == "__main__":
    main()
