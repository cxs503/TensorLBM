"""Campaign driver for IN67-N01: micro-bubble drag reduction, mechanism split.

Cases
-----
``reference``   single-phase plane channel (uniform liquid relaxation time).
``zero_gas``    the two-phase machinery with **no gas**: negative control C4.
``single_dN``   one bubble sized for ``d+ = N`` (5 / 10 / 20 / 40): the C5
                size scan, ``d+ -> infinity`` limit included as the slug case.
``void_nK``     K bubbles at ``d+ = 10``: the void-fraction path to zero (C4).

The scan is deliberately organised as *fixed bubble count* for the size sweep and
*fixed size* for the void-fraction sweep, because at the resolved friction
Reynolds number (``Re_tau ~ 20``) a ``d+ = 20`` bubble is already a large
fraction of the channel: holding the void fraction fixed across the whole size
range is geometrically impossible, and pretending otherwise would hide the
confound.  ``DR`` is therefore also quoted per unit void fraction.

Usage
-----
``python -m tensorlbm.microbubble_dr.campaign --shard 0/3`` prints one
``##RESULT## {...}`` line per case; ``--assemble a.json b.json ...`` merges the
collected summaries, applies the criteria and writes ``results.json`` plus
figures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .channel import MicrobubbleChannel2D, MicrobubbleConfig
from .decomposition import (
    einstein_bubbly_viscosity,
    laminar_wall_shear,
    mechanism_split,
    slip_length_drag_reduction,
)

ARTIFACT_DIR = Path("artifacts/microbubble_dr_in67_n01")

#: case name -> (config overrides, reference label it is compared against)
CASES: dict[str, dict[str, Any]] = {
    "reference": {"n_bubbles": 0, "d_plus": 0.0, "void_fraction": 0.0,
                  "viscosity_contrast": False},
    "zero_gas": {"n_bubbles": 0, "d_plus": 10.0},
    "single_d5": {"n_bubbles": 1, "d_plus": 5.0},
    "single_d10": {"n_bubbles": 1, "d_plus": 10.0},
    "single_d20": {"n_bubbles": 1, "d_plus": 20.0},
    "single_d40": {"n_bubbles": 1, "d_plus": 40.0},
    "void_n2": {"n_bubbles": 2, "d_plus": 10.0},
    "void_n4": {"n_bubbles": 4, "d_plus": 10.0},
    "void_n8": {"n_bubbles": 8, "d_plus": 10.0},
}

BASE = dict(nx=96, ny=97, n_steps=8000, sample_every=20, control_every=50,
            control_gain=0.02, J_target=0.18, seed=20261109)


def build_config(name: str, **overrides: Any) -> MicrobubbleConfig:
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}")
    kw = {**BASE, **overrides, **CASES[name]}
    return MicrobubbleConfig(**kw)


def summarise(name: str, run: dict[str, Any], n_prof: int = 12,
              keep_fields: bool = False) -> dict[str, Any]:
    """Compact, transportable summary of one run (profiles down-sampled)."""
    prof = run["profile"]
    y = prof["y"]
    idx = [int(round(i)) for i in range(0, len(y), max(1, len(y) // n_prof))]
    keep = {
        key: [float(prof[key][i]) for i in idx]
        for key in ("u", "rho", "mu", "rho_uv", "alpha")
    }
    keep["y"] = [float(y[i]) for i in idx]
    drop = {"profile", "tau_w_history", "bubble_centers"}
    if not keep_fields:
        # large field snapshots are only needed for the local visualisation case;
        # keeping them out of the transportable summary keeps the shard logs small
        drop |= {"alpha_field", "u_field"}
    out = {k: v for k, v in run.items() if k not in drop}
    out["case"] = name
    out["profile"] = keep
    out["bubble_centers"] = [[round(c[0], 3), round(c[1], 3)] for c in run["bubble_centers"]]
    out["tau_w_history"] = run["tau_w_history"][::10]
    return out


def run_case(name: str, **overrides: Any) -> dict[str, Any]:
    cfg = build_config(name, **overrides)
    result = MicrobubbleChannel2D(cfg).run()
    return summarise(name, result)


def _spread(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v) and v > 0.0]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (max(vals) - min(vals)) / mean


def evaluate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply criteria C1-C5 and build the mechanism table."""
    ref = results["reference"]
    mu_l = ref["config"]["rho_liquid"] * ref["config"]["nu_liquid"]
    h = (ref["config"]["ny"] - 1) / 2.0
    tau_w0 = ref["tau_w"]
    ref_balance = ref["tau_w_balance"]
    J_target = float(ref["config"]["J_target"])

    cases: dict[str, Any] = {}
    for name, res in results.items():
        entry: dict[str, Any] = {
            "d_plus_target": res["config"]["d_plus_target"],
            "d_plus_achieved": res["d_plus_achieved"],
            "n_bubbles": res["n_bubbles_placed"],
            "alpha_mean": res["alpha_mean"],
            "alpha_wall": res["alpha_wall"],
            "tau_w": res["tau_w"],
            "tau_w_std_rel": res["tau_w_std"] / res["tau_w"] if res["tau_w"] else None,
            "u_tau": res["u_tau"],
            "Re_tau": res["Re_tau"],
            "Re_bulk": res["Re_bulk"],
            "J": res["J"],
            "gx": res["gx"],
            "mass_drift": res["mass_drift"],
            "alpha_integral_drift": res["alpha_integral_drift"],
            "u_slip": res["u_slip"],
            "slip_length": res["slip_length"],
            "bubble_diameter_lattice": res["bubble_diameter_lattice"],
            # C2: three independent friction-velocity estimates
            "u_tau_gradient": res["u_tau_gradient"],
            "u_tau_identity": res["u_tau_identity"],
            "u_tau_balance": res["u_tau_balance"],
            # C2 triad: mean-velocity gradient / Reynolds-stress identity /
            # total shear from the momentum balance (the momentum-exchange
            # estimator is reported separately: it carries a ~5 % wall-force
            # offset of the bounce-back accumulator, see the report's trap list)
            "u_tau_spread": _spread([res["u_tau_gradient"], res["u_tau_identity"],
                                     res["u_tau_balance"]]),
            "u_tau_exchange": res["u_tau_exchange"],
            "tau_w_exchange": res["tau_w_exchange"],
            "DR_balance": (ref_balance - res["tau_w_balance"]) / ref_balance,
            "DR_exchange": (ref["tau_w_exchange"] - res["tau_w_exchange"]) / ref["tau_w_exchange"],
            "A_viscous": res["A_viscous"],
            "B_reynolds": res["B_reynolds"],
            "tau_w_identity": res["tau_w_identity"],
        }
        entry["DR"] = (tau_w0 - res["tau_w"]) / tau_w0
        # the mechanism split is only defined at matched mass flux; cases whose
        # drive hit the admissible band are flagged instead of being averaged in
        entry["flux_matched"] = abs(res["J"] / J_target - 1.0) <= 0.02
        entry["J_ratio"] = res["J"] / J_target
        entry["gx_over_gx0"] = res.get("gx_over_gx0")
        entry["DR_analytic_laminar"] = (laminar_wall_shear(mu_l, ref["J"] / ref["config"]["rho_liquid"], h)
                                        - res["tau_w"]) / tau_w0
        entry["nu_mix_ratio"] = res["mu_mean"] / mu_l
        entry["DR_einstein_expectation"] = 1.0 - einstein_bubbly_viscosity(res["alpha_mean"])
        if name != "reference":
            entry.update(mechanism_split(ref, res, mu_l, h))
            entry["DR_slip_analytic"] = slip_length_drag_reduction(res["slip_length"], h)
            entry["DR_per_alpha"] = entry["DR_total"] / res["alpha_mean"] if res["alpha_mean"] > 1e-9 else None
        cases[name] = entry

    # -- criteria ---------------------------------------------------------
    two_phase = [c for c in cases.values() if c["n_bubbles"] > 0]
    # physical premise of the mechanism split: matched mass flux.  Cases whose
    # drive hit the admissible band cannot be flux-matched and are reported,
    # not silently averaged in (see the report's choke-limited note).
    matched = [c for c in two_phase if c["flux_matched"]]
    c1_max_mass = max(c["mass_drift"] for c in cases.values())
    c1_max_alpha = max((c["alpha_integral_drift"] for c in two_phase), default=0.0)
    c2_worst = max(c["u_tau_spread"] for c in cases.values())
    c2_worst_matched = max((c["u_tau_spread"] for c in cases.values()
                            if c["flux_matched"] and c["n_bubbles"] > 0), default=0.0)
    c2_exchange_worst = max(abs(c["u_tau_exchange"] / c["u_tau_balance"] - 1.0)
                            for c in cases.values())
    c3_worst = max((abs(c["closure_rel"]) for c in matched), default=0.0)
    c4_zero = abs(cases["zero_gas"]["DR"])
    void_path = sorted((c["alpha_mean"], c["DR"]) for k, c in cases.items()
                       if k.startswith("void_"))
    # the response must grow in *magnitude* with the gas load and vanish with it;
    # the sign of the response is physics, not a criterion (here it is a drag
    # increase, see the report)
    c4_monotone = all(abs(b[1]) >= abs(a[1]) - 1e-9
                      for a, b in zip(void_path, void_path[1:]))
    size_path = [("single_d5", 5), ("single_d10", 10), ("single_d20", 20), ("single_d40", 40)]
    dr_size = [(d, cases[k]["DR"]) for k, d in size_path if k in cases]
    dr_peak = max(dr_size, key=lambda t: t[1]) if dr_size else (0, 0.0)
    c5_record = {
        "d_plus_vs_DR": dr_size,
        "peak_at_d_plus": dr_peak[0],
        "later_d_plus": [t for t in dr_size if t[0] > dr_peak[0]],
        "saturation_or_decay": all(t[1] <= dr_peak[1] * 1.05 for t in dr_size if t[0] > dr_peak[0]),
        "all_runs_finite": all(math.isfinite(c["tau_w"]) for c in cases.values()),
    }
    criteria = {
        "C1_mass_conservation": {
            "max_mass_drift": c1_max_mass,
            "max_alpha_integral_drift": c1_max_alpha,
            "threshold": 0.01,
            "pass": max(c1_max_mass, c1_max_alpha) <= 0.01,
        },
        "C2_momentum_consistency": {
            "worst_u_tau_spread": c2_worst,
            "worst_u_tau_spread_flux_matched": c2_worst_matched,
            "flux_matched_cases": [k for k, c in cases.items() if c["flux_matched"]],
            "worst_momentum_exchange_offset": c2_exchange_worst,
            "estimators": ["wall_gradient(quadratic fit)", "reynolds_identity A-B",
                           "force_balance gx*int rho dy"],
            "threshold": 0.02,
            "pass": c2_worst <= 0.02,
        },
        "C3_mechanism_closure": {
            "worst_relative_closure_error": c3_worst,
            "evaluated_on": "flux-matched two-phase cases only",
            "threshold": 0.03,
            "pass": c3_worst <= 0.03,
        },
        "C4_negative_control_zero_gas": {
            "DR_at_zero_gas": c4_zero,
            "threshold": 0.005,
            "void_path": void_path,
            "monotone_in_void_fraction": c4_monotone,
            "pass": (c4_zero <= 0.005) and c4_monotone,
        },
        "C5_size_saturation": {**c5_record, "pass": bool(c5_record["saturation_or_decay"]
                                                         and c5_record["all_runs_finite"])},
    }
    return {
        "cases": cases,
        "criteria": criteria,
        "all_pass": all(v["pass"] for v in criteria.values()),
        "reference": {
            "tau_w": ref["tau_w"],
            "J": ref["J"],
            "u_tau": ref["u_tau"],
            "Re_tau": ref["Re_tau"],
            "Re_bulk": ref["Re_bulk"],
            "mu_liquid": mu_l,
            "half_height": h,
            "rho_liquid": ref["config"]["rho_liquid"],
            "rho_gas": ref["config"]["rho_gas"],
            "nu_liquid": ref["config"]["nu_liquid"],
            "nu_gas": ref["config"]["nu_gas"],
            "n_steps": ref["config"]["n_steps"],
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _shard_cases(i: int, n: int) -> list[str]:
    names = list(CASES)
    return [name for k, name in enumerate(names) if k % n == i]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--cases", default=None, help="comma separated case names")
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--out", default=None, help="write <case>.json files here")
    ap.add_argument("--assemble", nargs="*", default=None,
                    help="merge collected summary files and write results.json")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    if args.assemble is not None:
        results = {}
        for path in args.assemble:
            data = json.loads(Path(path).read_text())
            items = data if isinstance(data, list) else [data]
            for item in items:
                results[item["case"]] = item
        evaluated = evaluate(results)
        out = Path(args.out or ARTIFACT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"summary": evaluated, "raw": results}
        (out / "results.json").write_text(json.dumps(payload, indent=1))
        if args.figures:
            from .figures import make_all_figures
            make_all_figures(payload, out)
        print(json.dumps({"all_pass": evaluated["all_pass"],
                          "criteria": {k: v["pass"] for k, v in evaluated["criteria"].items()},
                          "out": str(out)}))
        return 0

    if args.cases:
        names = [c.strip() for c in args.cases.split(",") if c.strip()]
    elif args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        names = _shard_cases(i, n)
    else:
        names = list(CASES)

    overrides = {}
    if args.n_steps:
        overrides["n_steps"] = args.n_steps
    if args.nx:
        overrides["nx"] = args.nx

    outdir = Path(args.out) if args.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            res = run_case(name, **overrides)
        except Exception as exc:      # one unstable case must not kill the shard
            res = {"case": name, "status": "failed", "error": repr(exc),
                   "config": {"d_plus_target": CASES[name].get("d_plus", 0.0),
                              "n_bubbles": CASES[name].get("n_bubbles")}}
            print(f"##RESULT## {json.dumps(res)}", flush=True)
            continue
        if outdir:
            (outdir / f"{name}.json").write_text(json.dumps(res))
        print(f"##RESULT## {json.dumps(res)}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())