"""Benchmark runner for the sediment verification surface (mission P84W2C-N04).

Produces the mission deliverable ``artifacts/p84_w1/sediment_summary.json``
(the C6 contract, with every criterion number and its pass/fail flag) plus the
figures used in the report.  Every number in the JSON is produced by the same
functions the pytest suite calls -- there is no hand-copied value anywhere.

Usage::

    PYTHONPATH=src python3 -m tensorlbm.sediment_benchmarks --outdir artifacts/p84_w1
    PYTHONPATH=src python3 -m tensorlbm.sediment_benchmarks --outdir ... --stages c3 c4

Stages are cached in ``<outdir>/parts/<stage>.json`` so a slow anchor (C1, C2,
C5) can be run in its own process; ``--stages all`` runs everything and
assembles the summary.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from tensorlbm.sediment.verification import (
    SedimentScenario,
    build_results,
    c1_incipient_curve,
    c2_rouse_profiles,
    c3_mass_budget,
    c4_negative_controls,
    c5_pile_scour,
)
from tensorlbm.sediment.ukc_hook import build_payload

STAGES = ("c1", "c2", "c3", "c4", "c5")


def _jsonable(obj):
    """Recursively convert numpy types / tuples to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _part_path(outdir: str, stage: str) -> str:
    return os.path.join(outdir, "parts", f"{stage}.json")


def run_stage(stage: str, outdir: str, scen: SedimentScenario) -> dict:
    t0 = time.time()
    if stage == "c1":
        res = c1_incipient_curve(scen)
    elif stage == "c2":
        res = c2_rouse_profiles(scen)
    elif stage == "c3":
        res = c3_mass_budget(scen)
    elif stage == "c4":
        res = c4_negative_controls(scen)
    elif stage == "c5":
        res = c5_pile_scour()
    else:  # pragma: no cover
        raise ValueError(f"unknown stage {stage}")
    res = _jsonable(res)
    res["_runtime_s"] = round(time.time() - t0, 2)
    os.makedirs(os.path.join(outdir, "parts"), exist_ok=True)
    with open(_part_path(outdir, stage), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def load_stage(stage: str, outdir: str) -> dict | None:
    p = _part_path(outdir, stage)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def make_figures(outdir: str, summary: dict) -> list[str]:
    """Report figures (matplotlib, Agg backend).  Returns the file names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(outdir, "figs")
    os.makedirs(figdir, exist_ok=True)
    names = []

    # --- Fig 1: incipient-motion curve -------------------------------------
    from tensorlbm.sediment.shields import critical_shields

    ds = np.logspace(0, 2, 120)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(ds, [critical_shields(d) for d in ds], "k-", lw=1.6,
              label="Soulsby & Whitehouse (2005)")
    pts = summary["shields_curve"]["points"]
    ax.loglog([p["d_star"] for p in pts], [p["theta_num"] for p in pts], "o",
              ms=7, mfc="none", mew=1.6, color="C3",
              label="measured onset (LBM/Exner chain)")
    ax.set_xlabel(r"$d_*$")
    ax.set_ylabel(r"$\theta_{cr}$")
    ax.set_title(f"C1 incipient motion: max rel err "
                 f"{summary['shields_curve']['max_rel_err']*100:.2f}%")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    f1 = os.path.join(figdir, "c1_shields_curve.png")
    fig.tight_layout()
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    names.append("c1_shields_curve.png")

    # --- Fig 2: Rouse profiles --------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, case in enumerate(summary["rouse_profiles"]["cases"]):
        y = np.asarray(case["y"])
        c = np.asarray(case["c_normalised"])
        ax.plot(c, y, "o", ms=3.5, color=f"C{i}",
                label=fr"LBM $u_*$={case['u_star']:.4f} m/s ($Z$={case['Z_num']:.3f})")
        ax.plot(np.asarray(case["c_analytic"]), np.asarray(case["y_analytic"]), "-",
                lw=1.2, color=f"C{i}",
                label=fr"Rouse $Z$={case['Z_ref']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$c/c_a$")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title(f"C2 Rouse profiles: max rel err in Z "
                 f"{summary['rouse_profiles']['max_rel_err']*100:.2f}%")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    f2 = os.path.join(figdir, "c2_rouse_profiles.png")
    fig.tight_layout()
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    names.append("c2_rouse_profiles.png")

    return names


def assemble(outdir: str, scen: SedimentScenario) -> dict:
    c1 = load_stage("c1", outdir)
    c2 = load_stage("c2", outdir)
    c3 = load_stage("c3", outdir)
    c4 = load_stage("c4", outdir)
    c5 = load_stage("c5", outdir)
    missing = [n for n, v in zip(STAGES, (c1, c2, c3, c4, c5)) if v is None]
    if missing:
        raise SystemExit(f"missing stages: {missing} (run them first)")
    results = build_results(scen=scen, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5)
    results["ukc_payload_demo"] = build_payload(
        results, charted_depth_m=6.0, draft_m=5.0, squat_m=0.1, allowances_m=0.2
    )
    results["criteria"] = {
        "C1": {"criterion": "onset theta_cr vs Soulsby-Whitehouse, 8 d* points, <=15%",
               "value": c1["max_rel_err"], "pass": c1["pass"]},
        "C2": {"criterion": "Rouse exponent vs w_s/(kappa u*), rel err <5%",
               "value": c2["max_rel_err"], "pass": c2["pass"]},
        "C3": {"criterion": "sediment mass drift < 1e-6",
               "value": max(c3["exner"]["mass_drift"],
                            c3["coupled_column"]["mass_drift"]), "pass": c3["pass"]},
        "C4": {"criterion": "no-motion gate / w_s->0 uniform / Exner off",
               "value": {
                   "a_q_s": c4["a_no_motion"]["q_s"],
                   "b_uniformity": c4["b_zero_settling"]["uniformity"],
                   "c_bed_change_off": c4["c_exner_off"]["bed_off_change"],
               },
               "pass": bool(c4["a_no_motion"]["pass"] and c4["b_zero_settling"]["pass"]
                            and c4["c_exner_off"]["pass"])},
        "C5": {"criterion": "S/D in Melville-Chiew band [1.0,2.4] and S/D trend with D",
               "value": {"S_over_D": c5["S_over_D"],
                         "trend": c5.get("trend", {}).get("direction")},
               "pass": c5["pass"]},
        "C6": {"criterion": "JSON contract emitted and consumed by the UKC hook",
               "value": sorted(results["ukc_payload_demo"]["ukc"].keys())[:4],
               "pass": True},
    }
    results["all_pass"] = all(v["pass"] for v in results["criteria"].values())
    return _jsonable(results)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="artifacts/p84_w1")
    ap.add_argument("--stages", nargs="*", default=["all"])
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    scen = SedimentScenario()
    stages = list(STAGES) if args.stages == ["all"] else args.stages
    for stage in stages:
        if stage == "assemble":
            continue
        print(f"[stage {stage}] running ...", flush=True)
        run_stage(stage, args.outdir, scen)
        print(f"[stage {stage}] done", flush=True)

    summary = assemble(args.outdir, scen)
    if args.figures:
        summary["figures"] = make_figures(args.outdir, summary)
    out = os.path.join(args.outdir, "sediment_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v["pass"] for k, v in summary["criteria"].items()}, indent=2))
    print("written:", out)
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())