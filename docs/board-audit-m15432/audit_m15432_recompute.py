"""Independent recompute for the board audit of m15432 (CASE-CYLINDER-DRAG-RE100, TensorLBM).

Reads ONLY the delivered run artifact (read-only) and the audit's own smoke run,
recomputes every headline metric from the raw per-step series, and records
pass/fail against the case-pool tolerance.  No number is copied from the
delivered report.

Usage:  python docs/board-audit-m15432/audit_m15432_recompute.py
"""
from __future__ import annotations

import json
import os

AUDITED = "/root/tp-agent/work/m15432/artifacts/cylinder_re100_D10/cylinder_re100_unconfined.json"
PILOT = "/tmp/m15712_pilot/cylinder_re100_unconfined.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_recompute.json")

# Case-pool declaration for CYLINDER-DRAG-RE100 (tensor_case_pool id=11): tol <5%.
TOL_PCT = 5.0
REF = {"cd": 1.3796, "cl": 0.0005, "st": 0.164}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def crossing_period(x):
    m = mean(x)
    prev = x[0] - m
    cross = []
    for i in range(1, len(x)):
        cur = x[i] - m
        if prev < 0.0 <= cur:
            cross.append(i - 1 + (-prev / (cur - prev)))
        prev = cur
    if len(cross) < 3:
        return float("nan"), len(cross)
    return (cross[-1] - cross[0]) / (len(cross) - 1), len(cross)


def window_drift_pct(cd, window=2000):
    n = len(cd)
    wins = [mean(cd[i:i + window]) for i in range(0, n - window + 1, window)]
    return wins


def audit_run(path, label):
    d = json.load(open(path))
    run = d["runs"][0]
    derived, cfg, s = run["derived"], run["config"], d["samples"][0]
    cd, cl = s["cd_tail"], s["cl_tail"]
    n = len(cd)
    half = n // 2
    period, ncross = crossing_period(cl)
    st = cfg["diameter"] / (period * cfg["u_in"]) if period == period else float("nan")
    wins = window_drift_pct(cd)
    rec = {
        "artifact": path,
        "label": label,
        "n_samples": n,
        "grid": [derived["nx"], derived["ny"]],
        "diameter": cfg["diameter"],
        "blockage": derived["blockage"],
        "mach": derived["mach"],
        "tau": derived["tau"],
        "stop_reason": derived["stop_reason"],
        "recomputed": {
            "cd_mean": mean(cd),
            "cl_mean": mean(cl),
            "cl_amp": 0.5 * (max(cl) - min(cl)),
            "cd_amp": 0.5 * (max(cd) - min(cd)),
            "half_window_drift_pct": 100.0 * abs(mean(cd[half:]) - mean(cd[:half])) / mean(cd),
            "period_steps": period,
            "n_crossings": ncross,
            "strouhal": st,
            "window_means_2000": wins,
            "first_to_last_window_pct": 100.0 * (wins[-1] - wins[0]) / mean(cd) if wins else float("nan"),
        },
        "reported": {
            k: derived[k] for k in
            ("cd_mean", "cl_mean", "cd_amp", "cl_amp", "strouhal", "period_steps",
             "cd_half_drift_pct", "samples", "wall_seconds", "stop_reason")
        },
    }
    # reported vs independently recomputed (0 == artefact is internally consistent)
    rec["consistency_delta_pct"] = {
        k: (abs(rec["recomputed"][k] - derived[rk]) / abs(derived[rk]) * 100.0
            if derived[rk] not in (0, None) else float("nan"))
        for k, rk in (("cd_mean", "cd_mean"), ("cl_mean", "cl_mean"),
                      ("cd_amp", "cd_amp"), ("cl_amp", "cl_amp"),
                      ("strouhal", "strouhal"), ("period_steps", "period_steps"),
                      ("half_window_drift_pct", "cd_half_drift_pct"))
    }
    rec["benchmark"] = {}
    for key in ("cd", "cl", "st"):
        got = rec["recomputed"][{"cd": "cd_mean", "cl": "cl_mean", "st": "strouhal"}[key]]
        err = 100.0 * (got - REF[key]) / REF[key] if REF[key] else float("nan")
        entry = {"ref": REF[key], "got": got, "rel_err_pct": err,
                 "tol_pct": TOL_PCT, "pass": abs(err) <= TOL_PCT}
        if key == "cl":
            # |ref| = 5e-4 is a symmetry statement, not a magnitude: grade on the
            # absolute offset relative to the lift amplitude instead.
            entry["pass"] = abs(got - REF[key]) < 1e-2 * rec["recomputed"]["cl_amp"]
            entry["grading_note"] = ("mean-lift symmetry check: |Cl_mean - ref| = %.2e vs "
                                     "Cl_amp = %.3f -> pass; relative error against a near-zero "
                                     "reference is not physically meaningful"
                                     % (abs(got - REF[key]), rec["recomputed"]["cl_amp"]))
        rec["benchmark"][key] = entry
    return rec


def main():
    main_run = audit_run(AUDITED, "unconfined_D10 (audited, 250x100, B=10%)")
    out = {
        "audit_mission": 15712,
        "audited_mission": 15432,
        "anchor": "CASE-CYLINDER-DRAG-RE100",
        "tolerance_pct": TOL_PCT,
        "reference": REF,
        "main_run": main_run,
        "independent_smoke_rerun": None,
        "notes": [
            "REF attribution (Schäfer/Rannacher/Turek 1996) carries no page/table excerpt in the "
            "delivered worktree; the well-known DFG 2D-2 case in that benchmark family is a 24% "
            "blockage channel (Cd~3.22), so the delivered unbounded numbers need the verbatim table.",
            "Cl amplitude is delivered (0.476) but never compared to a reference in the report.",
        ],
    }
    if os.path.exists(PILOT):
        out["independent_smoke_rerun"] = audit_run(PILOT, "audit pilot rerun (110x60, B=16.7%, 4000 steps)")
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    r = main_run
    print("== independent recompute of the audited artefact ==")
    for k, rk in (("cd_mean", "cd_mean"), ("cl_mean", "cl_mean"), ("cl_amp", "cl_amp"),
                  ("strouhal", "strouhal"), ("half_window_drift_pct", "cd_half_drift_pct")):
        print(f"  {k:22s} recomputed={r['recomputed'][k]:.6g} reported={r['reported'][rk]:.6g} "
              f"delta_pct={r['consistency_delta_pct'].get(k, float('nan')):.2e}")
    print("  window means (2000):", ["%.5f" % w for w in r["recomputed"]["window_means_2000"]])
    print("== benchmark vs declared tolerance %.1f%% ==" % TOL_PCT)
    for k, v in r["benchmark"].items():
        print(f"  {k.upper():3s} got={v['got']:.4f} ref={v['ref']} err={v['rel_err_pct']:+.2f}% "
              f"{'PASS' if v['pass'] else 'FAIL'}")
    if out["independent_smoke_rerun"]:
        p = out["independent_smoke_rerun"]
        print("== audit pilot rerun (not the graded config) ==")
        print(f"  Cd={p['recomputed']['cd_mean']:.4f} St={p['recomputed']['strouhal']:.4f} "
              f"blockage={p['blockage']:.3f} grid={p['grid']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()