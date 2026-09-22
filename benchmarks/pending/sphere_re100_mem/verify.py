"""W5-B independent verification — recompute the judgment from raw outputs.

Reads the raw per-sample force histories written by run.py (no reliance on
any in-run aggregation), recomputes every number with independent code
(numpy, different accumulation path), re-derives the reference from the
locked formula, applies the pre-registered gate, and writes result.json.

Usage:
  PYTHONPATH=src_patched python verify.py [--write]

With --write the recomputed result.json is (re)written; without it the
script only checks an existing result.json against the recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

STAGE = os.path.dirname(os.path.abspath(__file__))


# ---- locked reference (NOTES §1) — re-coded independently --------------
def cd_ref(re=100.0):
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def cd_ref_sn0681(re=100.0):
    return 24.0 / re * (1.0 + 0.15 * re**0.681)


def cd_ref_clift_gauvin(re=100.0):
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


GATE = 0.03


def window_stats(steps, values, frac=0.8):
    """Last-(1-frac) window mean and drift vs the IMMEDIATELY preceding
    window of equal length (NOTES §2: last 20% vs preceding 20%)."""
    arr_s = np.asarray(steps, dtype=np.float64)
    arr_v = np.asarray(values, dtype=np.float64)
    n = len(arr_v)
    i0 = int(round(n * frac))
    last = arr_v[i0:]
    prev = arr_v[2 * i0 - n : i0]  # equal length, immediately preceding
    mean = float(np.mean(last))
    prev_mean = float(np.mean(prev)) if len(prev) and abs(np.sum(prev)) > 1e-30 else float("nan")
    drift = (
        float((mean - prev_mean) / prev_mean)
        if prev_mean == prev_mean and abs(prev_mean) > 1e-30
        else float("nan")
    )
    return {
        "i0": i0,
        "n_window": int(len(last)),
        "mean": mean,
        "prev_mean": prev_mean,
        "drift_frac": drift,
    }


def analyze_run(path):
    with open(path) as fh:
        d = json.load(fh)
    meta, hist = d["meta"], d["history"]
    steps = [h["step"] for h in hist]
    if meta["route"] == "bfl":
        key, cl_key = "cd_bfl", "cl_bfl"
    else:
        key, cl_key = "cd_mem", "cl"
    cd = [h[key] for h in hist]
    cl = [h[cl_key] for h in hist]

    # independent dpS cross-check
    dpS_indep = 0.5 * meta["u_lb"] ** 2 * math.pi * meta["R_lb"] ** 2
    dpS_rel_err = abs(dpS_indep - meta["dpS"]) / dpS_indep

    w = window_stats(steps, cd)
    wl = window_stats(steps, cl)
    ref = cd_ref()
    out = {
        "file": os.path.basename(path),
        "route": meta["route"],
        "D": meta["D"],
        "domain_lu": meta["domain_lu"],
        "lat": meta.get("lat"),
        "tau": meta["tau"],
        "u_lb": meta["u_lb"],
        "dpS_meta": meta["dpS"],
        "dpS_independent": dpS_indep,
        "dpS_rel_diff": dpS_rel_err,
        "steps_completed": steps[-1] if steps else 0,
        "steady": {
            "window_mean": w["mean"],
            "prev_window_mean": w["prev_mean"],
            "drift_pct": w["drift_frac"] * 100.0 if w["drift_frac"] == w["drift_frac"] else None,
            "n_window": w["n_window"],
            "criterion_pct": 0.3,
            "min_steps": 8000,
            "meets_criterion": (
                w["drift_frac"] == w["drift_frac"]
                and abs(w["drift_frac"]) * 100.0 < 0.3
                and (steps[-1] if steps else 0) >= 8000
            ),
        },
        "cd_judged": w["mean"],
        "cl_mean": wl["mean"],
        "cd_ref_primary": ref,
        "err_pct": (w["mean"] - ref) / ref * 100.0,
        "gate": GATE,
        "within_gate": abs(w["mean"] - ref) / ref <= GATE,
        "mass_drift_ppm": meta.get("mass_drift_ppm"),
    }
    if meta["route"] == "bfl":
        # CV cross-check: exact momentum budget In - Out over a box around
        # the body, evaluated on the pre-streaming state each sample step
        cv = np.asarray([h["cv_mom_cd"][0] for h in hist], dtype=np.float64)
        cvw = cv[int(round(len(cv) * 0.8)) :]
        out["cv_mom_cd"] = float(np.mean(cvw))
        out["cv_vs_ledger_rel_diff"] = abs(out["cv_mom_cd"] - w["mean"]) / abs(w["mean"])
    return out


def chain_check():
    """Recompute the sha256 chain over NOTES.md revisions."""
    chain_path = os.path.join(STAGE, "NOTES_sha256_chain.txt")
    notes_path = os.path.join(STAGE, "NOTES.md")
    with open(chain_path) as fh:
        entries = [ln for ln in fh.read().splitlines() if "sha256" in ln]
    current = hashlib.sha256(open(notes_path, "rb").read()).hexdigest()
    m = re.search(r"sha256\s*:?\s*([0-9a-f]{64})", entries[-1])
    last_locked = m.group(1) if m else None
    return {
        "chain_entries": len(entries),
        "last_locked_hash": last_locked,
        "current_notes_hash": current,
        "notes_matches_last_rev": current == last_locked,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    runs = []
    for name in sorted(os.listdir(STAGE)):
        if name.startswith("formal_") and name.endswith(".json"):
            runs.append(analyze_run(os.path.join(STAGE, name)))
    runs.sort(key=lambda r: (r["route"], r["D"]))

    # monotonic convergence on the judged route
    judged = [r for r in runs if r["route"] == "bfl"]
    judged.sort(key=lambda r: r["D"])
    mono = None
    if len(judged) >= 2:
        errs = [abs(r["err_pct"]) for r in judged]
        mono = all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))

    result = {
        "case": "sphere Re=100 direct-force benchmark (W5-B)",
        "reference": {
            "formula": "Cd = 24/Re * (1 + 0.15*Re^0.687)",
            "re": 100.0,
            "cd_ref": cd_ref(),
            "gate_pct": 3.0,
            "cross_references": {
                "schiller_naumann_0681": cd_ref_sn0681(),
                "clift_gauvin": cd_ref_clift_gauvin(),
            },
        },
        "runs": runs,
        "judgment": {
            "route": "bfl" if judged else None,
            "observable": (
                "directly measured BFL per-link momentum ledger (laboratory frame), window-mean Cd"
            ),
            "n_grids": len(judged),
            "all_within_gate": all(r["within_gate"] for r in judged) if judged else None,
            "monotone_error_decrease": mono,
            "pass": bool(judged) and all(r["within_gate"] for r in judged) and bool(mono),
        },
        "notes_chain": chain_check(),
    }
    for r in runs:
        print(
            f"{r['file']:28s} D={r['D']:3d} route={r['route']:3s} "
            f"Cd={r['cd_judged']:.4f} err={r['err_pct']:+.2f}% "
            f"steady={r['steady']['meets_criterion']} within3%={r['within_gate']}"
        )
    j = result["judgment"]
    print(
        f"JUDGMENT route={j['route']}: pass={j['pass']} "
        f"(grids={j['n_grids']}, all_within_gate={j['all_within_gate']}, "
        f"monotone={j['monotone_error_decrease']})"
    )

    out_path = os.path.join(STAGE, "result.json")
    if args.write:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print("written", out_path)
    else:
        if not os.path.exists(out_path):
            sys.exit("result.json missing; run with --write first")
        with open(out_path) as fh:
            stored = json.load(fh)
        ok = json.dumps(stored, sort_keys=True) == json.dumps(result, sort_keys=True)
        print("stored result.json matches independent recomputation:", ok)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
