"""sphere_re200 verify — recompute every judgment number from raw force histories.

Self-contained archive version.  Everything judged here is recomputed from the
per-sample force histories of the two BFL big-domain tiers:

  * steady window Cd (mean over the last 20% of samples) and drift
    (last-20% vs previous-20%); steady contract |drift| < 0.3%, steps >= 8000
    and tau == 3*u_lb*D/200 + 0.5 (floating Re locked at 200)
  * err vs the LOCKED Schiller-Naumann reference Cd_ref(200)=0.8056147 (gate 3%)
  * control-volume ledger closure (per-sample and window mean)
  * grid span |Cd(D=30)-Cd(D=40)|/Cd_ref and monotonicity |err(D30)| > |err(D40)|

Usage: python verify.py            # prints tables
       python verify.py --write    # additionally rewrites result.json
"""

from __future__ import annotations

import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

RE_LOCK = 200.0
# LOCKED reference (Schiller-Naumann), same family anchor as sphere_re100
CD_REF_LOCK = 24.0 / RE_LOCK * (1.0 + 0.15 * RE_LOCK**0.687)  # 0.8056146866904639
CROSS_REFS = {
    "schiller_naumann_0681": 24.0 / RE_LOCK * (1.0 + 0.15 * RE_LOCK**0.681),  # 0.7841619
    "clift_grace_weber": 24.0 / RE_LOCK * (1.0 + 0.1935 * RE_LOCK**0.6305),  # 0.7756342
    "clift_gauvin": 24.0
    / RE_LOCK
    * (1.0 + 0.1315 * RE_LOCK ** (0.82 - 0.05 * __import__("math").log10(RE_LOCK))),  # 0.7810201
    "turton_levenspiel": 24.0 / RE_LOCK * (1.0 + 0.173 * RE_LOCK**0.657)
    + 0.413 / (1.0 + 16300.0 * RE_LOCK**-1.09),  # 0.8025398
}
GATE = 0.03

# tier files (top grid first)
TIERS = [
    ("re200_D40_12k.json", "T1_D40_12k"),
    ("re200_D30_12k.json", "T2_D30_12k"),
]


def steady_window(hist, key="cd", frac=0.2):
    vals = [h[key] for h in hist]
    n = len(vals)
    w = max(1, int(n * frac))
    tail = vals[-w:]
    prev = vals[-2 * w : -w] if n >= 2 * w else vals[:w]
    m_tail = sum(tail) / len(tail)
    m_prev = sum(prev) / len(prev)
    drift = (m_tail - m_prev) / m_prev * 100.0
    return m_tail, m_prev, drift, w


def cv_closure(hist):
    pairs = [(h["cd"], h["cv_mom_cd"]) for h in hist if "cv_mom_cd" in h]
    if not pairs:
        return None, None, 0
    rel = [abs(b - a) / abs(a) for a, b in pairs]
    mean_ledger = sum(a for a, _ in pairs) / len(pairs)
    mean_cv = sum(b for _, b in pairs) / len(pairs)
    window_rel = abs(mean_cv - mean_ledger) / mean_ledger
    return sum(rel) / len(rel), window_rel, len(pairs)


def load(path):
    with open(path) as fh:
        return json.load(fh)


def tier_row(path, tier):
    d = load(path)
    m = d["meta"]
    hist = d["history"]
    cd, cd_prev, drift, w = steady_window(hist)
    err = (cd - CD_REF_LOCK) / CD_REF_LOCK
    cv_rel, cv_win, cv_n = cv_closure(hist)
    tau_expect = 3.0 * m["u_lb"] * m["D"] / RE_LOCK + 0.5
    tau_ok = abs(m["tau"] - tau_expect) < 1e-9
    re_ok = abs(m["re_eff"] - RE_LOCK) < 1e-6
    steady_ok = abs(drift) < 0.3 and m["steps"] >= 8000 and tau_ok
    return {
        "file": os.path.basename(path),
        "tier": tier,
        "kernel": m["kernel"],
        "D": m["D"],
        "tau": m["tau"],
        "u_lb": m["u_lb"],
        "re_eff": m["re_eff"],
        "re_ok": re_ok,
        "domain_lu": m["domain_lu"],
        "lat": m["lat"],
        "up": m["up"],
        "down": m["down"],
        "blockage": m["blockage"],
        "bfl_links": m["bfl_links"],
        "steps": m["steps"],
        "diverged": m["diverged"],
        "tau_expected": tau_expect,
        "tau_ok": tau_ok,
        "cd_window": cd,
        "cd_prev_window": cd_prev,
        "drift_pct": drift,
        "n_window": w,
        "steady": steady_ok,
        "cl_mean": sum(abs(h["cl"]) for h in hist) / len(hist),
        "cd_ref": CD_REF_LOCK,
        "err_pct": err * 100.0,
        "within_gate": abs(err) <= GATE,
        "cv_closure_rel_samples": cv_rel,
        "cv_closure_rel_window": cv_win,
        "cv_samples": cv_n,
        "mass_drift_ppm": m.get("mass_drift_ppm"),
        "wall_s": m.get("wall_s"),
    }


def main(write=False):
    print(f"locked Cd_ref(Re=200) = {CD_REF_LOCK!r}")
    print(f"cross refs = { {k: round(v, 6) for k, v in CROSS_REFS.items()} }")

    rows = []
    for fname, tier in TIERS:
        path = os.path.join(BASE, fname)
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        rows.append(tier_row(path, tier))

    for r in rows:
        cv = (
            f"{r['cv_closure_rel_samples'] * 100:.3f}%/win{r['cv_closure_rel_window'] * 100:.3f}%"
            if r["cv_closure_rel_samples"] is not None
            else "n/a"
        )
        print(
            f"{r['tier']:14s} Cd={r['cd_window']:.6f} err={r['err_pct']:+.3f}% "
            f"drift={r['drift_pct']:+.3f}% steady={r['steady']} "
            f"tau_ok={r['tau_ok']} re_eff={r['re_eff']:.4f} gate3%={r['within_gate']} "
            f"blockage={r['blockage'] * 100:.2f}% CV={cv}(n={r['cv_samples']}) "
            f"steps={r['steps']} links={r['bfl_links']}"
        )

    judgment = {"pass": None, "note": "tiers incomplete"}
    if len(rows) >= 2:
        # rows[0] = top grid (D40), rows[1] = bottom grid (D30)
        cds = [r["cd_window"] for r in rows]
        span_pct = abs(cds[0] - cds[1]) / CD_REF_LOCK * 100.0
        all_gate = all(r["within_gate"] and r["steady"] and not r["diverged"] for r in rows)
        mono = abs(rows[1]["err_pct"]) > abs(rows[0]["err_pct"])
        judgment = {
            "case": "sphere Re=200, BFL big-domain ladder",
            "observable": "BFL per-link momentum ledger (laboratory frame), window-mean Cd",
            "tiers": [r["tier"] for r in rows],
            "cd_by_grid": cds,
            "cd_span_pct": span_pct,
            "all_within_gate_and_steady": all_gate,
            "grid_span_within_3pct": span_pct <= 3.0,
            "monotone_error_decrease": mono,
            "pass": bool(all_gate and span_pct <= 3.0),
        }
        print(
            f"\nladder: within_gate&steady={all_gate} span={span_pct:.3f}% "
            f"monotone|err|decrease={mono} PASS={all_gate and span_pct <= 3.0}"
        )

    if write:
        out = {
            "case": "sphere_re200 BFL big-domain ladder",
            "reference": {
                "formula": "Cd = 24/Re * (1 + 0.15*Re^0.687)",
                "re": RE_LOCK,
                "cd_ref": CD_REF_LOCK,
                "gate_pct": 3.0,
                "cross_references": CROSS_REFS,
            },
            "runs": rows,
            "judgment": judgment,
        }
        with open(os.path.join(BASE, "verify_result.json"), "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
        print("wrote verify_result.json")


if __name__ == "__main__":
    main(write="--write" in sys.argv)