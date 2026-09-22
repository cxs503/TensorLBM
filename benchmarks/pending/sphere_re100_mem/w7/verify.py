"""W7-B verify — recompute every judgment number from raw force histories.

Reads formal_*.json (and signature diagnostics), recomputes steady windows,
err vs the LOCKED reference, gate booleans, monotonicity, CV closure, and
blockage. No reliance on run-time meta beyond geometry/dpS fields.

Usage: python verify.py            # check mode, prints table
       python verify.py --write    # additionally writes result.json
"""

from __future__ import annotations

import json
import os
import sys

STAGE = os.path.dirname(os.path.abspath(__file__))

# LOCKED reference (W5-B carry-over, unchanged)
CD_REF_LOCK = 24.0 / 100.0 * (1.0 + 0.15 * 100.0**0.687)
CROSS_REFS = {
    "schiller_naumann_0681": 1.0685190542670633,
    "clift_gauvin": 1.1092345787735252,
}
GATE = 0.03


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
    """Per-sample mean |cv-ledger|/ledger AND window-mean comparison
    (W5-B comparable: window-mean rel diff)."""
    pairs = [(h["cd"], h["cv_mom_cd"]) for h in hist if "cv_mom_cd" in h]
    if not pairs:
        return None, None, 0
    rel = [abs(b - a) / abs(a) for a, b in pairs]
    mean_ledger = sum(a for a, _ in pairs) / len(pairs)
    mean_cv = sum(b for _, b in pairs) / len(pairs)
    window_rel = abs(mean_cv - mean_ledger) / mean_ledger
    return sum(rel) / len(rel), window_rel, len(pairs)


def load(fname):
    with open(os.path.join(STAGE, fname)) as fh:
        return json.load(fh)


def main(write=False):
    rows = []
    tiers = []
    for fname, tier in [
        ("formal_bfl_D40.json", "bfl_D40"),
        ("formal_bfl_D60.json", "bfl_D60"),
        ("formal_bb_D40.json", "bb_D40"),
        ("formal_bb_D60.json", "bb_D60"),
    ]:
        path = os.path.join(STAGE, fname)
        if not os.path.exists(path):
            print(f"MISSING {fname}")
            continue
        d = load(fname)
        m = d["meta"]
        hist = d["history"]
        cd, cd_prev, drift, w = steady_window(hist)
        err = (cd - CD_REF_LOCK) / CD_REF_LOCK
        cv_rel, cv_win, cv_n = cv_closure(hist)
        # Amendment #3 guard: tau must match Re=100 at this D (3*ulb*D/Re+0.5)
        tau_expect = 3.0 * m["u_lb"] * m["D"] / 100.0 + 0.5
        tau_ok = abs(m["tau"] - tau_expect) < 1e-9
        steady_ok = abs(drift) < 0.3 and m["steps"] >= 8000 and tau_ok
        rows.append(
            {
                "file": fname,
                "tier": tier,
                "route": m["route"],
                "collision": m["collision"],
                "D": m["D"],
                "tau": m["tau"],
                "u_lb": m["u_lb"],
                "re_eff": m["re_eff"],
                "domain_lu": m["domain_lu"],
                "lat": m["lat"],
                "up": m["up"],
                "down": m["down"],
                "blockage": m["blockage"],
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
            }
        )
        if tier.startswith("bfl"):
            tiers.append(rows[-1])

    print(f"locked Cd_ref = {CD_REF_LOCK!r}")
    for r in rows:
        cv = (
            f"{r['cv_closure_rel_samples'] * 100:.3f}%/win{r['cv_closure_rel_window'] * 100:.3f}%"
            if r["cv_closure_rel_samples"] is not None
            else "n/a"
        )
        print(
            f"{r['tier']:8s} Cd={r['cd_window']:.5f} err={r['err_pct']:+.2f}% "
            f"drift={r['drift_pct']:+.3f}% steady={r['steady']} "
            f"tau_ok={r['tau_ok']} "
            f"gate3%={r['within_gate']} blockage={r['blockage'] * 100:.2f}% "
            f"CV={cv}(n={r['cv_samples']}) steps={r['steps']}"
        )

    if len(tiers) >= 2:
        mono = abs(tiers[0]["err_pct"]) > abs(tiers[1]["err_pct"])
        all_gate = all(t["within_gate"] for t in tiers)
        print(
            f"\nbfl tiers: within_gate={all_gate} "
            f"monotone|err|decrease={mono} PASS={all_gate and mono}"
        )
        judgment = {
            "route": "bfl",
            "observable": "BFL per-link momentum ledger (laboratory frame), window-mean Cd",
            "n_grids": len(tiers),
            "all_within_gate": all_gate,
            "monotone_error_decrease": mono,
            "pass": bool(all_gate and mono),
        }
    else:
        judgment = {"pass": None, "note": "tiers incomplete"}

    # domain-convergence signature (attribution columns, NOT judgment)
    sig_files = [
        ("up1.25", "diag/t1_mrt_base.json"),
        ("up2.0", "diag/h2_up2.0.json"),
        ("up2.75 (12k formal)", "diag/formal12k_bfl_D40.json"),
        ("up3.0", "diag/h2_up3.0.json"),
        ("up4.0", "diag/p1_up4.0.json"),
        ("big lat3/up3/down4", "diag/h2_big.json"),
    ]
    sig = []
    for label, f in sig_files:
        p = os.path.join(STAGE, f)
        if not os.path.exists(p):
            continue
        d = load(f)
        h = d["history"]
        cd, _, drift, _ = steady_window(h)
        sig.append(
            {
                "probe": label,
                "file": f,
                "cd": cd,
                "err_pct": (cd - CD_REF_LOCK) / CD_REF_LOCK * 100.0,
                "drift_pct": drift,
            }
        )
    print("\ndomain-convergence signature (D40):")
    for s in sig:
        print(f"  {s['probe']:22s} Cd={s['cd']:.5f} err={s['err_pct']:+.2f}%")

    if write:
        out = {
            "case": "sphere Re=100 solution-side floor (W7-B)",
            "reference": {
                "formula": "Cd = 24/Re * (1 + 0.15*Re^0.687)",
                "re": 100.0,
                "cd_ref": CD_REF_LOCK,
                "gate_pct": 3.0,
                "cross_references": CROSS_REFS,
            },
            "runs": rows,
            "judgment": judgment,
            "domain_convergence_signature": sig,
        }
        with open(os.path.join(STAGE, "result.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        print("\nwrote result.json")


if __name__ == "__main__":
    main("--write" in sys.argv)
