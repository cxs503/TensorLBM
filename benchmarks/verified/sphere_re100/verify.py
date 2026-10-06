"""W8-A verify — recompute every judgment number from raw force histories.

Self-contained archive version: all inputs sit next to this script (the
W7 dense-kernel archive one level up).  No reliance on run-time meta
summaries beyond geometry/config fields that define the experiment.
Everything judged here is recomputed from the per-sample force histories:

  * steady window Cd (mean over the last 20% of samples) and drift
    (last-20% vs previous-20%), steady contract |drift| < 0.3% and
    steps >= 8000 and tau == 3*u_lb*D/100 + 0.5 (floating Re locked at 100)
  * err vs the LOCKED Schiller-Naumann reference (gate 3%)
  * control-volume ledger closure (per-sample and window mean)
  * bitwise reproduction checks:
      chain  — W7's own dense driver vs this track's dense driver (D20)
      T1     — this track's sparse D40-big 12k vs the W7 archive
               diag/h2_big.json (same config, dense kernel)
  * monotonicity |err(D40)| > |err(D60)| and the 3% gate on both tiers

Usage: python verify.py            # prints tables
       python verify.py --write    # additionally rewrites result.json
"""

from __future__ import annotations

import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
# Dense-kernel archive for the T1 bitwise reproduction check (W7 h2_big ==
# this track's sparse D40-big 12k).  Kept at the case root for a
# self-contained verified directory; falls back to the in-tree w7/ copy.
W7_H2_CANDIDATES = [
    os.path.join(BASE, "w7_dense_h2_big.json"),
    os.path.join(BASE, "w7", "diag", "h2_big.json"),
]

# LOCKED reference (carried unchanged from W5-B / W7-B)
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


def history_bitwise_equal(hist_a, hist_b, label_a, label_b):
    """Compare per-sample quantities exactly on the shared sample steps
    (JSON float round-trip is exact, so == compares the original doubles)."""
    steps_a = [h["step"] for h in hist_a]
    steps_b = {h["step"] for h in hist_b}
    shared = [s for s in steps_a if s in steps_b]
    if not shared:
        return False, "no shared sample steps"
    bmap = {h["step"]: h for h in hist_b}
    keys = ["cd", "cl", "cs", "mass"]
    n_bad = 0
    first = None
    for s in shared:
        a, b = next(h for h in hist_a if h["step"] == s), bmap[s]
        for k in keys:
            if a.get(k) != b.get(k):
                n_bad += 1
                if first is None:
                    first = (s, k, a.get(k), b.get(k))
    if n_bad == 0:
        return True, f"{len(shared)} shared steps x {len(keys)} quantities exact"
    return False, f"{n_bad} mismatches, first {first}"


def tier_row(path, tier):
    d = load(path)
    m = d["meta"]
    hist = d["history"]
    cd, cd_prev, drift, w = steady_window(hist)
    err = (cd - CD_REF_LOCK) / CD_REF_LOCK
    cv_rel, cv_win, cv_n = cv_closure(hist)
    tau_expect = 3.0 * m["u_lb"] * m["D"] / 100.0 + 0.5
    tau_ok = abs(m["tau"] - tau_expect) < 1e-9
    steady_ok = abs(drift) < 0.3 and m["steps"] >= 8000 and tau_ok
    return {
        "file": os.path.basename(path),
        "tier": tier,
        "kernel": m["kernel"],
        "D": m["D"],
        "tau": m["tau"],
        "u_lb": m["u_lb"],
        "re_eff": m["re_eff"],
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
    }


def main(write=False):
    print(f"locked Cd_ref = {CD_REF_LOCK!r}")

    # --- bitwise chain ------------------------------------------------------
    chain = {}
    p_a = os.path.join(BASE, "chain_w7diag_D20.json")
    p_b = os.path.join(BASE, "loop_dense_D20.json")
    if os.path.exists(p_a) and os.path.exists(p_b):
        ok, detail = history_bitwise_equal(
            load(p_a)["history"], load(p_b)["history"], "w7_diag", "w8a_dense"
        )
        chain["w7_driver_dense_vs_w8a_driver_dense_D20"] = {
            "exact": ok,
            "detail": detail,
        }
        print(f"[chain] W7 driver == W8-A driver (dense, D20): {ok} ({detail})")

    # --- T1: sparse D40 big 12k vs W7 archive h2_big -------------------------
    repro = {}
    p_t1 = os.path.join(BASE, "t1_d40big_12k.json")
    p_h2 = next((p for p in W7_H2_CANDIDATES if os.path.exists(p)), W7_H2_CANDIDATES[0])
    if os.path.exists(p_t1) and os.path.exists(p_h2):
        ok, detail = history_bitwise_equal(
            load(p_t1)["history"], load(p_h2)["history"], "w8a_sparse", "h2_big"
        )
        repro["t1_sparse_d40big_vs_w7_h2big"] = {"exact": ok, "detail": detail}
        print(f"[T1] sparse rerun == W7 h2_big (D40 big, 12k): {ok} ({detail})")

    # --- tiers ---------------------------------------------------------------
    rows = []
    tiers = []
    for path, tier in [
        (p_t1, "T1_d40big_12k"),
        (os.path.join(BASE, "t2_d60big_24k.json"), "T2_d60big_24k"),
    ]:
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        rows.append(tier_row(path, tier))
        tiers.append(rows[-1])

    for r in rows:
        cv = (
            f"{r['cv_closure_rel_samples'] * 100:.3f}%/win{r['cv_closure_rel_window'] * 100:.3f}%"
            if r["cv_closure_rel_samples"] is not None
            else "n/a"
        )
        print(
            f"{r['tier']:14s} Cd={r['cd_window']:.6f} err={r['err_pct']:+.3f}% "
            f"drift={r['drift_pct']:+.3f}% steady={r['steady']} "
            f"tau_ok={r['tau_ok']} gate3%={r['within_gate']} "
            f"blockage={r['blockage'] * 100:.2f}% CV={cv}(n={r['cv_samples']}) "
            f"steps={r['steps']} links={r['bfl_links']}"
        )

    judgment = {"pass": None, "note": "tiers incomplete"}
    if len(tiers) >= 2:
        mono = abs(tiers[0]["err_pct"]) > abs(tiers[1]["err_pct"])
        all_gate = all(t["within_gate"] and t["steady"] for t in tiers)
        judgment = {
            "case": "sphere Re=100, BFL big-domain ladder (W8-A stretch)",
            "observable": "BFL per-link momentum ledger (laboratory frame), window-mean Cd",
            "tiers": [t["tier"] for t in tiers],
            "all_within_gate_and_steady": all_gate,
            "monotone_error_decrease": mono,
            "pass": bool(all_gate and mono),
        }
        print(
            f"\nladder: within_gate&steady={all_gate} "
            f"monotone|err|decrease={mono} PASS={all_gate and mono}"
        )

    if write:
        out = {
            "case": "W8-A BFL sparse-kernel memory fix + sphere big-domain ladder",
            "reference": {
                "formula": "Cd = 24/Re * (1 + 0.15*Re^0.687)",
                "re": 100.0,
                "cd_ref": CD_REF_LOCK,
                "gate_pct": 3.0,
                "cross_references": CROSS_REFS,
            },
            "bitwise_chain": chain,
            "t1_reproduction": repro,
            "runs": rows,
            "judgment": judgment,
        }
        with open(os.path.join(BASE, "result.json"), "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main(write="--write" in sys.argv)
