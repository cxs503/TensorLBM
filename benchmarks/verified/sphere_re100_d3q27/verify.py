#!/usr/bin/env python3
"""sphere_re100_d3q27 verifier (frozen gates in prereg.md).

Reads ONLY JSON files (tier histories + reflocked archive); writes
gate/verdict27.json; exit 0 iff G1 ∧ G2 ∧ G3 (G4 rows mandatory content).
Zero hand transcription: every reference number is read from the
machine-pulled archive under diag/remote_pull/.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(BASE, "diag", "remote_pull", "verified__sphere_re100")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def window_cd(hist, steps, lo, hi):
    vals = [e["cd"] for e in hist if lo * steps < e["step"] <= hi * steps]
    return sum(vals) / len(vals), len(vals)


def tier_verdict(path, cd_ref):
    j = json.load(open(path))
    m, hist = j["meta"], j["history"]
    steps = m["steps_requested"] if not m["diverged"] else m["steps"]
    w_cd, n_w = window_cd(hist, steps, 0.8, 1.0)
    p_cd, n_p = window_cd(hist, steps, 0.6, 0.8)
    drift = abs(w_cd - p_cd) / abs(w_cd) * 100.0
    cv_tail = [e for e in hist if e["step"] > 0.8 * steps and "cv_mom_cd" in e]
    cv_closure = (
        sum(abs(e["cv_mom_cd"] - e["cd"]) for e in cv_tail) / len(cv_tail) / abs(w_cd)
        if cv_tail else None
    )
    err = abs(w_cd - cd_ref) / cd_ref * 100.0
    g3 = {
        "drift_pct": drift,
        "drift_ok": drift < 0.3,
        "cv_closure": cv_closure,
        "cv_ok": (cv_closure is not None and cv_closure <= 1e-2),
        "mass_drift_ppm": m["mass_drift_ppm"],
        "mass_ok": m["mass_drift_ppm"] is not None and abs(m["mass_drift_ppm"]) <= 100,
        "finite": m["diverged"] is False,
        "archive_twin": m.get("archive_twin_check"),
        "twin_ok": m.get("archive_twin_check") is not None
        and all(m["archive_twin_check"]["checks"].values()),
    }
    return {
        "file": os.path.basename(path), "steps_used": steps,
        "cd_window": w_cd, "n_window": n_w, "cd_prev20": p_cd, "n_prev20": n_p,
        "err_pct": err, "err_ok": err <= 3.0,
        "g3": g3, "meta": {k: m[k] for k in
                           ("D", "domain_lu", "tau", "u_lb", "re_eff", "bfl_links",
                            "wall_s", "mass_drift_ppm", "diverged")},
    }


def main() -> None:
    ref = json.load(open(os.path.join(ARCH, "result.json")))
    cd_ref = ref["reference"]["cd_ref"]
    t1 = tier_verdict(os.path.join(BASE, "gate", "t1_D40_27.json"), cd_ref)
    t2 = tier_verdict(os.path.join(BASE, "gate", "t2_D60_27.json"), cd_ref)
    g1 = t1["err_ok"] and t2["err_ok"]
    g2 = t2["err_pct"] < t1["err_pct"]
    g3 = all(v for k, v in t1["g3"].items() if k.endswith("_ok")) and \
         all(v for k, v in t2["g3"].items() if k.endswith("_ok"))
    # G4 disclosure rows (machine-read archive)
    g19 = ref["grids"]
    bgk = json.load(open(os.path.join(ARCH, "w7", "diag", "t1_bgk.json")))
    mrt = json.load(open(os.path.join(ARCH, "w7", "diag", "t1_mrt_base.json")))
    def w20(j):
        s = j["meta"]["steps"]
        v = [e["cd"] for e in j["history"] if e["step"] > 0.8 * s]
        return sum(v) / len(v)
    bgk_cd, mrt_cd = w20(bgk), w20(mrt)
    span27 = abs(t2["cd_window"] - t1["cd_window"]) / t1["cd_window"] * 100.0
    g4 = {
        "cd19_mrt_D40": g19["D40"]["cd"], "cd27_D40": t1["cd_window"],
        "delta27_19_D40_pct": (t1["cd_window"] - g19["D40"]["cd"]) / g19["D40"]["cd"] * 100.0,
        "cd19_mrt_D60": g19["D60"]["cd"], "cd27_D60": t2["cd_window"],
        "delta27_19_D60_pct": (t2["cd_window"] - g19["D60"]["cd"]) / g19["D60"]["cd"] * 100.0,
        "collision_decomp_domain": bgk["meta"]["domain_lu"],
        "cd19_bgk_D40small": bgk_cd, "cd19_mrt_D40small": mrt_cd,
        "delta_collision_pct": (bgk_cd - mrt_cd) / mrt_cd * 100.0,
        "span27_pct": span27, "span19_pct_archived": ref["convergence"]["cd_span_pct"],
        "err19_archived_D40": g19["D40"]["err_pct"], "err19_archived_D60": g19["D60"]["err_pct"],
    }
    verdict = {
        "case": "sphere_re100_d3q27",
        "gates_frozen_in": "prereg.md (md5 receipt)",
        "cd_ref": cd_ref,
        "cd_ref_source": "remote_pull/verified__sphere_re100/result.json",
        "ref_sha256": {
            "result.json": sha256(os.path.join(ARCH, "result.json")),
            "t1_d40big_12k.json": sha256(os.path.join(ARCH, "t1_d40big_12k.json")),
            "t2_d60big_24k.json": sha256(os.path.join(ARCH, "t2_d60big_24k.json")),
            "manifest.json": sha256(os.path.join(BASE, "diag", "remote_pull", "manifest.json")),
        },
        "t1": t1, "t2": t2,
        "G1_accuracy": g1, "G2_monotone": g2, "G3_health": g3,
        "G4_disclosure": g4,
        "verdict": "PASS" if (g1 and g2 and g3) else "FAIL",
    }
    out = os.path.join(BASE, "gate", "verdict27.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(json.dumps({k: verdict[k] for k in
                      ("G1_accuracy", "G2_monotone", "G3_health", "verdict")}, indent=1))
    print("G4:", json.dumps(g4, indent=1))
    sys.exit(0 if verdict["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
