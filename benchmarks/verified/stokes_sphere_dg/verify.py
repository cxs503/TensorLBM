#!/usr/bin/env python3
"""Independent re-verification of stokes_sphere_dg from the archived series.

Recomputes every judged quantity from case_*.json series arrays (never
reading the stored err/S fields) and re-checks the prereg gates.
--write regenerates result.json (bit-stable against the archived one).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RIDS = ["phi0p%s_R%d_tau1p0" % (p, r) for p in ("005", "01", "02") for r in (8, 12, 16, 24)]
PHIS = ("005", "01", "02")
MAIN_TOL, S0_TOL, S1_TOL, S2_TOL, S3_TOL = 0.03, 1e-6, 1e-3, 1e-3, 1e-2
BUDGET_LIMIT_H = 46.0


def tail(n, f):
    k = max(1, math.ceil(f * n))
    return slice(n - k, n)


def main(write=False):
    out, wall = {}, 0.0
    for rid in RIDS:
        d = json.load(open(os.path.join(HERE, "case_%s.json" % rid)))
        s = d["series"]
        n = len(s)
        w, w40 = s[tail(n, 0.2)], s[tail(n, 0.4)]
        ksim = sum(x["K"] for x in w) / len(w)
        err = ksim / d["inputs"]["K_ref_bie_P5"] - 1.0
        u20 = sum(x["U_sup"] for x in w) / len(w)
        u40 = sum(x["U_sup"] for x in w40) / len(w40)
        fled = d["estimators"]["F_ledger"]
        s3 = sum(abs(x["F_ME"] - fled) / fled for x in w) / len(w)
        s0 = max(abs(x["mass"] / s[0]["mass"] - 1.0) for x in s)
        W = max(50, min(200, math.ceil(0.2 * n)))
        wm = [
            sum(x["U_sup"] for x in s[i : i + 50]) / 50
            for i in range(max(0, n - W), n - 50 + 1, 10)
        ]
        s1 = (max(wm) - min(wm)) / (sum(wm) / len(wm)) if len(wm) > 1 else 0.0
        wall += d["wall_s"]
        out[rid] = {
            "K_sim": ksim,
            "err": err,
            "S0": s0,
            "S1": s1,
            "S2": abs(u20 - u40) / u20,
            "S3": s3,
        }
    main_ok = all(abs(v["err"]) <= MAIN_TOL for v in out.values())
    sg_ok = all(
        v["S0"] <= S0_TOL and v["S1"] <= S1_TOL and v["S2"] <= S2_TOL and v["S3"] <= S3_TOL
        for v in out.values()
    )
    conv_ok = True
    for p in PHIS:
        seq = [abs(out["phi0p%s_R%d_tau1p0" % (p, r)]["err"]) for r in (8, 12, 16, 24)]
        conv_ok &= all(seq[i + 1] < seq[i] for i in range(3))
    verdict = {
        "recomputed_tiers": out,
        "wall_total_h": wall / 3600.0,
        "main_gate_all_pass": main_ok,
        "s_gates_all_pass": sg_ok,
        "convergence_gate_all_pass": bool(conv_ok),
        "OVERALL": "PASS"
        if (main_ok and sg_ok and conv_ok and wall / 3600.0 <= BUDGET_LIMIT_H)
        else "FAIL",
    }
    if write:
        base = json.load(open(os.path.join(HERE, "result.json")))
        base["verify_recompute"] = verdict
        json.dump(base, open(os.path.join(HERE, "result.json"), "w"), indent=1)
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["OVERALL"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(write="--write" in sys.argv))
