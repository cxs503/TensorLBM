#!/usr/bin/env python3
"""couette_3d verifier (frozen gates in prereg.md).

Reads ONLY the case JSONs written by run_v2.py; applies G1/G2a/G2b/G3 and
assembles the mandatory G2-disclosure rows; writes gate/verdict_v2.json.
"""

from __future__ import annotations

import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def load(dtype, H):
    return json.load(open(os.path.join(BASE, "gate", dtype, f"case_H{H}_{dtype}.json")))


def main() -> None:
    fp32_H = [32, 64, 128]
    fp64_H = [16, 32, 64, 128]
    t32 = {H: load("fp32", H) for H in fp32_H}
    t64 = {H: load("fp64", H) for H in fp64_H}

    g1 = all(r["l2_rel_err"] <= 3e-2 and r["max_rel_err_masked_pct"] <= 3.0 for r in t32.values())
    g2a = all(r["l2_rel_err"] <= 1e-3 for r in t32.values())
    g2b = all(r["l2_rel_err"] <= 1e-7 for r in t64.values())

    def health(r):
        return (
            r["steady"] is True
            and r["n_steps"] <= 24000
            and r["mass_drift_pct"] <= 0.05
            and r["uz_max"] <= 1e-7
            and r["ux_spanwise_std"] <= 1e-6
            and r["finite"] is True
        )

    g3 = all(health(r) for r in list(t32.values()) + list(t64.values()))

    l2_32 = {H: t32[H]["l2_rel_err"] for H in fp32_H}
    l2_64 = {H: t64[H]["l2_rel_err"] for H in fp64_H}
    disclosure = {
        "fp32_l2_by_tier": l2_32,
        "fp64_l2_by_tier": l2_64,
        "fp32_mass_drift_pct": {H: t32[H]["mass_drift_pct"] for H in fp32_H},
        "fp32_n_steps": {H: t32[H]["n_steps"] for H in fp32_H},
        "statement": (
            "fp32 residuals are time-linear fp32 rounding accumulation "
            "(evidence diag_dtype.json: fp64 same kernels zero drift, l2 "
            "1e-13..2e-12), not resolution error; monotone-decreasing "
            "clause replaced by G2a+G2b per prereg.md (owner sign-off)."
        ),
    }
    verdict = {
        "case": "couette_3d",
        "gates_frozen_in": "prereg.md (md5 receipt)",
        "G1_accuracy": g1,
        "G2a_fp32_floor": g2a,
        "G2b_scheme_exactness_fp64": g2b,
        "G3_health": g3,
        "G2_disclosure": disclosure,
        "per_tier_fp32": {
            str(H): {
                k: t32[H][k]
                for k in (
                    "l2_rel_err",
                    "max_rel_err_masked_pct",
                    "u_top_row_err_pct",
                    "tau_err_pct",
                    "mass_drift_pct",
                    "uz_max",
                    "ux_spanwise_std",
                    "n_steps",
                    "steady",
                )
            }
            for H in fp32_H
        },
        "per_tier_fp64": {
            str(H): {k: t64[H][k] for k in ("l2_rel_err", "mass_drift_pct", "n_steps")}
            for H in fp64_H
        },
        "verdict": "PASS" if (g1 and g2a and g2b and g3) else "FAIL",
    }
    out = os.path.join(BASE, "gate", "verdict_v2.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(
        json.dumps(
            {
                k: verdict[k]
                for k in (
                    "G1_accuracy",
                    "G2a_fp32_floor",
                    "G2b_scheme_exactness_fp64",
                    "G3_health",
                    "verdict",
                )
            },
            indent=1,
        )
    )
    print("fp32 l2:", l2_32)
    print("fp64 l2:", l2_64)
    sys.exit(0 if verdict["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
