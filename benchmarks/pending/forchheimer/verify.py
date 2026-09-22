#!/usr/bin/env python3
"""W6-B independent verification: closed-form recomputation of every number
in result.json from the raw case JSONs + the locked reference JSON.

Independence rules (mirrors the campaign convention):
  * nothing is imported from run.py;
  * R_sim, re_darcy, k_hat, err, verdicts are re-derived from the raw
    fields (a_body, uxf_plateau, uxd_plateau, d_p_lu, nu, phi_s_mask);
  * the reference ratio R_ref_A at each tier's Re_target is taken from the
    LOCKED table directly (tier Re values 25.20 / 46.88 / 103.64 are exact
    series-A grid points, so the interpolation is exact there by
    construction) and additionally from a fresh log-log linear
    interpolation at the achieved Re -- the two must agree to < 0.05%,
    bounding the interpolation term of the criterion;
  * exits non-zero on any failed check.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ART = Path(__file__).resolve().parent
C_INERTIAL = 6.56e-4
TIERS = {
    "A": {"re": 25.20, "tau": 0.55, "ladder": [52, 104, 208]},
    "B": {"re": 46.88, "tau": 0.55, "ladder": [104, 156, 208]},
    "C": {"re": 103.64, "tau": 0.55, "ladder": [208, 312, 416]},
}
SA_TABLE = [
    (0.05, 15.56),
    (0.10, 24.83),
    (0.20, 51.53),
    (0.30, 102.90),
    (0.40, 217.89),
    (0.50, 532.55),
    (0.60, 1.763e3),
    (0.70, 1.352e4),
    (0.75, 1.263e5),
]

failures: list[str] = []
notes: list[str] = []


def check(cond: bool, msg: str) -> bool:
    if not cond:
        failures.append(msg)
    return cond


def loglog_interp(table, x):
    xs = [math.log(p) for p, _ in table]
    ys = [math.log(v) for _, v in table]
    lx = math.log(x)
    for i in range(len(xs) - 1):
        if xs[i] <= lx <= xs[i + 1]:
            t = (lx - xs[i]) / (xs[i + 1] - xs[i])
            return math.exp(ys[i] + t * (ys[i + 1] - ys[i]))
    raise ValueError(f"{x} outside table")


def main() -> int:
    result = json.loads((ART / "result.json").read_text())
    ref = json.loads((ART / "forchheimer_ref_fig6.json").read_text())
    # drop the Re->0 intercept markers (digitised at Re = -0.37 / +0.42):
    # they carry no inertial information and break log interpolation
    rawA = [p for p in ref["A_koch_ladd_1997"]["points"] if p["re_darcy"] > 1.0]
    rawB = [p for p in ref["B_fdm_2023"]["points"] if p["re_darcy"] > 1.0]
    ptsA = {round(p["re_darcy"], 2): p["R"] for p in rawA}
    ptsA_list = [(p["re_darcy"], p["R"]) for p in rawA]
    ptsB_list = [(p["re_darcy"], p["R"]) for p in rawB]
    check(
        len(rawA) == 8 and len(rawB) == 9, f"reference point counts {len(rawA)}/{len(rawB)} != 8/9"
    )

    # --- sanity of the locked reference itself --------------------------
    check(abs(ptsA[25.20] - 1.182235) < 5e-6, "locked R(25.20) mismatch")
    check(abs(ptsA[46.88] - 1.284402) < 5e-6, "locked R(46.88) mismatch")
    check(abs(ptsA[103.64] - 1.435497) < 5e-6, "locked R(103.64) mismatch")

    rows = {r["tier"] + str(r["d"]): r for r in result["rows"]}
    check(len(rows) == 9, f"expected 9 rows, got {len(rows)}")

    for tier, cfg in TIERS.items():
        for d in cfg["ladder"]:
            key = f"{tier}{d}"
            if key not in rows:
                failures.append(f"missing row {key}")
                continue
            row = rows[key]
            st = json.loads((ART / row["stokes_file"]).read_text())
            rr = json.loads((ART / row["re_file"]).read_text())

            # geometry / pairing consistency
            check(st["d"] == d == rr["d"], f"{key}: d mismatch in case files")
            check(abs(st["tau"] - cfg["tau"]) < 1e-12, f"{key}: anchor tau")
            check(rr["anchor_case"] == row["stokes_file"], f"{key}: finite case anchored elsewhere")

            # closed-form recomputation from raw fields
            nu = (cfg["tau"] - 0.5) / 3.0
            check(abs(rr["nu"] - nu) < 1e-15, f"{key}: nu field")
            re_ach = rr["uxd_plateau"] * rr["d_p_lu"] / nu
            check(abs(re_ach - rr["re_darcy"]) < 1e-6 * re_ach, f"{key}: re_darcy recomputation")
            check(
                abs(re_ach / cfg["re"] - 1.0) < 0.002,
                f"{key}: Re_ach off target by >0.2% ({re_ach})",
            )
            k_st = nu * st["uxf_plateau"] / st["a_body"]
            check(abs(k_st - st["k_hat"]) < 1e-9 * k_st, f"{key}: k_hat recomputation")

            r_anchor = 1.0 + C_INERTIAL * st["re_darcy"] ** 2
            R_sim = (
                (rr["a_body"] / rr["uxf_plateau"]) / (st["a_body"] / st["uxf_plateau"]) * r_anchor
            )
            check(
                abs(R_sim - row["R_sim"]) < 1e-9 * R_sim,
                f"{key}: R_sim recomputation {R_sim} vs {row['R_sim']}",
            )

            # reference evaluation: result.json evaluates the reference at
            # the ACHIEVED Re (up to 0.2% off the node); with dlnR/dlnRe
            # ~0.14 that shifts R_ref by up to ~3e-4 relative.  The real
            # check is agreement between the two INDEPENDENT interpolants
            # (locked-node value + log-log) and the recorded PCHIP value.
            R_locked = ptsA[round(cfg["re"], 2)]
            R_interp = loglog_interp(ptsA_list, re_ach)
            node_off = abs(row["R_ref_A"] / R_locked - 1.0)
            check(
                node_off < 4e-4,
                f"{key}: R_ref_A vs locked node {node_off:.2e} exceeds the Re_ach-offset band",
            )
            spread = abs(R_interp / row["R_ref_A"] - 1.0)
            check(spread < 5e-4, f"{key}: independent-interpolant spread {spread:.2e} > 0.05%")
            notes.append(
                f"{key}: R_ref node-offset {node_off:.3e}, interpolant spread {spread:.3e}"
            )

            err = abs(R_sim / row["R_ref_A"] - 1.0)
            check(abs(err - row["err_A"]) < 1e-12, f"{key}: err_A recomputation")

            # err_B disclosure column: recompute with the independent
            # log-log interpolant; PCHIP-vs-loglog spread on the wide B
            # intervals is the interpolation uncertainty of that column
            errB_ind = abs(R_sim / loglog_interp(ptsB_list, re_ach) - 1.0)
            bspread = abs(errB_ind - row["err_B"])
            check(bspread < 6e-3, f"{key}: err_B interpolant spread {bspread:.2e} > 0.6%")
            notes.append(f"{key}: err_B interpolant spread {bspread:.3e}")

            # invariants (mass-drift bound is a sanity ceiling, not a
            # pre-registered gate; drifts are disclosed per case)
            check(row["ma_umax"] < 0.1, f"{key}: Ma(u_max) {row['ma_umax']} >= 0.1")
            check(abs(row["mass_drift"]) < 5e-3, f"{key}: mass drift {row['mass_drift']}")
            check(row["steady"], f"{key}: steady flag false")
            check(row["converged_re"], f"{key}: secant did not land Re")

            # Stokes-branch cross-checks (not part of the criterion)
            y0_sim = d * d / k_st
            f_sa = loglog_interp(SA_TABLE, st["phi_s_mask"])
            check(
                0.30 < st["phi_s_mask"] < 0.45,
                f"{key}: phi_s_mask {st['phi_s_mask']} far from 0.40",
            )

            print(
                f"[{key}] Re_ach={re_ach:.3f}/{cfg['re']} R_sim={R_sim:.6f} "
                f"R_ref={R_locked:.6f} err={err * 100:.3f}% Ma={row['ma_umax']:.4f} "
                f"osc={row['osc']:.2e} y0_sim={y0_sim:.2f} f(SA)={f_sa:.1f}"
            )

    # tier verdicts
    for tier, cfg in TIERS.items():
        tr = sorted((r for r in result["rows"] if r["tier"] == tier), key=lambda r: r["d"])
        errs = [r["err_A"] for r in tr]
        mono = all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
        finest = errs[-1] <= 0.03
        want = "PASS" if (mono and finest) else "FAIL"
        got = result["tier_verdicts"][tier]["tier_verdict"]
        check(got == want, f"tier {tier}: verdict {got} != recomputed {want}")
        print(
            f"[tier {tier}] errs={['%.4f' % e for e in errs]} "
            f"monotone={mono} finest={errs[-1] * 100:.3f}% -> {want}"
        )

    overall = result["overall"]
    want_overall = (
        "PASS"
        if all(v["tier_verdict"] == "PASS" for v in result["tier_verdicts"].values())
        else "FAIL"
    )
    check(overall == want_overall, f"overall {overall} != {want_overall}")

    print()
    for n in notes:
        print("note:", n)
    if failures:
        print("VERIFY FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("VERIFY PASS: all result.json numbers independently reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
