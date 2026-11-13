#!/usr/bin/env python3
"""Machine referee for anchor ``XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN``.

Reads ``result.json`` produced by ``run.py`` and adjudicates every gate the
ledger entry names.  Exit code 0 = pass, 1 = fail (first failing gate is named).
Nothing here re-derives physics: it only checks the recorded numbers against the
locked constants in this file, so a stale or edited result file cannot pass by
accident (parameters are re-checked against the locked values).

    python benchmarks/verified/fused_push_pull_aa_pattern/verify.py [--result DIR]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

LOCKED = {
    "nu": 1.0 / 6.0,
    "u_max": 0.02,
    "gate_pct": 0.15,
    "target_pct": 0.03,
}
TOL_PARAM = 1e-12
TOL_EQUIV_STATE = 1e-12
TOL_EQUIV_VELOCITY = 1e-10
TOL_MASS_DRIFT = 1e-12
ORDER_MIN = 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(pathlib.Path(__file__).resolve().parent))
    args = ap.parse_args()
    root = pathlib.Path(args.result)
    path = root if root.suffix == ".json" else root / "result.json"
    if not path.exists():
        print(f"FAIL: {path} missing (run run.py first)")
        return 1
    res = json.loads(path.read_text())
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(("PASS " if ok else "FAIL ") + f"{name}: {detail}")
        if not ok:
            fails.append(name)

    # ---- locked parameters ------------------------------------------------
    par = res["parameters"]
    check("anchor", res.get("anchor") == "XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN", res.get("anchor", "?"))
    check("nu", abs(par["nu"] - LOCKED["nu"]) < TOL_PARAM, f"{par['nu']:.12f}")
    check("u_max", abs(par["u_max"] - LOCKED["u_max"]) < TOL_PARAM, f"{par['u_max']}")
    check("gate", abs(par["gate_pct"] - LOCKED["gate_pct"]) < TOL_PARAM, f"{par['gate_pct']}")

    arms = res["arms"]
    finest_grid = max(a["grid"] for a in arms)
    for kernel in ("ab", "fused_push_pull"):
        finest = [a for a in arms if a["grid"] == finest_grid and a["kernel"] == kernel]
        if not finest:
            check(f"arm[{kernel}]", False, "missing")
            continue
        row = finest[0]
        check(
            f"ux_rel_l2[{kernel}]@n={finest_grid}",
            row["ux_lu2"] < LOCKED["target_pct"],
            f"{100 * row['ux_lu2']:.4f}% < 3% target (gate 15%)",
        )
        check(f"uy_rel_l2[{kernel}]", row["uy_lu2"] < LOCKED["target_pct"], f"{100 * row['uy_lu2']:.4f}%")
        check(
            f"mass_drift[{kernel}]",
            row["mass_drift_rel"] < TOL_MASS_DRIFT,
            f"{row['mass_drift_rel']:.2e} < 1e-12",
        )

    # ---- scheme equivalence ----------------------------------------------
    eq = res["equivalence"]
    check(
        "equivalence_state(fused == collide(ab))",
        eq["max_abs_fused_vs_collided_ab"] < TOL_EQUIV_STATE,
        f"{eq['max_abs_fused_vs_collided_ab']:.3e} < 1e-12",
    )
    check(
        "equivalence_velocity(cross-arm)",
        eq["rel_l2_velocity_cross_arm"] < TOL_EQUIV_VELOCITY,
        f"{eq['rel_l2_velocity_cross_arm']:.3e} < 1e-10",
    )

    # ---- convergence ------------------------------------------------------
    for entry in res["ladder"]:
        orders = [o for o in entry["orders"] if o == o]  # drop NaN
        check(
            f"order[{entry['kernel']}]",
            bool(orders) and max(orders) >= ORDER_MIN,
            f"observed={['%.3f' % o for o in entry['orders']]} >= {ORDER_MIN}",
        )

    # ---- negative controls must fail the gate ----------------------------
    names = {c["name"] for c in res["controls"]}
    check("controls_present", {"nu_bias_25pct", "mirror_we_initial"} <= names, sorted(names))
    for c in res["controls"]:
        check(
            f"control_bites[{c['name']}]",
            c["ux_lu2"] >= LOCKED["gate_pct"],
            f"{100 * c['ux_lu2']:.2f}% >= 15% gate (gate is not toothless)",
        )

    print()
    if fails:
        print(f"VERDICT FAIL ({len(fails)} gate(s)): {', '.join(fails)}")
        return 1
    print(f"VERDICT PASS — {len(arms)} arms, {len(res['ladder'])} ladders, {len(res['controls'])} controls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
