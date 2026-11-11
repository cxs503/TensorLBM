"""Board-audit (m15044) of mission m14689 — GAP-LBM-MCMP-POISEUILLE.

Non-redo audit: reads the delivered artifacts from the m14689 work tree, re-derives
every acceptance number independently from the raw records, re-runs the repository
reproduction commands, and emits audit_summary.json + two figures.

Usage:
    python3 scripts/audit_m14689.py [--src /root/tp-agent/work/m14689]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "artifacts/audit_m14689"


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/tp-agent/work/m14689")
    args = ap.parse_args()
    src = args.src
    sys.path.insert(0, os.path.join(src, "cases", "mcmp_stratified_poiseuille"))
    from analytic import layer_viscosities, layered_reference, sample_profile, wall_conventions  # noqa: E402

    os.makedirs(OUT, exist_ok=True)
    res = load(os.path.join(src, "benchmarks/verified/mcmp_stratified_poiseuille/result_r2.json"))
    out = {"mission": "m14689", "anchor": "GAP-LBM-MCMP-POISEUILLE",
           "source_tree": src, "checks": {}, "criteria": {}}

    # ---- 1. independent re-derivation of the reference and of criteria 1/2 ----
    rows = []
    for ny in (25, 33, 49):
        rec = load(os.path.join(
            src, f"artifacts/mcmp_stratified_poiseuille/records/r2_ny{ny}_mirror.json"))
        conv = wall_conventions(ny)["halfway"]
        row = {"ny": ny, "steady": rec["steady"],
               "y_int_row": rec["interface_crossing"],
               "y_int_target_row": conv["h"] + conv["y_offset"],
               "Q_mix": rec["Q_mix"], "sum_u_profile": sum(rec["u_profile_mix"]),
               "S_all_mix": rec["spurious_mix"], "u_max_mix": rec["u_max_mix"],
               "rules": {}}
        row["err_interface"] = (row["y_int_row"] / row["y_int_target_row"]) - 1.0
        for rule in ("component", "mixture"):
            nl, nu = layer_viscosities(1.0, 0.75, rule=rule)
            ref = layered_reference(ny, conv, 5e-5, nl, nu)
            prof = sample_profile(ref, conv["y_offset"], ny)
            stored = load(os.path.join(
                src, f"artifacts/mcmp_stratified_poiseuille/records/r2_ny{ny}_mirror.json"
            ))["analytical"][f"halfway/{rule}"]
            row["rules"][rule] = {
                "Q_ref_independent": ref["Q_total"],
                "Q_ref_stored": stored["Q_total"],
                "dQ_ref": ref["Q_total"] - stored["Q_total"],
                "err_Q_mix_independent": row["Q_mix"] / ref["Q_total"] - 1.0,
                "err_Q_mix_stored": stored["err_Q_mix"],
                "u_max_ref": max(prof),
            }
        row["ratio_S"] = row["S_all_mix"] / row["u_max_mix"]
        rows.append(row)
    out["checks"]["flow_rows"] = rows
    out["checks"]["Q_mix_equals_sum_profile_max_abs_diff"] = max(
        abs(r["Q_mix"] - r["sum_u_profile"]) for r in rows)
    out["checks"]["ref_recompute_max_abs_diff"] = max(
        abs(r["rules"][k]["dQ_ref"]) for r in rows for k in ("component", "mixture"))

    thresholds = res["thresholds"]
    out["criteria"] = {
        "c1_interface": {
            "threshold": thresholds["err_iface"],
            "max_abs_err": max(abs(r["err_interface"]) for r in rows),
            "verdict_independent": "PASS" if max(
                abs(r["err_interface"]) for r in rows) < thresholds["err_iface"] else "FAIL",
            "verdict_delivered": res["criteria"]["c1_interface"]["verdict"]},
        "c2_flow": {
            "threshold": thresholds["err_Q_mix"],
            "max_abs_err_mixture": max(abs(r["rules"]["mixture"]["err_Q_mix_independent"])
                                       for r in rows),
            "max_abs_err_component": max(abs(r["rules"]["component"]["err_Q_mix_independent"])
                                         for r in rows),
            "verdict_independent": "PASS" if max(
                abs(r["rules"][k]["err_Q_mix_independent"]) for r in rows
                for k in ("component", "mixture")) < thresholds["err_Q_mix"] else "FAIL",
            "verdict_delivered": res["criteria"]["c2_flow"]["verdict"]},
        "c3_spurious": {
            "threshold": thresholds["ratio_S"],
            "ratios": {r["ny"]: r["ratio_S"] for r in rows},
            "gate_miss_factor": {r["ny"]: r["ratio_S"] / thresholds["ratio_S"] for r in rows},
            "verdict_independent": "PASS" if max(r["ratio_S"] for r in rows)
            < thresholds["ratio_S"] else "FAIL",
            "verdict_delivered": res["criteria"]["c3_spurious"]["verdict"]},
    }

    # ---- 2. static control / limit-cycle probe (uses the *work-tree* long run) ----
    long_path = os.path.join(
        src, "artifacts/mcmp_stratified_poiseuille/diag/sp_gx0_ny49_g25_mirror_long.json")
    short_path = os.path.join(
        src, "artifacts/mcmp_stratified_poiseuille/diag/sp_gx0_ny49_g25_mirror.json")
    ctl = {}
    for tag, path in (("40000", short_path), ("80000", long_path)):
        if not os.path.exists(path):
            ctl[tag] = None
            continue
        d = load(path)
        sif = [p["S_ifc"] for p in d["series"][len(d["series"]) // 2:]]
        ctl[tag] = {"steps": d["config"]["steps"],
                    "S_ifc_final": d["series"][-1]["S_ifc"],
                    "S_ifc_min_second_half": min(sif),
                    "S_ifc_max_second_half": max(sif),
                    "ratio_final_over_min": (d["series"][-1]["S_ifc"] / min(sif)) if min(sif) else None,
                    "series": [[p["step"], p["S_ifc"], p["S_all"]] for p in d["series"]]}
    out["checks"]["static_control_ny49_g25_mirror"] = ctl

    # ---- 3. wall-convention calibration (independent re-run output) ----
    cal_path = "/tmp/m15044_walls_ny25.json"
    if os.path.exists(cal_path):
        cal = load(cal_path)["wall_calibration"]
        out["checks"]["wall_calibration_rerun"] = {
            k: {"u_max_error_rel": v["u_max_error_rel"], "Q_error_rel": v["Q_error_rel"]}
            for k, v in cal.items()}
    out["checks"]["wall_calibration_delivered"] = load(os.path.join(
        src, "benchmarks/verified/mcmp_stratified_poiseuille/result.json"))["wall_calibration"]

    # ---- figures -------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    nys = [r["ny"] for r in rows]
    ax1.axhspan(-5, 5, color="tab:green", alpha=0.12, label="criterion 1 gate +/-5%")
    ax1.axhspan(-10, 10, color="tab:blue", alpha=0.08, label="criterion 2 gate +/-10%")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.plot(nys, [100 * r["err_interface"] for r in rows], "o-", label="interface position")
    ax1.plot(nys, [100 * r["rules"]["mixture"]["err_Q_mix_independent"] for r in rows],
             "s-", label="Q_mix (mixture rule)")
    ax1.plot(nys, [100 * r["rules"]["component"]["err_Q_mix_independent"] for r in rows],
             "^--", label="Q_mix (component rule)")
    ax1.set_xlabel("ny (channel rows)"); ax1.set_ylabel("relative error [%]")
    ax1.set_title("audit re-derivation: criteria 1-2 vs closed form")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ax2.axhline(thresholds["ratio_S"], color="r", ls="--", label="gate 0.005")
    ax2.semilogy(nys, [r["ratio_S"] for r in rows], "o-", label="driven S_all/u_max,mix")
    sp = [e for e in res["criteria"]["c3_spurious"]["static_floor"] if e["ny"] == 49]
    if sp:
        ax2.semilogy([49], [sp[-1]["ratio_ifc_analytic"]], "D",
                     label="static control (40k, g25)")
    if ctl.get("80000"):
        ax2.semilogy([49], [ctl["80000"]["S_ifc_final"] / 0.115], "D", mfc="none",
                     label="static control (80k, g25, untracked)")
    ax2.set_xlabel("ny"); ax2.set_ylabel("S / u_max  [-]")
    ax2.set_title("criterion 3: parasitic current misses the gate")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3, which="both")
    fig1.tight_layout()
    f1 = os.path.join(OUT, "audit_m14689_criteria.png")
    fig1.savefig(f1, dpi=140)

    fig2, ax = plt.subplots(figsize=(6.4, 4.2))
    for tag, style in (("40000", "C0"), ("80000", "C3")):
        if not ctl.get(tag):
            continue
        s = ctl[tag]["series"]
        ax.semilogy([p[0] for p in s], [p[2] for p in s], style + "-", lw=0.9,
                    label=f"S_ifc, static ny=49 g25 mirror, {tag} steps")
    ax.axhline(thresholds["ratio_S"] * 0.115, color="r", ls="--",
               label="gate level at u_max=0.115")
    ax.set_xlabel("step"); ax.set_ylabel("S_ifc  [lattice]")
    ax.set_title("static control is a limit cycle, not a converged floor")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    f2 = os.path.join(OUT, "audit_m14689_static_control.png")
    fig2.savefig(f2, dpi=140)

    out["figures"] = [f1, f2]
    with open(os.path.join(OUT, "audit_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out["criteria"][k] for k in out["criteria"]}, indent=1))
    print("wrote", os.path.join(OUT, "audit_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())