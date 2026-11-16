"""Round-3 analysis: re-score the anchor against the corrected unsteady-Stokes reference.

Every number in ``REPORT3.md`` / ``summary_r3.md`` is produced here from the committed
raw ``result.json`` files (round 2) and the round-3 runs.  Nothing is typed by hand and
nothing is recomputed from prose.

What it does
------------
1. recovers the *shipped* reference from each raw file (``reference.stokes`` -- written by
   the old code, so the audit is reproducible from committed evidence alone);
2. rebuilds the reference with the corrected mapping ``delta/R = 2/sqrt(pi beta)``,
   ``k = sqrt(pi beta/2)`` (``tensorlbm.oscillatory_cylinder_inertia``, patched);
3. reports both error tables side by side, the true vs shipped Stokes-layer thickness,
   the resolution trend, and the per-arm acceptance verdicts of the predeclared rules.

Usage
-----
    python xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/r3/analyze_r3.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorlbm.oscillatory_cylinder_inertia import stokes_cylinder_coefficients

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R2_RUNS = ROOT / "r2" / "runs"


def load_arms() -> list[dict]:
    arms = []
    for base in (R2_RUNS, HERE / "runs"):
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            f = d / "result.json"
            if not f.exists():
                continue
            raw = json.loads(f.read_text())
            if raw.get("status") != "ok":
                continue
            if "kpi" not in raw:            # reference-only runs are merged, not arms
                continue
            arms.append(_score(raw, f))
    return arms


def _score(raw: dict, path: Path) -> dict:
    case = raw["case"]
    beta = float(case["beta"])
    D = float(case["D"])
    nx = int(raw["numerics"]["nx"])
    g_ship = raw["reference"]["stokes"]                    # shipped (buggy) reference
    g_corr = stokes_cylinder_coefficients(beta)            # corrected reference
    kpi = raw["kpi"]
    c_a1 = float(raw["reference"]["c_a_level1"])           # same-mask potential, arm grid
    c_a2 = float(raw["reference"]["c_a_area"])             # 2x refined grid (mask sensitivity)
    ref_ship = c_a1 * float(g_ship["Gamma_re"])
    ref_corr = c_a1 * float(g_corr["Gamma_re"])
    delta_true = D / math.sqrt(math.pi * beta)             # = sqrt(2 nu/omega)
    delta_ship = float(raw["numerics"]["delta_over_R"]) * D / 2.0
    rows: dict[str, dict] = {}
    for est in ("E1_mem_area", "E2_trac_area", "E3_global_area"):
        rows[est] = {
            "C_I": float(kpi[est]["C_I"]),
            "C_Q": float(kpi[est]["C_Q"]),
            "err_vs_shipped_pct": 100.0 * (float(kpi[est]["C_I"]) - ref_ship) / ref_ship,
            "err_vs_corrected_pct": 100.0 * (float(kpi[est]["C_I"]) - ref_corr) / ref_corr,
            "CQ_err_vs_shipped_pct": 100.0 * (float(kpi[est]["C_Q"]) - float(g_ship["C_Q_viscous"]))
                                     / float(g_ship["C_Q_viscous"]),
            "CQ_err_vs_corrected_pct": 100.0 * (float(kpi[est]["C_Q"]) - float(g_corr["C_Q_viscous"]))
                                       / float(g_corr["C_Q_viscous"]),
        }
    proto = raw["protocol"]
    ach = raw["achieved"]
    healthy = bool(
        proto.get("gate_closed")
        and abs(float(ach["mass_rel_drift_window"])) <= 1e-6
        # predeclared: O(Ma^2) = 3e-4 for a warm arm; a cold start rings the box
        # acoustically to ~1e-2 (2e-2 envelope). The defect arm of round 2 sits at 0.30.
        and float(ach["drho_max_window"]) <= 2e-2
        and abs(rows["E1_mem_area"]["C_I"] - rows["E3_global_area"]["C_I"]) <= 1e-3
    )
    name = case["name"]
    control = any(tag in name for tag in ("int4", "kc_half", "spin", "proto_"))
    return {
        "arm": name,
        "is_control": control,
        "source": str(path.relative_to(ROOT.parent.parent)),
        "D": D,
        "nx": nx,
        "beta": beta,
        "omega": float(case["omega"]),
        "box": float(case["box"]),
        "u0": float(case["u0"]),
        "interior": case.get("interior", "8"),
        "delta_true_cells": delta_true,
        "delta_shipped_cells": delta_ship,
        "delta_ratio_shipped_over_true": delta_ship / delta_true,
        "c_a_level1": c_a1,
        "c_a_level2": c_a2,
        "reference_mask_sensitivity_pct": 100.0 * abs(c_a2 - c_a1) / c_a1,
        "k_shipped": float(g_ship["k"]),
        "k_corrected": float(g_corr["k"]),
        "gamma_re_shipped": float(g_ship["Gamma_re"]),
        "gamma_re_corrected": float(g_corr["Gamma_re"]),
        "cq_ref_shipped": float(g_ship["C_Q_viscous"]),
        "cq_ref_corrected": float(g_corr["C_Q_viscous"]),
        "ref_shipped": ref_ship,
        "ref_corrected": ref_corr,
        "estimators": rows,
        "healthy": healthy,
        "diagnostics": {
            "gate_closed": bool(proto.get("gate_closed")),
            "spin_up_periods": float(proto.get("spin_up_periods") or 0.0),
            "mass_rel_drift": float(ach["mass_rel_drift_window"]),
            "drho_max": float(ach["drho_max_window"]),
            "U_over_u0": float(ach["U_over_u0"]),
            "E1_E3_spread": abs(rows["E1_mem_area"]["C_I"] - rows["E3_global_area"]["C_I"]),
        },
    }


def fit_power_law(arms: list[dict]) -> dict:
    """Fit ``err = A * delta_true^-p`` on the healthy in-phase E1 errors."""
    sel = [a for a in arms if a["healthy"] and not a["is_control"]]
    if len(sel) < 2:
        return {"status": "insufficient_arms", "n": len(sel)}
    x = np.log([a["delta_true_cells"] for a in sel])
    y = np.log([abs(a["estimators"]["E1_mem_area"]["err_vs_corrected_pct"]) / 100.0 for a in sel])
    p, logA = np.polyfit(x, y, 1)
    pred = np.exp(logA) * np.exp(p * x)
    ss_res = float(np.sum((np.exp(y) - pred) ** 2))
    ss_tot = float(np.sum((np.exp(y) - np.mean(np.exp(y))) ** 2))
    out = {
        "status": "ok",
        "n": len(sel),
        "arms": [a["arm"] for a in sel],
        "p": float(-p),
        "A": float(math.exp(logA)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }
    # delta at which a pure power law would cross 3 % / 15 %
    for level, tag in ((0.03, "delta_for_3pct_cells"), (0.15, "delta_for_15pct_cells")):
        # err = A delta^-p with p > 0 (error falling as the layer is resolved)
        out[tag] = (float((math.exp(logA) / level) ** (1.0 / (-p))) if -p > 0 else None)
    return out


def scaling_diagnostic(arms: list[dict]) -> dict:
    """Is the residual explained by the Stokes layer (delta) or by the mask (1/D)?

    Three independent read-outs of the same residual:
      * flat in beta at fixed D  -> not a delta effect;
      * two-arm fit |err| ~ D^-q -> geometric/mask scaling if q ~ 1;
      * implied effective Stokes parameter: the measured force divided by the same-mask
        potential reference is ``Gamma_re`` at some ``k_eff``; inverting the corrected
        analytic curve gives how much thicker the *numerical* layer behaves.
    """
    from scipy.optimize import brentq

    out: dict = {}
    byD: dict[float, list] = {}
    for a in arms:
        if a["is_control"] or not a["healthy"]:
            continue
        byD.setdefault(a["D"], []).append(a)
    out["delta_spread_at_fixed_D"] = {
        f"D={int(D)}": {
            "delta_true_cells": [round(x["delta_true_cells"], 3) for x in v],
            "err_pct": [round(x["estimators"]["E1_mem_area"]["err_vs_corrected_pct"], 2) for x in v],
            "spread_pp": round(max(x["estimators"]["E1_mem_area"]["err_vs_corrected_pct"] for x in v)
                               - min(x["estimators"]["E1_mem_area"]["err_vs_corrected_pct"] for x in v), 2),
        } for D, v in byD.items()
    }

    med = {D: float(np.median([x["estimators"]["E1_mem_area"]["err_vs_corrected_pct"] for x in v]))
           for D, v in byD.items()}
    out["err_by_D_pct"] = {f"D={int(D)}": round(e, 2) for D, e in sorted(med.items())}
    if len(med) >= 2:
        Ds = np.array(sorted(med))
        es = np.array([abs(med[D]) for D in Ds])
        q, logC = np.polyfit(np.log(Ds), np.log(es), 1)
        out["fit_err_eq_C_over_D_pow_q"] = {"q": float(-q), "C": float(math.exp(logC))}
        out["err_predicted_at_D128_pct"] = round(100.0 * math.exp(logC) * 128.0 ** q, 2)

    implied = {}
    for a in arms:
        if a["is_control"] or not a["healthy"]:
            continue
        ratio = a["estimators"]["E1_mem_area"]["C_I"] / a["c_a_level1"]

        def f(k: float) -> float:
            g = stokes_cylinder_coefficients(2.0 * k * k / math.pi)
            return g["Gamma_re"] - ratio

        k_eff = brentq(f, 0.05, math.sqrt(math.pi * a["beta"] / 2.0))
        implied[a["arm"]] = {
            "C_I_over_Ca": round(ratio, 4),
            "k_nominal": round(a["k_corrected"], 3),
            "k_eff": round(k_eff, 3),
            "delta_eff_cells": round(math.sqrt(2.0) * a["D"] / 2.0 / k_eff, 3),
            "delta_true_cells": round(a["delta_true_cells"], 3),
            "delta_eff_over_true": round((math.sqrt(2.0) * a["D"] / 2.0 / k_eff) / a["delta_true_cells"], 2),
            "blocked_area_excess_halfway_pct": None,
        }
    out["implied_stokes_parameter"] = implied
    return out


def main() -> int:
    arms = load_arms()
    if not arms:
        print("no raw arms found")
        return 1
    table = []
    for a in sorted(arms, key=lambda r: (r["beta"], r["D"])):
        table.append({
            "arm": a["arm"], "D": a["D"], "nx": a["nx"], "beta": a["beta"],
            "delta_true_cells": round(a["delta_true_cells"], 3),
            "delta_shipped_cells": round(a["delta_shipped_cells"], 3),
            "delta_ratio": round(a["delta_ratio_shipped_over_true"], 4),
            "k_shipped": round(a["k_shipped"], 4),
            "k_corrected": round(a["k_corrected"], 4),
            "gamma_re_shipped": round(a["gamma_re_shipped"], 4),
            "gamma_re_corrected": round(a["gamma_re_corrected"], 4),
            "cq_ref_shipped": round(a["cq_ref_shipped"], 4),
            "cq_ref_corrected": round(a["cq_ref_corrected"], 4),
            "c_a_level1": round(a["c_a_level1"], 4),
            "ref_shipped": round(a["ref_shipped"], 4),
            "ref_corrected": round(a["ref_corrected"], 4),
            "CI_E1": round(a["estimators"]["E1_mem_area"]["C_I"], 4),
            "CI_E2": round(a["estimators"]["E2_trac_area"]["C_I"], 4),
            "CI_E3": round(a["estimators"]["E3_global_area"]["C_I"], 4),
            "err_shipped_pct": round(a["estimators"]["E1_mem_area"]["err_vs_shipped_pct"], 2),
            "err_corrected_pct": round(a["estimators"]["E1_mem_area"]["err_vs_corrected_pct"], 2),
            "err_corrected_E2_pct": round(a["estimators"]["E2_trac_area"]["err_vs_corrected_pct"], 2),
            "err_corrected_E3_pct": round(a["estimators"]["E3_global_area"]["err_vs_corrected_pct"], 2),
            "CQ_E1": round(a["estimators"]["E1_mem_area"]["C_Q"], 4),
            "CQ_err_shipped_pct": round(a["estimators"]["E1_mem_area"]["CQ_err_vs_shipped_pct"], 2),
            "CQ_err_corrected_pct": round(a["estimators"]["E1_mem_area"]["CQ_err_vs_corrected_pct"], 2),
            "healthy": a["healthy"],
            "gate_closed": a["diagnostics"]["gate_closed"],
            "dm_rel": f"{a['diagnostics']['mass_rel_drift']:.1e}",
            "drho_max": round(a["diagnostics"]["drho_max"], 4),
            "U_over_u0": round(a["diagnostics"]["U_over_u0"], 3),
            "E1_E3_spread": round(a["diagnostics"]["E1_E3_spread"], 4),
        })

    fit = fit_power_law(arms)
    scaling = scaling_diagnostic(arms)

    # H5: same delta_true, different (D, beta)
    h5 = {}
    for a in arms:
        if a["is_control"]:
            continue
        key = round(a["delta_true_cells"], 2)
        h5.setdefault(key, []).append((a["arm"], a["D"], a["beta"],
                                       round(a["estimators"]["E1_mem_area"]["err_vs_corrected_pct"], 2)))
    h5_groups = {f"{k}": v for k, v in h5.items() if len(v) > 1}

    # predeclared verdicts
    healthy = [a for a in arms if a["healthy"]]
    e1 = {a["arm"]: a["estimators"]["E1_mem_area"] for a in arms}
    r2_healthy = [a for a in healthy if not a["is_control"] and a["source"].find("r2/") >= 0]
    h1_pass = bool(r2_healthy) and all(abs(a["estimators"]["E1_mem_area"]["err_vs_corrected_pct"]) > 15.0
                                       for a in r2_healthy)
    h2_pass = any(r["CQ_err_vs_corrected_pct"] > 0 for a in healthy
                  for r in [a["estimators"]["E1_mem_area"]])
    trend = []
    for a in sorted([a for a in healthy if a["beta"] == 50.0 and not a["is_control"]],
                    key=lambda r: r["delta_true_cells"]):
        trend.append((a["arm"], round(a["delta_true_cells"], 3),
                      round(a["estimators"]["E1_mem_area"]["err_vs_corrected_pct"], 2)))
    h3_pass = None
    if len(trend) >= 2:
        errs = [abs(t[2]) for t in trend]
        h3_pass = all(b < a for a, b in zip(errs, errs[1:]))
    d64 = [a for a in arms if a["D"] == 64.0]
    if d64:
        e = abs(d64[0]["estimators"]["E1_mem_area"]["err_vs_corrected_pct"])
        h4 = "upgrade_to_3pct" if e <= 3.0 else ("partial" if e <= 15.0 else "honest_fail")
    else:
        h4 = "not_run"

    summary = {
        "generated_by": "r3/analyze_r3.py",
        "n_arms": len(arms),
        "table": table,
        "resolution_fit": fit,
        "scaling_diagnostic": scaling,
        "same_delta_groups": h5_groups,
        "trend_beta50": trend,
        "verdicts": {
            "H1_reference_correction_reverses_round2_pass": bool(h1_pass),
            "H2_quadrature_flips_to_high": bool(h2_pass),
            "H3_error_falls_with_delta": h3_pass,
            "H4_3pct_caliber": h4,
        },
        "shipped_vs_corrected_reference": {
            "delta_ratio": sorted({round(a["delta_ratio_shipped_over_true"], 4) for a in arms}),
            "gamma_re_overstatement_pct": {
                str(int(a["beta"])): round(100.0 * (a["gamma_re_shipped"] / a["gamma_re_corrected"] - 1.0), 2)
                for a in arms
            },
            "cq_reference_ratio": {
                str(int(a["beta"])): round(a["cq_ref_shipped"] / a["cq_ref_corrected"], 3)
                for a in arms
            },
        },
    }
    (HERE / "summary_r3.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---- markdown tables ---------------------------------------------------- #
    def md(rows, cols, headers):
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for r in rows:
            out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(out)

    lines = ["# R3 tables (generated by analyze_r3.py)", "",
             "## T1 identification and reference (true vs shipped Stokes-layer thickness)", "",
             md(table, ["arm", "D", "nx", "beta", "delta_true_cells", "delta_shipped_cells",
                        "k_shipped", "k_corrected", "gamma_re_shipped", "gamma_re_corrected",
                        "c_a_level1", "ref_shipped", "ref_corrected"],
                ["arm", "D", "nx", "β", "δ true [cells]", "δ shipped [cells]", "k shipped",
                 "k corrected", "Γ_re shipped", "Γ_re corrected", "C_a same-mask",
                 "ref shipped", "ref corrected"]), "",
             "## T2 measured coefficients and the re-scored error", "",
             md(table, ["arm", "CI_E1", "CI_E2", "CI_E3", "err_shipped_pct", "err_corrected_pct",
                        "err_corrected_E2_pct", "err_corrected_E3_pct"],
                ["arm", "C_I E1", "C_I E2", "C_I E3", "err vs shipped ref [%]",
                 "err vs corrected ref [%] (E1)", "E2 [%]", "E3 [%]"]), "",
             "## T3 quadrature channel (sign of the discrepancy)", "",
             md(table, ["arm", "CQ_E1", "cq_ref_shipped", "cq_ref_corrected",
                        "CQ_err_shipped_pct", "CQ_err_corrected_pct"],
                ["arm", "C_Q E1", "−ImΓ shipped", "−ImΓ corrected",
                 "C_Q err vs shipped [%]", "C_Q err vs corrected [%]"]), "",
             "## T4 protocol / conservation diagnostics", "",
             md(table, ["arm", "healthy", "gate_closed", "dm_rel", "drho_max", "U_over_u0",
                        "E1_E3_spread"],
                ["arm", "healthy", "gate closed", "ΔM/M", "max|ρ−1|", "U/u₀", "|E1−E3|"]), "",
             "## T5 resolution trend at β = 50 (corrected reference)", ""]
    if trend:
        lines.append("| arm | δ true [cells] | err E1 [%] |")
        lines.append("|---|---|---|")
        for t in trend:
            lines.append(f"| `{t[0]}` | {t[1]} | {t[2]:+} |")
        lines.append("")
    if fit.get("status") == "ok":
        lines += [f"Power-law fit `|err| = A δ^-p` on {fit['n']} healthy arms: "
                  f"`A = {fit['A']:.4f}`, `p = {fit['p']:.3f}`, `R² = {fit['r2']:.3f}`; "
                  f"3 % crossing at δ = {fit['delta_for_3pct_cells']:.2f} cells, "
                  f"15 % crossing at δ = {fit['delta_for_15pct_cells']:.2f} cells.", ""]
    lines += ["## T5b scaling diagnostic (δ vs mask)", ""]
    lines.append("| D | δ true [cells] | err E1 [%] |")
    lines.append("|---|---|---|")
    for D, v in sorted(scaling.get("delta_spread_at_fixed_D", {}).items()):
        for dl, e in zip(v["delta_true_cells"], v["err_pct"]):
            lines.append(f"| {D} | {dl} | {e:+} |")
    lines.append("")
    if "fit_err_eq_C_over_D_pow_q" in scaling:
        f2 = scaling["fit_err_eq_C_over_D_pow_q"]
        lines.append(f"`|err| = C D^-q`: q = {f2['q']:.3f}, C = {f2['C']:.4f} "
                     f"(err at D=128 {scaling['err_predicted_at_D128_pct']:+.2f} %)")
    lines.append("")
    if scaling.get("implied_stokes_parameter"):
        lines.append("| arm | C_I/C_a | k nominal | k implied | δ eff [cells] | δ true [cells] | ratio |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, v in scaling["implied_stokes_parameter"].items():
            lines.append(f"| `{k}` | {v['C_I_over_Ca']} | {v['k_nominal']} | {v['k_eff']} | "
                         f"{v['delta_eff_cells']} | {v['delta_true_cells']} | {v['delta_eff_over_true']} |")
        lines.append("")
    lines += ["## T6 same-δ groups (H5: error is a function of δ, not of D)", ""]
    if h5_groups:
        for k, v in h5_groups.items():
            lines.append(f"* δ = {k} cells: " + ", ".join(f"`{n}` (D={d}, β={int(b)}) "
                                                          f"err {e:+.2f} %" for n, d, b, e in v))
    else:
        lines.append("* no two arms share δ in this round")
    lines += ["", "## T7 predeclared verdicts", "",
              f"* H1 (round-2 PASS was a reference artefact): **{h1_pass}**",
              f"* H2 (quadrature flips to positive error): **{h2_pass}**",
              f"* H3 (error falls with δ at fixed β): **{h3_pass}**",
              f"* H4 (3 % caliber): **{h4}**", ""]
    text = "\n".join(lines)
    (HERE / "summary_r3.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())