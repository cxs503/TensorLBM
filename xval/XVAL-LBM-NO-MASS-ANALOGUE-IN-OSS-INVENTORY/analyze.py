"""Analyze the XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY re-runs.

Reads ``runs/*/result.json`` (written by ``run_xval.py``) and emits
``summary.json`` + ``summary.md``; every number in the report is taken from here.

Checks implemented
------------------
1. ``rep``      : does the m21062 ladder (C_I = 2.16 / 2.10 / 1.99 at D = 8/12/16)
                  reproduce with the current library code at the stated setting?
2. ``protocol`` : transient / window convergence -- a first-harmonic decomposition
                  is only meaningful when the fit residual is small.
3. ``beta``     : Stokes-number ladder against the same-mask potential-flow value
                  (the anchor's reference).  If C_I -> c_a^pot as beta -> inf, the
                  in-phase excess is viscous contamination, not an estimator defect.
4. ``e3``       : three mutually independent force estimators on the same run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# m21062 report_payload table (the claim under review)
M21062 = {
    8: {"C_I": 2.1588, "C_Q": 0.3772, "drift": 0.020, "resid": 0.0185},
    12: {"C_I": 2.1002, "C_Q": 0.4595, "drift": -0.005, "resid": 0.0042},
    16: {"C_I": 1.9897, "C_Q": 0.4595, "drift": -0.018, "resid": 0.0043},
}


def load(group: str) -> list[dict]:
    out = []
    for p in sorted(RUNS.glob("*/result.json")):
        r = json.loads(p.read_text())
        if r.get("group") == group:
            out.append(r)
    return out


def rows_rep():
    rows = []
    for r in load("rep"):
        if r["status"] != "ok":
            rows.append({"D": r["case"]["D"], "status": r["status"]})
            continue
        k = r["kpi"]
        D = r["case"]["D"]
        ca = r["reference"].get("potential_same_mask_area")
        rows.append({
            "D": D, "nx": int(r["numerics"]["nx"]), "tau": round(r["numerics"]["tau"], 4),
            "C_I_E1_area": round(k["E1_mem_area"]["C_I"], 4),
            "C_I_E1_halfway": round(k["E1_mem_halfway"]["C_I"], 4),
            "C_Q_E1_area": round(k["E1_mem_area"]["C_Q"], 4),
            "C_I_E2_area": round(k["E2_trac_area"]["C_I"], 4),
            "resid_E1": round(r["harmonic"]["E1_mem"]["resid_rel"], 4),
            "c_a_pot_area": round(ca, 4),
            "err_vs_pot_pct": round(100 * (k["E1_mem_area"]["C_I"] - ca) / ca, 2),
            "m21062_C_I": M21062[D]["C_I"],
            "ratio_current_over_m21062": round(k["E1_mem_area"]["C_I"] / M21062[D]["C_I"], 3),
            "a0_meas_over_nominal": round(r["achieved"]["a0_over_nominal"], 4),
        })
    return rows


def rows_protocol():
    rows = []
    for r in load("protocol"):
        if r["status"] != "ok":
            continue
        k, h = r["kpi"], r["harmonic"]
        ca = r["reference"].get("potential_same_mask_area")
        rows.append({
            "transient_periods": r["case"]["transient"],
            "measure_periods": r["case"]["measure"],
            "C_I_E1": round(k["E1_mem_area"]["C_I"], 4),
            "C_Q_E1": round(k["E1_mem_area"]["C_Q"], 4),
            "C_I_E2": round(k["E2_trac_area"]["C_I"], 4),
            "C_Q_E2": round(k["E2_trac_area"]["C_Q"], 4),
            "resid_E1": round(h["E1_mem"]["resid_rel"], 4),
            "resid_E2": round(h["E2_trac"]["resid_rel"], 4),
            "c_a_pot_area": round(ca, 4),
        })
    rows.sort(key=lambda d: (d["transient_periods"], d["measure_periods"]))
    return rows


def rows_beta():
    rows = []
    for r in load("beta"):
        if r["status"] != "ok":
            rows.append({"beta": r["case"]["beta"], "status": r["status"]})
            continue
        k, h, n = r["kpi"], r["harmonic"], r["numerics"]
        ca = r["reference"].get("potential_same_mask_area")
        rows.append({
            "beta_nominal": r["case"]["beta"],
            "beta_achieved": round(r["achieved"]["beta_halfway"], 1),
            "tau": round(n["tau"], 4),
            "C_I_E1": round(k["E1_mem_area"]["C_I"], 4),
            "C_Q_E1": round(k["E1_mem_area"]["C_Q"], 4),
            "C_I_E2": round(k["E2_trac_area"]["C_I"], 4),
            "C_Q_E2": round(k["E2_trac_area"]["C_Q"], 4),
            "resid_E1": round(h["E1_mem"]["resid_rel"], 4),
            "c_a_pot_area": round(ca, 4),
            "err_E1_vs_pot_pct": round(100 * (k["E1_mem_area"]["C_I"] - ca) / ca, 2),
            "estimator_spread": round(abs(k["E1_mem_area"]["C_I"] - k["E2_trac_area"]["C_I"]), 4),
        })
    rows.sort(key=lambda d: d.get("beta_nominal", 0))
    return rows


def rows_e3():
    rows = []
    for r in load("e3"):
        if r["status"] != "ok":
            continue
        k = r["kpi"]
        ca = (r["reference"].get("potential_same_mask_area")
              or r["reference"].get("c_a_area"))
        rows.append({
            "beta_nominal": r["case"]["beta"],
            "C_I_E1": round(k["E1_mem_area"]["C_I"], 4),
            "C_I_E2": round(k["E2_trac_area"]["C_I"], 4),
            "C_I_E3": round(k["E3_momentum_area"]["C_I"], 4),
            "C_Q_E1": round(k["E1_mem_area"]["C_Q"], 4),
            "C_Q_E2": round(k["E2_trac_area"]["C_Q"], 4),
            "C_Q_E3": round(k["E3_momentum_area"]["C_Q"], 4),
            "resid_P": round(k["E3_momentum_area"]["resid_rel"], 4),
            "c_a_pot_area": round(ca, 4),
            "E1_vs_E3_delta": round(k["E1_mem_area"]["C_I"] - k["E3_momentum_area"]["C_I"], 4),
        })
    return rows


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no runs)_\n"
    keys = list(rows[0])
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    rep, proto, beta, e3 = rows_rep(), rows_protocol(), rows_beta(), rows_e3()
    summary = {"rep": rep, "protocol": proto, "beta": beta, "e3": e3}

    # headline reproducibility check
    repro = {}
    for r in rep:
        if "ratio_current_over_m21062" in r:
            repro[str(r["D"])] = {"m21062": r["m21062_C_I"], "current": r["C_I_E1_area"],
                                  "ratio": r["ratio_current_over_m21062"]}
    summary["reproducibility_of_m21062"] = repro

    # does the beta ladder approach the potential anchor?
    ok = [r for r in beta if "err_E1_vs_pot_pct" in r]
    if ok:
        summary["beta_trend"] = {
            "err_E1_vs_pot_first_pct": ok[0]["err_E1_vs_pot_pct"],
            "err_E1_vs_pot_last_pct": ok[-1]["err_E1_vs_pot_pct"],
            "beta_first": ok[0]["beta_nominal"], "beta_last": ok[-1]["beta_nominal"],
            "monotone_decreasing": all(ok[i + 1]["err_E1_vs_pot_pct"] <= ok[i]["err_E1_vs_pot_pct"]
                                       for i in range(len(ok) - 1)),
        }
    diag = {}
    for f in sorted(RUNS.glob("*/result.json")):
        r = json.loads(f.read_text())
        if r.get("diagnostics"):
            diag[r["arm"]] = {k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in r["diagnostics"].items()}
    summary["diagnostics"] = diag
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = ["# XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY -- re-run summary", "",
          "## A. m21062 ladder reproduction (D-ladder, beta=50, box=4D, KC=0.2)", "",
          md_table(rep), "## B. protocol / window convergence (D=16, beta=50)", "",
          md_table(proto), "## C. Stokes-number ladder (D=16, box=4D, t=8/5 periods)", "",
          md_table(beta), "## D. independent estimator E3 (global momentum balance)", "",
          md_table(e3), "## E. diagnostics (compressibility / mass budget / pressure dipole)", "",
          md_table([{"arm": k, **v} for k, v in diag.items()])]
    (HERE / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())