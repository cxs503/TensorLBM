"""Analyze the R2 gated added-mass ladder: every number in the report comes from here.

Reads ``runs/*/result.json`` (written by ``oc_added_mass_r2.py``) and emits

    summary_r2.json   machine-readable digest
    summary_r2.md     markdown tables (the report embeds these verbatim)

Nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

ORDER = [
    # local (nx = 64) ladder -- the evidence base of this round
    "r2l_b25_d16", "r2_b50_d16", "r2l_b100_d16",
    "r2l_b50_d16_kc_half", "r2l_b50_d16_int4",
    # nx = 128 ladder submitted to the cluster (cj4267); listed when present
    "r2_b25_d32", "r2_b50_d32", "r2_b100_d32", "r2_b50_d32_spin40",
    "r2_b50_box3_d32", "r2_b50_d32_kc_half", "r2_b50_d32_int4",
]


def load() -> list[dict]:
    arms = []
    for name in ORDER:
        p = RUNS / name / "result.json"
        if p.exists():
            d = json.loads(p.read_text())
            d["_name"] = name
            arms.append(d)
    return arms


def fmt(v, nd=3, pct=False):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        if pct:
            return f"{v:+.{nd}f} %"
        return f"{v:+.{nd}f}" if abs(v) < 100 else f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    arms = load()
    ok = [a for a in arms if a.get("status") == "ok"]

    ident = ["| arm | D | nx | box/D | β | τ | δ/R | δ [cells] | ω | ωD | detune [Γ] | KC | Ma |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ok:
        n, g, c = a["numerics"], a["geometry"], a["case"]
        ident.append(
            f"| `{a['_name']}` | {n['D']} | {n['nx']} | {c['box']:.1f} | {c['beta']:.0f} | "
            f"{fmt(n['tau'],3)} | {fmt(n['delta_over_R'],3)} | {fmt(a['stokes_layer_cells'],2)} | "
            f"{fmt(n['omega'],5)} | {fmt(n['omega']*n['D'],3)} | "
            f"{fmt(min(n['detune_in_linewidths']),0)} | {fmt(n['KC'],3)} | {fmt(n['Ma'],4)} |")

    coeff = ["| arm | E1 C_I | E2 C_I | E3 C_I | E1–E3 spread | E1 C_Q | E2 C_Q | E3 C_Q |",
             "|---|---|---|---|---|---|---|---|"]
    for a in ok:
        k = a["kpi"]
        coeff.append(
            f"| `{a['_name']}` | {fmt(k['E1_mem_area']['C_I'],4)} | {fmt(k['E2_trac_area']['C_I'],4)} | "
            f"{fmt(k['E3_global_area']['C_I'],4)} | "
            f"{fmt(abs(k['E1_mem_area']['C_I']-k['E3_global_area']['C_I']),4)} | "
            f"{fmt(k['E1_mem_area']['C_Q'],4)} | {fmt(k['E2_trac_area']['C_Q'],4)} | "
            f"{fmt(k['E3_global_area']['C_Q'],4)} |")

    refs = ["| arm | C_a same-mask (pot.) | Γ_re(β) | C_a·Γ_re | E1 vs pot. | E1 vs C_a·Γ_re | E2 vs C_a·Γ_re | E3 vs C_a·Γ_re | per-πR² vs Γ_re |",
            "|---|---|---|---|---|---|---|---|---|"]
    for a in ok:
        r, e = a["reference"], a["err_inphase_pct"]
        refs.append(
            f"| `{a['_name']}` | {fmt(r['c_a_level1'],4)} | {fmt(r['stokes']['Gamma_re'],4)} | "
            f"{fmt(r['c_ref_viscous'],4)} | {fmt(e['E1_vs_potential'],1,True)} | "
            f"**{fmt(e['E1_vs_stokes'],1,True)}** | {fmt(e['E2_vs_stokes'],1,True)} | "
            f"{fmt(e['E3_vs_stokes'],1,True)} | {fmt(e['E1_per_piR2_vs_Gamma_re'],1,True)} |")

    diag = ["| arm | spin-up [per.] | gate closed | E1 resid | U resid | U/u₀ | KC meas. | max&#124;ρ−1&#124; | ΔM/M | C_Q vs −ImΓ |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ok:
        p, ac, e = a["protocol"], a["achieved"], a["err_inphase_pct"]
        diag.append(
            f"| `{a['_name']}` | {fmt(p['spin_up_periods'],1)} | {fmt(p['gate_closed'])} | "
            f"{fmt(p['E1_resid_rel'],3)} | {fmt(p['U_resid_rel'],3)} | {fmt(ac['U_over_u0'],3)} | "
            f"{fmt(ac['KC_measured'],3)} | {fmt(ac['drho_max_window'],4)} | "
            f"{ac['mass_rel_drift_window']:+.1e} | {fmt(e['C_Q_vs_Gamma_im'],1,True)} |")

    verdict = ["| arm | δ [cells] | E1 vs C_a·Γ_re | owner gate &#124;err&#124;≤15 % | platform caliber ≤3 % |",
               "|---|---|---|---|---|"]
    n_pass = 0
    for a in ok:
        e = a["err_inphase_pct"]["E1_vs_stokes"]
        d = a["stokes_layer_cells"]
        p15 = abs(e) <= 15.0
        p3 = abs(e) <= 3.0
        n_pass += int(p15)
        verdict.append(f"| `{a['_name']}` | {fmt(d,2)} | {fmt(e,1,True)} | "
                       f"{'**PASS**' if p15 else 'FAIL'} | {'**PASS**' if p3 else 'FAIL'} |")

    def get(name, path, default=None):
        for a in ok:
            if a["_name"] == name:
                cur = a
                for key in path.split("."):
                    cur = cur[key]
                return cur
        return default

    digest = {
        "n_arms": len(ok),
        "acceptance": {"owner_15pct_pass": n_pass, "of": len(ok)},
        "primary": {
            "arm": "r2_b50_d16",
            "C_I_E1": get("r2_b50_d16", "kpi.E1_mem_area.C_I"),
            "C_I_E2": get("r2_b50_d16", "kpi.E2_trac_area.C_I"),
            "C_I_E3": get("r2_b50_d16", "kpi.E3_global_area.C_I"),
            "ref_potential": get("r2_b50_d16", "reference.c_a_level1"),
            "Gamma_re": get("r2_b50_d16", "reference.stokes.Gamma_re"),
            "ref_viscous": get("r2_b50_d16", "reference.c_ref_viscous"),
            "err_vs_stokes_pct": get("r2_b50_d16", "err_inphase_pct.E1_vs_stokes"),
            "err_vs_potential_pct": get("r2_b50_d16", "err_inphase_pct.E1_vs_potential"),
        },
        "defect_ab": {
            "interior8": {"drho": get("r2_b50_d16", "achieved.drho_max_window"),
                          "mass_drift": get("r2_b50_d16", "achieved.mass_rel_drift_window"),
                          "C_I_E1": get("r2_b50_d16", "kpi.E1_mem_area.C_I")},
            "interior4": {"drho": get("r2l_b50_d16_int4", "achieved.drho_max_window"),
                          "mass_drift": get("r2l_b50_d16_int4", "achieved.mass_rel_drift_window"),
                          "C_I_E1": get("r2l_b50_d16_int4", "kpi.E1_mem_area.C_I")},
        },
        "resolution_trend": [
            {"arm": a["_name"], "delta_cells": a["stokes_layer_cells"],
             "err_vs_stokes_pct": a["err_inphase_pct"]["E1_vs_stokes"]} for a in ok
        ],
    }

    (HERE / "summary_r2.json").write_text(json.dumps(digest, indent=2, sort_keys=True) + "\n")
    md = ["# R2 tables (generated by analyze_r2.py)", "",
          "## T1 identification", *ident, "",
          "## T2 force coefficients", *coeff, "",
          "## T3 reference comparison", *refs, "",
          "## T4 protocol and conservation diagnostics", *diag, "",
          "## T5 acceptance", *verdict, ""]
    (HERE / "summary_r2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())