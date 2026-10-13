"""Assemble W11-A floor-diagnosis summary (diagnostic-only artifact)."""

import json

BASE = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady"
OUT = f"{BASE}/diag/floor_20261008"

rows = []
base = json.load(open(f"{BASE}/formal/re40_D40.json"))
rows.append(
    {
        "run": "formal_re40_D40 (BASELINE)",
        "up_d": base["up_d"],
        "lx_d": base["lx_d"],
        "h_d": base["h_d"],
        "cd": base["cd"],
        "err_pct": base["err_pct"],
        "delta_vs_baseline_pp": 0.0,
    }
)
for tag in ["E-B_up20_lx50_h20", "E-A_up20_lx60_h20", "E-C_up10_lx40_h40", "E-D_up40_lx70_h20"]:
    d = json.load(open(f"{OUT}/{tag}.json"))
    rows.append(
        {
            "run": tag,
            "up_d": d["up_d"],
            "lx_d": d["lx_d"],
            "h_d": d["h_d"],
            "cd": d["cd"],
            "err_pct": d["err_pct"],
            "delta_vs_baseline_pp": d["err_pct"] - base["err_pct"],
        }
    )

by_up = sorted(
    [r for r in rows if r["h_d"] == 20.0 and r["lx_d"] - r["up_d"] == 30.0], key=lambda r: r["up_d"]
)
summary = {
    "diagnostic_only": True,
    "date": "2026-10-08",
    "question": "decompose the +5.4-5.9% systematic Cd floor vs Fornberg "
    "(D40, Re=40, tau=0.6500216597361282, mask n_solid=1257 "
    "R_area=20.002887964817102 IDENTICAL in all runs)",
    "single_variable_ladder": rows,
    "upstream_convergence_h20_down30": by_up,
    "decomposition_pp_at_D40": {
        "baseline_err": base["err_pct"],
        "upstream_10D_to_20D_E-B": "2.9854 - 5.8218 = -2.8365 pp",
        "downstream_30D_to_40D_E-A_minus_E-B": "3.0385 - 2.9854 = +0.0531 pp (nil)",
        "lateral_20D_to_40D_E-C": "5.7375 - 5.8218 = -0.0843 pp (nil)",
        "note": "streamwise effect is ~entirely the upstream Dirichlet free-stream inlet",
    },
    "verdict_impact": "none: formal/ untouched, gates unchanged, honest FAIL stands",
}
with open(f"{OUT}/summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
