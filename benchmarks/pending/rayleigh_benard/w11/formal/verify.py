#!/usr/bin/env python
"""W11-C verify: consumes ONLY ref_table.json + formal/results/f_*.json.
Gates G1-G4 per prereg §4 + E1/E2/E3 errata; zero hand-copied numbers.
G3 scatter reading: 散布 = 三窗斜率半幅 (max-min)/2 -> clause |gamma_sim| >= 2*half-range
  == spread_rel <= 1 (recorded literally).
G4 e_folds clause per E2: |ln(a_last/a_first)| >= 2.2 on gate arms only
  (arm 1.0 report-only, exempt from e-fold sub-clause; other health reported).
R8: ny256 single-arm wallclock cap 14400 s (fail that arm honestly).
"""

import json
import math
from pathlib import Path

BASE = Path("/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard")
REF = json.loads((BASE / "ref" / "ref_table.json").read_text())
RES = BASE / "formal" / "results"
GATE_ARMS = (0.85, 0.95, 1.05, 1.15, 1.3)
LEVELS = (64, 128, 256)

rows, level_pass = [], {}
for ny in LEVELS:
    lev = next(x for x in REF["levels"] if x["ny"] == ny)
    arm_ok = []
    for frac in GATE_ARMS:
        p = RES / f"f_ny{ny}_arm{frac}.json"
        if not p.exists():
            rows.append({"ny": ny, "frac": frac, "error": "missing"})
            arm_ok.append(False)
            continue
        d = json.loads(p.read_text())
        h = d["health"]
        rel = d["rel_err"]
        spread = d["slope_spread_rel"]
        sign_ok = (d["gamma_sim"] < 0) == (d["gamma_ref"] < 0)
        scatter_ok = spread <= 1.0
        g1 = rel is not None and abs(rel) <= 0.03
        g3 = sign_ok and scatter_ok
        r8 = not (ny == 256 and d["wall_seconds"] > 14400)
        g4 = (
            abs(h["mass_drift_rel"]) <= 1e-8
            and h["u_max"] <= 1e-3
            and max(abs(h["theta_min"]), abs(h["theta_max"])) <= 0.01
            and spread <= 0.02
            and h["e_folds"] >= 2.2
            and r8
            and d["abort"] is None
        )
        arm_ok.append(g1 and g3 and g4)
        rows.append(
            {
                "ny": ny,
                "frac": frac,
                "Ra": d["Ra"],
                "gamma_sim": d["gamma_sim"],
                "gamma_ref": d["gamma_ref"],
                "rel_pct": 100 * rel if rel is not None else None,
                "sign_ok": sign_ok,
                "slope_spread_rel": spread,
                "scatter_ok": scatter_ok,
                "e_folds": h["e_folds"],
                "mass_drift_rel": h["mass_drift_rel"],
                "u_max": h["u_max"],
                "theta_absmax": max(abs(h["theta_min"]), abs(h["theta_max"])),
                "wall_s": d["wall_seconds"],
                "steps": d["steps"],
                "G1_rel3": g1,
                "G3_sign_scatter": g3,
                "G4_health": g4,
                "R8_wall": r8,
                "arm_pass": g1 and g3 and g4,
            }
        )
    level_pass[ny] = all(arm_ok) if arm_ok else False

# E(ny) and monotone gate
Es = {}
for ny in LEVELS:
    rels = [r["rel_pct"] / 100 for r in rows if r.get("ny") == ny and r.get("rel_pct") is not None]
    if rels:
        Es[ny] = math.sqrt(sum(x * x for x in rels) / len(rels))
mono = all(Es[a] > Es[b] for a, b in zip(LEVELS, LEVELS[1:])) if len(Es) == 3 else False
n_pass_levels = sum(level_pass.values())
g1_overall = n_pass_levels >= 2

# report-only arm 1.0 rows
report_rows = []
for ny in LEVELS:
    p = RES / f"f_ny{ny}_arm1.0.json"
    if p.exists():
        d = json.loads(p.read_text())
        report_rows.append(
            {
                "ny": ny,
                "gamma_sim": d["gamma_sim"],
                "gamma_ref": d["gamma_ref"],
                "steps": d["steps"],
                "sign": "neg" if d["gamma_sim"] < 0 else "pos",
            }
        )

verdict = {
    "case": "rayleigh_benard",
    "tier": "formal",
    "prereg_md5_prefix": "3670555e1a0c8b0f624a53c45086dc00",
    "prereg_md5_with_errata": "44ee931819b80bdcb1b635a969c68a50",
    "gates": {
        "G1_rel3_at_least_2_levels": g1_overall,
        "levels_all_gate_arms_pass": level_pass,
        "G2_monotone_E": mono,
        "E_ny": Es,
        "G3_sign": all(
            r.get("arm_pass") is not None and r["G3_sign_scatter"] for r in rows if "frac" in r
        ),
        "G4_health": all(r.get("G4_health") for r in rows if "frac" in r),
    },
    "arm_rows": rows,
    "arm1_0_report_only": report_rows,
}
verdict["VERDICT"] = (
    "PASS"
    if (g1_overall and mono and all(r.get("arm_pass") for r in rows if "frac" in r))
    else "FAIL"
)
out = BASE / "formal" / "verdict.json"
out.write_text(json.dumps(verdict, indent=1))
print(json.dumps({k: verdict[k] for k in ("gates", "VERDICT")}, indent=1))
print("VERDICT_JSON:", out)
