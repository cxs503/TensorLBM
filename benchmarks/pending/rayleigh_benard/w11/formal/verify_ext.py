#!/usr/bin/env python
"""W11-C verify EXTENSION (prereg frozen L61-63): identical gates to verify.py,
with ny256 arms 0.95/1.05 substituted by their single preregistered N x1.5
extension runs (ext_f_*). Writes verdict_ext.json; original verdict.json
untouched. Machine JSON in / machine numbers out."""

import json
import math
from pathlib import Path

BASE = Path("/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard")
REF = json.loads((BASE / "ref" / "ref_table.json").read_text())
RES = BASE / "formal" / "results"
GATE_ARMS = (0.85, 0.95, 1.05, 1.15, 1.3)
LEVELS = (64, 128, 256)
EXT = {(256, 0.95), (256, 1.05)}  # arms replaced by ext_f_ runs


def load(ny, frac):
    pref = "ext_f" if (ny, frac) in EXT else "f"
    return json.loads((RES / f"{pref}_ny{ny}_arm{frac}.json").read_text()), pref


rows, level_pass = [], {}
for ny in LEVELS:
    arm_ok = []
    for frac in GATE_ARMS:
        try:
            d, pref = load(ny, frac)
        except FileNotFoundError:
            rows.append({"ny": ny, "frac": frac, "error": "missing"})
            arm_ok.append(False)
            continue
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
                "prefix": pref,
                "extended": (ny, frac) in EXT,
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
                "gammas_windows": d["gammas_windows"],
                "G1_rel3": g1,
                "G3_sign_scatter": g3,
                "G4_health": g4,
                "R8_wall": r8,
                "arm_pass": g1 and g3 and g4,
            }
        )
    level_pass[ny] = all(arm_ok) if arm_ok else False

Es = {}
for ny in LEVELS:
    rels = [r["rel_pct"] / 100 for r in rows if r.get("ny") == ny and r.get("rel_pct") is not None]
    if rels:
        Es[ny] = math.sqrt(sum(x * x for x in rels) / len(rels))
mono = all(Es[a] > Es[b] for a, b in zip(LEVELS, LEVELS[1:])) if len(Es) == 3 else False
g1_overall = sum(level_pass.values()) >= 2

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

ext_note = {
    "protocol": "prereg.md frozen L61-63: spread>2% -> single N x1.5 re-run; "
    "still >2% -> arm honest-FAIL (fit-quality clause, gate values unchanged)",
    "trigger": "base-run spreads: arm0.95=2.0942%, arm1.05=2.5539% (f_ny256 files, untouched)",
    "extension_rule": "N_ext = ceil(1.5 * N_base); N is the ONLY change (steps_for override; "
    "frozen serial runner code path, seed/protocol identical)",
    "arms": {
        "0.95": {"N_base": 2008782, "N_ext": 3013173},
        "1.05": {"N_base": 2057664, "N_ext": 3086496},
    },
}

verdict = {
    "case": "rayleigh_benard",
    "tier": "formal_extension",
    "prereg_md5_prefix": "3670555e1a0c8b0f624a53c45086dc00",
    "prereg_md5_with_errata": "44ee931819b80bdcb1b635a969c68a50",
    "extension_note": ext_note,
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
out = BASE / "formal" / "verdict_ext.json"
out.write_text(json.dumps(verdict, indent=1))
print(json.dumps({k: verdict[k] for k in ("gates", "VERDICT")}, indent=1))
print("VERDICT_EXT_JSON:", out)
