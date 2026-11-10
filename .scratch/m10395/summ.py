import sys, json
sys.path.insert(0, "src")
from pathlib import Path
from tensorlbm.sloshing_forced.campaign import summarize, case_table, BASE_CPH, REFINED_CPH, SWEEP_RATIOS, LINEAR_SWAY_FRAC, IMPACT_SWAY_FRAC
from tensorlbm.sloshing_forced.sweep import CASE_DEFAULTS

out = Path("artifacts/p88w1")
cases = {}
for p in out.glob("*.json"):
    if p.stem == "campaign":
        continue
    cases[p.stem] = json.loads(p.read_text())
zeta = cases["decay_m1"]["zeta"]
config = {
    "base_cells_per_height": BASE_CPH,
    "refined_cells_per_height": REFINED_CPH,
    "sweep_ratios": list(SWEEP_RATIOS),
    "linear_sway_frac": LINEAR_SWAY_FRAC,
    "impact_sway_frac": IMPACT_SWAY_FRAC,
    "case_defaults": dict(CASE_DEFAULTS),
    "zeta_used_for_forced": zeta,
    "cases": [c["id"] for c in case_table()],
}
payload = summarize(out, cases, config)
print("cases:", len(cases))
for k, v in payload["criteria"].items():
    if isinstance(v, dict) and "pass" in v:
        print(" ", k, v.get("value_pct", v.get("value")), "->", v["pass"])