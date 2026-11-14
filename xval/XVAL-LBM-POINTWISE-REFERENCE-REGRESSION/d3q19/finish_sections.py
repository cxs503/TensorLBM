"""Complete the sections that the timed full run did not reach (ABC physics,
sensitivity controls, criteria verdicts) and re-read the persisted summary."""
import json
from pathlib import Path

import math

import run_xval as R

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "summary.json").read_text())

rows: list[list] = []
# shear wave: recomputed here because the timed full run was killed before it
# could persist this section (its stdout is reproduced exactly by this call).
S["physics_shear_wave"] = R.physics_shear_wave(rows)
S["physics_abc"] = R.physics_abc(rows)
S["sensitivity"] = R.sensitivity_controls()

equiv = S["code_equivalence"]
worst = {}
for rec in equiv:
    worst[rec["arm"]] = max(worst.get(rec["arm"], 0.0), rec["eps_inf_t30"])
S["worst_eps_inf_t30_by_arm"] = worst

C = R.PRE["acceptance_criteria"]
gaps = [v for k, v in R._REF_CACHE.items() if isinstance(k, tuple) and k[0] == "mrt_gap"]
v = {
    "C1_code_equivalence": {
        "threshold": C["C1_code_equivalence"]["threshold"],
        "worst": max(worst[a] for a in ("A1_solver3d_bgk", "A2_diff3d_step", "A3_adv_bgk_d3q19")),
        "passed": all(worst[a] <= C["C1_code_equivalence"]["threshold"]
                      for a in ("A1_solver3d_bgk", "A2_diff3d_step", "A3_adv_bgk_d3q19")),
    },
    "C2_mrt_native_vs_symbolic": {
        "threshold": C["C2_mrt_native_vs_symbolic"]["threshold"],
        "worst": worst["A5_mrt_native"],
        "passed": bool(worst["A5_mrt_native"] <= C["C2_mrt_native_vs_symbolic"]["threshold"]),
    },
    "C3_mrt_basis_reconstruction": {
        "threshold": C["C3_mrt_basis_reconstruction"]["threshold"],
        "value": max(gaps) if gaps else None,
        "passed": bool(max(gaps) <= C["C3_mrt_basis_reconstruction"]["threshold"]) if gaps else False,
    },
    "C4_conservation": {
        "threshold": C["C4_conservation"]["threshold"],
        "worst_mass": max(r["mass_drift_max"] for r in equiv),
        "worst_momentum": max(r["momentum_drift_max"] for r in equiv),
    },
    "C6_physics_shear_wave": {
        "threshold": C["C6_physics_shear_wave"]["threshold"],
        "worst_err_vs_continuum": max(r["err_vs_continuum"] for r in S["physics_shear_wave"]),
        "worst_err_vs_lattice_dispersion": max(r["err_vs_lattice_dispersion"] for r in S["physics_shear_wave"]),
    },
    "C7_physics_abc": {
        "threshold": C["C7_physics_abc"]["threshold"],
        "worst_err": max(r["err_vs_analytic"] for r in S["physics_abc"]),
    },
}
v["C4_conservation"]["passed"] = bool(
    v["C4_conservation"]["worst_mass"] <= C["C4_conservation"]["threshold"]
    and v["C4_conservation"]["worst_momentum"] <= C["C4_conservation"]["threshold"])
v["C6_physics_shear_wave"]["passed"] = bool(
    v["C6_physics_shear_wave"]["worst_err_vs_continuum"] <= C["C6_physics_shear_wave"]["threshold"])
v["C7_physics_abc"]["passed"] = bool(
    v["C7_physics_abc"]["worst_err"] <= C["C7_physics_abc"]["threshold"])
v["C8_sensitivity_controls"] = {"passed": all(x["detected"] for x in S["sensitivity"].values()), **S["sensitivity"]}
S["verdicts"] = v
(HERE / "summary.json").write_text(json.dumps(S, indent=1, default=float))

import csv
with (HERE / "table.csv").open("a", newline="") as fh:
    csv.writer(fh).writerows(rows)

print(json.dumps({"verdicts": v, "worst_by_arm": worst,
                  "abc": S["physics_abc"], "shear": S["physics_shear_wave"],
                  "convention": S["convention"], "frozen": S["frozen_bundle"],
                  "findings_F1_tau0.55": [f for f in S["findings"] if f["tau"] == 0.55][:2]},
                 indent=1, default=float))
