#!/usr/bin/env python
"""Aggregate the MHD node acceptance verdicts (campaign p422w3-n08-mhd-lbm).

Reads the artefacts of the two branches and emits a single machine-checkable
verdict file mapping the campaign's six acceptance criteria onto the measured
numbers and their evidence paths:

(a) ``profile_l2``          Hartmann profile vs closed form, L2 <= 1 %
(b) ``mean_velocity``       mean velocity vs closed form <= 1 %
(c) ``negative_poiseuille`` B0 -> 0 collapses onto plane Poiseuille
(d) ``shercliff``           insulating-wall branch vs closed form <= 2 %
(e) ``gci``                 grid-convergence index <= 3 %
(f) ``thrust``              thruster thrust vs ``F = sigma |E| B V`` <= 5 %

Nothing is re-derived by hand: every number is read from the artefact JSONs
written by ``scripts/run_mhd_campaign.py``, ``scripts/run_mhd_shercliff.py`` and
this script's own thruster re-run (cheap, ``run_lattice=False``).

Output: ``artifacts/mhd_acceptance/node_acceptance.json`` (+ a markdown table).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.mhd.thruster import (  # noqa: E402
    MHDThrusterConfig,
    run_thruster_ledger,
    thruster_ideal_check,
)

#: Campaign thresholds as written in the node specification (fixed before runs).
NODE_THRESHOLDS = {
    "a_profile_l2": 1.0e-2,
    "b_mean_velocity": 1.0e-2,
    "c_negative_poiseuille": 1.0e-2,
    "d_shercliff": 2.0e-2,
    "e_gci": 3.0e-2,
    "f_thrust": 5.0e-2,
}


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artefact: {path}")
    return json.loads(path.read_text())


def collect(hartmann_dir: Path, shercliff_dir: Path) -> dict:
    """Assemble every number the six node criteria need."""
    ladder = _load(hartmann_dir / "hartmann_ladder.json")
    criteria = _load(hartmann_dir / "criteria.json")["criteria"]
    thruster = _load(hartmann_dir / "thruster_ledger.json")
    sl_criteria = _load(shercliff_dir / "shercliff_criteria.json")
    sl_ladder = _load(shercliff_dir / "shercliff_ladder.json")

    # (a) L2 error of the profile: the campaign reports the mean-normalised
    # inf-norm as c1; the L2 value lives in the per-run table.
    runs = ladder["runs"]
    finest: dict[str, dict] = {}
    for run in runs:
        key = f"{float(run['ha']):g}"
        if key not in finest or run["ny"] > finest[key]["ny"]:
            finest[key] = run
    l2_worst = max((r["rel_l2"], float(r["ha"]), r["ny"]) for r in finest.values())

    # (e) GCI: gate on the dimensionless wall-shear functional (a nonzero
    # functional); the profile-error functional is informational only because a
    # GCI applied to a vanishing functional is inflated to O(1).
    gci_rows = {k: v for k, v in ladder["gci"].items() if k.startswith("wall_shear_coefficient")}
    gci_worst = max(
        ((v["gci_fine"] if v.get("gci_fine") is not None else float("inf")), k)
        for k, v in gci_rows.items()
    )
    sl_gci = [
        (row.get("gci"), key)
        for key, row in sl_criteria["per_ha"].items()
        if row.get("gci") is not None and float(key) > 0.0
    ]
    sl_gci_worst = max(sl_gci, default=(0.0, "n/a"))

    # (f) thrust vs the ideal sigma|E|B V
    cfg = MHDThrusterConfig()
    ledger = run_thruster_ledger(cfg, run_lattice=False)
    thrust = thruster_ideal_check(ledger, threshold=NODE_THRESHOLDS["f_thrust"])
    thrust["ledger_design_point"] = {
        "ha": ledger.ha,
        "loading_K": ledger.loading,
        "u_mean_ms": ledger.u_mean_m_s,
        "u_jet_ms": ledger.u_jet_m_s,
        "delta_p_pa": ledger.delta_p_pa,
        "flow_rate_m3_s": ledger.flow_rate_m3_s,
        "thrust_density_pa": ledger.thrust_density_pa,
        "eta_total": ledger.efficiency,
        "eta_jet_ideal": ledger.efficiency_jet_ideal,
        "eta_propeller_ideal_same_size": ledger.efficiency_propeller_ideal,
        "P_electric_W": ledger.power_electric_w,
        "P_joule_W": ledger.power_joule_w,
        "P_viscous_W": ledger.power_viscous_w,
        "P_hydraulic_W": ledger.power_hydraulic_w,
        "joule_share_of_electric": ledger.power_joule_w / ledger.power_electric_w,
        "lattice_check": thruster["ledger"].get("lattice", {}),
    }

    return {
        "hartmann": {
            "criteria": criteria,
            "l2_worst": {"value": l2_worst[0], "ha": l2_worst[1], "ny": l2_worst[2]},
            "gci_worst": {"value": gci_worst[0], "functional": gci_worst[1]},
            "gci_wall_shear_all": {k: v.get("gci_fine") for k, v in gci_rows.items()},
            "gci_wall_shear_rows": gci_rows,
        },
        "shercliff": {
            "criteria": sl_criteria,
            "gci_worst": {"value": sl_gci_worst[0], "ha": sl_gci_worst[1]},
            "plan": sl_ladder["plan"],
        },
        "thruster": thrust,
        "thruster_sweep": thruster["sweep"],
    }


def evaluate(data: dict) -> dict:
    """Map the six node criteria onto verdicts with the measured values."""
    h = data["hartmann"]
    s = data["shercliff"]
    t = data["thruster"]
    th = NODE_THRESHOLDS

    rows = {
        "a_profile_l2": {
            "description": "Hartmann profile vs closed form, L2 error (worst Ha)",
            "value": h["l2_worst"]["value"],
            "worst_ha": h["l2_worst"]["ha"],
            "ny": h["l2_worst"]["ny"],
            "threshold": th["a_profile_l2"],
            "passed": h["l2_worst"]["value"] <= th["a_profile_l2"],
            "artefact": "artifacts/mhd_hartmann/hartmann_ladder.json#runs[].rel_l2",
        },
        "b_mean_velocity": {
            "description": "mean velocity vs closed form (worst Ha)",
            "value": h["criteria"]["c2_flowrate"]["value"],
            "worst_ha": h["criteria"]["c2_flowrate"].get("worst_ha"),
            "threshold": th["b_mean_velocity"],
            "passed": h["criteria"]["c2_flowrate"]["passed"],
            "artefact": "artifacts/mhd_hartmann/criteria.json#c2_flowrate",
        },
        "c_negative_poiseuille": {
            "description": "B0 -> 0 negative control: closed form and LBM collapse onto Poiseuille",
            "value": max(
                h["criteria"]["c6_negative"]["value"],
                h["criteria"]["c6b_negative_lbm"]["value"],
            ),
            "closed_form_deviation": h["criteria"]["c6_negative"]["value"],
            "lbm_centre_ratio_error": h["criteria"]["c6b_negative_lbm"]["value"],
            "lbm_centre_ratio": h["criteria"]["c6b_negative_lbm"].get("ratio"),
            "threshold": th["c_negative_poiseuille"],
            "passed": bool(
                h["criteria"]["c6_negative"]["passed"] and h["criteria"]["c6b_negative_lbm"]["passed"]
            ),
            "artefact": "artifacts/mhd_hartmann/criteria.json#c6_negative",
        },
        "d_shercliff": {
            "description": "insulating-wall (Shercliff) branch vs closed form, worst of L2/Linf/mean",
            "value": max(
                s["criteria"]["worst_case"]["c_shercliff_profile_l2"],
                s["criteria"]["worst_case"]["c_shercliff_profile_linf"],
                s["criteria"]["worst_case"]["c_shercliff_mean"],
            ),
            "profile_l2": s["criteria"]["worst_case"]["c_shercliff_profile_l2"],
            "profile_linf": s["criteria"]["worst_case"]["c_shercliff_profile_linf"],
            "mean": s["criteria"]["worst_case"]["c_shercliff_mean"],
            "wall_shear": s["criteria"]["worst_case"]["c_shercliff_wall_shear"],
            "zero_net_current": {
                "discrete": s["criteria"]["worst_case"]["c_shercliff_zero_current"],
                "integral": s["criteria"]["worst_case"]["c_shercliff_zero_current_integral"],
            },
            "threshold": th["d_shercliff"],
            "passed": bool(
                s["criteria"]["verdicts"]["c_shercliff_profile_l2"]
                and s["criteria"]["verdicts"]["c_shercliff_profile_linf"]
                and s["criteria"]["verdicts"]["c_shercliff_mean"]
            ),
            "artefact": "artifacts/mhd_shercliff/shercliff_criteria.json",
        },
        "e_gci": {
            "description": "grid-convergence index (worst of Hartmann wall-shear and Shercliff wall-shear/u_mean)",
            "value": max(h["gci_worst"]["value"], s["gci_worst"]["value"]),
            "hartmann": h["gci_worst"],
            "shercliff": s["gci_worst"],
            "threshold": th["e_gci"],
            "passed": max(h["gci_worst"]["value"], s["gci_worst"]["value"]) <= th["e_gci"],
            "artefact": "artifacts/mhd_hartmann/hartmann_ladder.json#gci",
            "note": (
                "GCI is applied to nonzero functionals (dimensionless wall shear, u_mean). "
                "The profile-error functional is reported as informational only: its GCI is "
                "O(1) by construction because the functional itself tends to zero."
            ),
        },
        "f_thrust": {
            "description": "thruster thrust vs ideal F = sigma*E*B*V",
            "value": t["relative_error"],
            "F_ideal_N": t["F_ideal_N"],
            "thrust_lorentz_N": t["thrust_lorentz_N"],
            "thrust_momentum_N": t["thrust_momentum_N"],
            "threshold": th["f_thrust"],
            "passed": bool(t["passed"]),
            "artefact": "artifacts/mhd_acceptance/node_acceptance.json#thruster",
        },
    }
    return {
        "all_passed": all(row["passed"] for row in rows.values()),
        "thresholds": th,
        "criteria": rows,
    }


def markdown_table(verdict: dict) -> str:
    rows = [
        "| id | criterion | measured | threshold | verdict | artefact |",
        "|---|---|---|---|---|---|",
    ]
    for key, row in verdict["criteria"].items():
        rows.append(
            f"| {key} | {row['description']} | {row['value']:.3e} | "
            f"{row['threshold']:.0e} | {'PASS' if row['passed'] else 'FAIL'} | "
            f"`{row['artefact']}` |"
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hartmann-dir", default="artifacts/mhd_hartmann")
    parser.add_argument("--shercliff-dir", default="artifacts/mhd_shercliff")
    parser.add_argument("--outdir", default="artifacts/mhd_acceptance")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = collect(Path(args.hartmann_dir), Path(args.shercliff_dir))
    verdict = evaluate(data)
    payload = {
        "campaign": "p422w3-n08-mhd-lbm",
        "node": "TensorLBM MHD physics surface: Hartmann/Shercliff analytic ladder + Lorentz force + thruster ledger",
        "verdict": verdict,
        "thruster": data["thruster"],
        "thruster_sweep": data["thruster_sweep"],
        "shercliff_criteria": data["shercliff"]["criteria"],
        "hartmann_criteria": data["hartmann"]["criteria"],
        "hartmann_gci_wall_shear": data["hartmann"]["gci_wall_shear_all"],
        "hartmann_gci_wall_shear_rows": data["hartmann"]["gci_wall_shear_rows"],
        "shercliff_gci": data["shercliff"]["gci_worst"],
    }
    (outdir / "node_acceptance.json").write_text(json.dumps(payload, indent=2, default=float))
    (outdir / "node_acceptance.md").write_text(
        "# MHD node acceptance (p422w3-n08-mhd-lbm)\n\n"
        + markdown_table(verdict)
        + f"\n\n**all_passed = {verdict['all_passed']}**\n"
    )
    print(markdown_table(verdict))
    print("all_passed:", verdict["all_passed"])
    return 0 if verdict["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())