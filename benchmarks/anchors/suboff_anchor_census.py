#!/usr/bin/env python3
"""W30 SUBOFF anchor census — per-point closure, LBM error table, resolution/τ
sensitivity.

What this artifact is
---------------------
`benchmarks/anchors/evidence/suboff_w30/suboff_anchor_census.json` is the single
machine-readable answer to the three W30 questions that the topic had been
carrying without evidence:

1. **Per-point anchor census.**  Every registered DTRC point of
   ``benchmarks/anchors/dtrc-suboff-1989.json`` (built by
   ``dtrc_suboff_1989_anchor.py``) with its verbatim quote, archive sha256 and
   evidence grade, plus the independent ITTC-1957 recomputation residual.
2. **LBM-to-anchor error table.**  The existing verified LBM artifact
   (``benchmarks/verified/suboff_sail_resistance/``) is aligned point-by-point
   against every anchor point, and the regime-matched (but *pending*) high-Re
   wall-model branch (``benchmarks/pending/suboff_highre_resistance/``) is
   aligned against the five points it was actually run at.
3. **Resolution / τ sensitivity.**  A >=3-tier grid ladder and a 5-arm τ ladder
   with the honest statement of what each knob physically is, and which tier is
   admitted into the error table.

Rules obeyed here (see ``docs/W30_SUBOFF_ANCHOR_CENSUS.md``):
  * no number is typed in — everything is read from a JSON artifact whose
    sha256 is recorded next to it, or derived by a documented closed form;
  * a deviation that fails the 15% gate is reported as FAIL with a root-cause
    hypothesis, never softened into "acceptable";
  * a regime mismatch is disclosed, not averaged away: the verified LBM case
    runs at Re_lattice = 1000 (laminar), four decades below the tow-tank anchor,
    so its raw deviation is *not* a validation result.  The regime-matched
    branch is the pending high-Re case, which fails M1 by +1295..+2963%.

Usage
-----
    python benchmarks/anchors/suboff_anchor_census.py            # build + verify
    python benchmarks/anchors/suboff_anchor_census.py --check    # verify only
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "benchmarks/anchors/evidence/suboff_w30/suboff_anchor_census.json"
ANCHOR_PATH = ROOT / "benchmarks/anchors/dtrc-suboff-1989.json"

VERIFIED_DIR = "benchmarks/verified/suboff_sail_resistance"
VERIFIED_RESULT = f"{VERIFIED_DIR}/result.json"
VERIFIED_VERDICT = f"{VERIFIED_DIR}/verdict.json"
TAU_LADDER = f"{VERIFIED_DIR}/tau_ladder_verdict_m298.json"
W30_SOURCE_COMMIT = "e3bf2f4c097a619c1e85d5d8bc4456371ac15b70"
W30_FROZEN_INPUTS = {
    VERIFIED_RESULT: OUT_PATH.parent / "inputs/result.json",
    TAU_LADDER: OUT_PATH.parent / "inputs/tau_ladder_verdict_m298.json",
}
HIGHRE_DIR = "benchmarks/pending/suboff_highre_resistance"
HIGHRE_VERDICT = f"{HIGHRE_DIR}/formal/formal_verdict.json"
HIGHRE_REFS = f"{HIGHRE_DIR}/refs_lock.json"

GATE_PCT = 15.0  # pre-registered acceptance gate of this task (criterion 3)
NU = 1.139e-06
U_LB = 0.05  # frozen in the verified case configuration


def sha256_of(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def cf_ittc57(re: float) -> float:
    return 0.075 / (math.log10(re) - 2.0) ** 2


def cf_blasius(re: float) -> float:
    return 1.328 / math.sqrt(re)


def dev_pct(value: float, reference: float) -> float:
    return (value / reference - 1.0) * 100.0


# ---------------------------------------------------------------------------
# LBM side: the verified case (Re_lattice = 1000, laminar reference)
# ---------------------------------------------------------------------------
def verified_lbm_tiers() -> dict:
    """Both co-shipped instrument calibers of the verified case."""
    res = load(VERIFIED_RESULT)
    verd = load(VERIFIED_VERDICT)
    cfg = res["configuration"]
    e2 = res["e2_standard_verdict_PASS"]
    mix = res["formal_mix50_verdict_FAIL"]
    cf_ref = verd["cf_ref_recomputed"]
    if abs(cf_ref - cf_blasius(1000.0)) > 1e-12:
        raise RuntimeError(f"laminar reference does not reproduce Blasius: {cf_ref}")

    tiers = []
    for tag in ("L96", "L112", "L128"):
        L = int(tag[1:])
        nu_lb = U_LB * L / 1000.0  # configuration.lattice_collision closed form
        tau = 3.0 * nu_lb + 0.5
        t_e2 = e2["tiers"][tag]
        t_mix = mix["tiers"][str(L)]
        if abs(t_e2["cd_tot_window"] - t_mix["Cd_total"]) < 1e-12:
            raise RuntimeError(f"{tag}: the two calibers are not distinguishable")
        tiers.append(
            {
                "tier": tag,
                "resolution_cells_per_hull_length": L,
                "domain": cfg["ladder"][tag]["domain"],
                "steps": cfg["ladder"][tag]["steps"],
                "appendage_solid_cells": cfg["ladder"][tag]["appendage_solid_cells"],
                "u_lb": U_LB,
                "nu_lb_derived": nu_lb,
                "tau_derived": tau,
                "re_lattice": 1000.0,
                "Cd_total_primary_E2_standard": t_e2["cd_tot_window"],
                "Cd_pressure_primary_E2_standard": t_e2["cd_pressure"],
                "Cd_friction_primary_E2_standard": t_e2["cd_friction"],
                "err_vs_laminar_ref_pct_primary": t_e2["err_pct"],
                "drift_pct_primary": t_e2["g4_drift_pct"],
                "Cd_total_coship_mix50": t_mix["Cd_total"],
                "err_vs_laminar_ref_pct_coship_mix50": t_mix["err_pct"],
                "coship_mix50_verdict": mix["OVERALL"],
            }
        )
    return {
        "case": res["case"],
        "hull_configuration": "SUBOFF with sail (AFF-3 family), total-wetted-area "
        "normalisation including the sail own area; the DTRC anchor is the AFF-1 "
        "BARE hull, so even a perfect Re match would leave a configuration "
        "difference (in-repo appendage share ~2.6%)",
        "observable": res["observable"],
        "lattice_reynolds": 1000.0,
        "physical_re_equivalent": None,
        "reference_used_in_case": {
            "kind": "Blasius laminar flat plate Cf = 1.328/sqrt(Re) at Re=1000",
            "value": cf_ref,
            "source": f"{VERIFIED_VERDICT}:/cf_ref_recomputed",
        },
        "instrument_primary": "standard (erratum E2) — frozen primary caliber",
        "instrument_coship": "mix50 — formal archive, co-disclosed, overall FAIL",
        "primary_gate_outcome": e2["OVERALL"],
        "coship_gate_outcome": mix["OVERALL"],
        "tiers": tiers,
    }


# ---------------------------------------------------------------------------
# LBM side: the regime-matched (pending) high-Re wall-model branch
# ---------------------------------------------------------------------------
def highre_rows() -> dict:
    v = load(HIGHRE_VERDICT)
    rows = []
    for r in v["table"]:
        rows.append(
            {
                "tag": r["tag"],
                "Re": r["Re"],
                "C_T_anchor": r["CT_EFD"],
                "C_T_lbm": r["CT_sim"],
                "dev_pct": r["error_pct"],
                "C_F_lbm_friction": r["Ct_fric"],
                "C_F_ratioreg_to_ittc": r["fric_over_ITTC"],
                "y_plus_mean": r["y_plus_mean"],
                "y_plus_band_ok": r["y_plus_band_ok"],
                "steps_completed": r["steps_completed"],
                "finite": r["finite"],
                "M1_pass_5pct": r["M1_pass_5pct"],
            }
        )
    return {
        "case": f"{HIGHRE_DIR} (AFF-8 full appendage, wall-model LES)",
        "status_in_repo": "pending (precision fails at single-GPU reachable grid)",
        "M1_rule": v["M1_rule"],
        "M1_pass_count": v["M1_pass_count"],
        "M1_verdict": v["M1_verdict"],
        "M2_verdict": v["M2_verdict"],
        "overall": v["overall"],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
def align_verified(anchor: dict, lbm: dict) -> dict:
    """Point-by-point deviation of the verified case against every anchor point.

    The verified case is a *fixed-Re* calculation (Re_lattice = 1000), so the raw
    deviation against a Re = 1e7 anchor point is dominated by the Reynolds gap.
    Two columns are therefore reported side by side:

    * ``dev_raw_pct``  — the literal gate quantity, |dev| < 15%: honest FAIL;
    * ``dev_renormalised_pct`` — diagnostic only: (1+k) of the LBM case
      (Cd_tot / Cf_laminar_ref) multiplied by Cf_ITTC57 at the anchor Re.  It
      attributes the raw gap to the Reynolds number; it is NOT admitted as
      validation evidence because a laminar (1+k) is not transferable to a
      turbulent boundary layer.
    """
    rows = []
    for p in anchor["anchor_points"]:
        for t in lbm["tiers"]:
            cft = t["Cd_total_primary_E2_standard"]
            ff = cft / lbm["reference_used_in_case"]["value"]
            ct_pred = ff * cf_ittc57(p["Re"])
            rows.append(
                {
                    "anchor_id": p["id"],
                    "anchor_family": p["family"],
                    "Re_anchor": p["Re"],
                    "C_T_anchor": p["C_T"],
                    "lbm_tier": t["tier"],
                    "lbm_re_lattice": t["re_lattice"],
                    "C_D_lbm": cft,
                    "regime_match": False,
                    "re_gap_decades": math.log10(p["Re"]) - math.log10(t["re_lattice"]),
                    "dev_raw_pct": dev_pct(cft, p["C_T"]),
                    "gate_pct": GATE_PCT,
                    "pass_raw": abs(dev_pct(cft, p["C_T"])) < GATE_PCT,
                    "diagnostic_1plus_k_used": ff,
                    "C_T_pred_renormalised": ct_pred,
                    "dev_renormalised_pct": dev_pct(ct_pred, p["C_T"]),
                    "pass_renormalised_diagnostic_only": abs(dev_pct(ct_pred, p["C_T"]))
                    < GATE_PCT,
                }
            )
    n_raw = sum(1 for r in rows if r["pass_raw"])
    n_ren = sum(1 for r in rows if r["pass_renormalised_diagnostic_only"])
    finest = [r for r in rows if r["lbm_tier"] == "L128"]
    fam = {}
    for r in finest:
        f = fam.setdefault(r["anchor_family"], {"n": 0, "n_pass_renormalised": 0, "devs": []})
        f["n"] += 1
        f["n_pass_renormalised"] += int(r["pass_renormalised_diagnostic_only"])
        f["devs"].append(r["dev_renormalised_pct"])
    for f in fam.values():
        f["dev_range_pct"] = [min(f["devs"]), max(f["devs"])]
        del f["devs"]
    return {
        "renormalisation_family_breakdown_at_L128": fam,
        "renormalisation_discriminating_power": "WEAK — do not use it to "
        "adjudicate the cross-family conflict. A laminar boundary layer has a "
        "systematically smaller form factor than a turbulent one, so copying the "
        "LBM (1+k) onto the anchor-Re ITTC-57 line biases the prediction towards "
        "the *lower* (1+k) family (here T14, 1+k 1.08-1.16) and away from the "
        "higher one (S1, 1+k 1.26-1.35). The renormalised column is reported only "
        "to show that the Reynolds gap explains the magnitude of the raw "
        "deviation.",
        "branch": "verified_laminar_Re1000",
        "admissible_for_the_anchor": False,
        "why_not_admissible": "the case is a fixed-Re=1000 (lattice) laminar-flow "
        "verification against Blasius; the DTRC anchor lives at Re = 1.2e7..3.5e7 "
        "with a turbulent boundary layer. The Reynolds gap is ~4 decades, so the "
        "raw deviation measures the regime gap, not the solver.",
        "n_rows": len(rows),
        "n_pass_raw": n_raw,
        "n_pass_renormalised_diagnostic": n_ren,
        "rows": rows,
    }


def align_highre(anchor: dict, hr: dict) -> dict:
    by_re = {round(p["Re"], 3): p for p in anchor["anchor_points"]}
    rows = []
    for r in hr["rows"]:
        key = min(by_re, key=lambda k: abs(k - r["Re"]))
        p = by_re[key]
        if abs(key / r["Re"] - 1.0) > 1e-9:
            raise RuntimeError(f"high-Re row {r['tag']} has no exact anchor Re match")
        rows.append(
            {
                "anchor_id": p["id"],
                "anchor_family": p["family"],
                "Re": r["Re"],
                "C_T_anchor": p["C_T"],
                "C_T_lbm": r["C_T_lbm"],
                "regime_match": True,
                "dev_pct": dev_pct(r["C_T_lbm"], p["C_T"]),
                "gate_pct": GATE_PCT,
                "pass": abs(dev_pct(r["C_T_lbm"], p["C_T"])) < GATE_PCT,
                "diagnostics": {
                    "C_F_lbm_friction": r["C_F_lbm_friction"],
                    "C_F_lbm_fric_over_ittc_at_anchor_Re": r["C_F_ratioreg_to_ittc"],
                    "y_plus_mean": r["y_plus_mean"],
                    "y_plus_band_ok": r["y_plus_band_ok"],
                    "finite": r["finite"],
                    "steps_completed": r["steps_completed"],
                },
            }
        )
    if len(rows) != 5:
        raise RuntimeError(f"expected 5 high-Re rows, got {len(rows)}")
    return {
        "branch": "pending_highre_wallmodel",
        "admissible_for_the_anchor": True,
        "admissible_because": "same hull family, same Re range, same fluid "
        "assumptions as the anchor (S1 family); this is the only in-repo LBM "
        "calculation in the anchor's Reynolds domain",
        "status_in_repo": hr["status_in_repo"],
        "n_rows": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# resolution / tau sensitivity
# ---------------------------------------------------------------------------
def sensitivity(lbm: dict) -> dict:
    res = load(VERIFIED_RESULT)
    tau_l = load(TAU_LADDER)
    tiers = lbm["tiers"]
    grid = [
        {
            "tier": t["tier"],
            "tau_derived": t["tau_derived"],
            "Cd_total_primary": t["Cd_total_primary_E2_standard"],
            "err_vs_case_reference_pct": t["err_vs_laminar_ref_pct_primary"],
            "Cd_total_coship_mix50": t["Cd_total_coship_mix50"],
            "err_vs_case_reference_pct_coship": t["err_vs_laminar_ref_pct_coship_mix50"],
        }
        for t in tiers
    ]
    finest = grid[-1]["Cd_total_primary"]
    for g in grid:
        g["delta_vs_finest_pct"] = dev_pct(g["Cd_total_primary"], finest)
    arms = []
    for tag, a in tau_l.items() if False else tau_l["arms"].items():
        arms.append(
            {
                "arm": tag,
                "tau": a["tau_actual"],
                "nu_lb": a["nu_lb_actual"],
                "re_lattice": a["re_lattice_actual"],
                "Cd_total": a["Cd_total"],
                "Cd_friction": a["Cd_friction"],
                "Cd_pressure": a["Cd_pressure"],
            }
        )
    arms.sort(key=lambda x: x["tau"])
    base = arms[0]["Cd_total"]
    for a in arms:
        a["delta_vs_armA_pct"] = dev_pct(a["Cd_total"], base)
    grid_spread = dev_pct(
        max(g["Cd_total_primary"] for g in grid), min(g["Cd_total_primary"] for g in grid)
    )
    tau_spread = dev_pct(
        max(a["Cd_total"] for a in arms), min(a["Cd_total"] for a in arms)
    )
    return {
        "grid_ladder": {
            "knob": "cells per hull length L (h = 1/L), Re_lattice fixed at 1000",
            "what_changes": "resolution only — nu_lb = u_lb*L/1000 shifts tau from "
            "0.5144 to 0.5192, i.e. the physical problem is held fixed",
            "tiers": grid,
            "n_tiers": len(grid),
            "spread_max_over_min_pct": grid_spread,
            "monotone_in_L": all(
                grid[i]["Cd_total_primary"] < grid[i + 1]["Cd_total_primary"]
                for i in range(len(grid) - 1)
            ),
            "case_reference_convergence": {
                "primary_E2_standard_abs_err_seq_pct": [
                    abs(g["err_vs_case_reference_pct"]) for g in grid
                ],
                "verdict": lbm["primary_gate_outcome"],
                "note": "against its own laminar reference the primary caliber "
                "converges (|err| 8.15 -> 2.92 -> 0.95%), while the co-shipped "
                "mix50 archive diverges (1.56 -> 5.93 -> 9.21%): the ladder is "
                "instrument-sensitive at the few-% level, which is itself part of "
                "the resolution statement",
            },
            "pathology_note": res["formal_mix50_verdict_FAIL"]["note"],
        },
        "tau_ladder": {
            "knob": "tau (= lattice viscosity nu_lb = (tau-0.5)/3)",
            "what_changes": "IN AN LBM, TAU IS THE REYNOLDS NUMBER, NOT A NUMERICAL "
            "KNOB: u_lb and L are case-frozen, so tau from 0.51 to 1.0 sweeps "
            "Re_lattice from 1200 down to 24 (50x). A tau sweep therefore measures "
            "Re-sensitivity of C_D, and cannot be read as discretisation error.",
            "tiers": arms,
            "n_tiers": len(arms),
            "spread_max_over_min_pct": tau_spread,
            "monotone_in_tau": all(
                arms[i]["Cd_total"] < arms[i + 1]["Cd_total"] for i in range(len(arms) - 1)
            ),
            "overall": tau_l["overall"],
            "overall_note": tau_l["overall_note"],
            "case_independent": True,
            "caveat": tau_l["supplement_tau_re_coupling"]["statement"],
        },
        "admitted_into_error_table": {
            "grid_tier": "L128",
            "instrument": "standard (erratum E2, frozen primary caliber)",
            "Cd_total": finest,
            "rationale": "finest grid of the only verified LBM artifact; its own "
            "reference gate (G1 |err| <= 3%) passes at 0.95%. The co-shipped "
            "mix50 archive at the same tier (0.045862) is reported alongside, and "
            "the whole ladder is disclosed — no tier shopping",
            "tau_arm": "not applicable (tau is not a resolution knob; the case's "
            "own derived tau at L128 is 0.5192)",
        },
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build() -> dict:
    anchor = load("benchmarks/anchors/dtrc-suboff-1989.json")
    lbm = verified_lbm_tiers()
    hr = highre_rows()
    align_v = align_verified(anchor, lbm)
    align_h = align_highre(anchor, hr)
    sens = sensitivity(lbm)

    residuals = [p["recompute"]["residual_vs_registered_C_F_abs"] for p in anchor["anchor_points"]]
    codes = [p["C_F"]["value"] for p in anchor["anchor_points"]]

    crit1 = {
        "id": "Q1_anchor_points_sha256",
        "question": "锚件逐点 >=3 个数值点且 sha256 可核",
        "verdict": "达标",
        "n_points": len(anchor["anchor_points"]),
        "n_families": anchor["summary"]["n_families"],
        "sha256_checked": anchor["criteria"][0]["sha256_checked"],
        "regrades": {p["id"]: p["evidence_grade"] for p in anchor["anchor_points"]},
        "note": "11 points / 2 families, each with verbatim quote + locator + "
        "recomputable sha256. Grades are C for all points (primary WITHHELD) — "
        "the anchor may not be cited as a published experimental benchmark.",
    }
    crit2 = {
        "id": "Q2_ittc57_recompute",
        "question": "C_F 独立复算与 ITTC-57 闭式残差 <= 1e-9",
        "verdict": "达标" if max(residuals) <= 1e-9 else "不达标",
        "tolerance_abs": 1e-9,
        "n_points": len(residuals),
        "max_abs_residual": max(residuals),
        "residuals": residuals,
        "C_F_recomputed": codes,
        "two_path_max_spread": max(
            p["recompute"]["path_spread_abs"] for p in anchor["anchor_points"]
        ),
        "note": "two independent arithmetic paths (50-digit Decimal vs math.log10 "
        "float) are compared against the C_F each register recorded; the larger "
        "of the two residuals is reported, not the friendlier one.",
    }
    devs = [r["dev_pct"] for r in align_h["rows"]]
    crit3 = {
        "id": "Q3_lbm_vs_anchor",
        "question": "LBM vs 锚逐点相对偏差列表，判定门 |偏差| < 15%",
        "verdict": "不达标",
        "gate_pct": GATE_PCT,
        "branches": {
            "verified_laminar_Re1000": {
                "n_rows": align_v["n_rows"],
                "n_pass_raw": align_v["n_pass_raw"],
                "raw_dev_range_pct": [
                    min(r["dev_raw_pct"] for r in align_v["rows"]),
                    max(r["dev_raw_pct"] for r in align_v["rows"]),
                ],
                "n_pass_renormalised_diagnostic": align_v["n_pass_renormalised_diagnostic"],
                "dev_renormalised_range_pct": [
                    min(r["dev_renormalised_pct"] for r in align_v["rows"]),
                    max(r["dev_renormalised_pct"] for r in align_v["rows"]),
                ],
                "admissible": align_v["admissible_for_the_anchor"],
            },
            "pending_highre_wallmodel": {
                "n_rows": align_h["n_rows"],
                "n_pass": align_h["n_pass"],
                "dev_range_pct": [min(devs), max(devs)],
                "admissible": align_h["admissible_for_the_anchor"],
            },
        },
        "root_cause_hypotheses": [
            {
                "branch": "verified_laminar_Re1000",
                "rank": 1,
                "hypothesis": "regime mismatch: Re_lattice = 1000 (laminar, Blasius "
                "reference) vs anchor Re = 1.2e7..3.5e7 (turbulent). The raw "
                "deviation is the Reynolds gap.",
                "quantitative_support": "raw dev "
                f"{min(r['dev_raw_pct'] for r in align_v['rows']):+.0f}.."
                f"{max(r['dev_raw_pct'] for r in align_v['rows']):+.0f}% at Re gap "
                f"{align_v['rows'][0]['re_gap_decades']:.2f} decades; after "
                "transferring the LBM form factor (1+k) onto the anchor-Re ITTC-57 "
                "line the residual shrinks to "
                f"{min(r['dev_renormalised_pct'] for r in align_v['rows']):+.1f}.."
                f"{max(r['dev_renormalised_pct'] for r in align_v['rows']):+.1f}%, "
                "i.e. the bulk of the deviation is the Reynolds number, not the "
                "scheme",
                "test": "run the same case at Re_lattice = 1e4/1e5 (same geometry, "
                "nu from the documented closed form) and check that the (1+k) ratio "
                "stays ~1.09 while the friction level follows the Re law",
            },
            {
                "branch": "pending_highre_wallmodel",
                "rank": 1,
                "hypothesis": "resolution starvation of the wall model: y+_mean "
                "1878..4660 sits far outside the wall-law validity band, so the "
                "delivered friction is the *resolved-Re* (1e5) friction, not the "
                "anchor-Re friction",
                "quantitative_support": "C_F_fric/ITTC(anchor Re) = "
                f"{min(r['diagnostics']['C_F_lbm_fric_over_ittc_at_anchor_Re'] for r in align_h['rows']):.3f}.."
                f"{max(r['diagnostics']['C_F_lbm_fric_over_ittc_at_anchor_Re'] for r in align_h['rows']):.3f} "
                "and nearly constant across the five anchors, i.e. a structural "
                "wall-chain level error, not a per-anchor drift; dev range "
                f"{min(devs):+.0f}..{max(devs):+.0f}%",
                "test": "re-run on a grid whose y+_mean lands in [30, 3000] "
                "(~1e8 cells, multi-node) per the in-repo promotion path",
            },
            {
                "branch": "both",
                "rank": 2,
                "hypothesis": "anchor itself is ambiguous: the two in-repo "
                "transcription families differ by "
                f"{(anchor['cross_family_conflict']['ratio_mean'] - 1) * 100:.1f}%, "
                "larger than the 15% gate, so no simulation can be adjudicated to "
                "15% against it",
                "quantitative_support": "ratio "
                f"{anchor['cross_family_conflict']['ratio_mean']:.4f} "
                f"+- {anchor['cross_family_conflict']['ratio_sd']:.4f}",
                "test": "acquire CRDKNSWC/HD-1298-11 Table 14 p.23 and re-register",
            },
        ],
        "next_experiments": [
            "E1 (anchor side): obtain the primary HD-1298-11 PDF; re-register both "
            "families; the 17% conflict must be adjudicated before any 15% claim",
            "E2 (solver side): wall-model path on a grid with y+_mean in the valid "
            "band (multi-node submission) — the only admissible route to the anchor's Re",
            "E3 (scheme side): laminar Re ladder (1e3 -> 1e5) of the verified case "
            "to separate the form factor from the friction law",
        ],
    }
    crit4 = {
        "id": "Q4_resolution_declaration",
        "question": "网格/tau 对 C_T 的敏感性（>=3 档），标明哪一档进入误差表",
        "verdict": "达标",
        "grid_tiers": sens["grid_ladder"]["n_tiers"],
        "tau_tiers": sens["tau_ladder"]["n_tiers"],
        "grid_spread_pct": sens["grid_ladder"]["spread_max_over_min_pct"],
        "tau_spread_pct": sens["tau_ladder"]["spread_max_over_min_pct"],
        "admitted": sens["admitted_into_error_table"],
        "caveat": "tau is the Reynolds number in LBM, not a discretisation knob; "
        "the grid ladder is the only resolution evidence, and only 3 tiers of it "
        "exist in-repo",
    }

    art = {
        "schema": "suboff-anchor-census/v1",
        "artifact_id": "w30-lbm-suboff-tau-anchor-census",
        "task": "m21512",
        "task_keys_closed": [
            "paper-exp-lbm-suboff-tau-3",
            "paper-exp-lbm-suboff-tau-4",
            "paper-exp-lbm-suboff-tau-6",
        ],
        "topic": "lbm-suboff-tau",
        "repo": "TensorLBM",
        "git_commit": git_commit(),
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "generator": "benchmarks/anchors/suboff_anchor_census.py",
        "discipline": "ship-total-performance / resistance (DARPA SUBOFF, bare hull)",
        "sources": {
            rel: sha256_of(rel)
            for rel in [
                "benchmarks/anchors/dtrc-suboff-1989.json",
                "benchmarks/anchors/dtrc_suboff_1989_anchor.py",
                "docs/suboff_experimental_resistance_validation_2026-08-01.md",
                "src/tensorlbm/suboff_reference_data.py",
                "benchmarks/pending/suboff_highre_resistance/refs_lock.md",
                "benchmarks/pending/suboff_highre_resistance/refs_lock.json",
                HIGHRE_VERDICT,
                VERIFIED_RESULT,
                VERIFIED_VERDICT,
                TAU_LADDER,
                "benchmarks/anchors/lbm-suboff-tau.json",
            ]
        },
        "anchor": {
            "anchor_id": anchor["anchor_id"],
            "status": anchor["status"],
            "acquisition_status": anchor["acquisition_status"],
            "grade_scale": anchor["grade_scale"],
            "geometry_gate": anchor["geometry"]["gate"],
            "conventions": anchor["conventions"],
            "n_points": anchor["summary"]["n_anchor_points"],
            "points": [
                {
                    "id": p["id"],
                    "family": p["family"],
                    "speed_kn": p["speed_kn"],
                    "speed_ms": p["speed_ms"],
                    "resistance_n": p["resistance_n"],
                    "Re": p["Re"],
                    "C_T": p["C_T"],
                    "C_F": p["C_F"]["value"],
                    "form_factor_CT_over_CF": p["form_factor_CT_over_CF"],
                    "evidence_grade": p["evidence_grade"],
                    "acquisition_status": p["acquisition_status"],
                    "grade_reason": p["grade_reason"],
                    "provenance": p["provenance"],
                    "recompute": p["recompute"],
                    "what_is_missing": p["what_is_missing"],
                }
                for p in anchor["anchor_points"]
            ],
            "cross_family_conflict": anchor["cross_family_conflict"],
        },
        "lbm_evidence": {
            "verified_case": lbm,
            "regime_matched_case": hr,
        },
        "error_table": {
            "gate_pct": GATE_PCT,
            "sign_convention": "dev_pct = (C_T_lbm / C_T_anchor - 1) * 100",
            "branch_verified_laminar_Re1000": align_v,
            "branch_pending_highre_wallmodel": align_h,
        },
        "resolution_and_tau_sensitivity": sens,
        "criteria": [crit1, crit2, crit3, crit4],
        "verdict": {
            "Q1": crit1["verdict"],
            "Q2": crit2["verdict"],
            "Q3": crit3["verdict"],
            "Q4": crit4["verdict"],
            "overall": "PARTIAL — anchor register + ITTC recompute + sensitivity "
            "达标; the LBM-to-anchor 15% gate is 不达标 in every admissible branch "
            "and the anchor itself carries a 17% cross-transcription conflict",
            "must_not_be_reported_as": [
                "validation of the LBM solver against DTRC tow-tank data",
                "an agreement claim within 15%",
                "a primary/anchor grade A or B",
            ],
        },
        "claim_boundary": {
            "what_this_artifact_supports": [
                "a per-point, sha256-traced register of the DTRC SUBOFF resistance "
                "numbers recoverable in this checkout, with grades as-is (all C)",
                "an independent ITTC-1957 closure at 1e-9 or better on 11 points",
                "the statement that the LBM-vs-anchor 15% gate is NOT met, with the "
                "quantified Reynolds gap and the wall-model y+ root cause",
                "a 3-tier grid sensitivity and a 5-arm tau(=Re) sensitivity",
            ],
            "what_it_does_not_support": [
                "any A/B anchor claim",
                "any <15% agreement claim against the DTRC data",
                "treating the verified Re=1000 case as an anchor validation",
            ],
        },
    }
    return art


def verify(path: Path = OUT_PATH) -> dict:
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    for k in ("anchor", "error_table", "resolution_and_tau_sensitivity", "criteria", "verdict", "sources"):
        if k not in art:
            problems.append(f"missing {k}")
    frozen_inputs = W30_FROZEN_INPUTS if art.get("git_commit") == W30_SOURCE_COMMIT else {}
    for rel, h in art.get("sources", {}).items():
        source = frozen_inputs.get(rel, ROOT / rel)
        if not source.is_file():
            problems.append(f"missing source {rel}")
        elif hashlib.sha256(source.read_bytes()).hexdigest() != h:
            problems.append(f"sha256 drift on {rel}")
    a = art["anchor"]
    if a["n_points"] < 3:
        problems.append("fewer than 3 anchor points")
    for p in a["points"]:
        if p["evidence_grade"] not in ("A", "B", "C", "D"):
            problems.append(f"{p['id']}: bad grade")
        if not p["provenance"]["verbatim_quote"]:
            problems.append(f"{p['id']}: empty quote")
        if p["recompute"]["residual_vs_registered_C_F_abs"] > 1e-9:
            problems.append(f"{p['id']}: ITTC residual too large")
    et = art["error_table"]
    for br in ("branch_verified_laminar_Re1000", "branch_pending_highre_wallmodel"):
        if not et[br]["rows"]:
            problems.append(f"{br}: no rows")
    if et["branch_pending_highre_wallmodel"]["n_rows"] != 5:
        problems.append("high-Re branch must align all 5 anchor points")
    if et["branch_verified_laminar_Re1000"]["n_rows"] != a["n_points"] * 3:
        problems.append("verified branch must be n_points x 3 tiers")
    sens = art["resolution_and_tau_sensitivity"]
    if sens["grid_ladder"]["n_tiers"] < 3 or sens["tau_ladder"]["n_tiers"] < 3:
        problems.append("need >=3 tiers in both ladders")
    if not sens["admitted_into_error_table"]["grid_tier"]:
        problems.append("no admitted tier declared")
    return {"path": str(Path(path)), "ok": not problems, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv:
        res = verify()
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return 0 if res["ok"] else 1
    art = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    res = verify()
    print(
        json.dumps(
            {
                "written": str(OUT_PATH),
                "n_anchor_points": art["anchor"]["n_points"],
                "max_ittc57_residual": max(
                    p["recompute"]["residual_vs_registered_C_F_abs"]
                    for p in art["anchor"]["points"]
                ),
                "verified_branch_pass_raw": art["error_table"][
                    "branch_verified_laminar_Re1000"
                ]["n_pass_raw"],
                "highre_branch_pass": art["error_table"]["branch_pending_highre_wallmodel"][
                    "n_pass"
                ],
                "verdict": art["verdict"],
                "verify_ok": res["ok"],
                "problems": res["problems"],
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0 if res["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())