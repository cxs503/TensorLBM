#!/usr/bin/env python3
"""m20151 board audit — independent re-derivation of every number claimed by m20070.

Reads only the delivered artifact (snapshot, sha256-identical to the m20070 work
tree) plus the delivered markdown, and re-computes:

  * the 18 (closure, parameter) AD-vs-FD relative errors and the 1e-2 gate,
  * non-degeneracy of every sensitivity (zero-vs-zero comparisons are vacuous),
  * the plane Couette / plane Poiseuille exact-solution anchor errors,
  * the DFG cylinder error against BOTH references that appear in the delivery
    (artifact-side 2-D-1 range and doc-side 2-D-2 range),
  * text-vs-artifact drift of the parameter tables in the report.

No torch is imported: this is a recomputation of arithmetic the artifact states,
not a rerun of the solver (the solver rerun is evidence/repro_grad_main_reproduced.json).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
ART = BASE / "delivered_snapshot" / "artifacts" / "diff_boundary_grad_full.json"
DOC = BASE / "delivered_snapshot" / "GAP-DIFFLBM-BOUNDARY-GRAD.md"
OUT = BASE / "evidence"


def rel(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref)


def main() -> int:
    d = json.loads(ART.read_text())
    OUT.mkdir(exist_ok=True)
    rep: dict = {}

    # ---- 1. gradient matrix ------------------------------------------------
    rows = []
    worst = 0.0
    degenerate = 0
    for r in d["gradient_matrix"]:
        for obj in ("drag", "lift"):
            ad = r["ad"][obj]
            for tag in ("h1", "h2"):
                fd = r["fd"][tag][obj]
                e = rel(ad, fd) if fd else float("inf")
                rows.append({"closure": r["closure"], "theta0": r["theta0"], "objective": obj,
                             "h": r["fd"][tag]["h"], "ad": ad, "fd": fd, "rel_err": e,
                             "pass_1e2": e <= 1e-2})
                worst = max(worst, e)
            if abs(r["ad"][obj]) < 1e-3 or abs(r["fd"]["h2"][obj]) < 1e-3:
                degenerate += 1
        if abs(r["ad"]["drag"] - r["ad"]["lift"]) < 1e-3:
            degenerate += 1
    rep["gradient_matrix"] = {
        "n_rows_reported": len(d["gradient_matrix"]),
        "n_pairs_h1h2": len(rows),
        "closures": sorted({r["closure"] for r in rows}),
        "objectives": sorted({r["objective"] for r in rows}),
        "worst_rel_err": worst,
        "n_fail": sum(1 for r in rows if not r["pass_1e2"]),
        "n_degenerate": degenerate,
        "artifact_summary": d["summary"],
        # F3: the delivered summary aggregates only the FINER FD step (h2), so its
        # headline "worst 1.31e-03" understates the worst error over both step sizes
        # (coarse step h1 worst = 1.77e-03).  Pass/fail is unaffected (1e-2 gate).
        "worst_rel_err_fine_step_only": max(
            max(r["rel_err_drag"], r["rel_err_lift"]) for r in d["gradient_matrix"]),
        "worst_rel_err_coarse_step": max(
            max(r["fd"]["h1"]["rel_err_drag"], r["fd"]["h1"]["rel_err_lift"])
            for r in d["gradient_matrix"]),
        "summary_scope_defect": abs(worst - d["summary"]["worst_rel_err"]) > 1e-12,
        "artifact_summary_consistent": (
            len(rows) == d["summary"]["n_pairs"]
            and all(r["pass_1e2"] for r in rows) == d["summary"]["all_pass"]
        ),
        "rows": rows,
    }

    # ---- 2. exact-solution anchors ----------------------------------------
    anchors = {}
    c = d["anchors"]["couette"]
    anchors["couette"] = {
        "profile_max_rel_err_reported": c["profile_max_rel_err"],
        "profile_max_rel_err_recomputed": max(
            rel(g, e) for g, e in zip(c["profile_got"], c["profile_exact"])),
        "force_on_moving_wall": c["force_on_moving_wall"],
        "force_analytic": c["force_analytic_magnitude"],
        "force_rel_err_reported": c["force_rel_err"],
        "force_rel_err_recomputed": rel(abs(c["force_on_moving_wall"]), c["force_analytic_magnitude"]),
        "sign_consistent": c["sign_consistent"],
        "reference": c["reference"],
    }
    p = d["anchors"]["poiseuille"]
    anchors["poiseuille"] = {
        "profile_max_rel_err_reported": p["profile_max_rel_err"],
        "profile_max_rel_err_recomputed": max(
            rel(g, e) for g, e in zip(p["profile_got"], p["profile_exact"])),
        "force_on_lower_wall": p["force_on_lower_wall"],
        "force_analytic_per_wall": p["force_analytic_per_wall"],
        "force_rel_err_recomputed": rel(p["force_on_lower_wall"], p["force_analytic_per_wall"]),
        "sign_consistent": p["sign_consistent"],
        "reference": p["reference"],
    }
    # ---- 3. DFG cylinder against both references quoted in the delivery ----
    g = d["anchors"]["dfg"]
    dfg = {
        "params": g["params"],
        "C_D_got": g["C_D_mean"],
        "C_L_got": g["C_L_amplitude"],
        "artifact_side_reference": g["C_D_ref"],
        "artifact_side_rel_err_to_mid": g["C_D_rel_err_to_mid"],
        "artifact_side_rel_err_recomputed": rel(
            g["C_D_mean"], sum(g["C_D_ref"]) / 2.0),
        "doc_side_reference_2D2_Re100": [3.22, 3.24],
        "doc_side_rel_err_to_mid": rel(g["C_D_mean"], 3.23),
        "doc_side_rel_err_vs_artifact_halfwidth": None,
        "C_D_history_first_last": [g["samples"]["C_D"][0], g["samples"]["C_D"][-1]],
        "C_D_history_monotone_growth": all(
            b >= a for a, b in zip(g["samples"]["C_D"], g["samples"]["C_D"][1:])),
        "verdict": "FAIL against both references (honest FAIL disclosed in the report)",
    }
    # the doc prints 1241.08 % while quoting ref 3.23; 74.83/5.58-1 = 1241 % -> the
    # percentage belongs to the 2-D-1 range, not to the 3.23 it is printed next to.
    dfg["doc_side_rel_err_vs_artifact_halfwidth"] = rel(g["C_D_mean"], sum(g["C_D_ref"]) / 2.0)

    # ---- 4. text-vs-artifact drift of the report parameter tables ---------
    doc = DOC.read_text()

    def has(pat: str) -> bool:
        return re.search(pat, doc) is not None

    lat = d["lattice"]
    drift = {
        "artifact_lattice_main": lat["main"],
        "artifact_lattice_coarse": lat["coarse"],
        "doc_claim_repro_grad_flags": re.findall(r"--grad-ny (\d+) --grad-nx (\d+) --grad-dcells ([\d.]+) --grad-steps (\d+)", doc),
        "doc_claim_1200_steps": has(r"1200 steps"),
        "doc_claim_800_steps": has(r"800 steps"),
        "doc_claim_dfg_2D2_Re100": has(r"DFG 2-D-2") and has(r"Re=100"),
        "artifact_dfg_re": d["anchors"]["dfg"]["params"]["re"],
        "artifact_matrix_steps_effective": 200,
        "artifact_horizon_steps_effective": 100,
        "artifact_resolution_scan_lattice": {
            "ny": d["resolution_scan"][0]["ny"], "nx": d["resolution_scan"][0]["nx"],
            "d_cells": d["resolution_scan"][0]["d_cells"],
        },
        "note": ("delivered `lattice` block was written by an earlier --part invocation "
                 "(setdefault in run_gap_diff_boundary_grad.py) and does not describe the "
                 "gradient_matrix actually contained in the file"),
    }
    assert lat["main"]["ny"] != 30, "lattice.main unexpectedly matches the reproduced matrix"

    rep["anchors"] = anchors
    rep["dfg"] = dfg
    rep["doc_artifact_drift"] = drift
    rep["overall"] = {
        "gradient_claim_holds": rep["gradient_matrix"]["n_fail"] == 0
        and rep["gradient_matrix"]["n_degenerate"] == 0
        and rep["gradient_matrix"]["artifact_summary_consistent"],
        "exact_anchors_hold": anchors["couette"]["force_rel_err_recomputed"] < 1e-6
        and anchors["poiseuille"]["force_rel_err_recomputed"] < 1e-6,
        "external_benchmark": "FAIL (DFG cylinder diverges; honest FAIL in report)",
        "level_implication": ("L2: real implementation + run artifacts + verified AD-vs-FD "
                              "self-consistency + exact channel anchors; no integration caller "
                              "and no passing external benchmark -> L3/L4 not earned"),
    }
    (OUT / "verification.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("gradient_matrix",)}, indent=1)[:1500])
    print("worst_rel_err", worst, "deg", degenerate, "fails",
          sum(1 for r in rows if not r["pass_1e2"]))
    return 0 if rep["overall"]["gradient_claim_holds"] and rep["overall"]["exact_anchors_hold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())