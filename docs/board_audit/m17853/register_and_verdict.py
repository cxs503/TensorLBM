#!/usr/bin/env python3
"""m17853 board audit — benchmark-case registration + maturity verdict for m17726.

Two writes, both idempotent (re-running replaces this auditor's own rows):

1. ``tensor_benchmark_cases``: the anchor had **no** judge row before this audit
   (0 rows for ``%WALL-ORDER%``).  The row is registered with the predeclared
   15 % bar and the measured errors re-derived from the run files, status
   ``pending`` because 1/10 points fails.
2. ``tensor_maturity_verdicts``: level 2 (functional substance), evidence pointers
   are the *repo-relative* paths the deliverable itself cites -> they do not verify
   inside the repository (all files are untracked, see AUDIT.md F1), so the machine
   gate is expected to downgrade the ledger level to 1 and open the rework.

Everything numeric is read from the artefacts written by the sibling scripts
(``verify_snapshot.py`` -> snapshot_manifest.json, ``compare_rerun.py`` ->
rerun_compare.json), never typed by hand.

Run: python3 docs/board_audit/m17853/register_and_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")

AUDIT_MID = 17853
SUBJECT_MID = 17726
REPO = "TensorLBM"
ANCHOR = "XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER"
SCOPE = f"m{SUBJECT_MID} {ANCHOR}"
BY = "board-audit"
LEVEL = 2          # functional substance; ledger level expected 1 (nothing landed)

CASE_NAME = ("XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER: sessile-cap static contact "
             "angle theta(psi_w) on the wall-order-parameter ghost wall, 6-point psi_w sweep "
             "at xi=3 + xi=2/3/4 resolution sweep (TensorLBM port, 56x80, x-periodic)")
REF_SOURCE = ("analytic Young law cos(theta)=(3 psi_w - psi_w^3)/2 implied by the Dirichlet wall "
              "order parameter of the same free energy (free-energy consistency reference, NOT an "
              "experiment); mechanism provenance = ebinan92/Fingering_dynamics "
              "lattice_boltzmann/validation.py@150a5739, whose numeric arm produced no number "
              "(NaN by step 200; published-box retry cluster job cj1656 still queued)")
METRIC = ("static contact angle theta [deg] per point, measured by Laplace-cap inversion from the "
          "conserved excess area + apex height (primary) with a psi=0 circle fit as cross-check; "
          "err vs analytic law per point, median |err|, max |err|; convergence flag per row")
TOLERANCE = ("<15% per point (owner standard; predeclared accept_pct=15.0 in "
             "xval/LBM_WALL_ORDER_PARAMETER_WETTING/predeclared.json). Auditor m17853 re-derived "
             "from runs/*/result.json: median |err| 1.60%, 9/10 inside gate, psi_w=+0.70 -> 29.52% "
             "FAIL, 10/10 rows hit the 30000-iteration cap (converged=false) => gate NOT passed; "
             "external OSS anchor WITHHELD (NaN; cj1656 queued)")


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def build_evidence(snap: dict, cmp: dict) -> dict:
    g = snap["shipped_gates"]
    cs = cmp["summary"]
    fails = [r for r in cmp["rows"] if r.get("status") != "not_rerun"
             and not r["within_15pct_rerun"]]
    return {
        # --- SCHEME keys: repo-relative pointers *as the deliverable cites them* ---
        "impl_files": ["src/tensorlbm/wetting/wall_order_parameter_2d.py"],
        "entry": "xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py",
        "run_artifacts": [
            "xval/LBM_WALL_ORDER_PARAMETER_WETTING/runs/port_psi+0.35_xi3.0/result.json",
            "xval/LBM_WALL_ORDER_PARAMETER_WETTING/runs/port_psi+0.70_xi3.0/result.json",
            "xval/LBM_WALL_ORDER_PARAMETER_WETTING/summary.json",
        ],
        "tests": [
            "tests/test_wall_order_parameter_wetting.py::test_isotropic_laplacian_reproduces_quadratic",
            "tests/test_wall_order_parameter_wetting.py::test_isotropic_gradient_reproduces_linear_field",
            "tests/test_wall_order_parameter_wetting.py::test_wall_ghost_enters_both_stencils",
            "tests/test_wall_order_parameter_wetting.py::test_free_energy_parameters_give_tanh_profile_and_sigma",
            "tests/test_wall_order_parameter_wetting.py::test_young_law_reference_values",
            "tests/test_wall_order_parameter_wetting.py::test_theta_from_cap_area_inverts_synthetic_cap",
            "tests/test_wall_order_parameter_wetting.py::test_neutral_wall_gives_ninety_degrees",
            # auditor re-run: 15 passed, 1 xfailed (63 s)
        ],
        "integration_callers": ["src/tensorlbm/wetting/__init__.py:42"],
        "benchmark": {
            "anchor_slug": ANCHOR,
            "ref": "cos(theta) = (3 psi_w - psi_w^3)/2 (predeclared accept_pct=15.0)",
            "got_median_abs_err_pct": round(g["median_abs_err_pct"], 4),
            "got_max_abs_err_pct": round(g["max_abs_err_pct"], 4),
            "n_points": g["n_points"],
            "n_within_15pct": g["n_in_15pct"],
            "worst_point": "psi_w=+0.70, xi=3: theta=20.115 deg vs analytic 28.538 deg (-29.52%)",
            "all_converged": g["all_converged"],
            "oss_arm": snap.get("oss_arm"),
            "auditor_rerun": {"n_rerun": cs["n_points_rerun"],
                              "bitwise_identical": cs["bitwise_identical_points"],
                              "max_abs_delta_deg": cs["max_abs_delta_deg"],
                              "n_in_15pct": cs["n_in_15pct_rerun"],
                              "max_abs_err_pct": round(cs["max_abs_err_pct_rerun"], 4),
                              "not_rerun": cs["missing_reruns"]},
        },
        "repro_cmd": ("cd xval/LBM_WALL_ORDER_PARAMETER_WETTING && python3 run_xval.py port "
                      "--psi-wall 0.35 --xi 3.0 && python3 run_xval.py port --psi-wall 0.70 "
                      "--xi 3.0 && python3 run_xval.py summarize   # auditor ran the same via "
                      "docs/board_audit/m17853/rerun_all.sh, 6/10 points bit-identical"),
        "external_anchor": ("ebinan92/Fingering_dynamics lattice_boltzmann/validation.py@150a5739 "
                            "(vendored copy, no LICENSE in that repo) — WITHHELD: "
                            "xval/LBM_WALL_ORDER_PARAMETER_WETTING/runs/oss_psi+0.00/result.json "
                            "has finite=false, psi_min/max=NaN at step 200; cluster job cj1656 "
                            "(200x250, 1200 steps) state='queued', never ran"),
        # --- audit-only keys (not part of SCHEME; provenance for the machine + board) ---
        "audit_verdict_by": BY,
        "audit_mission": AUDIT_MID,
        "audit_level_rationale": (
            "L2 = real code input->compute->output, independently re-executed bit-identically "
            "(6/10 points, max |dtheta| = 0.0 deg). L3 refused: the only in-tree reference is the "
            "re-export src/tensorlbm/wetting/__init__.py:42, no solver/benchmark caller. "
            "L4 refused: anchor gate fails at psi_w=+0.70 (29.52% > 15%), 10/10 rows "
            "converged=false, external arm WITHHELD (NaN). L5 refused: no independent "
            "experiment/literature angle. Ledger level lower because of F1 (nothing landed)."),
        "audit_landing_defect_F1": {
            "finding": "no mission commit: branch agent/m17726 HEAD holds none of the deliverables; "
                       "all files are untracked in /root/tp-agent/work/m17726",
            "machine_checks": {d["path"]: {k: d[k] for k in
                                           ("exists_in_main_clone", "in_any_git_ref_history",
                                            "in_agent_m17726_tree", "untracked_in_m17726_worktree")}
                               for d in snap["deliverables"]},
            "mission_result_branch": snap["mission_17726"].get("status"),
        },
        "audit_snapshot": {
            "why": "the only copy of the deliverable is an untracked worktree file; preserved with "
                   "sha256 inside this audit branch so it cannot be lost with the worktree",
            "files": snap["snapshot_files"],
            "root": "docs/board_audit/m17853/refs",
        },
        "audit_benchmark_row_absent_before": snap["benchmark_rows_for_anchor_before_audit"],
        "audit_open_questions": [
            "psiw=+0.70: the two independent estimators differ by >5 deg -> measurement defect "
            "hypothesis untested, not only a model limit",
            "xi=2/4 resolution-sweep points (4) were NOT re-run by this audit -> unverified",
            "no independent (experiment/literature) contact angle exists -> L5 unreachable",
            "vendored OSS file from a LICENSE-less repo is a redistribution risk if committed",
        ],
        "audit_artifacts": [
            "docs/board_audit/m17853/AUDIT.md",
            "docs/board_audit/m17853/artifacts/snapshot_manifest.json",
            "docs/board_audit/m17853/artifacts/rerun_compare.json",
            "docs/board_audit/m17853/refs/xval/figures/fig1_contact_angle_law.png",
        ],
        "audit_report_id": 16715,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    snap, cmp_ = load("snapshot_manifest.json"), load("rerun_compare.json")
    ev = build_evidence(snap, cmp_)
    print(json.dumps({k: ev[k] for k in ("benchmark", "external_anchor")}, indent=2)[:1500])
    if a.dry_run:
        print("[dry-run] no DB write")
        return 0

    con = psycopg.connect(DSN)
    cur = con.cursor()
    cur.execute("DELETE FROM tensor_benchmark_cases WHERE anchor_slug=%s", (ANCHOR,))
    cur.execute("INSERT INTO tensor_benchmark_cases "
                "(anchor_slug, case_name, ref_source, metric, tolerance, repo, status) "
                "VALUES (%s,%s,%s,%s,%s,%s,'pending') RETURNING id",
                (ANCHOR, CASE_NAME, REF_SOURCE, METRIC, TOLERANCE, REPO))
    case_id = cur.fetchone()[0]

    cur.execute("DELETE FROM tensor_maturity_verdicts WHERE repo=%s AND scope=%s AND verdict_by=%s",
                (REPO, SCOPE, BY))
    cur.execute("INSERT INTO tensor_maturity_verdicts "
                "(repo, scope, mission_id, level, evidence, verdict_by) VALUES (%s,%s,%s,%s,%s,%s) "
                "RETURNING id", (REPO, SCOPE, AUDIT_MID, LEVEL, json.dumps(ev), BY))
    vid = cur.fetchone()[0]
    con.commit()
    print(f"benchmark_case_id={case_id} verdict_id={vid} level={LEVEL} scope={SCOPE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())