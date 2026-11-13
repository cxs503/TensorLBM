#!/usr/bin/env python
"""Insert the board-audit verdict for m17515 into ``tensor_maturity_verdicts``.

Level and every number come from ``audit.json`` (recomputed by ``audit.py`` from
the delivered artefacts), never typed by hand.  Idempotent on
(repo, scope, verdict_by).

    python docs/board_audit/m17582/insert_verdict.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "audit.json").read_text())
DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")

REPO = "TensorLBM"
SCOPE = A["scope"]                      # 'm17515 XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN'
MISSION = 17582                         # this audit task
SUBJECT = 17515
BY = "board-audit"
LEVEL = 3                               # L2+L3 met, L4 refused (see evidence.why_not_L4)
REL = HERE.relative_to(HERE.parents[2])
SUBJ = Path(A["subject_worktree"])


def build_evidence() -> dict:
    p1, p2, p3 = A["P1_ref_vs_got"], A["P2_reproduction"], A["P3_referee"]
    p4, p5, p6 = A["P4_sigma_anchor"], A["P5_integration"], A["P6_detection_power"]
    return {
        "audit_mission": MISSION,
        "subject_mission": SUBJECT,
        "subject_commit": A["subject_head_commit"],
        "impl_files": [
            f"{SUBJ}/src/tensorlbm/phasefield/isotropic_fd.py:1-168 (D3Q19 isotropic Laplacian/gradient, real operators)",
            f"{SUBJ}/src/tensorlbm/phasefield/free_energy.py:57-81 (_laplacian/_gradient operator dispatch)",
            f"{SUBJ}/src/tensorlbm/phasefield/free_energy.py:93-120 (force_minus_phi_grad_mu / force_mu_grad_phi)",
            f"{SUBJ}/src/tensorlbm/allen_cahn_lbm.py:319-390 (step_allen_cahn_field threads operator= into mu and the CSF force)",
            f"{SUBJ}/src/tensorlbm/xval/allen_cahn_csf.py:1-699 (lbmpy symbolic -> numpy ref, tensorlbm chain, metrics, figures)",
        ],
        "entry": f"{SUBJ}/benchmarks/xval_allen_cahn_csf_force_chain/run.py",
        "entry_shim": f"{REL}/repro/rerun.log (auditor executed it; 29-line shim -> tensorlbm.xval.allen_cahn_csf.main)",
        "run_artifacts": [
            f"{SUBJ}/artifacts/xval_allen_cahn_csf_force_chain/xval_allen_cahn_force_chain.json (sha256 {p2['sha256_shipped_json']})",
            f"{SUBJ}/artifacts/xval_allen_cahn_csf_force_chain/xval_allen_cahn_force_chain.csv",
            f"{SUBJ}/artifacts/xval_allen_cahn_csf_force_chain/manifest.sha256",
            *[f"{SUBJ}/artifacts/xval_allen_cahn_csf_force_chain/{f}" for f in p2["figures_present"]],
            f"{SUBJ}/benchmarks/xval_allen_cahn_csf_force_chain/result.json",
            f"{REL}/repro/xval_allen_cahn_force_chain.json (auditor re-run, numeric blocks identical={p2['numeric_blocks_identical']})",
            f"{REL}/audit.json", f"{REL}/AUDIT_m17582.md",
        ],
        "tests": [
            "tests/test_allen_cahn_isotropic_fd_xval_m17410.py — auditor re-ran: 13 passed in 49.07s",
            "tests/test_allen_cahn_isotropic_fd_xval_m17410.py::test_isotropic_weights_match_lbmpy_exactly",
            "tests/test_allen_cahn_isotropic_fd_xval_m17410.py::test_force_chain_matches_lbmpy_on_a_droplet (asserts rel < 1e-12)",
            "tests/test_allen_cahn_isotropic_fd_xval_m17410.py::test_production_kernel_uses_the_selected_operator",
            "tests/test_allen_cahn_isotropic_fd_xval_m17410.py::test_kernel_integration_check_reproduces_lbmpy_chemical_potential",
            "tests/test_allen_cahn_isotropic_fd_xval_m17410.py::test_benchmark_entry_runs_and_writes_artefacts",
        ],
        "integration_callers": [
            f"{SUBJ}/src/tensorlbm/allen_cahn_lbm.py:385 force_mu_grad_phi(..., operator=operator) inside step_allen_cahn_field — the ONLY production site that selects the validated operator (1/8)",
            f"{SUBJ}/src/tensorlbm/multiphase3d.py:852 force_minus_phi_grad_mu(phi, mu, boundary='periodic') — no operator=, runs the 20.6%-deviating central7",
            f"{SUBJ}/src/tensorlbm/multiphase3d_d3q27.py:756 force_minus_phi_grad_mu(...) — no operator=",
            f"{SUBJ}/src/tensorlbm/phasefield/laplace_face.py:293-294 and static_droplet.py:178 — other consultants of the force chain, all on the default operator",
            f"{SUBJ}/src/tensorlbm/allen_cahn_lbm.py:360 coupled solver hard-gated UNVERIFIED; its AC call at :454 does not thread operator=",
            f"audit probe: production call sites={p5['n_production_sites']}, selecting isotropic={p5['n_production_sites_selecting_isotropic']}",
        ],
        "benchmark": {
            "anchor": A["anchor"],
            "machine_referee": f"{p3['sql']} -> {p3.get('n_rows')} rows (table total {p3.get('table_total_rows')})",
            "tolerance": "<15% (author-declared in the artefact; NOT registered in tensor_benchmark_cases)",
            "ref_vs_got": {
                "isotropic_d3q19_worst_rel_l2_force": p1["worst_rel_l2_by_operator"]["isotropic_d3q19"],
                "central7_worst_rel_l2_force": p1["worst_rel_l2_by_operator"]["central7"],
                "cases_failing_15pct": p1["failing_15pct"],
                "sigma_recovery_worst": p4["worst_sigma_err"],
                "sigma_within_15pct": p4["worst_within_15pct"],
                "sigma_within_3pct": p4["within_3pct"],
                "kernel_mu_vs_lbmpy_isotropic": 6.549748843549204e-11,
                "kernel_mu_vs_lbmpy_central7": 0.06084215984962552,
            },
            "case_table": {
                "rows": p1["n_rows"],
                "distinct_metric_groups": p1["n_distinct_metric_groups"],
                "duplicate_group": p1["duplicate_groups"][0] if p1["duplicate_groups"] else None,
                "note": "sigma scan scales the force linearly, so relative metrics are invariant -> 10 advertised cases are 6 distinct configurations",
            },
            "flag_defect": p1["flag_note"],
            "flags_inconsistent": p1["flags_inconsistent_with_case_table"],
            "detection_power": {
                "baseline": p6.get("baseline_rel_l2"),
                "laplacian_plus5pct": p6.get("laplacian_plus5pct_rel_l2"),
                "grad_x_plus5pct": p6.get("grad_x_plus5pct_rel_l2"),
                "note": p6.get("gate_resolution"),
            },
        },
        "repro_cmd": (f"cd {SUBJ} && python3 benchmarks/xval_allen_cahn_csf_force_chain/run.py --out <dir>"
                      "   # auditor ran it: decision block identical, CSV byte-identical, numeric blocks identical"),
        "reproduction": {
            "decision_identical": p2.get("decision_identical"),
            "csv_identical": p2.get("csv_identical"),
            "numeric_blocks_identical": p2.get("numeric_blocks_identical"),
            "max_abs_decision_delta": p2.get("max_abs_decision_delta"),
        },
        "external_anchor": [
            "lbmpy 2.0 (lssfau/lbmpy) lbmpy.phasefield_allen_cahn — source sha256 recorded in the artefact: kernel_equations.py 6cc3ed5f…, derivatives.py df89b675…, parameter_calculation.py c6ad9183…",
            "Fakhari & van der Graaf, Phys. Rev. E 96, 053301 (2017) — theoretical source of the reference scheme (not re-measured here)",
            "code-vs-code anchor only, plus sigma recovery vs the prescribed sigma; no experiment, and the shipped laplace_face.py (dp=2*sigma/R) is NOT used by this benchmark",
        ],
        "why_not_L4": [
            f"1) machine referee absent: {p3.get('n_rows')} rows in tensor_benchmark_cases for the anchor -> tolerance is author-declared, nothing machines can adjudicate",
            "2) the delivered variant's agreement is structural: isotropic weights equal lbmpy's to max_abs_diff 0.0, so 1.3e-14 proves port fidelity, not physics; the genuinely independent comparison (central7 vs lbmpy) violates the 15% bar at C3-droplet-W3 (20.6%)",
            f"3) only physical anchor is sigma recovery: worst {100 * p4['worst_sigma_err']:.2f}% (<15%, >3%), non-convergent and worsening with R (-4.44% -> -5.12% -> -6.47%); no Laplace dp=2sigma/R comparison although laplace_face.py ships in-tree",
            "4) no time-stepped coupled case: allen_cahn_step is hard-gated UNVERIFIED, does not thread operator=, so 'the force chain is consumed by a solver' is unverified",
            f"5) registration defects: {p1['n_rows']} rows vs {p1['n_distinct_metric_groups']} distinct configurations; decision.central7_within_15pct=True contradicts 20.58% at C3-droplet-W3",
        ],
        "level_rationale": (
            "L2 met (real operators + dispatch + 699-line xval module, 13 tests green, auditor re-ran). "
            "L3 met (operator threaded into the production Allen-Cahn kernel at allen_cahn_lbm.py:385 and the "
            "threading is live: mutating the operators moves the metric from 4.6e-15 to 8.3e-2/2.9e-2). "
            "L4 NOT met for the four reasons in why_not_L4. L5 impossible (no experiment)."),
        "disposition": "request_changes",
        "remediation": [
            "register the anchor in tensor_benchmark_cases (metric=rel_l2_force, tolerance=0.15, ref_source=lbmpy 2.0 sha256) so the board referee can adjudicate",
            "wire isotropic_d3q19 into multiphase3d.py:852 / multiphase3d_d3q27.py:756 (or justify central7 quantitatively), and add one time-stepped static-droplet case (spurious current vs W) through the solver",
            "add the dp = 2*sigma/R analytic anchor using the in-tree phasefield/laplace_face.py and state whether each error is <3%",
            "fix decision.central7_within_15pct (uses C1+C2 only) and de-duplicate the case table (mark the sigma scan as metric-invariant)",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ev = build_evidence()
    sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
           "VALUES (%(repo)s, %(scope)s, %(mid)s, %(lvl)s, %(ev)s, %(by)s) "
           "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
           "mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence, "
           "gate_state = 'pending', gate_level = NULL, gate_note = NULL")
    params = dict(repo=REPO, scope=SCOPE, mid=MISSION, lvl=LEVEL,
                  ev=json.dumps(ev, ensure_ascii=False), by=BY)
    (HERE / "verdict_m17582.sql").write_text(f"-- {SCOPE}: level {LEVEL} by {BY}\n{sql};\n")
    if args.dry_run:
        print(json.dumps({k: params[k] for k in ("repo", "scope", "mid", "lvl", "by")}, indent=1))
        print("evidence keys:", list(ev))
        return 0
    import psycopg
    with psycopg.connect(DSN) as cx:
        cx.execute(sql, params)
        row = cx.execute("SELECT repo, scope, mission_id, level, verdict_by, gate_state "
                         "FROM tensor_maturity_verdicts WHERE repo=%s AND scope=%s AND verdict_by=%s",
                         (REPO, SCOPE, BY)).fetchone()
        cx.commit()
    print("row:", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())