"""Land the board-audit verdict for m15432 (CASE-CYLINDER-DRAG-RE100) into tensor_maturity_verdicts.

Idempotent: ON CONFLICT (repo, scope, verdict_by) updates and resets gate_state to pending.
"""
import json
import subprocess

EVIDENCE = {
    "impl_files": [
        "tests/validation/cylinder_re100_unconfined.py:138 run_cfg() (318 lines, real D2Q9 BGK loop)",
        "tests/validation/cylinder_re100_unconfined.py:166-167 compute_obstacle_forces + far_field_bc_2d call order",
        "tests/validation/make_cylinder_re100_figures.py:42 main()",
    ],
    "entry": "python -m tests.validation.cylinder_re100_unconfined --full --diameter 10 --emit-dir artifacts/cylinder_re100_D10",
    "run_artifacts": [
        "artifacts/cylinder_re100_D10/cylinder_re100_unconfined.json (18000 cd/cl samples + ux/uy/wz fields; audit recompute delta 0.00e+00)",
        "figures/fig_cd_history.png (1120x630, ink 4.7%)",
        "figures/fig_centerline.png (1120x630, ink 2.8%)",
        "figures/fig_vorticity.png (1540x588, ink 29.3%)",
        "audit-side: docs/board-audit-m15432/audit_recompute.json, fig_audit_m15432.png",
    ],
    "tests": [
        "NO automated assert binds this module: tests/validation/cylinder_re100_unconfined.py has no test_* function and no pytest file references it (grep over repo) -> reported values carry no regression test",
        "audit smoke rerun (not the graded config): PYTHONPATH=src python3 -m tests.validation.cylinder_re100_unconfined --pilot -> 22 s, no NaN, Cd=1.8134 (+31.4%), St=0.1655 (+0.9%)",
    ],
    "integration_callers": [
        "tensorlbm.solver:221 collide_bgk_matmul, :423 stream",
        "src/tensorlbm/boundary/boundaries.py:9 cylinder_mask, :40 bounce_back_cells, :164 compute_obstacle_forces (Ladd F=2*sum c_i f_i), :201 far_field_bc_2d",
        "platform registry: tensor_case_pool id=11 CYLINDER-DRAG-RE100 repo=TensorLBM tol_policy='<5%' status=dispatched mission_id=15432",
        "platform report: tensor_reports id=14637 scenario_key=CASE-CYLINDER-DRAG-RE100",
    ],
    "benchmark": {
        "referee_row": "ABSENT: SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-CYLINDER-DRAG-RE100' = 0 (table has 21 rows) -> no machine referee; fell back to tensor_case_pool id=11 tol '<5%' + task-goal ref",
        "ref": {"cd": 1.3796, "cl": 0.0005, "st": 0.164,
                "source": "Schäfer/Rannacher/Turek 1996 as claimed by the mission; NO page/table excerpt, PDF or sha256 archived in the worktree -> unverifiable (finding F4)"},
        "got_D10_250x100_B10": {"cd": 1.7300971896035802, "cl_mean": -8.748044156366002e-05,
                                "cl_amp": 0.4760362207889556, "st": 0.17119452080121134},
        "rel_err_pct": {"cd": 25.405711046939707, "st": 4.386902927567884},
        "tolerance_pct": 5.0,
        "verdict": "CD FAIL (+25.41% >> 5%); St PASS (+4.39%); Cl mean symmetry PASS (|delta|=5.9e-4 vs amp 0.476); Cl amplitude delivered but never compared to a reference (ungraded, report tick unsupported) -> benchmark NOT met",
        "cross_runs_same_repo": {"D10_B16.7_audit_pilot": 1.8134, "D10_B10_audited": 1.7301,
                                 "D16_B7.14_m15407_audited": 1.5445, "literature_band": [1.33, 1.3796],
                                 "note": "monotone decrease with finer grid + lower blockage, but D and B co-vary -> no separated blockage correction, no converged unconfined point"},
    },
    "repro_cmd": "cd /root/tp-agent/work/m15432 && PYTHONPATH=src OMP_NUM_THREADS=1 python3 -m tests.validation.cylinder_re100_unconfined --full --diameter 10 --emit-dir artifacts/cylinder_re100_D10  (342 s recorded). CAVEAT: the delivered command omits PYTHONPATH=src and 'import tensorlbm' then resolves to the editable install /root/tp-agent/work/m13963/src, a different worktree; diff vs the audited src is input-contract validation only, so numerics for tau=0.53 are equivalent (finding F6)",
    "external_anchor": "Schäfer, Rannacher & Turek (1996) - cited only, no archived excerpt; cross-checked against the repo's own prior audited run m15407 CASE-KARMAN-RE100 (knowledge_base id=79) and the literature band 1.33-1.38",
    "audit_artifacts": [
        "docs/board-audit-m15432/AUDIT_m15432_CASE-CYLINDER-DRAG-RE100.md",
        "docs/board-audit-m15432/audit_m15432_recompute.py",
        "docs/board-audit-m15432/audit_recompute.json",
        "docs/board-audit-m15432/make_audit_figures.py",
        "docs/board-audit-m15432/fig_audit_m15432.png",
        "docs/board-audit-m15432/artifacts/audit_pilot_rerun.json",
    ],
    "audit_findings": [
        "F1 blocking: Cd +25.41% >> 5% tolerance -> benchmark FAIL (report already honest-FAIL, no whitewash)",
        "F2 high: delivered geometry has 10% blockage (height 10D, free-stream Dirichlet on lateral faces) while compared to an unbounded reference; no blockage correction discussed",
        "F3 high: no grid-convergence chain - the D=20 leg never ran (cluster cj1159 still queued 55.6 min at audit time); report's 'D=10 已收敛' conflates temporal stationarity with grid convergence",
        "F4 medium: reference provenance not closed (no page/table excerpt, PDF or sha256); attribution of the unbounded numbers to the confined DFG S&T benchmark family is unverified",
        "F5 medium: Cl amplitude 0.476 delivered but not graded; report ticks 'magnitude correct' without any reference (~0.32-0.33 typical for 2D Re=100 => potential +45%)",
        "F6 medium: repro_cmd is not hermetic (editable install resolves to worktree m13963); numerics equivalent for tau=0.53 but the command does not pin the audited code",
        "F7 medium systemic: tensor_benchmark_cases has no row for this anchor (same gap found for CASE-KARMAN-RE100) -> case pool and referee table are out of sync",
        "F8 low: stationarity gate uses 9000-vs-9000 half windows and hides a residual +0.40% creep (first/last 1000-sample means 1.7258 -> 1.7327); immaterial to the 25% verdict",
        "no-fabrication check PASSED: all headline metrics recompute exactly (delta 0.00e+00) from the delivered 18000-sample series and fields",
    ],
    "level_rationale": "L2 = real code input->compute->output (18000 samples + fields + figures, auditor-reproducible). L3 = built exclusively on main-solver primitives (collide_bgk_matmul/stream/bounce_back_cells/compute_obstacle_forces/far_field_bc_2d), registered in tensor_case_pool id=11 and reported to tensor_reports id=14637. L4 NOT earned: the primary benchmark quantity Cd fails by 5x tolerance (+25.41% vs 5%), there is no grid-convergence study (D=20 never executed) and the reference is not provenance-closed. Report volume and honest-FAIL labelling do not change the level.",
    "residual_scope_for_successor": [
        "run a genuinely unconfined configuration (B<=2%, height_d>=50) with D=16/20 grid points and reach Cd within 1.38 +-5%",
        "register CASE-CYLINDER-DRAG-RE100 in tensor_benchmark_cases (case/ref/tolerance) so the gate has a machine referee",
        "archive the reference table page + sha256 (zero-hand-copy discipline)",
        "add Cl amplitude to the graded quantity list (or declare it explicitly ungraded)",
        "make repro_cmd hermetic (PYTHONPATH=src or CI-pinned install)",
    ],
}

SCOPE = "m15432 CASE-CYLINDER-DRAG-RE100"
SQL = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    "VALUES ('TensorLBM', %s, 15712, 3, %s::jsonb, 'board-audit') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=3, mission_id=15712, "
    "evidence=EXCLUDED.evidence, gate_state='pending', gate_level=NULL, gate_note=NULL "
    "RETURNING id, repo, scope, level, gate_state;"
)

if __name__ == "__main__":
    ev = json.dumps(EVIDENCE, ensure_ascii=False)
    script = "\\set ON_ERROR_STOP on\n" + "PREPARE p(text,text) AS " + SQL.replace("%s", "$1") + ";\n"
    path = "/tmp/m15712_verdict.sql"
    with open(path, "w") as fh:
        fh.write(
            "\\set ON_ERROR_STOP on\n"
            "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES\n"
            "('TensorLBM', '" + SCOPE + "', 15712, 3, $ev$" + ev + "$ev$::jsonb, 'board-audit')\n"
            "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=3, mission_id=15712, "
            "evidence=EXCLUDED.evidence, gate_state='pending', gate_level=NULL, gate_note=NULL;\n"
            "SELECT id, repo, scope, level, verdict_by, coalesce(gate_state,'pending'), gate_level "
            "FROM tensor_maturity_verdicts WHERE repo='TensorLBM' AND scope='" + SCOPE + "';\n"
        )
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-f", path],
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    print(r.stderr[-800:])