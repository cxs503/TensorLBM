"""Board-audit verdict for m21802 (audit of m21672 / CASE-BACKSTEP-FVM-RE100, LBM arm).

Inserts one row into tensor_maturity_verdicts with real pointers only; every
pointer was verified in the audit (see docs/board_audit/m21802/*.md).  The audited
delivery lives on agent/m21672 @ 0fb4c2ebe (not merged), so artefact pointers carry
that commit id; the code/test pointers resolve in main as well.
"""
from __future__ import annotations

import json
import subprocess

PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}

EVIDENCE = {
    "impl_files": [
        "src/tensorlbm/bfs_armaly.py:1-485 (485 lines, 21 functions, sha256 fbbd9601eb9dcb963351a8ab9ecd8e72c8853a2e7ba6f825478a46c008d65702; real D2Q9 TRT/BGK BFS solver + wall-shear observables, no placeholder) @0fb4c2ebe",
        "benchmarks/case_pool/backward_step_armaly/run_case.py:1-108 (CLI entry, GRID of 9 preset points) @0fb4c2ebe",
    ],
    "entry": "PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run_case.py --preset h1_re100 --steps 200000 --avg-from 0.5 --steady-tol 5e-3  [auditor executed the same entry as --smoke: rc=0, 400 steps, wrote result.json + wall_shear.csv(15KB) + field.npz(19KB), X1/s=2.0761]",
    "run_artifacts": [
        "artifacts/case_backward_step_armaly/h1_re100/result.partial.json @0fb4c2ebe (partial=true, steps_run=20000 of n_steps=200000, steady_at_step missing => NOT a benchmark point)",
        "artifacts/case_backward_step_armaly/h1_re100_partial/history.json @0fb4c2ebe (X1/s_instant 2.7649/2.9680/2.9420/2.9037/2.9298; max_speed still drifting 0.15567->0.15311)",
        "artifacts/case_backward_step_armaly/h1_re100_partial/partial_manifest.json @0fb4c2ebe (kind=partial, not_a_finished_point=true, provenance job cj3613 / node n330)",
        "NOT in delivery: converged 88000-step result.json exists only on unmerged branch agent/m17034 @f3a211d78 (steady_at_step=88000, X1/s=2.92021, mass_flux_rel_error=8.93e-5); git merge-base --is-ancestor f3a211d78 0fb4c2ebe => false",
    ],
    "tests": "tests/test_case_pool_backward_step_armaly.py @0fb4c2ebe (171 lines): auditor ran PYTHONPATH=src pytest -q => 7 passed, 2 skipped; the 2 skips ARE the benchmark assertions (test_committed_benchmark_meets_reference_tolerance :110 requires artifacts/.../g2_re100/result.json; :163 'no committed Re=100 artefact yet'). L1 invariants asserted: ER exact 101/52 (:43), parabolic inlet row-mean exact (:67), quadratic wall-shear derivative 1e-12 (:80), zero-crossing interpolation (:89)",
    "integration_callers": [
        "src/tensorlbm/backward_facing_step.py:78 'from .bfs_armaly import reattachment_from_wall_shear, wall_shear_from_near_wall' and :482 metadata['final_reattachment_xr_star_wallshear'] = reattachment_from_wall_shear(...) [present in main]",
        "benchmarks/case_pool/run_pool.py:41-44 run_case() imports 'case_pool.<name>.run_case'; auditor ran 'run_pool.py --list' => backward_step_armaly listed (discovery real). CAVEAT found+fixed on branch agent/m21802: the documented '--case' invocation raised ModuleNotFoundError: case_pool (benchmarks/ never on sys.path) and TypeError for the 3 zero-arg-main cases",
    ],
    "benchmark": "NO ROW: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BACKSTEP-FVM-RE100' => 0 rows (140 rows total / 60 TensorLBM; fuzzy %BACKSTEP%/%backward%/%armaly%/%reattach% also 0). Case-local machine verdict artifacts/case_backward_step_armaly/verdict.json @0fb4c2ebe => runs=[] n_runs=0 n_pass=0 max_abs_error_pct=null all_judged_pass=false. Only number available: transient X1/s=2.9298 vs Armaly 3.05 = -3.94% (and +1.80% vs Erturk 2008 2.878) => inside 15% band but NOT a benchmark point",
    "repro_cmd": "PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run_case.py --preset h1_re100 --steps 200000 --avg-from 0.5 --steady-tol 5e-3 && ... --preset g1_re100 --steps 100000 ... && PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/verdict.py --figures --maturity_evidence? : entry verified executable by the auditor (--smoke rc=0) but the declared primary point NEVER completed (cj3613 stopped at 20000/200000 steps => partial only). NOT RUN by the auditor: the 88k-step convergence rerun (audit ticket is 交差核验, not redo; ~21 min/job projected from cj3613's 287 s per 20000 steps, queue depth 60-84 at close)",
    "external_anchor": "benchmarks/case_pool/backward_step_armaly/refs.json: tier=SECONDARY_IN_REPO, acquisition_status=PRIMARY_PDF_CLOSED (Armaly 1983 JFM 127:473 primary table not machine-readable from this sandbox; 4 fetch attempts logged in-file), adopted Re=100 -> X1/s=3.05 (experimental digitisation), tolerance 15%; cross-check Erturk 2008 Comput.Fluids 37(6):633 Table 1 -> 2.878 (band 2.792-2.965), i.e. 6.0% irreducible 2D-vs-experiment spread at this point",
    "audit": {
        "mission": 21802,
        "audited_delivery": "agent/m21672 @0fb4c2ebe",
        "level": 3,
        "why_not_4": "F1 no converged point in delivery (steady_at_step null, all_judged_pass=false) + F2 anchor not registered in tensor_benchmark_cases (no referee)",
        "why_not_2": "L3 callers real: library BFS solver imports the observables (backward_facing_step.py:78/:482) and the pool driver discovers the case (run_pool.py --list)",
        "findings": {
            "F3_overstated": "docs/CASE_BACKWARD_STEP_LBM_ROUND3.md still claims 'PASS X1/s=2.920 (-4.3%)' and points at artifacts/.../h1_re100/{result.json,wall_shear.csv,field.npz} which do NOT exist at 0fb4c2ebe; the number comes from unmerged f3a211d78. No SUPERSEDED marker => stale PASS claim in the delivered tree",
            "F5_repo_defect_fixed_on_m21802": "benchmarks/case_pool/run_pool.py: (1) ModuleNotFoundError case_pool under its own documented invocation for 13/13 cases, (2) TypeError for main() without argv (couette_2d, karman_re100, mcmp_laplace), (3) after (1)+(2) fixed, zero-arg mains still parsed the driver's argv. Patched + tests/test_case_pool_run_pool.py (7 passed)",
            "F6_minor": "src/tensorlbm/bfs_armaly.py calls macroscopic(f) twice for the same field (redundant, not a correctness bug); left untouched",
        },
        "report_check": "report id 19655 (md 4626 chars, 1 figure, 6 sections) explicitly states WITHHELD / 'No verdict exists' => no overstatement in the report itself",
        "not_run": ["88k-step convergence rerun (audit does not redo compute)", "g1_re100 exact-ER grid", "grid-convergence study", "TensorFVM arm of the anchor slug", "registering the referee row (left to an independent ticket)"],
        "disposition": "L3 confirmed-grade; request changes for L4: converged result.json committed + referee row registered + verdict.py all_judged_pass=true + SUPERSEDED header on round-3 doc",
        "evidence_doc": "docs/board_audit/m21802/CASE-BACKSTEP-FVM-RE100_m21672_audit.md",
    },
}


def main() -> int:
    payload = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
    sql = (
        "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
        "VALUES ('TensorLBM', 'm21672 CASE-BACKSTEP-FVM-RE100', 21802, 3, "
        f"'{payload}'::jsonb, 'board-audit') RETURNING id, level, gate_state;"
    )
    r = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-F", "|", "-c", sql],
        capture_output=True, text=True, env=PGENV,
    )
    print("rc", r.returncode)
    print(r.stdout.strip() or r.stderr.strip()[:800])
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())