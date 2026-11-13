#!/usr/bin/env python3
"""m18779 board-audit 落表: hand-over verdict for m18749 -> tensor_maturity_verdicts.

Every evidence value below is a real pointer produced by collect_evidence.py
(verify_output.json), not an inference. Idempotent on (repo, scope, verdict_by).
"""
from __future__ import annotations

import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
W = pathlib.Path("/root/tp-agent/work/m18779")
D = json.loads((HERE / "verify_output.json").read_text())
DB = "tensorplatform"
SCOPE = "m18749 XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES"
MISSION = 18779
LEVEL = D["D_level_gates"]["assigned_level"]

a, b, c, t = D["A_delivered_files"], D["B_benchmark_cases"], D["C_repro_rerun"], D["D_tests"]

evidence = {
    "impl_files": [
        f"TensorLBM/src/tensorlbm/ibm.py 1-{a['ibm.py']['lines']} sha256={a['ibm.py']['sha256'][:16]}… "
        "(peskin_delta_4pt:22, build_stencil_2d:35, spread_force_2d:72, interpolate_velocity_2d:102, "
        "direct_forcing_static:129, build_cylinder_markers_2d:179) — untracked in m18749 worktree, not on branch history",
        f"TensorLBM/src/tensorlbm/verification/cylinder_flow/__init__.py 1-{a['cylinder_flow']['lines']} "
        f"sha256={a['cylinder_flow']['sha256'][:16]}… (CylinderFlowSolver:110, run_re20_validation:301) — untracked",
    ],
    "entry": "TensorLBM/src/tensorlbm/verification/cylinder_flow/__init__.py:301 run_re20_validation(); "
             "declared repro_cmd at xval/XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES/REPORT.md §6.2",
    "run_artifacts": "NONE. xval/XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES/artifacts/ is an empty dir (0 files); "
                     "the artifact declared in docs/oss_solutions/XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES.md "
                     "(artifacts/cylinder_flow_re20.json) does not exist (exists=False). Cluster job cj2024 declared "
                     "as the producer was observed state=queued (never terminal) — no run output was ever delivered.",
    "tests": f"{t['cmd']} -> {t['tail']} (rc={t['returncode']}); invariants: partition of unity 1e-10, "
             "spread-force conservation, uniform-field interpolation exact, zero-velocity->zero-force. "
             "Gap: no test of the coupled solver, no Cd/convergence assertion.",
    "integration_callers": "NONE. grep -rn 'ibm' over TensorLBM/src (excluding ibm.py, verification/cylinder_flow) "
                           "and TensorLBM/app -> 0 hits; production solver.py / boundary/ has no IBM obstacle hook "
                           "=> L3 not met.",
    "benchmark": f"tensor_benchmark_cases rows for this anchor_slug = {b['n_rows']} "
                 f"(table total {b['n_rows_table']}); platform /tensor/bench/l4-verdict/18749 = "
                 f"{D['E_cross_checks']['platform_l4_verdict_mission_18749']['status']} "
                 f"(failed_checks={D['E_cross_checks']['platform_l4_verdict_mission_18749']['failed_checks']}). "
                 f"Self-declared judge: Cd_ref=2.05 (Dennis & Chang 1970), tol 15%. Auditor re-run of the delivered "
                 f"repro_cmd: Cd_computed={c.get('Cd_computed'):.4f}, rel_error={c.get('rel_error_pct'):.1f}% "
                 f"(>15% => FAIL); Cd history sign-changing, second-half mean {c.get('second_half_Cd_mean')} "
                 f"std {c.get('second_half_Cd_pstd')}, predeclared stopping rule ('Cd variation < 0.01 over last "
                 f"2000 steps') met={c.get('stopping_rule_met')}.",
    "repro_cmd": f"EXECUTED (rc=0, {c.get('wall_sec')}s, 8000 steps): "
                 "cd /root/tp-agent/work/m18779 && PYTHONPATH=TensorLBM/src python3 "
                 "docs/board_audit/m18779/rerun_re20.py -> artifacts/m18779/xval_audit_re20.json "
                 f"sha256={c.get('sha256')[:16]}…, stdout artifacts/m18779/repro_stdout.log; "
                 "figures docs/board_audit/m18779/figures/fig1_cd_history.png, fig2_field_snapshot.png",
    "external_anchor": "Dennis & Chang (1970) J. Fluid Mech. 42:471 — Cd(Re=20)=2.05, Lr/D=0.93; "
                       "cited by the mission itself (REPORT.md §1, CYLINDER_CD_2D[20]). Not an independent "
                       "verification source here: only the mission's own citation, no platform-registered case.",
    "report_six_sections": D["D_report_six_sections"],
    "level_rationale": (
        "L0/L1/L2 met (docs + 8/8 unit tests + real operator code executing end-to-end, 8000 steps in "
        f"{c.get('wall_sec')}s). L3 NOT met: no caller outside the verification module. L4 NOT met: zero run "
        "artifacts, zero registered benchmark rows, and the auditor's faithful re-run of the delivered "
        f"repro_cmd yields {c.get('rel_error_pct'):.1f}% error against the mission's own 15% criterion — the "
        "reported 'expected Cd ~2.0-2.3 within 15%' is a prediction that is falsified. Level = 2 (functional "
        "implementation, validation absent/failing)."),
    "audit_artifacts": [
        "docs/board_audit/m18779/verify_output.json (machine evidence bundle)",
        "docs/board_audit/m18779/audit_report.md (six-section report)",
        "docs/board_audit/m18779/collect_evidence.py / make_figures.py / rerun_re20.py / field_snapshot.py",
        "artifacts/m18779/xval_audit_re20.json, artifacts/m18779/repro_stdout.log, artifacts/m18779/field_1500steps.npz",
    ],
}

sql = """
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', $1$, $2$, $3$, $4$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
      evidence = EXCLUDED.evidence, created_at = now()
RETURNING id, repo, scope, mission_id, level, verdict_by, gate_state, gate_level;
""".replace("$1$", "'" + SCOPE.replace("'", "''") + "'").replace("$2$", str(MISSION)) \
   .replace("$3$", str(LEVEL)).replace("$4$", "'" + json.dumps(evidence, ensure_ascii=False).replace("'", "''") + "'")

r = subprocess.run(["psql", "-U", "foundation", DB, "-At", "-F", "|", "-c", sql],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
if r.returncode != 0:
    raise SystemExit(r.returncode)

q = subprocess.run(["psql", "-U", "foundation", DB, "-At", "-F", "|", "-c",
                    "SELECT id, level, gate_state, gate_level, left(gate_note::text,300) "
                    f"FROM tensor_maturity_verdicts WHERE repo='TensorLBM' AND scope='{SCOPE}' AND verdict_by='board-audit';"],
                   capture_output=True, text=True)
print("ROW:", q.stdout.strip())
(HERE / "insert_result.txt").write_text(q.stdout)