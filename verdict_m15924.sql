-- m15924 交差审计 verdict: m15519 CASE-BACKWARD-STEP-LBM (TensorLBM)
-- evidence 全部为真实指针；数字来源 artifacts/m15924_audit/audit_matrix.json（程序化重算）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15519 CASE-BACKWARD-STEP-LBM',
  15924,
  2,
  $json${
    "impl_files": [
      "/root/tp-agent/work/m15519/benchmarks/case_pool/backward_step_armaly/run.py:1-193 (real scalar BFS loop: build_solid:44-50, inlet_profile:53-62, half-way BB:130-134, Zou/He in/out:136-137, sponge:139-140, time-average window:120-143, graded observable x1_over_s_row1:162+175, shear observable:157-158+174, printed RESULT:188)",
      "/root/tp-agent/work/m15519/benchmarks/case_pool/backward_step_armaly/refs.json:1-31 (reference provenance tier SECONDARY_IN_REPO + tolerance_pct=15.0)"
    ],
    "entry": "python3 benchmarks/case_pool/backward_step_armaly/run.py --re 100 --m 16 --steps 12000 --u 0.1   (run.py:83-100 argparse defaults re=100 m=16 u=0.05 steps=60000)",
    "run_artifacts": [
      "/root/tp-agent/work/m15519/runs/case_backward_step_armaly/re100_m16.json",
      "/root/tp-agent/work/m15519/runs/case_backward_step_armaly/re100_m8.json",
      "/root/tp-agent/work/m15519/runs/case_backward_step_armaly/re200_m16.json",
      "/root/tp-agent/work/m15519/runs/case_backward_step_armaly/re200_m8.json",
      "/root/tp-agent/work/m15519/runs/case_backward_step_armaly/re400_m16.json",
      "/root/tp-agent/work/m15519/runs/case_backward_step_armaly/re400_m8.json",
      "/root/tp-agent/work/m15519/runs/case_backward_step_armaly/results_table.json",
      "/root/tp-agent/work/m15924/artifacts/m15924_audit/repro_check.json (audit re-run of repro_cmd: rc=0, 71.9 s, row1 rel-diff 0.000%, shear 0.038%)",
      "/root/tp-agent/work/m15924/artifacts/m15924_audit/metric_re100_m16.json (5-operator probe, field fidelity check 4e-10)",
      "/root/tp-agent/work/m15924/artifacts/m15924_audit/metric_re200_m16.json",
      "/root/tp-agent/work/m15924/artifacts/m15924_audit/audit_matrix.json (programmatic re-derivation of every number)",
      "/root/tp-agent/work/m15924/artifacts/m15924_audit/fig1_operator_spread.png",
      "/root/tp-agent/work/m15924/artifacts/m15924_audit/fig2_error_and_rho.png"
    ],
    "run_artifacts_committed": false,
    "run_artifacts_note": "the 13 runs/*.json live only in worktree m15519: git ls-tree -r b63efb0e shows just run.py/refs.json/docs/CASE_BACKWARD_STEP_LBM.md/figures/CASE_BACKWARD_STEP_LBM.png",
    "tests": "none: no test references backward_step_armaly (grep -rln over tests/ tools/ scripts/ = 0 hits); run.py asserts no invariant (no convergence-window criterion, no rho-spread gate, no isfinite gate) -> Re=400 silently emitted a non-physical 0.343 instead of aborting",
    "integration_callers": "none: standalone script merged into main (0ac198e4 'Merge #1831 (m15519) via merge-master') but nothing imports it and the bench harness never calls it -> L3 not met",
    "benchmark": {
      "registered_case": "none: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BACKWARD-STEP-LBM' -> 0 rows (2026-11-12); 25 anchor_slugs in ledger, no backward-step anchor -> sole referee silent",
      "ref_tier": "SECONDARY_IN_REPO: values 3.0/5.5/8.0 @ Re=100/200/400 hand-entered by an earlier campaign; Armaly primary PDF unreachable (mission-declared PRIMARY_PDF_UNREACHABLE)",
      "delivered_ref_vs_got": {
        "Re100_S16_row1": {"got": 2.6931608, "ref": 3.0, "tol_pct": 15.0, "rel_err_pct": -10.228, "audit_recompute": "match"},
        "Re100_S8_row1": {"got": 2.4705546, "ref": 3.0, "rel_err_pct": -17.648, "verdict": "FAIL"},
        "Re200_S16_row1": {"got": 4.689232, "ref": 5.5, "rel_err_pct": -14.741, "verdict": "PASS at the line"},
        "Re200_S8_row1": {"got": 4.197563, "ref": 5.5, "rel_err_pct": -23.681, "verdict": "FAIL"},
        "Re400_S16_row1": {"got": 0.34298867, "ref": 8.0, "rel_err_pct": -95.713, "verdict": "INVALID (rho 0.730-1.485, max_speed 0.230)"},
        "Re400_S8_row1": {"got": null, "ref": 8.0, "verdict": "NaN, no crossing"}
      },
      "same_field_shear_operator": {"Re100_S16": 1.88351, "Re200_S16": 3.32398, "err_pct": [-37.22, -39.56], "verdict": "0/2 PASS"},
      "caliber_conflict": "reference chain benchmarks/pending/bstep_3d/result_H40.json::reference.re_definition = 'Re = U_max * step_h / nu (最大入口速度, 台阶高)' but run.py:97-99 uses Re = ubar*H/nu with H=2S and U_max=1.5*ubar -> Re_ref = 0.75*Re_sim (25% offset); refs.json meanwhile claims the Armaly ubar*D_h/nu caliber. One of the two must be wrong and the delivery never resolved or disclosed it."
    },
    "repro_cmd": "cd /root/tp-agent/work/m15924 && python3 benchmarks/case_pool/backward_step_armaly/run.py --re 100 --m 16 --steps 12000 --u 0.1 --out /tmp/m15924/re100_m16_rerun.json  # AUDIT EXECUTED rc=0 in 71.9 s: row1 2.6931608 == committed (0.000%), shear 1.88422 vs 1.88351 (0.038%) -> deterministic",
    "external_anchor": "Armaly, Durst, Pereira & Schoenung, JFM 127(4):473-496 (1983) — primary table UNREACHABLE offline (mission self-declared); machine-verified surrogate arXiv:2507.16509 p.2 confirms ER=1.942, Re on inlet-channel hydraulic diameter, 2D numerics valid to Re_D<=400 -> per-value external anchor WITHHELD, 3.0/5.5/8.0 not traced to the paper",
    "metric_operator_spread": {
      "method": "artifacts/m15924_audit/metric_sensitivity.py — same numerics, 5 operators on the time-averaged field (K-row shear reproduces run.py result.x1_over_s to 4e-10 -> field identity proven)",
      "Re100_S16": {"x1_shear_krow": 1.8835, "x1_row1": 2.6913, "x1_row2": 2.6297, "x1_row3": 2.5274, "x1_parab_fit": 2.8280, "err_pct_range": [-37.2, -5.7], "band_pct_of_ref": 31.5},
      "Re200_S16": {"x1_shear_krow": 3.3240, "x1_row1": 4.6819, "x1_row2": 4.5827, "x1_row3": 4.4139, "x1_parab_fit": 4.9004, "err_pct_range": [-39.6, -10.9], "band_pct_of_ref": 28.7},
      "implication": "operator band 29-32% >> 15% tolerance: the two PASS rows are operator-selected, not physics-robust; run.py's own documented observable and its printed RESULT line are the shear one (1.884/3.324), which is 42.9% away from the graded row1 value at Re=100"
    },
    "findings": [
      "F1 HIGH: Reynolds-caliber contradiction — reference inheritance file declares Re=U_max*S/nu while run.py solves at Re=ubar*H/nu (ratio 4/3); every graded row compares at ~25% lower Reynolds number, and the direction of the systematic under-prediction (-10..-24%) is exactly what that offset produces. Undetected and undisclosed by the delivery.",
      "F2 HIGH (structural): tensor_benchmark_cases has 0 rows for anchor CASE-BACKWARD-STEP-LBM -> metric+tolerance exist only in prose (refs.json tolerance_pct=15.0); sole machine referee silent -> L4 cannot be granted by rule",
      "F3 HIGH->MEDIUM: graded observable != documented observable. run.py:14-16 docstring and run.py:188 RESULT line report the K-row shear slope; the delivered table grades x1_over_s_row1, which is a SINGLE FINAL-SNAPSHOT row-1 crossing (run.py:162) that bypasses the mission's own time-average machinery. Reproducing from the printed line disagrees with the delivered table by 42.9%.",
      "F4 MEDIUM: no invariants asserted — no convergence-window criterion, no rho-spread gate (Re400_S16 gave rho 0.730-1.485, max_speed 0.230 vs ubar 0.1), no isfinite gate; a non-physical 0.343 was emitted as a number instead of aborting",
      "F5 MEDIUM: provenance/durability — runs/*.json (13 files) uncommitted; no results_table generator script anywhere in the repo although the result summary claims 'programmatic injection' (numbers themselves DO recompute exactly: 6/6 match); graded field serialized as JSON string \"2.6931608\" (numpy.float32 + default=str) -> downstream arithmetic breaks",
      "F6 LOW: inherited chain hygiene — benchmarks/pending/bstep_3d/result_re400_H49.json::reference.source repeats the Re=200 text ('Re=200 X_r/h≈5.5') while xr_h_ref=8.0 (copy-paste); the 8.0 value has no self-consistent source in the chain",
      "F7 INFO: mission was honest about the two biggest holes it could see (PRIMARY_PDF_UNREACHABLE, Re=400 pseudo-crossing, sandbox process reaping) — this is the reason the verdict is L2-with-request-changes and not a fabrication finding"
    ],
    "positive": [
      "repro_cmd executes and is deterministic (row1 0.000%, shear 0.038%) — artifacts are genuinely produced by the delivered entry",
      "all 6 delivered rows recompute exactly from the raw JSONs -> numbers were not hand-typed",
      "mesh-sensitivity statement (S=8->16 error -17.6%->-10.2%) is supported by its own artifacts",
      "figure figures/CASE_BACKWARD_STEP_LBM.png (80 362 B) and report tensor_reports id=14624 exist"
    ],
    "level_rationale": "L2: real input->compute->output implementation (run.py, library kernels only, 193 lines) + genuine, deterministic, recomputable run artifacts + merged into main. L3 NOT met: no integration caller, no test, nothing imports the module. L4 NOT met for two independent reasons: (a) the sole referee (tensor_benchmark_cases) has 0 rows for this anchor; (b) the reference caliber contradicts the delivered caliber (F1) and only 2/6 rows land inside 15%, both on an operator-selected proxy whose 5-operator band (29-32% of ref) is twice the tolerance. L5 no external anchor (Armaly table WITHHELD).",
    "not_run": [
      "Did not obtain Armaly (1983) Table 1 (offline; DOI paywalled) -> per-value external anchoring WITHHELD, not FAIL",
      "Did not re-run Re=200/400 converged cases (60000+ steps) — out of scope for a verification ticket",
      "Did not modify the audited worktree m15519 (read-only); the operator probe imports its run.py helpers untouched and writes only into /root/tp-agent/work/m15924/artifacts",
      "Pi_xy non-equilibrium momentum-flux operator returns a corner artifact (0.0315 S at both Re) -> declared unusable in this probe, excluded from the band statistics rather than tuned"
    ],
    "scope_note": "Audited m15519 (succeeded, auto-approve, result.head_commit=null, PR #1831 merged as 0ac198e4), worktree /root/tp-agent/work/m15519 commit b63efb0e. Cross-checked: tensor_benchmark_cases (25 anchors, 0 backward-step), tensor_reports id=14624, git ls-tree/main ancestry, 6 raw run JSONs, refs.json provenance chain (3 pending/bstep_3d files). Audit artifacts: /root/tp-agent/work/m15924/artifacts/m15924_audit/*, report docs/audit_m15924.md."
  }$json$::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      mission_id = EXCLUDED.mission_id,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL
RETURNING id, repo, scope, level, gate_state;