-- m21704 board audit verdict for mission m21640
-- anchor: XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM (TensorLBM)
-- full report: docs/board_audit/m21704/VERDICT-M21640.md
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
 'TensorLBM',
 'm21640 XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM',
 21704,
 4,
 $json${
  "impl_files": [
   "TensorLBM/src/tensorlbm/verification/layered_poiseuille/two_component_sc.py:1-401 (TwoComponentSCConfig/TwoComponentSCSolver, MCMP D2Q9, EDM at Martys-Chen common velocity)",
   "TensorLBM/src/tensorlbm/multiphase/__init__.py:492 sc_two_component_force (shipped library kernel)",
   "TensorLBM/src/tensorlbm/multiphase/__init__.py:558 collide_edm_common (shipped library kernel)",
   "TensorLBM/src/tensorlbm/multiphase/__init__.py:470 _wall_extend (mirror/dry wall stencil)",
   "TensorLBM/src/tensorlbm/verification/layered_poiseuille/analytic.py:120 solve_layered (exact 2Nx2N LU reference, solver-independent)",
   "TensorLBM/src/tensorlbm/verification/layered_poiseuille/metrics.py:1-191"
  ],
  "entry": [
   "TensorLBM/src/tensorlbm/verification/layered_poiseuille/run_two_component_sc_validation.py:238-409 (python -m tensorlbm.verification.layered_poiseuille.run_two_component_sc_validation --matrix/--case/--figure-only; writes results.json+sha256.txt+png, gate verdicts, exit 1 when all_gates_pass=false)"
  ],
  "run_artifacts": [
   "TensorLBM/artifacts/xval_two_component_sc_layered_poiseuille_edm/results.json (28303 B, sha256 c902d46f6f51c5e82eb36539e30273e1e116b4ba7312ae1f5c64961e1f41ce8f, matches sha256.txt)",
   "TensorLBM/artifacts/xval_two_component_sc_layered_poiseuille_edm/two_component_sc_layered.png (155418 B)",
   "TensorLBM/artifacts/audit_m21704/results_rerun_all6.json (this audit: 6/6 cases rerun on merged main 15ef2a4c0, metrics bitwise identical to the artifact for the 4 delivered cases)",
   "TensorLBM/artifacts/audit_m21704/static_kpi_vacuity.json (F2 counter-probe)",
   "TensorLBM/artifacts/audit_m21704/fig1_layered_profile_vs_exact.png",
   "TensorLBM/artifacts/audit_m21704/fig2_kpi_errors.png"
  ],
  "tests": "TensorLBM/tests/test_two_component_sc_layered_poiseuille.py (10 tests); audit rerun at main 15ef2a4c0: 10 passed in 58.23 s. Weaknesses: test_layered_poiseuille_within_15pct:144 asserts min(nominal, interface_matched) (best-of-two, looser than the pre-registered nominal KPI); test_static_interface_has_no_spurious_flow:121 is structurally vacuous (x-uniform fields).",
  "integration_callers": "TensorLBM/src/tensorlbm/verification/layered_poiseuille/two_component_sc.py:57-58 imports sc_two_component_force/collide_edm_common from tensorlbm.multiphase and calls them at :231 and :273 (shipped library kernels driven by the verification solver); tests/test_two_component_sc_layered_poiseuille.py:17 imports the solver. No caller outside the verification harness: the two-component SC kernels are library-module wiring, not yet consumed by the main LBM solver path.",
  "benchmark": "tensor_benchmark_cases id=149 (anchor_slug=XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM, status=passed, owner gate 15%). got vs exact layered solution, layered_r167 H=32 nx=1 G=2.5 g_x=5e-5 12000 steps: u_peak_err 7.44%, eps_inf 8.84%, eps_l2_rel 8.23%, Q_err 8.16%; controls eps_inf 0.3675% (single) and 0.3673% (mixture) <= 3% self target. Ref vs got within tolerance for all four 15% KPIs. CAVEAT: the artifact's own pre-registered subgate interface_position<=5%H is FALSE (y_iface 19.21 vs nominal 16.0 = 10.02%H) and summary.all_gates_pass=false, which the ledger row does not record. Audit-run withheld cases: layered_r167_h48 all gates PASS (interface 0.52%H, u_peak 0.07%, eps_inf 4.87%, eps_l2 3.82%, Q 3.36%); layered_r233 eps_inf 13.80% (1.2 pp margin), interface 8.22%H FAIL.",
  "repro_cmd": "cd TensorLBM && PYTHONPATH=src python -m tensorlbm.verification.layered_poiseuille.run_two_component_sc_validation --matrix core --out artifacts/xval_two_component_sc_layered_poiseuille_edm  (documented at docs/XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM-STATUS.md:90-104; executable, exits 1 by design while all_gates_pass=false). Verified in this audit: per-case rerun on main 15ef2a4c0 reproduces every metric bitwise (u_peak_err 0.07437226515465295, eps_inf 0.08837783187947695, eps_l2_rel 0.08234053870518139, Q_err 0.08159435994696285, y_iface 19.207644419279987) and the full ux_total profile array is identical.",
  "external_anchor": "NONE. The reference is an analytic exact piecewise-quadratic solution (L2/L3/L5 anchor internally re-verified: u(0)=u(H)=0, per-layer u''=-g/nu to 0.3%, u and nu*u' continuity, peak 0.0490 < single-fluid 0.064). No second independent LBM/CFD code and no experiment were compared, so no cross-code external anchor and no L5.",
  "findings": {
   "F1_major": "ledger row status=passed while the delivery's own pre-registered subgate interface_position<=5%H = false (10.02%H) and summary.all_gates_pass=false; ledger metric/tolerance text omits the failing subgate. Author disclosed the FAIL in STATUS.md section 3/4 and in the mission result.",
   "F2_major": "layered_static spurious-current KPI is structurally zero: nx=1 and x-uniform fields make F_x cancel pairwise, so u_x==0 bitwise and the gate cannot fail. Counter-probe: nx=8 uniform -> 0.0 exactly; 1e-3 noise -> 9.1e-14; noise 0.2 -> 5.6e-10. The registered 'spurious current of the static interface' KPI therefore carries no information.",
   "F3_medium": "control figure 0.25% (commit message, STATUS section 3, ledger tolerance text) is not reproducible; artifact and rerun both give 0.3675% (single) / 0.3673% (mixture). No pass/fail change, but violates programmatic transcription.",
   "F4_medium": "artifact environment.git_commit=1e5e0891b (parent; tree dirty at run time) while the artifact was committed in 8a8d86d60; environment.argv records only the last --case although records were merged from 4 invocations. Reproducibility itself proven bitwise (see repro_cmd).",
   "F5_medium": "results.json contains 3 bare NaN tokens (control_single y_iface, layered_static eps_l2_rel, ...): not RFC-8259 JSON; strict parsers and PostgreSQL jsonb reject it.",
   "F6_minor": "test_layered_poiseuille_within_15pct uses min(nominal, interface_matched) reference selection; test_static_interface_has_no_spurious_flow inherits F2 vacuity.",
   "F7_minor": "STATUS.md section 3 hand-copied u_peak absolute columns (6.30e-2/6.79e-2) do not match the artifact (4.530e-2/4.894e-2); doc claims '11 physical assertions' while the file has 10 tests.",
   "F8_minor": "2 of 6 pre-registered cases (layered_r233, layered_r167_h48) were not run at delivery (honestly declared); this audit ran them, see benchmark."
  },
  "not_checked": [
   "no external/experimental anchor (analytic reference only)",
   "no formal grid-convergence study (only H=32 vs H=48, no order estimate)",
   "spurious currents not actually measured (F2); no 3D, no curved interface, no large density ratio",
   "CPU/float64/torch path only; no GPU or parallel consistency check",
   "STATUS.md claim 'wall_mode=dry raises eps_inf from 6.8% to 14.4%' has no artifact in the repo and was not re-verified",
   "tensorlbm.multiphase.step_sc single-component production path not audited"
  ],
  "level_note": "L4 by functional substance: real MCMP D2Q9 + EDM solver driving shipped library kernels (L2/L3), one complete registered case vs an independent exact reference with 4 KPIs at 7.4-8.8% (owner 15% gate), error analysis incl. layer-wise viscosity fit and interface-matched diagnostic, sha256-stamped artifacts, and bitwise reproducibility on merged main. Not L5: no external code/experiment anchor and no research novelty. F1/F2 do not remove the capability but must be corrected in the ledger (R1-R3 in docs/board_audit/m21704/VERDICT-M21640.md section 7).",
  "verdict": "L4 / approve with notes (3 remediation items: R1 ledger must record the failing 5%H subgate and the H=48 all-gates-pass result; R2 replace the vacuous static spurious-current KPI; R3 fix 0.25%->0.3675%, stale git_commit, bare NaN, and the min() reference in the test)"
 }$json$::jsonb,
 'board-audit'
);