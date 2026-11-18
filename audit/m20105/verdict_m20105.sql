-- m20105 board-audit verdict on mission m20073 (GAP-DIFFLBM-INVERSE-SHAPE, TensorLBM)
-- Level = L2 (real end-to-end multi-parameter inverse code, independently re-run and tested by
-- the audit). L3 withheld: no in-repo caller / no platform wiring. L4 withheld: no registered
-- benchmark row, key case cj2492 never executed, parameter error grows 12.6% -> 16.4%.
-- Disposition: request changes / reopen. Report: tensor_reports id=18303 (rev2 of 18296).
-- Row id=825 was created by this audit (draft with prose-only pointers) and is rewritten here
-- with machine-checkable file pointers so maturity_gate can re-verify it; level/verdict unchanged.
UPDATE tensor_maturity_verdicts
SET repo = 'TensorLBM',
    scope = 'm20073 GAP-DIFFLBM-INVERSE-SHAPE',
    mission_id = 20105,
    level = 2,
    verdict_by = 'board-audit',
    evidence = jsonb_build_object(
    'impl_files', jsonb_build_array(
      'audit/m20073_external/inverse_shape_multiobs.py',
      'audit/m20073_external/probe_identifiability.py'
    ),
    'entry', 'audit/m20073_external/inverse_shape_multiobs.py',
    'run_artifacts', jsonb_build_array(
      'audit/m20073_external/run5/results.json',
      'audit/m20073_external/run20/results.json',
      'audit/m20073_external/probe/identifiability.json',
      'audit/m20073_external/fig_audit_m20073.png'
    ),
    'tests', jsonb_build_array(
      'audit/m20073_external/test_m20073_audit.py::test_i1_drag_is_blind_to_lateral_centre',
      'audit/m20073_external/test_m20073_audit.py::test_i2_loss_is_flat_in_cy_at_true_radius',
      'audit/m20073_external/test_m20073_audit.py::test_i3_delivered_run_diverges_away_from_truth',
      'audit/m20073_external/test_m20073_audit.py::test_i4_acceptance_line_not_met',
      'audit/m20073_external/test_m20073_audit.py::test_i5_forward_drag_invariance_recomputed'
    ),
    'integration_callers', 'audit/m20073_external/inverse_shape_multiobs.py',
    'integration_note', 'delivered script has NO in-repo caller and is not wired into apps/API/CLI; the only edge it uses is the pre-existing src/tensorlbm/autograd_path.py:877 rollout (obstacle_force :839) and src/tensorlbm/soft_geometry.py:121 SoftGeometry, untouched by m20073 — hence L3 is withheld although the file pointer above greps.',
    'benchmark', 'ref vs got: criterion ">=2 params, rel err <=3%" (mission text, no tensor_benchmark_cases row: anchor GAP-DIFFLBM-INVERSE-SHAPE -> 0 rows of 121, checked 2026-11-14); got 5-iter cy 12.584% r 19.424%, 20-iter cy 16.405% r 5.090% -> error > 3% by 1.7x-6.5x, cy error GROWS',
    'repro_cmd', 'PYTHONPATH=src python audit/m20073_external/inverse_shape_multiobs.py --iters 20 --log-every 4 --out audit/m20073_external/run20 ; PYTHONPATH=src python audit/m20073_external/probe_identifiability.py ; PYTHONPATH=src python -m pytest audit/m20073_external/test_m20073_audit.py -q',
    'external_anchor', 'none — no experimental or literature reference value is defined for this anchor (both the anchor and the delivery are self-referential); L5 not claimed',
    'verdict_basis', 'audit/m20073_external/AUDIT_M20073_INVERSE_SHAPE.md',
    'measured_vs_criterion', jsonb_build_object(
      'criterion', 'recover >=2 geometry parameters with relative error <=3%',
      'got_5iter', jsonb_build_object('cy_rel_err', 0.12584, 'r_rel_err', 0.19424, 'verdict', 'FAIL'),
      'got_20iter', jsonb_build_object('cy_rel_err', 0.16405, 'r_rel_err', 0.05090, 'verdict', 'FAIL — cy error grows 12.50%->16.41% while loss falls to 9.68e-03'),
      'script_success_flag', false
    ),
    'root_cause', 'observable pair is non-identifiable in cy: forward scan at r=2.30 without any optimiser gives |drag rel. change| = 4.44e-08 at cy=9.0 (drag blind) and wake-centre sensitivity ~0.085 lattice/lattice (<1% for dcy=1, below the contractual 1% noise floor); loss is flat to <1e-4 across cy in [7,9] (cy=7.0 -> 8.96e-05, cy=9.0 -> 7.55e-05)',
    'overstated_claims', 'docs/inverse_shape_final_summary.md ticks "Bootstrap CI (20 samples, 1% noise)" and "Achievement of <3% error target" inside a block titled Pending (bootstrap never run; same document reports 12.6%/19.4%); states "Both parameters move in correct direction" although cy moves away from truth; claims full validation needs only a cluster run although the 20-iter re-run shows a wrong attractor',
    'honest_parts', 'the 5-iteration numbers reproduce bit-for-bit (loss 3.290495e-01 -> 1.712428e-01, cy=9.0067, r=2.7468) and the wake-field gradient failure (~1e-9) is documented instead of hidden',
    'process_defects', 'tensor_dev_missions.id=20073 head_commit=NULL and all 5 delivered files untracked (git_status "??"); cluster_jobs.id=2492 (150-iter validation) state=queued, rc=NULL, finished_at=NULL — never ran; no run artifact shipped; server_quality_audit={gate_valid:false, key_present:false, gate_missing_warning:true} yet the mission was auto-approved to succeeded',
    'level_cap_reason', 'L3 needs a real in-repo caller/platform wiring (none); L4 needs a registered benchmark row + executed convergence case + error analysis (none; error diverges); if committed code were required the repo substance would be docs-only (L0/L1) since head_commit=NULL',
    'audit_independence', 'audit re-ran the delivered script from a verbatim copy on the same host and torch build 2.7.1; independence lies in the independent re-run, the optimiser-free forward identifiability scan and the reconstructed criterion, not in a third-party implementation; the 150-iteration cluster rerun and the bootstrap CI rerun were NOT performed',
    'report_id', 18303,
    'disposition', 'request changes / reopen'
  ),
    gate_state = 'pending',
    gate_level = NULL,
    gate_note = NULL
WHERE id = 825;

SELECT id, repo, scope, mission_id, level, verdict_by, gate_state,
       jsonb_array_length(evidence->'impl_files') AS impl_n,
       jsonb_array_length(evidence->'run_artifacts') AS art_n,
       jsonb_array_length(evidence->'tests') AS test_n
FROM tensor_maturity_verdicts WHERE id = 825;
