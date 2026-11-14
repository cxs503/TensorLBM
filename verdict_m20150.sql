-- m20150 board audit verdict on mission 20069 (TensorLBM, GAP-DIFFLBM-ADJOINT-VERDICT)
-- Auditor mission 20150. Every value below is reproduced by
--   python3 docs/board_audit/m20150/verify_m20069.py   (8 checks, ~10 s)
-- -> artifacts/m20150_audit/verify_output.json
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
 'TensorLBM',
 'm20069 GAP-DIFFLBM-ADJOINT-VERDICT',
 20150,
 2,
 jsonb_build_object(
  'impl_files', jsonb_build_array(
    'src/tensorlbm/shape_gradient.py:1-612 (sha256 f6906e112ee0d964…; real forward D2Q9 BGK+Guo, soft NoDynamics skip, periodic streaming, full-way bounce-back blend, Ladd momentum-exchange probe; three gradient sources shape_gradient_through_flow/frozen_field_gradient/ad)',
    'src/tensorlbm/adjoint.py:76 SHAPE_DERIVATIVE_ONLY=True; :95-109 optimization_path_eligible() gate (default reject, requires shape_derivative_only is False AND includes_flow_response is True); :81-93 FROZEN_FIELD_VERDICT literals (sha256 7bee2474e74b8f22…)',
    'benchmarks/adjoint_verdict_diff_lbm.py:1-543 (--mode quick|paper, --ad-tol 0.05; sha256 109415277ec3f051…)',
    'benchmarks/adjoint_verdict_report.py:1-418 (report generator; DEFECT :198 and :294-296 hardcode the headline verdict and the cos ticks as literals; sha256 08d91c6937e2085d…)',
    'docs/GAP_DIFFLBM_ADJOINT_VERDICT.md:1-163 (six sections present; sha256 97801416cb2c6153…)'
  ),
  'entry', 'PYTHONPATH=src python3 benchmarks/adjoint_verdict_diff_lbm.py --mode quick --out artifacts/adjoint_verdict_diff_lbm.json  (shipped smoke entry, auditor-independent artifact); report: python3 benchmarks/adjoint_verdict_report.py --json artifacts/adjoint_verdict_diff_lbm.json --doc docs/GAP_DIFFLBM_ADJOINT_VERDICT.md --figdir figures',
  'run_artifacts', jsonb_build_array(
    'artifacts/adjoint_verdict_diff_lbm.json (mode=quick, generated_at 2026-11-14T08:55:36Z, wall_seconds=57.2; Q_yc050 g_fd=-0.2882637514609104 g_ad=0.15470128056529037 g_frozen=0.7992991787194612 g_ship=0.16199036537916833 ad_rel_err_vs_fd=1.5366657437200126; Q_yc055 g_fd=-0.14155646225169066 g_ad=0.14965600171375068 ad_rel_err_vs_fd=2.057217730177933)',
    'artifacts/adjoint_verdict_diff_lbm/fields_Q_yc050.npz (25133 B) + fields_Q_yc055.npz (33200 B)',
    'figures/adjoint_verdict_diff_lbm_{cos,convergence,trajectory}.png (37261/44916/39104 B)',
    'PAPER-RESOLUTION ARTIFACT ABSENT: cj2485 produced only fields_R1_yc050.npz (238093 B); no mode=paper JSON anywhere; node log /root/tp-agent/cluster/jobs/cj2485.log = rc 255, output_tail "[R1_yc050] steady: 20000 steps, C_L=8.881784e-13, C_D=-4.399347e+01" (nonphysical C_D)'
  ),
  'tests', jsonb_build_array(
    'tests/test_shape_gradient.py (13 passed in 6.46 s, auditor re-run 2026-11-14, PYTHONPATH=src python3 -m pytest -q; sha256 2eafb20c67bb2596…): streaming permutation mass conservation atol 1e-12, disk weight -> hard mask, Poiseuille u_max vs analytic gH^2/(8nu), unrolled AD vs central FD rel<2e-4 at n=30 40x32 D=8 epsilon=0.6 (:111-137), flow-response term non-zero, gate rejects surrogate / accepts flow-response reporter',
    'GAP: no test reads artifacts/adjoint_verdict_diff_lbm.json and no test binds FROZEN_FIELD_VERDICT to a run -> the literal cos=-1.000/ratio=2.289 in adjoint.py cannot be caught (finding F6)'
  ),
  'integration_callers', jsonb_build_array(
    'NONE in src/: grep "shape_gradient|optimization_path_eligible" over src/ (excluding shape_gradient.py itself) returns only doc-comment cross-references in src/tensorlbm/adjoint.py:31,91,231 -> no solver, no opt loop, no platform API imports the new flow-response gradient',
    'call sites are benchmarks/adjoint_verdict_diff_lbm.py:64,179,183,433 and benchmarks/adjoint_verdict_report.py:199,257-259,391 only',
    'doc conclusion #3 itself defers wiring to round 2 ("把该入口接到形状优化环") => L3 not reached'
  ),
  'benchmark', 'tensor_benchmark_cases: 0 rows for anchor_slug=GAP-DIFFLBM-ADJOINT-VERDICT (table total 118 rows, TensorLBM 55; ILIKE %ADJOINT%/%DIFFLBM% matches only GAP-DIFFLBM-3D-COST) -> no machine judge exists, L4 impossible by the board criterion. The mission pre-declared its own thresholds (gate cos>=0.99, fd_steps [1.0,0.5], ad_tol 5%) inside the artifact, not in the ledger. Got vs its own thresholds: AD-vs-FD 153.67% (Q_yc050) and 205.72% (Q_yc055) both fail the 5% tolerance by 30-41x; cos(ship,AD)=1.000/0.996 and cos(frozen,AD)=1.000/0.980 satisfy >=0.99 in 3 of 4 rows, contradicting the report headline that hardcodes "NO" (report generator :198, ticks :294-296).',
  'repro_cmd', 'python3 -m pytest tests/test_shape_gradient.py -q -> 13 passed, 6.46 s (auditor, exit 0). Independent re-run of the shipped quick entry at DEFAULT settings (compute gate job cj2519, node n330, 2 threads, 293.4 s, log artifacts/m20150_audit/cj2519_repro.log): 6000 steps instead of 700, FD reference -0.002986389/-0.003196747, cos(ship,AD) -1.000/-0.7585 => the shipped JSON is NOT reproducible at default settings and requires the undocumented --max-steps 700 override (finding F0). The shipped repro_cmd (--mode paper --max-steps 20000 --tol 1e-9) is itself the failing path: cj2485 node rc=255 after the R1 base state, C_D=-43.99.',
  'external_anchor', 'Poiseuille analytic u(y)=g/(2nu)(y-y0)(y1-y), u_max=gH^2/(8nu) — enforced by tests/test_shape_gradient.py::test_poiseuille_profile_matches_analytic_maximum (passes, ~1e-12). Literature basis cited but NOT numerically reproduced: Pironneau JFM 59 (1973) 97; Mohammadi & Pironneau (2010) ch.2-3; Ataei & Salehipour CPC 300 (2024) 109187 (XLB); Ladd JFM 271 (1994). No external C_L reference exists for disk-in-channel Re=15, so the gradient arbitration is self-referential central FD on a non-converged state (residual 1.00 in the artifact base block) -> external_anchor WITHHELD at the verdict level.',
  'verdict_basis', jsonb_build_object(
    'L0', 'PASS: docs/GAP_DIFFLBM_ADJOINT_VERDICT.md (163 lines, six sections) + 3 figures',
    'L1', 'PASS: 13 pytest invariants re-run green (6.46 s) incl. analytic Poiseuille, 1e-12 mass conservation, AD-vs-FD 2e-4 on a 30-step window',
    'L2', 'PASS: shape_gradient.py is genuine code (input->forward LBM->gradient->dict); adjoint gate is genuine logic with default reject',
    'L3', 'FAIL: 0 src/ callers, gate not called by any optimisation loop',
    'L4', 'FAIL: 0 benchmark rows for the anchor; paper campaign failed at R1 with nonphysical C_D=-43.99 and node rc=255; AD-vs-FD 153-206% vs 5% tolerance; headline criterion hardcoded',
    'L5', 'FAIL: no external numerical anchor, no integration, no unsteady extension'
  ),
  'findings', jsonb_build_array(
    'F0 blocking: the shipped artifact reproduces only with the undocumented --max-steps 700 (residual=1.00, not converged); the shipped entry at default settings (cluster cj2519, node n330, 2 threads, 293.4 s, 6000 steps) gives FD reference -0.002986389 / -0.003196747 (96.5x / 44.3x smaller than the published -0.2882637515 / -0.1415564623) and cos(ship,AD) = -1.000 / -0.7585 instead of +1.000 / +0.996; log artifacts/m20150_audit/cj2519_repro.log, parsed as verify_output.json:C9',
    'F0b high: the same re-run shows an AD sensitivity to x_c of 3.47e-3 where FD gives 1.66e-11 (rel err 2.09e8) — a spurious gradient in the periodic direction that the shipped artifact (g_ad_xc=3.5e-15) hides',
    'F1 blocking: paper-resolution verdict absent; report claims cj2485 "still queued, not fetched" while queue.log shows dispatch 08:59:28 and terminal state 09:05:28 with node rc=255 and only one R1 field snapshot',
    'F2 high: report headline (cos>=0.99 -> NO) and the three cos tick marks are literals in adjoint_verdict_report.py:198,294-296, contradicted by the artifact''s own cos values (1.000/0.996/1.000/0.980)',
    'F3 high: adjoint.py:81-93 FROZEN_FIELD_VERDICT hardcodes cos_vs_unrolled_ad=-1.000 and amplitude_ratio_vs_true=2.289, contradicting this mission''s artifact (cos 1.000/0.980, |g_frozen/g_ad| 5.17/5.37) with no run pointer',
    'F4 medium: artifact configs/*/ad/n=100 carries window=50, so the non-monotone 0.522/0.530/0.155 is confounded by truncated backprop; the single-cause attribution "transient dominated" was never separated from window truncation',
    'F5 medium: commit message "matches central finite differences (rel<2e-4)" holds only for a 30-step rollout at 40x32 epsilon=0.6, not at engineering scale (153-206%)',
    'F6 medium: no test binds the published artifact or the module literal to a run -> drift undetected',
    'F7 minor: commit 3dabe126c lives only on agent/m20069; main has no shape_gradient.py'
  ),
  'disposition', 'request changes / reopen L4: (1) fix paper path (R1 C_D=-43.99 root cause and cj2485 rc=255 crash) and run R1/R2 x 2 parameters x dual-step FD + Richardson to produce a mode=paper JSON; (2) register a tensor_benchmark_cases row for this anchor; (3) compute the report ticks from the JSON instead of literals; (4) delete or re-derive FROZEN_FIELD_VERDICT literals and add an artifact-binding test; (5) run a backprop-window sweep to split transient vs truncation; (6) wire shape_gradient_through_flow + the gate into the shape-optimisation loop (L3).'
 ),
 'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
SET level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id
RETURNING id, level, coalesce(gate_state,'pending'), coalesce(gate_level,-1);