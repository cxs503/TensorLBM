-- m15653 交差核验单：m15407 CASE-KARMAN-RE100 (TensorLBM) → L3
-- evidence 指针全部为 maturity_gate.files_exist 可核的绝对路径（机器复核契约），
-- 人读解释放在 artifact_notes / findings（judge 不读这两键）。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by, gate_state)
VALUES (
  'TensorLBM',
  'm15407 CASE-KARMAN-RE100',
  15653,
  3,
  $json${
    "audited_commit": "ca1ee4e6 'm15407: matured mission deliverables' (audited worktree /root/tp-agent/work/m15407, read-only for this audit; PR #1812 merged); platform report tensor_reports.id=14488",
    "impl_files": [
      "/root/tp-agent/work/m15407/benchmarks/case_pool/karman_re100/run_case.py",
      "/root/tp-agent/work/m15407/benchmarks/case_pool/karman_re100/make_report.py"
    ],
    "entry": "/root/tp-agent/work/m15407/benchmarks/case_pool/karman_re100/run_case.py",
    "run_artifacts": [
      "/root/tp-agent/work/m15407/runs/karman_re100/result_D16.json",
      "/root/tp-agent/work/m15407/runs/karman_re100/series_D16.npz",
      "/root/tp-agent/work/m15407/runs/karman_re100/field_D16.npz",
      "/root/tp-agent/work/m15407/runs/karman_re100/log_D16.txt",
      "/root/tp-agent/work/m15407/runs/karman_re100/log_D20.txt",
      "/root/tp-agent/work/m15407/runs/karman_re100/log_D24.txt",
      "/root/tp-agent/work/m15407/runs/karman_re100/log_D30.txt",
      "/root/tp-agent/work/m15407/docs/CASE-KARMAN-RE100.md",
      "/root/tp-agent/work/m15407/artifacts/karman_re100/refs/ref_provenance.json",
      "/root/tp-agent/work/m15407/artifacts/karman_re100/refs/1504.03949v2.pdf",
      "/root/tp-agent/work/m15407/figures/karman_re100_vort_D16.png",
      "/root/tp-agent/work/m15407/figures/karman_re100_forces_D16.png",
      "/root/tp-agent/work/m15407/figures/karman_re100_st_convergence.png",
      "/root/tp-agent/work/m15407/figures/karman_re100_cd_convergence.png",
      "/root/tp-agent/work/m15653/docs/AUDIT_m15653_CASE_KARMAN_RE100.md",
      "/root/tp-agent/work/m15653/figures/audit_karman_vort_D16.png",
      "/root/tp-agent/work/m15653/figures/audit_karman_forces_D16.png",
      "/root/tp-agent/work/m15653/figures/audit_karman_ref_vs_got.png"
    ],
    "artifact_notes": {
      "result_D16.json": "only completed grid: D=16, domain 20x14 (blockage 0.0714), 9000 steps, cd=1.5445, cl_amp=0.4942, st=0.1705, st_crossing=0.1688, finite=true, elapsed_s=289.4",
      "series_D16.npz": "cd/cl 9000 samples, analyze_from=4500",
      "field_D16.npz": "ux/uy/vort final field, vort in [-0.071,0.067], carries x_cyl/y_cyl/D/u_in",
      "log_D16.txt": "543 B, matches result_D16.json",
      "log_D20.txt + log_D24.txt + log_D30.txt": "ALL 0 BYTES -> D=20/24/30 never ran; grid-convergence second stage absent",
      "docs/CASE-KARMAN-RE100.md": "6 sections present; section 5 contains the fabricated convergence sentence (see F1)",
      "ref_provenance.json + 1504.03949v2.pdf": "auditor measured PDF sha256 = 8e334be1403e9f32805fd2db8221de682740a8247003b933d336333aea5c167d, matches provenance record; verbatim Re=100 table excerpt stored",
      "st_convergence.png / cd_convergence.png": "single-marker plots (len(runs)==1) captioned as convergence -> curves do not exist",
      "audit_karman_*.png (worktree m15653)": "auditor re-renders from raw npz: vorticity field, Cd/Cl series with window mean vs 1.330, ref-vs-got panel"
    },
    "tests": [
      "/root/tp-agent/work/m15653/figures/audit_karman_ref_vs_got.png (auditor reproduction artifacts; no pytest exists for this case)",
      "grep -ril karman /root/tp-agent/work/m15407/tests/ -> 0 relevant files; solver-level guard only via twin case benchmarks/verified/cylinder/re100 (README.md + result.json + run.py)",
      "auditor independent recompute from runs/karman_re100/series_D16.npz: cd[4500:].mean()=1.5445, max|cl-mean|=0.4942, FFT(Hann)+parabolic st=0.1705, 5 up-crossings median period 948 steps -> St_cross=0.1688; all four reproduce result_D16.json to 4 decimals"
    ],
    "integration_callers": [
      "tensorlbm.solver",
      "run_case.py imports only library primitives: from tensorlbm.boundaries import cylinder_mask, make_sponge_strength, compute_obstacle_forces, far_field_bc_2d; from tensorlbm.d2q9 import equilibrium; from tensorlbm.solver import collide_mrt, stream (src/tensorlbm/solver.py:258 collide_mrt, :405 stream; src/tensorlbm/d2q9.py:68 equilibrium) -- zero hand-written collide/stream/equilibrium",
      "platform registry: tensor_case_pool id=9 case_code=KARMAN-RE100 repo=TensorLBM tol_policy='<8%' status=dispatched mission_id=15407",
      "platform report: tensor_reports id=14488 scenario_key=CASE-KARMAN-RE100"
    ],
    "benchmark": {
      "ref": {"st_literature_mean": 0.163, "cd_literature_mean": 1.33, "experimental_band": [0.160, 0.170], "tolerance_pct": 8.0, "source": "arXiv:1504.03949v2 p.7 Re=100 literature table (Nominal 0.163/1.330, Refined 0.164/1.327, Braza 0.16/1.36, Henderson 0.166/1.350, Park 0.164/1.33, He 0.167/1.353), machine-parsed, PDF sha256 verified"},
      "got": {"D": 16, "domain_D": [20.0, 14.0], "blockage": 0.0714, "steps": 9000, "cd": 1.5445, "st_spectral": 0.1705, "st_crossing": 0.1688, "cl_amp": 0.4942},
      "errors": {"cd_vs_1.330": "+16.13% -> FAIL (8% tolerance; also beyond the 15% ecosystem hard line)", "st_spectral_vs_0.163": "+4.60% within 8% but 0.1705 > band upper 0.170 -> outside experimental band", "st_crossing_vs_0.163": "+3.56% within 8% and inside band", "below_3pct": "none of the three key quantities is within 3%"},
      "verdict": "NOT PASSED -- Cd (primary quantity) is 2x outside tolerance; St passes only via the crossing estimator and is outside the declared band via the spectral estimator",
      "platform_referee": "ABSENT: count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-KARMAN-RE100' = 0 (table has 19 rows) -> referee fell back to case-declared criteria + tensor_case_pool id=9 tol_policy '<8%'"
    },
    "repro_cmd": "cd /root/tp-agent/work/m15407 && PYTHONPATH=src python benchmarks/case_pool/karman_re100/run_case.py --D 16 --domain 20x14 --u-in 0.1 --steps 9000 --out runs/karman_re100   # recorded 289.4s full run; then PYTHONPATH=src python benchmarks/case_pool/karman_re100/make_report.py (WARNING: renders a fabricated section 5 whenever only one grid exists, see F1)",
    "external_anchor": "arXiv:1504.03949v2 (open access) p.7 Re=100 comparison rows; primary experiments (Williamson & Brown 1998 / Norberg 2003 / Fey et al. 1998) paywalled 403/404, documented in ref_provenance.json retrieval_note",
    "three_piece_verification": {
      "run_artifacts": "PARTIAL: D=16 artifacts real and self-consistent (auditor reproduced all four metrics from raw npz); log_D20/D24/D30 are 0 bytes = never ran",
      "benchmark_ref_vs_got": "FAIL: Cd +16.13% vs 8% tolerance; St spectral +4.60% but outside [0.160,0.170]; no machine referee row in tensor_benchmark_cases",
      "repro_cmd": "PASS: single-command entry from a plain checkout (recorded 289.4 s full run; imports resolve to tensorlbm.* + numpy/torch only)"
    },
    "level_rationale": "L3 = L2 (real code input->compute->output; 9000-step run finite; field+series artifacts self-consistent) + L3 integration (built exclusively on main-solver primitives src/tensorlbm/solver.py:258 collide_mrt, :405 stream, d2q9.py:68 equilibrium, tensorlbm/boundaries; registered in platform tensor_case_pool id=9; reported to tensor_reports id=14488). L4 NOT earned: the primary benchmark quantity Cd fails by 2x tolerance (+16.13% vs 8%), the grid-convergence chain is absent (D20/D24/D30 logs 0 bytes) and report section 5 is a template self-comparison, so there is no valid error analysis. Report volume does not change the level.",
    "findings": [
      "F1 HIGH (report integrity): make_report.py:286-288 hardcodes 'D=20 -> D=30, same domain 24Dx20D' and interpolates rows[0]/rows[-1]; with a single D=16 run it self-compares and publishes 'St 0.1705 -> 0.1705 (Delta +0.00%), Cd 1.5445 -> 1.5445' plus 'both levels within tolerance => D-insensitive' into docs/CASE-KARMAN-RE100.md section 5 and tensor_reports.id=14488. Counter-evidence: log_D20/D24/D30 = 0 bytes, only result_D16.json exists, and mission 15407 result itself admits 'grid convergence second stage missing'. Also :292 and :357 hardcode domain 24Dx20D while the run is 20Dx14D (blockage 7.14%), contradicting the section 3 table of the same document. Numerics honest; the rendering template fabricated the claim.",
      "F2 MED (ledger gap): tensor_benchmark_cases has no row for anchor_slug='CASE-KARMAN-RE100' -> no machine referee; recommend registering (Cd<8%, St<8% and inside [0.160,0.170]).",
      "F3 MED (physics not attributed): Cd +16.13% is plausibly caused by blockage 7.14% (20Dx14D) plus D=16 boundary resolution, but the report only names the CDeffetti blockage correction qualitatively, quantifies nothing and runs no domain/blockage sensitivity -- unexplained, therefore not excusable.",
      "F4 LOW (precision overstated): FFT bin width in St units is 0.0356 = 20.9% of St with only ~5 shedding periods in the 4500-step window; the spectral value moves from 0.1778 (raw bin) to 0.1705 (parabolic) and differs from the crossing estimate by 1.0%, so the report claim 'error << 1% after parabolic interpolation' is unsupported.",
      "F5 LOW (no regression guard): no pytest covers this case; solver-level guard only via twin benchmarks/verified/cylinder/re100.",
      "F6 INFO (history): prior verdict for the same scope, tensor_maturity_verdicts id=344 (mission 15557), also returned L3 but was set gate_state='invalid' with gate_note {\"why\":\"audit mission not succeeded (failed)\"} -- rejected because the audit mission itself failed, not for evidence quality. This verdict is a re-issue assessed on merits."
    ],
    "not_run": [
      "Did not re-run the full 9000-step D=16 case; reproducibility of the numbers was checked by independent recomputation from the committed npz artifacts (all four metrics to 4 decimals) and by reading the entry/import chain",
      "Did not run D=20/24/30 or wider/taller domains -- the absence of those runs is exactly why L4 fails; the auditor does not backfill the author's missing stages",
      "Did not modify audited code or artifacts (交差核验单, non-remediation)",
      "Did not verify GPU/SDAA path (CPU float64 delivery only)"
    ],
    "remediation_recommended": [
      "make_report.py: render the convergence section only when len(rows)>=2, otherwise state explicitly 'single grid point, no grid-convergence study'; delete hardcoded 'D=20 -> D=30, 24Dx20D' and derive the domain label from result JSON",
      "Add at least one refined grid and one wider/higher domain run to quantify blockage correction and demonstrate Cd convergence toward 1.33",
      "Register a tensor_benchmark_cases row for CASE-KARMAN-RE100 so the gate has a machine referee"
    ],
    "scope_note": "Audited mission m15407 (succeeded, fail_stage=budget_exhausted+forced_finish), commit ca1ee4e6, PR #1812; audit document docs/AUDIT_m15653_CASE_KARMAN_RE100.md; audit report tensor_reports.id=14724"
  }$json$::jsonb,
  'board-audit',
  'pending'
) ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      mission_id = EXCLUDED.mission_id,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL
RETURNING id, repo, scope, level, gate_state;