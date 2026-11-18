-- Audit verdict: mission 19475 reviewing mission 19252
-- (XVAL-LBM-ROUND2-METHOD-SELECTION-CONTENT-ADDRESSE, TensorLBM)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
 'TensorLBM',
 'm19252 XVAL-LBM-ROUND2-METHOD-SELECTION-CONTENT-ADDRESSE',
 19475,
 3,
 jsonb_build_object(
  'impl_files', jsonb_build_array(
    'xval/benchmark_method_selection/run_arm.py:1-408 (real LBM arm runner; production solver imports at :54-60 collide_bgk/collide_mrt/collide_trt/stream; KPI analyze() at :127-152; sha256 cf5a003e24e8... matches bundle)',
    'src/tensorlbm/method_selection.py:83-158 (production selection surface select_collision/collide_dispatch; sha256 b446dfa5a207...)',
    'xval/benchmark_method_selection/run_round2_entry.py:1-133 (entry; sha256 bc77b96b9b40...)',
    'xval/benchmark_method_selection/benchmark_l4.py:1-290 (L4 table builder; DEFECT at :105-126: :115-116 read oss["k1"]/["k2"] instead of k1_collision/k2_streaming, :117/:122 set within_15pct hardcoded True)',
    'xval/benchmark_method_selection/oss_crosscheck.py:1-271 (kernel xval vs unmodified OSS source; sha256 e2f1089579ae...)'
  ),
  'entry', 'python3 xval/benchmark_method_selection/run_round2_entry.py --smoke  (shipped entry); auditor entry: python3 artifacts/m19475_audit/verify_m19252.py --out artifacts/m19475_audit/verify_output.json (exit 0, ~10 s, regenerates every number in this verdict)',
  'run_artifacts', jsonb_build_array(
    'xval/benchmark_method_selection/runs/A1_D10/result.json (Cd_mean=1.6143262481245584, Cl_amp=0.6579248352005606, St_fft=0.16899580031695963, n=21259) + trace.npz (855270 B)',
    'xval/benchmark_method_selection/runs/A2_D10/result.json (Cd=1.4661360417278881, Cl_amp=0.3515523646082385, St=0.17662926112262423) + trace.npz (859073 B)',
    'xval/benchmark_method_selection/runs/A4_D10/result.json (Cd=1.6423615055096823, Cl_amp=0.556625096634633, St=0.1717673074998151) + trace.npz (860823 B)',
    'xval/benchmark_method_selection/runs/A3_D10/fail.json (non_finite_population @step 600/21259), runs/A5_D10/fail.json (@step 5800)',
    'xval/benchmark_method_selection/runs/A6_D10/state.npz ONLY (2.45 MB, no result.json/fail.json -> 4th factorial cell has no KPI record)',
    'artifacts/m19475_audit/verify_output.json (C3_kpi_replay max_rel_diff=0.0; C1 6/6 sha match; C4/C5/C7/C8/C9 findings)'
  ),
  'tests', jsonb_build_array(
    'tests/test_xval_benchmark_method_selection.py (6 passed in 41.04 s under the auditor; invariants: A1..A6 arm table, 0.1<Cd<10, smoke spread<50%, OSS kernel agreement)',
    'tests/test_xval_round2_method_selection.py (6 tests collected: aggregation on synthetic matrix, diverging arm reported not dropped)',
    'tests/test_m19475_m19252_audit.py (auditor; 6 passed 0.39 s: sha match, bit-exact KPI replay, 0 benchmark rows, NaN-marked-pass, ladder/determinism/factorial gaps, uncommitted delivery)'
  ),
  'integration_callers', jsonb_build_array(
    'benchmarks/case_pool/xval_method_selection/run_case.py:32 imports tensorlbm.method_selection.select_collision; EXECUTED by auditor (`PYTHONPATH=src python3 benchmarks/case_pool/xval_method_selection/run_case.py --smoke`) -> artifacts/case_xval_method_selection/a1_mrt_sponge_d6/{result.json,selection.json,trace.npz} and a2_bgk_d6/ (A2 Cd=1.2393432470464272, St=0.17168233427728155 at D=6)',
    'src/tensorlbm/xval/method_selection_integration_demo.py:28 imports select_collision (demo only)',
    'grep: src/tensorlbm/solver.py does NOT import method_selection -> integration is case-pool/bench level, the default solver path is not wired to the selection decision'
  ),
  'benchmark', 'xval/benchmark_method_selection/benchmark_l4.json: 5 comparison rows, all case-level rows use ref_arm="A1" = TensorLBM own MRT+sponge (self-referential), deltas -9.18%/+4.52%/-46.57% (A2) and +1.74%/+1.64%/-15.40% (A4); 2 of 6 KPIs exceed 15% vs A1; OSS kernel row renders got=NaN/delta=NaN with within_15pct=true (reproduced: `python3 benchmark_l4.py` prints "k1_collision_rel_L2: delta=nan%, within_15pct=True"). tensor_benchmark_cases: 0 rows for the anchor (ILIKE %ROUND2-METHOD-SELECTION% -> 0; table total 118) -> machine judge impossible.',
  'external_anchor', 'WITHHELD at case level: report §5 "External reference (Norberg 1993) is WITHHELD"; predeclared_round2.json baseline_control.external_caliber. Auditor had no literature access either -> observed absent, not proven unobtainable. Only external anchor is kernel-level: OSS Maarten-vd-Sande/lbm@573a01c executed source, rel_l2_collision=1.1757884876259976e-07 / rel_l2_streaming_interior=2.709855973692391e-08 (reproduced in 6.0 s by the auditor) -> validates primitives, not the case-level Cd/Cl/St.',
  'repro_cmd', 'python3 artifacts/m19475_audit/verify_m19252.py && python3 -m pytest tests/test_m19475_m19252_audit.py -q  (auditor, ~11 s, exit 0); delivery-side: python3 xval/benchmark_method_selection/oss_crosscheck.py (6.0 s, bit-identical rel-L2); python3 xval/benchmark_method_selection/benchmark_l4.py (regenerates the NaN/pass row)',
  'verdict_basis', jsonb_build_object(
    'L1', 'PASS kernel xval 1.18e-07 / 2.71e-08 reproduced + 6 pytest invariants green',
    'L2', 'PASS real input->compute->output, KPI replay bit-exact (max rel diff 0.0 over 5 KPIs x 3 arms)',
    'L3', 'PASS platform-side caller executed and wrote real artifacts (case_pool run_case.py --smoke)',
    'L4', 'BLOCKED: 0 tensor_benchmark_cases rows (F1), ref=A1 self-referential + external WITHHELD (F2), OSS row NaN+hardcoded pass (F3), ladder D=14/D=20 null though cj1872 succeeded (F4)'
  ),
  'findings', jsonb_build_array(
    'F1 blocking L4: 0 benchmark rows for anchor',
    'F2 high: case-level ref = own A1 arm; external caliber WITHHELD',
    'F3 high (reproduced): benchmark_l4.py:105-126 (defect at :115-116, hardcoded flags :117/:122) key mismatch -> NaN + hardcoded within_15pct=True',
    'F4 medium: predeclared closure unclosed (ladder 14/20 null, closure_H1 9/9 null, determinism null) yet cluster job cj1872 (A1 D=14) succeeded and artifacts were never retrieved (no runs/A1_D14)',
    'F5 medium: A6_D10 state.npz only -> factorial [] not closable',
    'F6 medium: cl_rms computed but not persisted -> table_round2.csv cl_rms empty, gate_pass nan without reason',
    'F7 high: delivery uncommitted (HEAD 95d2712ff074 = base merge; run_round2_entry.py/benchmark_l4.*/maturity_evidence_m19252.* untracked, tests file modified) -> auto-push would ship nothing',
    'F8 medium: 0 rows in tensor_reports for mission 19252 while report_owed=false'
  ),
  'audit_doc', 'docs/AUDIT_m19475_m19252_XVAL_LBM_ROUND2.md',
  'audit_artifacts', 'artifacts/m19475_audit/{verify_m19252.py,verify_output.json}',
  'independence_limits', 'Auditor re-ran shipped scripts inside the m19252 worktree: oss_crosscheck.json throughput field rewritten (5.126->4.767 s/step, rel_l2 bit-identical), benchmark_l4.json regenerated byte-identical (sha256 259684a77c36...), artifacts/case_xval_method_selection/ created by the integration smoke. No source file altered; all 6 impl sha256 still match after the audit.'
 ),
 'board-audit'
);