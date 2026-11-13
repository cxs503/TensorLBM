-- Verdict for m18750 XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE
-- Level: L4 (基准验证：完整算例+误差分析+可复现)
-- Reason: Real solver code with 4 I/I selector alternatives, predeclared hypotheses/thresholds,
--   9 benchmark runs (4 rules × 2 resolutions + tau sensitivity), operator-level conservation audit
--   on 4 synthetic configurations, 4 passing unit tests, 2 embedded figures, honest FAIL reporting.
--   L5 not met: no external peer review or adoption beyond this repo; OSS anchor is inventoried
--   snapshot not vendored source with hash.

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm18750 XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE',
  18780,
  4,
  '{"impl_files":["src/tensorlbm/free_surface_lbm_2d.py:214-215 (INTERFACE_EXCHANGE_RULES tuple)","src/tensorlbm/free_surface_lbm_2d.py:368-814 (koerner_step_2d with interface_exchange_rule parameter, 4-rule dispatch at lines 632-669)"],"entry":"src/tensorlbm/free_surface_lbm_2d.py::free_surface_step_2d -> koerner_step_2d(... interface_exchange_rule=)","run_artifacts":["xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/operator_audit.json","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/db_koerner_a24.json","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/db_koerner_a32.json","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/db_oss_a24.json","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/db_oss_a32.json","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/db_pair_symmetric_a24.json","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/db_pair_symmetric_a32.json","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/db_repo_a24.json","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/db_repo_a24_tau0.8.json","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/artifacts/db_repo_a32.json"],"tests":["tests/test_fs_mass_exchange_pair_symmetry.py (4 passed: test_pair_symmetric_selector_deadens_fewer_links, test_pair_symmetric_is_strictly_antisymmetric, test_rule_argument_is_validated, test_default_rule_is_bit_identical_to_repo)"],"integration_callers":["xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/run_dambreak_xval.py (integrated dam-break benchmark)","xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/run_operator_audit.py (operator-level conservation audit on 4 synthetic configs)"],"benchmark":{"anchor":"XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE","ref_source":"Martin & Moyce 1952 Phil. Trans. R. Soc. A 244:312-324, 14 digitised points in src/artifacts/martin_moyce_1952_dam_break.json","metric":"dambreak_Z_error_max_pct = max|Z(T)-Z_MM(T)|/Z_MM(T)*100, Z=x_front/a, T=t*sqrt(g/h0)","tolerance":"<15% (owner precision standard)","got_values":{"koerner_a24":16.67,"koerner_a32":13.81,"oss_a24":16.67,"oss_a32":21.76,"pair_symmetric_a24":16.67,"pair_symmetric_a32":13.81,"repo_a24":16.67,"repo_a32":13.81},"pass_fail":{"koerner_a24":"FAIL (>15%)","koerner_a32":"PASS","oss_a24":"FAIL","oss_a32":"FAIL","pair_symmetric_a24":"FAIL","pair_symmetric_a32":"PASS","repo_a24":"FAIL","repo_a32":"PASS"},"inventory_drift":{"koerner_a24":0.0575,"koerner_a32":0.0473,"oss_a24":62.9062,"oss_a32":79.7147,"pair_symmetric_a24":0.0575,"pair_symmetric_a32":0.0473,"repo_a24":0.0575,"repo_a32":0.0473},"operator_audit":{"surface_band_oss_net_rel":1.0,"surface_band_repo_net_rel":0.05556,"droplet_shell_all_net_rel":0.01714}},"repro_cmd":"python3 xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/run_operator_audit.py ; python3 xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/run_dambreak_xval.py --rule repo --cells 32 ; python3 -m pytest tests/test_fs_mass_exchange_pair_symmetry.py -q","external_anchor":["Körner et al. 2005 J. Comput. Phys. 202:616-628 eq. 9 (paired I/I exchange)","Thürey 2007 PhD thesis Univ. Erlangen table 4.1 (selector), eq. 4.5 (ABB)","Martin & Moyce 1952 Phil. Trans. R. Soc. A 244:312-324 (14 digitised points)","OSS oracle: Maarten-vd-Sande/lbm lbm/streamcollide.py::interface_mass_exchange (inventoried snapshot in /root/oss-cache)"]}',
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id,
      level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL;
