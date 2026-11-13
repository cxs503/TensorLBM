-- Board audit verdict for m18142 XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT
-- Audit performed by m18214 (board-audit)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorLBM',
    'm18142 XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT',
    'm18214',
    4,
    '{"impl_files":["src/tensorlbm/models/ci_unit.py:1-1201","src/tensorlbm/api.py:list_model_units,run_model_unit_ci,write_model_unit_artifacts,model_unit_build"],"entry":"python -m tensorlbm.models.ci_unit all --root models/units --out artifacts/model_units","run_artifacts":["artifacts/model_units/summary.json","artifacts/model_units/SUMMARY.md","artifacts/model_units/builds/*.json","artifacts/model_units/ci/*.json","artifacts/model_units/sensitivity.json","artifacts/model_units/sensitivity.md"],"tests":["tests/models/test_model_dir_ci_unit.py:15 tests, all pass"],"integration_callers":["src/tensorlbm/api.py:list_model_units","src/tensorlbm/api.py:run_model_unit_ci","src/tensorlbm/api.py:write_model_unit_artifacts","src/tensorlbm/api.py:model_unit_build"],"benchmark":{"d2q9_bgk_channel_poiseuille":{"profile_l2_rel_err":{"got":1.2268e-03,"max":0.01,"pass":true},"umax_rel_err":{"got":9.1733e-04,"max":0.01,"pass":true},"mass_rel_drift":{"got":2.0022e-12,"max":1e-10,"pass":true}},"d2q9_bgk_shear_wave":{"nu_rel_err":{"got":1.1108e-03,"max":0.01,"pass":true},"amp0_rel_err":{"got":5.2042e-16,"max":1e-10,"pass":true},"mass_rel_drift":{"got":3.3429e-13,"max":1e-10,"pass":true}},"d2q9_mrt_shear_wave":{"nu_rel_err":{"got":1.1704e-03,"max":0.01,"pass":true},"amp0_rel_err":{"got":5.2042e-16,"max":1e-10,"pass":true},"mass_rel_drift":{"got":5.6843e-13,"max":1e-10,"pass":true}},"d2q9_trt_channel_poiseuille":{"profile_l2_rel_err":{"got":5.5931e-05,"max":0.001,"pass":true},"umax_rel_err":{"got":5.7752e-05,"max":0.001,"pass":true},"mass_rel_drift":{"got":2.0082e-12,"max":1e-10,"pass":true}}},"repro_cmd":"python -m tensorlbm.models.ci_unit all --root models/units --out artifacts/model_units","external_anchor":"CFD-GO/TCLB model-dir-as-build-and-CI-unit convention"}',
    'board-audit'
);
