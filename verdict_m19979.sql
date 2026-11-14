-- Verdict for m19979 CASE-POISEUILLE-2D-RELERR
-- Audited by m20062 (board-audit)
-- Date: 2026-11-14

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm19979 CASE-POISEUILLE-2D-RELERR',
  20062,
  4,
  '{"impl_files":["src/tensorlbm/ice_hydro/lbm_channel.py","src/tensorlbm/verification/xval/poiseuille_skeleton.py","src/tensorlbm/verification/xval/cases.py","src/tensorlbm/examples/registry.py"],"entry":["benchmarks/verified/poiseuille_2d/run.py","benchmarks/verified/poiseuille_convergence_order/run.py"],"run_artifacts":["runs/m19979_poiseuille2d/summary.json","runs/m19979_poiseuille2d/comparison.csv","runs/m19979_poiseuille2d/profile_rows.csv","runs/m19979_poiseuille2d/profiles.json","runs/m19979_poiseuille2d/raw/H60_tau0.8_ua0.04.json","runs/m19979_poiseuille2d/archived/case_H60_archived.json"],"tests":["tests/test_poiseuille_2d_bodyforce.py","tests/test_circular_poiseuille.py","tests/test_layered_poiseuille.py"],"integration_callers":["src/tensorlbm/ice_hydro/lbm_channel.py:D2Q9Channel","src/tensorlbm/forcing/body_force_driver.py"],"benchmark":{"anchor":"CASE-POISEUILLE-2D-RELERR","metric":"eps_L2_rel","tolerance":"<1%","H60_got":"0.193%","H40_got":"0.458%","H20_got":"1.198%(FAIL-coarse)","reproducibility":"0.94%<5%PASS"},"repro_cmd":"PYTHONPATH=src python benchmarks/verified/poiseuille_2d/run.py single 60 runs/m19979_poiseuille2d/raw/H60_tau0.8_ua0.04.json --tau 0.8 --umax 0.04","external_anchor":"Schlichting&Gersten(2017) Boundary-Layer Theory Ch.3 DOI:10.1007/978-3-662-52919-5_17"}',
  'board-audit'
);
