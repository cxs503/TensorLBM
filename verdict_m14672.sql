-- Verdict for m14672 GAP-LBM-SPHERE
-- Level: L2 (独立功能实现)
-- Reason: Real code + passing tests, but benchmark results are placeholders (Schiller-Naumann formula, not LBM simulation)

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm14672 GAP-LBM-SPHERE',
  14991,
  2,
  '{"impl_files":["src/tensorlbm/bfl_common.py:1-1417","src/tensorlbm/bfl_d3q19.py:1-455","src/tensorlbm/bfl_boundary.py:1-203","src/tensorlbm/sphere_bfl_control_volume.py","src/lbm/boundary/spherical_force.py:1-458"],"entry":"src/tensorlbm/__init__.py exports: SphereBFLControlVolumeConfig, bfl_surface_area_weights","run_artifacts":["benchmarks/verified/sphere_drag_bfl/run.py","benchmarks/verified/sphere_drag_bfl/results.json"],"tests":["tests/test_bfl_sparse.py (16 passed)","tests/test_bfl_fractional_distance.py (7 passed)"],"integration_callers":"benchmarks/verified/sphere_drag_bfl/run.py uses GeneralSimEngine","benchmark":{"status":"INCOMPLETE","issue":"results.json contains Schiller-Naumann placeholder values, not LBM simulation output","evidence":"elapsed_s=0, wall_s=0, all_results={}, note=Full LBM simulation pending cluster execution cj832"},"repro_cmd":"PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/run.py","external_anchor":"Bouzidi Firdaouss Lallemand 2001 Phys Fluids 13:3452; Schiller-Naumann 1933"}',
  'board-audit'
);
