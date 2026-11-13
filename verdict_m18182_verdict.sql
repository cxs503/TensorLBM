INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorLBM', 'm17759 XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S', 18182, 4, $json${
 "impl_files": [
  "src/tensorlbm/boundary/nebb.py:1-430 (force-corrected NEBB wall + Guo/EDM body-force driver: wall_directions:104, wall_density:141, force_corrected_velocity:167, nebb_wall_plane:190, guo_source:216, collide_guo:236, collide_edm:255, NebbPoiseuilleChannel:330)",
  "src/tensorlbm/boundary/__init__.py:19-30 (public exports of the 10 NEBB symbols)",
  "NOTE: m17759's report cited src/tensorlbm/wave_bc_common.py as the NEBB implementation; git grep 'nebb' src/tensorlbm/wave_bc_common.py -> 0 hits, i.e. the cited impl did not exist before this mission (that is what the L1 gate caught)"
 ],
 "entry": "src/tensorlbm/verification/nebb_bodyforce/run.py:1-330 (CLI --out/--ny/--nu/--gx/--max-steps/--quick), run via 'python -m tensorlbm.verification.nebb_bodyforce.run'",
 "run_artifacts": [
  "runs/m18182_nebb_force/result.json (9 arms, per-arm profile+analytic+error+steady residual+elapsed)",
  "runs/m18182_nebb_force/profiles.csv (arm,ny,y,ux_got,ux_ref long table)",
  "runs/m18182_nebb_force/nebb_bodyforce_xval.png (profile overlay + log-log convergence vs dy^2 + 15%/3% thresholds)",
  "runs/m18182_nebb_force/run.log (stdout of the run)"
 ],
 "tests": "tests/test_nebb_bodyforce_force_corrected.py: 11 passed in 70.8s (PYTHONPATH=src python -m pytest tests/test_nebb_bodyforce_force_corrected.py -q) — lattice-derived wall sets + one-hot streaming-convention probe, NEBB writes only the incoming populations of one plane, wall-density/equilibrium consistency, force-correction signs, parabola reference, grid-convergence (>=3x per doubling), steady residual<1e-10 and mass drift<1e-10, negative control (dropping -Gx/(2 rho_w) raises wall slip and global error), core-path wall_bc='nebb' wired + default 'halfway' path unchanged",
 "integration_callers": [
  "src/tensorlbm/gpu_core_path.py:381-416 apply_channel_walls_nebb (calls boundary.nebb.nebb_wall_plane/wall_density/wall_directions) invoked from core_step:415-419 when cfg.wall_bc=='nebb'",
  "src/tensorlbm/gpu_core_path.py:189 GPUCorePathConfig.wall_bc, :251 validate, :526 eager-backend guard, :285 channel_analytic_velocity_nebb",
  "runs/m18182_nebb_force/result.json arms.corepath_nebb / arms.corepath_halfway (same capability driven through GPUCorePathSolver.run/step)"
 ],
 "benchmark": {
  "ref": "closed-form steady plane Poiseuille ux(y)=Gx/(2 nu) y (H-y), H=ny-1, no-slip planes on the wall rows",
  "got": {
   "ny10": 0.044,
   "ny20": 0.00978,
   "ny40": 0.00232,
   "unit": "max|got-ref|/max|ref|"
  },
  "tolerance": "0.15 owner / 0.03 ecosystem stretch (registered as tensor_benchmark_cases id=100, tolerance 0.03)",
  "convergence": "error reduction 4.50x (ny10->ny20) and 4.22x (ny20->ny40) => 2nd order, consistent with dy^2",
  "controls": {
   "nebb_guo_ny20_nocorr": "0.0120 vs 0.00978 corrected; wall slip 0.0027 vs 0.0022 (force correction is load-bearing but small at Gx=1e-3)",
   "halfway_ny10/ny20": "0.349 / 0.144 vs the node-based reference — expected: half-way walls sit 0.5 cell off, so this arm is a convention control, not a like-for-like ranking",
   "nebb_edm_ny20": "0.541 — the equilibrium-shift-only forcing arm does NOT reproduce the parabola (honest negative result; the Guo arm is the shipped path)",
   "corepath_halfway": "0.00361 vs its own H=ny-2 reference (unchanged default path)",
   "corepath_nebb": "0.04695 vs the H=ny-1 reference at ny=18, 7000 steps (wall-node convention, not fully converged at that grid — reported as-is)"
  }
 },
 "repro_cmd": "cd <worktree> && PYTHONPATH=src python -m tensorlbm.verification.nebb_bodyforce.run --out runs/m18182_nebb_force  (full run ~4 min CPU float64; --quick for a 20 s smoke) ; tests: PYTHONPATH=src python -m pytest tests/test_nebb_bodyforce_force_corrected.py -q",
 "external_anchor": {
  "oss": "haimingz/vivsim (boundary/nebb.py + forcing/{edm,guo}.py + basic.py velocity correction)",
  "status": "algorithm family confirmed by reading the source; the vivsim-port arm committed at xval/xval_nebb_bodyforce_driver.py reports 5.0% at NY=10 (m17759 artifact), this implementation reports 4.4% at the same NY with clean 2nd-order convergence",
  "caveat": "/tmp/vivsim_src is not present in this container, so the port was NOT re-executed here; the cross-arm comparison is against the committed m17759 artifact numbers, not a fresh OSS run"
 },
 "original_m17759_audit": {
  "verdict_then": "L1 (m17963): impl_files/entry/run_artifacts resolved to a standalone script + a non-existent wave_bc_common NEBB",
  "now": "real package implementation + entry + run artifacts + 11 tests + main-solver caller + analytic benchmark at 0.23% (ny=40)",
  "overstated_claims_found": [
   "report attributes NEBB to src/tensorlbm/wave_bc_common.py (0 grep hits)",
   "report's 'repo Guo forcing in gpu_core_path' is real but was re-implemented in numpy rather than imported"
  ]
 }
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;

