-- Board-audit verdict for mission m17603 (anchor XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE).
-- Produced by audit mission m17664. Idempotent: ON CONFLICT (repo, scope, verdict_by)
-- updates and resets gate_state to pending so maturity_gate re-adjudicates.
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17603 XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE', 17664, 2, $json${
  "impl_files": [
    "/root/tp-agent/work/m17603/src/tensorlbm/boundary/inamuro_wetnode.py:306 inamuro_wetnode_scalar (525-line module, torch float64; 4 public operators + 1 helper)",
    "/root/tp-agent/work/m17603/src/tensorlbm/boundary/inamuro_wetnode.py:359 inamuro_zero_gradient",
    "/root/tp-agent/work/m17603/src/tensorlbm/boundary/inamuro_wetnode.py:422 inamuro_wetnode_velocity (anti-bounce-back: f_ibar = f_ibar^eq(rho_w,u_w) + f_i^eq(rho_w,u_w) - f_i)",
    "/root/tp-agent/work/m17603/src/tensorlbm/boundary/inamuro_wetnode.py:488 inamuro_corner_average",
    "PRE-EXISTING infrastructure only: d2q9.py / solver2d.py (equilibrium2d, macroscopic2d) were already in the base c05d7af3; the new capability is the wet-node closure family itself"
  ],
  "entry": "PYTHONPATH=src python3 benchmarks/verified/inamuro_wetnode_xval/run.py --part film --steps 12000 --devsteps 2000 --oss-ref artifacts/m17603_oss_src --out <outdir>   (film arm only; velocity arm is not_run)",
  "run_artifacts": [
    "/root/tp-agent/work/m17603/artifacts/m17603/xval_inamuro_wetnode.json (graded rows: rel L2 3.1747e-11 / 5.1186e-11 / 4.3562e-11 / 4.2212e-11 at steps 100/1000/10000/12000)",
    "/root/tp-agent/work/m17603/artifacts/m17603/film_fields.npz (TensorLBM fields + reference fields at every checkpoint)",
    "/root/tp-agent/work/m17603/artifacts/m17603/fig1_film_field_xval.png",
    "/root/tp-agent/work/m17603/artifacts/m17603/oss_ref/film_inamuro/film_inamuro{0000100,0001000,0010000,0050000,0012000}.dat (upstream fields)",
    "audit-side: /root/tp-agent/work/m17664/docs/board_audit/m17664/audit.json + audit_fields.npz + fig_audit_m17664.png + AUDIT_m17664.md"
  ],
  "tests": [
    "tests/test_inamuro_wetnode.py::14 property tests, re-run by the auditor: 'PYTHONPATH=src python3 -m pytest tests/test_inamuro_wetnode.py -q' -> 14 passed in 3.81 s (assertions are invariants: face direction sets, scalar mass consistency, all_but_rest leaves f_0, zero-gradient keeps a uniform field, closure touches only its own face, ABB preserves a wall-consistent equilibrium, ABB imposes the wall velocity, rho extrapolation, corner average vs upstream block, inplace semantics, lattice pairing)",
    "audit control probe (detection power): inamuro_wetnode_scalar -> identity at step 1000 gives rel L2 = 0.99666 (FAIL) vs 5.12e-11 intact -> the 1e-11 agreement is not a vacuous pass"
  ],
  "integration_callers": [
    "NONE: 'grep -rn inamuro /root/tp-agent/work/m17603/src --include=*.py' returns 14 hits, ALL inside src/tensorlbm/boundary/inamuro_wetnode.py itself; 0 call sites in solver / d2q9 / boundaries / free-surface code",
    "NOT registered: boundary_registry.py, boundary/__init__.py and boundary_capability_contract.py contain zero references to the new closures (checked reading all three files)",
    "the only caller is the xval driver benchmarks/verified/inamuro_wetnode_xval/run.py:139 film_apply_closures -> therefore L3 (system integration) NOT met"
  ],
  "benchmark": {
    "machine_referee": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE' -> 0 rows (anchor has no referee row; the 15% tolerance is re-declared inside the driver, not pre-registered)",
    "external_anchor_executed": "lbm-principles-practice/code @ 6e2c592fdc3592c14dfd52f860fc1ceea930bcb0 chapter8/film_inamuro.cpp - the auditor recompiled the archived source with g++ -O2 and re-ran it (94 s, N=50000); sha256 of film_inamuro{0000100,0001000,0010000,0012000}.dat MATCHES the m17603 archive byte-for-byte -> reference is genuinely executed, not synthesised",
    "ref_vs_got": {
      "step_100": {"rel_l2": 3.174711034176936e-11, "rel_linf": 4.99745245186034e-11, "declared_tol": 0.15, "pass": true},
      "step_1000": {"rel_l2": 5.1185883463411415e-11, "rel_linf": 3.516677640587728e-10, "declared_tol": 0.15, "pass": true},
      "step_10000": {"rel_l2": 4.356183890868777e-11, "rel_linf": 3.8146957978921796e-10, "declared_tol": 0.15, "pass": true},
      "step_12000": {"rel_l2": 4.221245099089233e-11, "rel_linf": 3.6127191906104005e-10, "declared_tol": 0.15, "pass": true}
    },
    "shipped_vs_audit_delta": "abs_delta = 0.0 for rel_l2 and got_mean at every overlapping checkpoint (deterministic re-run reproduces the archived numbers bit-for-bit)",
    "why_not_L4": "(1) no machine referee row -> tolerance not pre-registered; (2) the anchor is named ANTI-BOUNCE-BACK but the externally validated arm is the SCALAR residual-weight closure (film_inamuro.cpp) only; the anti-bounce-back velocity arm is velocity.status='not_run' (open driver defect: channel time loop does not constrain the wall rows across steps) so there is NO external error number for the headline closure; (3) the deliverables are untracked in git so repo-level reproducibility does not exist and the documented one-shot repro crashes (see repro_cmd); (4) no L3 integration"
  },
  "repro_cmd": {
    "as_documented": "benchmarks/verified/inamuro_wetnode_xval/reproduce_oss.sh /tmp/oss_lbm && PYTHONPATH=src python3 benchmarks/verified/inamuro_wetnode_xval/run.py --part film --steps 50000 --devsteps 5000 --oss-ref /tmp/oss_lbm --out artifacts/m17603",
    "status": "BROKEN on a fresh outdir - auditor executed step 2 verbatim and it died after 64 s with FileExistsError: '/tmp/audit17664/film/oss_ref/film_inamuro' (part_film compiles the reference binary to <out>/oss_ref/film_inamuro as a FILE at run.py:372-375, then run_oss_reference mkdirs the same path as the field DIRECTORY at run.py:204-205). It only works because artifacts/m17603/oss_ref/ ships a hand-built layout with the binary renamed film_inamuro_bin",
    "audit_side_working_path": "python3 /root/tp-agent/work/m17664/docs/board_audit/m17664/independent_audit.py --lbm-worktree /root/tp-agent/work/m17603 --oss-fields <dir with fresh upstream .dat> --out docs/board_audit/m17664   -> 68 s, regenerates every number in audit.json"
  },
  "external_anchor": [
    "lbm-principles-practice/code @ 6e2c592fdc3592c14dfd52f860fc1ceea930bcb0, chapter8/film_inamuro.cpp (MIT) - executed binary, sha256-verified",
    "Kruger, Kusumaatmaja, Kuzmin, Shardt, Silva & Viggen, The Lattice Boltzmann Method: Principles and Practice, Springer 2017 (chapter 5 wetnode=3 scripts, chapter 8 film code)",
    "Inamuro, Yoshino & Ogino, Phys. Fluids 11 (1999) 2924 (anti-bounce-back wet-node closure) - cited for the velocity arm, which was NOT benchmarked"
  ],
  "level_rationale": "L1 met (14 property tests, auditor re-run green). L2 met (new 525-line production module implementing input->compute->output torch operators, exercised end-to-end by the driver). L3 NOT met (0 call sites outside the module; not in boundary_registry / __init__ / capability contract -> unreachable from the solver API). L4 NOT met despite excellent numbers: 0 referee rows, anti-bounce-back arm not_run so the named capability lacks an external benchmark, deliverables untracked (git ls-files empty for all four paths) and the documented repro crashes with FileExistsError. Numerically the scalar closure is validated at 3-5e-11 (30x better than the 15% bar, bit-identical on re-run, with a 0.9967 control probe proving detection power) - the promotion blocker is coverage + registration + commit, not accuracy. L5 impossible (no experimental data).",
  "findings": [
    "F1 blocking: all four deliverables are untracked in agent/m17603 (git status '?? '; git log -- src/tensorlbm/boundary/inamuro_wetnode.py empty) -> the platform auto-push would ship nothing",
    "F2 medium: README repro crashes (FileExistsError, binary/dir name collision, run.py:372-375 vs :204)",
    "F3 medium: anchor names anti-bounce-back but only the scalar residual-weight closure is externally validated; velocity arm not_run",
    "F4 medium: tensor_benchmark_cases has 0 rows for the anchor -> the 15% criterion is self-declared in the driver, not pre-registered",
    "F5 medium: no solver integration (0 external call sites, not in boundary_registry)",
    "F6 minor: run_film() defaults to scalar_policy='all_but_rest', the variant upstream never executes (rel L2 ~0.79 vs reference); only part_film overrides it to all_known",
    "F7 minor: collision/streaming core of the driver runs in NumPy (documented, zero-copy) so the xval exercises the production torch path for the closures only",
    "F8 minor: no tensor_reports row for mission 17603 (paper-grade report owed)"
  ],
  "report_debt": "tensor_reports rows for mission 17603 = 0",
  "audit_limits": [
    "only the film arm was re-derived; the velocity arm is honestly not_run by the mission and was not re-run by the auditor",
    "the upstream binary was rebuilt from the archived source (network-restricted box) rather than a fresh git clone; sha256 identity with the archived fields plus an unpatched source file support authenticity",
    "the TensorLBM side was re-run to 12000 steps (the horizon of the graded rows); the 50000-step upstream horizon was produced but not replicated on the TensorLBM side",
    "no mesh-convergence study: this is a cross-implementation consistency anchor, so its numbers certify a port, not physical accuracy"
  ],
  "disposition": "request_changes",
  "remediation": [
    "commit the four untracked paths to agent/m17603",
    "fix the part_film binary/directory name collision so the README one-liner works on a clean outdir; archive film_ref_*.dat as offline-comparable reference",
    "insert a tensor_benchmark_cases referee row (case_name 'INAMURO-WETNODE-FILM', metric 'rel_l2_field_vs_oss@12000', tolerance '<15%')",
    "default run_film() to scalar_policy='all_known'",
    "complete the velocity arm (constrain wall rows across steps) and report Couette/Poiseuille errors, or narrow the anchor name to the scalar closure",
    "register the closures in boundary_registry + capability contract with at least one solver call site before claiming L3"
  ]
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL;