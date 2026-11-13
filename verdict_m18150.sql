INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorLBM', 'm18049 XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING', 18150, 3, $j${
 "entry": "/root/tp-agent/work/m18049/tools/codegen_pin_audit.py",
 "tests": "/root/tp-agent/work/m18049/tests/test_codegen_pin.py",
 "benchmark": {
  "rows": [],
  "query": "SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING';",
  "registered": false,
  "consequence": "no platform machine criterion for this anchor: ref/got comparison is not applicable; the only re-derivable machine gate is the mission's own drift check (exit code 1 on drift)"
 },
 "repro_cmd": "cd /root/tp-agent/work/m18049 && PYTHONPATH=src python tools/codegen_pin_audit.py && PYTHONPATH=src python -m pytest tests/test_codegen_pin.py -q",
 "impl_files": [
  "/root/tp-agent/work/m18049/src/tensorlbm/codegen_pin.py",
  "/root/tp-agent/work/m18049/src/tensorlbm/triton_fused_obstacle.py"
 ],
 "gate_preview": {
  "note": "the gate holds verdicts pending until the audit mission reaches a terminal state, and it greps the mainline ecosystem checkout for integration callers",
  "status": "previewed_without_side_effects",
  "reasons": [
   "L3 缺: integration_callers(grep 无引用)"
  ],
  "gate_level": 2,
  "link_original": 18049,
  "declared_level": 3
 },
 "run_artifacts": [
  "/root/tp-agent/work/m18049/artifacts/codegen_pin_manifest.json",
  "/root/tp-agent/work/m18049/artifacts/codegen_pin_audit.md",
  "/root/tp-agent/work/m18049/artifacts/codegen_pin_audit.png",
  "/root/tp-agent/work/m18150/artifacts/board_audit/m18150/audit_m18049.json",
  "/root/tp-agent/work/m18150/artifacts/board_audit/m18150/fig_audit_m18049.png",
  "/root/tp-agent/work/m18150/docs/board_audit/m18150/AUDIT_m18049.md"
 ],
 "impl_files_notes": {
  "git": "TRACKED: single deliverable commit ac1c9db6 on branch agent/m18049; worktree clean after audit",
  "new_module": "/root/tp-agent/work/m18049/src/tensorlbm/codegen_pin.py:141 fingerprint_of, :153 DECLARED_PINS, :267 resolve (rejects silent codegen drift), :324 launch_kwargs, :370 assert_comparable, :394 maxnreg_supported, :415 environment_snapshot, :476 PinnedRecorder, :601 audit_codegen (884 lines, 0 bare 'pass' -> passes gate impl_real)",
  "integration": "/root/tp-agent/work/m18049/src/tensorlbm/triton_fused_obstacle.py:62-79 codegen_pin import (local shim fallback), launch sites :1706/:1924/:2175 compose kwargs via _codegen_launch_kwargs, wrappers accept maxnreg at :1651/:1730/:2092"
 },
 "integration_callers": [
  "_codegen_launch_kwargs",
  "/root/tp-agent/work/m18049/src/tensorlbm/triton_fused_obstacle.py:1706",
  "/root/tp-agent/work/m18049/src/tensorlbm/triton_fused_obstacle.py:1924",
  "/root/tp-agent/work/m18049/src/tensorlbm/triton_fused_obstacle.py:2175"
 ],
 "run_artifacts_notes": {
  "png": "125831 B sha256 3edb5081504ab5f8a990e535ed71c07c14842a12f05ea447f8269e2ecf80586e (tools/codegen_pin_figure.py reads the manifest)",
  "manifest": "12783 B sha256 459e7b71675070a4b62fe3fd4eccba4abaf03089b2d247f71cea48e8b95f82e1; declared_pins 5, kernels 13, launch_sites 12, routed 5, maxnreg_supported false, drift 0",
  "reproduction": "auditor re-ran entry: exit 0, regenerated manifest+md byte-identical to committed except environment.timestamp; audited worktree restored to committed state (git status empty)",
  "auditor_artifacts": "/root/tp-agent/work/m18150/artifacts/board_audit/m18150/{audit_m18049.json,fig_audit_m18049.png}; /root/tp-agent/work/m18150/docs/board_audit/m18150/AUDIT_m18049.md"
 },
 "integration_callers_notes": "3 production wrappers route launch kwargs through codegen_pin.launch_kwargs => 5 of 12 launch sites. Runtime launch impossible on this node (no CUDA); routing asserted structurally by an AST test. 7 sites remain inline (tensorlbm._voxel_kernels x4, tensorlbm.triton_fused x3), listed not silently claimed. NOTE: branch agent/m18049 is not merged into the mainline checkout /root/tensor-ecosystem/TensorLBM yet, so the gate's in-mainline grep misses _codegen_launch_kwargs and previews L3->L2 ('integration_callers(grep 无引用)'); inside the audited worktree the identifier appears on 4 lines. Downward gap is mechanical (merge lag), not substantive."
}$j$, 'board-audit');
