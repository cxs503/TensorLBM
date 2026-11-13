# Board audit — m18049 XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING

**Disposition: approve with notes. Verdict level L3.**
Audited repo `TensorLBM`, branch `agent/m18049`, commit `ac1c9db6e951f0fd735e36b1ea22d6034a2ee6b4`,
worktree `/root/tp-agent/work/m18049` (restored to committed state after audit:
`worktree_restored=True`).
Audit time 2026-11-13T12:15:47.194141+00:00. Auditor mission 18150, `verdict_by='board-audit'`.

## 1. What the mission claims and what was re-derived

Reconstruction was done from files, not from the report: the mission's own tool was
re-executed (`/usr/local/bin/python tools/codegen_pin_audit.py`, exit **0**), its tests were re-run, and every
published number was re-read from `artifacts/codegen_pin_manifest.json`.

| id | claim as published | status | independent method | measured |
|---|---|---|---|---|
| C1 | 5 Triton modules / 13 @triton.jit kernels / 12 launch sites audited | **VERIFIED** | independent AST re-run of tools/codegen_pin_audit.py | `{"modules": 5, "kernels": 13, "launch_sites": 12}` |
| C2 | 5 of 12 launch sites routed through codegen_pin.launch_kwargs | **VERIFIED** | manifest launch_sites[].pinned_via == codegen_pin; tests/test_codegen_pin.py::test_production_obstacle_launches_are_routed_through_codegen_pin | `{"routed": 5, "total": 12}` |
| C3 | 0 of 5 declared pins fix a maxnreg value (backend cannot accept one here) | **VERIFIED** | manifest declared_pins[*].knobs.maxnreg; environment.maxnreg_supported | `{"pins_with_maxnreg": 0, "maxnreg_supported": false}` |
| C4 | drift_findings == 0 and the drift gate actually fires on a lying pin | **VERIFIED** | fresh tool run (exit code) + test_audit_detects_drift_when_a_pin_lies | `{"tool_exit_code": 0, "drift_findings": 0, "gate_tests": ["5 passed, 19 deselected, 1 warning in 5.44s"]}` |
| C5 | 24 tests green | **VERIFIED** | fresh pytest run of tests/test_codegen_pin.py | `{"pytest_exit": 0, "pytest_tail": ["24 passed, 1 warning in 6.92s"]}` |
| C6 | artifacts are reproducible (tool regenerates them from source) | **VERIFIED** | sha256 of committed vs regenerated artifacts; JSON compared with environment.timestamp stripped | `{"committed_sha256": {"codegen_pin_manifest.json": {"sha256": "459e7b71675070a4b62fe3fd4eccba4abaf03089b2d247f71cea48e8b95f82e1", "bytes": 12783}, "codegen_pin_audit.md": {"sha256": "cfbb30b43358801951e277dad70d0dbf0a97657915bb668b46ee3bf0b9834d36", "bytes": 3636}, "codegen_pin_audit.png": {"sha256": "3edb5081504ab5f8a990e535ed71c07c14842a12f05ea447f8269e2ecf80586e", "bytes": 125831}}, "regenerated_sha256": {"codegen_pin_manifest.json": "c3456d699ace187e387c432a613e02de3de362a7f47b262258398e2023f46160", "codegen_pin_audit.md": "6db1979327de3ed5b06e064beac8eb3c6d9f4bfc8102b6b52db328637787dfd4"}, "identical_except_timestamp": true, "worktree_restored": true}` |
| C7 | maxnreg honoured on hardware / register count measured | **WITHHELD** | no CUDA backend on the audit host; mission itself declares honest FAIL | `{"cuda_available": false, "maxnreg_supported": false, "nvcc_version": null}` |
| C8 | remaining inline launch sites = 'Six' (report section 5) with enumeration 3+4 | **INCORRECT_COUNT** | manifest arithmetic: 12 - 5 = 7; enumeration 3 + 4 = 7 | `{"unrouted": 7, "by_module": {"tensorlbm._voxel_kernels": 4, "tensorlbm.triton_fused": 3}}` |

Reproducibility: re-running the tool regenerated the manifest and the audit markdown
**byte-identical to the committed artifacts except `environment.timestamp`**
(`identical_to_committed_except_timestamp=True`).

## 2. Benchmark criterion (the only judge)

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING';
```

Rows returned: **0** — `registered=False`.
no platform machine criterion for this anchor: ref/got comparison is not applicable; the only re-derivable machine gate is the mission's own drift check (exit code 1 on drift).

Consequence for grading: there is no `ref` value to put next to a `got` value, so the
ref-vs-got error table required by a Level-4 verdict **cannot be produced for this anchor**.
The nearest machine-checkable gate is the mission's own drift gate, which was re-run and
**fires** (proven by `test_audit_detects_drift_when_a_pin_lies`, which asserts that a
deliberately wrong pin yields >=2 drift findings, and by the tool's zero-drift exit code).

## 3. Re-derived quantities

| quantity | value (from manifest, re-read) |
|---|---|
| `@triton.jit` kernels | 13 |
| launch sites | 12 |
| launch sites routed through `codegen_pin.launch_kwargs` | 5 |
| launch sites forwarding `maxnreg` | 5 |
| declared pins | 5 |
| declared pins that fix `maxnreg` | 0 |
| distinct pin fingerprints | 4 |
| drift findings (fresh run) | 0 |
| remaining inline launch sites | 7 (tensorlbm._voxel_kernels x4 / tensorlbm.triton_fused x3) |
| host | loongarch64, python 3.11.6, torch 2.7.1, triton 3.4.0 |
| `cuda_available` / `maxnreg_supported` / `nvcc` | False / False / None |

Tests re-run by the auditor: `['24 passed, 1 warning in 6.92s']` (exit 0);
drift/routing subset: `['5 passed, 19 deselected, 1 warning in 5.44s']`.

## 4. Findings (severity-ranked)

| id | severity | finding | evidence | impact |
|---|---|---|---|---|
| F1 | high-for-L4 | the anchor's hard mechanism (a fixed register cap) is declared but never fixed: 0/5 pins set maxnreg | `artifacts/codegen_pin_manifest.json declared_pins[*].knobs.maxnreg; environment.maxnreg_supported=false` | blocks L4; honestly declared by the mission itself (report section 6 'honest FAIL / not_run'), so it is a capability gap, not an overclaim |
| F2 | low | report section 5 says 'Six launch sites (triton_fused.py x3, _voxel_kernels.py x4)'; the enumeration and the manifest both give 7 | `manifest launch_sites with pinned_via == null` | documented remaining-work count is wrong by one; routing itself is honest |
| F3 | low | the generated audit markdown renders '<num_warps>/<num_stages>/<maxnreg>' placeholders for launch sites because the manifest stores null knob values there | `artifacts/codegen_pin_audit.md table rows for routed sites` | the md alone cannot be used to read the codegen numbers of a site; the pins table must be consulted |
| F4 | medium-governance | feature_ledger acceptance for this anchor is the generic owner accuracy floor '<15%' which is inapplicable to a launch-configuration governance change; hence 0 rows in tensor_benchmark_cases | `SELECT acceptance FROM feature_ledger WHERE slug=anchor -> '<15% (owner precision standard)'` | anchor cannot ever reach L4 until acceptance is restated in machine-checkable form (drift_findings==0, routed sites, gate exit code) |
| F5 | low | the import guard in triton_fused_obstacle.py falls back to a local shim that drops pins silently if the package import breaks | `src/tensorlbm/triton_fused_obstacle.py:66-75` | pins could degrade without an audit failure in a broken layout; the fallback preserves pre-change behaviour, so no regression |
| F6 | medium-governance | maturity_gate's L3 check greps the ecosystem MAINLINE checkout, which cannot see a freshly delivered branch; previewing judge() on this evidence yields 2 with reason 'L3 missing integration_callers(grep)' | `gate preview recorded in audit_m18049.json:gate_preview; ecosystem checkout main at a commit predating the audited commit, while branch agent/m18049 holds it` | a genuinely integrated delivery reads one level lower until its merge lands; mechanical, not substantive -- the verdict here stays L3 and the predicted gate outcome is stated explicitly instead of being hidden |

F1 is the only finding that touches the anchor's substance: the upstream practice is to **fix**
`-maxrregcount`; here the cap is `None` in every pin because no CUDA backend exists on the
audit node. The mission says so itself (report section 6, "honest FAIL / not_run"), so this is a
declared capability gap rather than an overstated claim — it is what holds the verdict at L3.

F2 is a real (if cosmetic) inaccuracy in the published report: the remaining-work sentence says
"Six" while the enumeration it prints (3 + 4) and the manifest both give
7. F4 is the systemic cause of the missing benchmark row and is the
actionable item for whoever reopens this anchor.

## 5. External anchor mapping

Upstream: `AminZar69/CudaPhaseField` — pin the register budget (-maxrregcount) before any performance tuning.
Triton analogue: `{"maxnreg": "-maxrregcount", "num_warps": "__launch_bounds__ threads", "num_stages": "pipeline depth / unroll"}`.
Hardware transfer verified: **False** — audit host is loongarch64 without CUDA; the analogue is declared and unit-tested with the support probe monkeypatched, not run on a GPU.

## 6. Level adjudication

Rubric: L0=documents only; L1=math/unit verification; L2=independent functional implementation (real code in->compute->out); L3=system integration (wired into the main solver/platform); L4=benchmark validation (full case + error analysis + reproducible); L5=research/engineering result.

**L3.** Not lower: 884-line real implementation + 24 passing tests + reproducible artifact generator + a drift gate that is proven to fire. Not L2:
the pin table is not an isolated module: 3 production wrappers / 5 launch sites in src/tensorlbm/triton_fused_obstacle.py compose kwargs through codegen_pin.launch_kwargs, and an AST test pins that routing. Not L4: no benchmark case is registered for this anchor in tensor_benchmark_cases (0 rows), no measured quantity exists (no GPU), and maxnreg is fixed in 0/5 pins -- so there is no ref-vs-got error to adjudicate.

## 7. Checks not run (unknown, not passed)

- GPU launch of any Triton kernel (no CUDA backend on loong64 audit node)
- maxnreg honoured by ptxas (requires nvcc + GPU)
- MLUPS / n_regs measurement (explicitly out of scope for the mission, honest FAIL)
- registry/capability-contract presence for codegen_pin in src/tensorlbm/__init__.py (absent by design; integration is via launch_kwargs import)

## 8. Platform gate preview (measured, not assumed)

`maturity_gate.judge(3, evidence, 'TensorLBM', 18150)`
was executed in-process (no side effects) on the exact evidence submitted above and returned
**gate_level=2** with reasons
`['L3 缺: integration_callers(grep 无引用)']` (link_original -> m18049). The gap is the gate's L3 rule: it greps the *mainline*
ecosystem checkout for integration callers, and branch `agent/m18049` is not merged there yet,
so a correctly wired L3 delivery reads L2 until the merge lands. This is finding F6: a
mechanical downgrade on merge lag, stated here instead of being hidden. The verdict stays
**L3** on functional substance.

## Artifacts
- `artifacts/board_audit/m18150/audit_m18049.json` (machine-readable evidence)
- `artifacts/board_audit/m18150/fig_audit_m18049.png` (claim status + pin coverage, drawn from the JSON)
- `verdict_m18150.sql` (row inserted into `tensor_maturity_verdicts`)
- audited worktree left clean: `git -C /root/tp-agent/work/m18049 status --porcelain` → empty
