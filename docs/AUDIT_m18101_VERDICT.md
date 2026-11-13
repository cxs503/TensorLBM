# Board audit — m18101 `XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE` (交差核验)

- **Auditor mission**: 18151 (`board-audit`), repo TensorLBM
- **Audited mission**: 18101 — status `succeeded`, model deepseek-v4-flash, 80 tool calls / 1224 s
- **Audited commit**: `f2f53bd5` on `agent/m18101` (present in `/root/tensor-ecosystem/TensorLBM` as a branch, **not merged to main**)
- **Verdict row**: `tensor_maturity_verdicts` id=700 — `level=4`, `gate_state=pending` (gate skips rows whose audit mission is still `running`; it will adjudicate on mission completion)
- **Gate dry-run on the stored JSON**: `maturity_gate.judge(4, evidence, 'TensorLBM', 18151) -> (4, [])` (all six L4 keys machine-verify)
- **Disposition**: **approve with notes** — L4 confirmed; no material overclaim; two process/coverage gaps recorded below.

---

## 1. What was reviewed

| Deliverable | Pointer | Exists / real |
|---|---|---|
| Kernel under test | `src/tensorlbm/multiphase.py:534` `sc_two_component_force`; called at `:788` from `collide_sc_two_component` | yes, real PyTorch implementation |
| Solver-side caller | `src/tensorlbm/cases/two_layer_poiseuille.py:43`, `src/tensorlbm/mcmp_verification.py:670` (`simulate_planar_slab`) | yes; slab arm time-marches `collide_sc_two_component` + `stream` for 2250–4500 steps |
| Entry point | `xval/.../run_xval.py` (arms A–F, `--slab-case` CLI) | yes, 22.5 kB |
| Independent references | `xval_lib.py:77` forward-gather OSS stencil, `:129` closed-form `S(k)`, `:105/:116` FFT-exact gradient | yes, NumPy only — no import of the kernel |
| Tests | `tests/test_xval_mcmp_cross_force.py` | yes, 9 tests asserting numeric invariants |
| Run artifacts | `artifacts/summary.json` (50 kB), `slab_cases.jsonl`, `fields.npz`, `fig_xval.png` (1650×1200) | yes, all non-empty |
| Platform report | `tensor_reports` id=16985 | yes — sections ①–⑥ present, 1 embedded figure, explicit `<3%` answer |
| Mission DB records | `tensor_xval_cases` id=8 (`status=settled`), `oss_solutions` id=430 | yes |

## 2. Independent reproduction (auditor-run, not read-off)

Scratch copy `/tmp/xval_audit`, `PYTHONPATH=/root/tp-agent/work/m18101/src`, so the **production kernel** was exercised.

- `pytest tests/test_xval_mcmp_cross_force.py -q` → **9 passed in 6.08 s** (claim: 9 passed).
- Analytic arms re-executed (`arm_mode_sweep`, `arm_symbol_refinement`, `arm_momentum`, `arm_invariants`, `arm_mutation`) → **bit-identical** to `summary.json`:

| Quantity | Committed | Auditor rerun | Gate |
|---|---|---|---|
| worst resolved mode, axis | 9.9684 % | 9.9684 % | <15 % PASS |
| worst resolved mode, diagonal | 7.7476 % | 7.7476 % | <15 % PASS |
| refinement order | 1.99912 | 1.99912 | ≈2 PASS |
| kernel ≡ OSS stencil | 7.4506e-09 | 7.4506e-09 | ≤1e-6 PASS |
| kernel ≡ closed form `S(k)` | 7.4386e-09 | (same arm) | ≤1e-6 PASS |
| uniform-field force | 0.0 | 0.0 | ≤1e-14 PASS |
| D2Q9 symmetry (rot90 / transpose) | 7.33e-16 / 7.06e-16 | same | ≤1e-12 PASS |
| net momentum std / edm | 1.83e-17 / 1.83e-19 | same | ≤1e-13 PASS |
| weakest planted defect (M5) | 12.45 % | 12.45 % | 6/6 ≥15 % **FAIL (honest)** |

- Slab arm re-run on the production solver (`artifacts/m18101_rerun_slab_cases.json`):

| Case | standard (committed → rerun) | edm (committed → rerun) | steps (committed → rerun) |
|---|---|---|---|
| n=64, G12=−2.5 | 3.6539 % → 3.6539 % | 3.5371 % → 3.5371 % | 2250 → 2250 (converged) |
| n=96, G12=−2.5 | 3.4865 % → 3.4877 % | 3.3836 % → 3.3845 % | 3000 → 4000 (not converged) |
| n=96, G12=−3.0 | 7.0118 % → 7.0118 % | 6.43039 % → 6.43038 % | 3000 → 4500 (converged) |

Primary claim **7.01 % / 6.43 % < 15 %** reproduces; the ≤0.001 pp drift on the n=96 cases is entirely explained by the different step counts (see finding F2).

Figure: `docs/audit_m18101/fig_reproduction.png` — (a) every error metric committed vs rerun against the 15 % line, (b) invariant residuals vs predeclared limits.

## 3. Severity-ranked findings

**F1 — benchmark case not registered on the platform (non-blocking process gap).**
`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE'` returns **0 rows**. The `<15 %` acceptance therefore lives only in the mission's own `predeclared.json` and in `tensor_xval_cases` id=8 (`status=settled`) — both self-declared. Consequence: the board cannot machine-adjudicate tolerance for this anchor; this audit checked the numbers manually. Related but different anchor `GAP-LBM-MCMP-POISEUILLE` exists in the table and is `pending`, so this is a registration omission, not a policy gap.

**F2 — committed slab metadata inconsistent with the report's own repro command (non-blocking).**
The committed cases record `steps_run = 3000` with `converged=false` for both n=96 cases, while `REPORT.md`'s repro block and this audit rerun use 4000/4500 steps. Re-running at the larger step counts changes the errors by ≤0.001 pp and makes n=96 G12=−3.0 converge — so the *numbers* are robust, only the provenance metadata of the C arm is inconsistent. No error number changes status.

**F3 — falsifiability arm mutates the reference, not the kernel (non-blocking, disclosed).**
`run_xval.py:335-388` plants M1–M6 in the independent NumPy stencil (`xval_lib.oss_force` with mutated weights/directions). This measures the *metric's* sensitivity to classic operator defects, not defect detection inside the kernel. The report labels it "在独立 NumPy 参考实现上种植 6 个缺陷", so this is a scope statement, not an overclaim.

**F4 — "OSS anchor" is a transcription, not executable upstream code (limitation).**
`xval_lib.oss_force` is a hand-written NumPy forward-gather stencil attributed to palabos `shanChenProcessor3D.hh:304` / listLBM `mcmp_mod.f90`; there is no vendored source or hash, and the machine is offline. The identity test therefore proves agreement with the transcription. The load-bearing independent anchors are the *analytic* ones — closed-form `S(k)` and FFT-exact gradient — which are transcription-free and reproduce to 1e-9.

**F5 — coverage boundaries are declared and correct (informational).**
D2Q9 / `ψ=ρ` only; no `wall_mode='dry'/'mirror'`, no D3Q19/D3Q27. `REPORT.md` §⑥ and the platform report state these limits. Not a defect.

**Overstated claims: none found.** Headline 7.01/6.43 %, 9.97/7.75 %, invariants, 5/6 mutation and `gate_pass=false` all match the artifacts bit-for-bit; the falsified hypothesis ("SC does not conserve momentum") is honestly recorded as negative knowledge rather than hidden.

## 4. Level adjudication

- **L2 (real functional implementation)** — met.
- **L3 (integration into the main solver)** — met: the tested symbol is the production D2Q9 MCMP force, reached through `collide_sc_two_component`, and the slab arm runs a real time-marching relaxation (collide → stream) rather than an equilibrium balance.
- **L4 (benchmark validation: complete case + error analysis + reproducible)** — met: 3 relaxed slab cases + 6 analytic arms, error analysis against two analytic references, artifacts committed, and the auditor reproduced all numbers from the kernel.
- **L5 (research/engineering result)** — **not met**: no new capability; external literature is cited but not machine-verifiable offline; no experimental anchor.

**Verdict: L4** (not downgraded by F1–F4, all of which are registration/coverage limitations rather than number or implementation defects).

## 5. Possible follow-up (not required for confirmation)

1. Register the anchor in `tensor_benchmark_cases` so tolerance is machine-adjudicated.
2. Re-emit the C-arm metadata at the step counts actually used (4000/4500) or state 3000 consistently.
3. Extend the XVAL to D3Q19/D3Q27 and to wall modes (`dry`/`mirror`) — new anchors, not a repair of this one.