# XVAL: AA-pattern fused stream-collide — operator-ordering audit and fix

**Anchor**: `XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE` (lanl/MF-LBM
`aa-pattern-fused-stream-collide`) · **Mission**: m21666 · **Date**: 2026-11-15
**Disposition of the review**: *request changes* → **changed and re-validated**
(module fixed, defect-characterisation tests retargeted, new non-degenerate gates).

---

## 1. Claim under review

`docs/xval_aa_pattern_fused_stream_collide.md` (m17976, 2026-11-13) concluded:

1. "For periodic BCs the AA-pattern and separate approach produce **bit-identical**
   results (max diff 5.55e-17)" → equivalence **PASS**;
2. cavity at n=33: `L2_u = 0.2472` for AA vs `0.0957` for separate, with a 93.9 %
   mass difference attributed to "BC ordering", and the accuracy gap blamed on
   "grid too coarse" (n=33);
3. the module `tensorlbm/aa_fused_stream_collide.py` "implements the approach
   used in lanl/MF-LBM".

The audit below tests (1)–(3) independently, from the code and from the operator
algebra, not from the report.

## 2. Findings (severity-ranked)

### F1 — BLOCKING (correctness): the module was **not** the AA-pattern scheme

The AA fused pull form takes the **collision increment at the source node**

```
f_i(x, t+1) = f_i(x - c_i, t) + omega [ f_i^eq(x - c_i, t) - f_i(x - c_i, t) ]
```

so the increment rides along with the same pull and the result is *identical* to
"collide everywhere, then stream". The audited code instead pulled
**pre-collision** populations and collided **at the destination**:

```
g_i(x) = f_i(x - c_i) ;  f_i(x, t+1) = g_i + omega (f_i^eq(<g>) - g_i)      # C∘S
```

Streaming `S` and BGK collision `C` do not commute (the equilibrium is a
nonlinear functional of the *local* moments), so this relaxes towards the
equilibrium of the **upwind-averaged** moments. Measured deviation from the
canonical operator: **2.31e-3 relative, single step**, and it persists over a
run (2.31e-3 → 8.5e-4 after 200 steps; `ordering_audit.json`,
`E1_E2_operator_ordering.divergence_trace_nonuniform`). The module docstring
itself mis-stated the pattern ("ordering of operations differs when boundaries
are present" — the ordering difference is in the *bulk* operator, not a boundary
effect).

### F2 — BLOCKING (evidence): the equivalence test was degenerate

`tests/test_aa_fused_stream_collide.py` compared the two operators starting from
`f = 0.1 * ones; f[0] = 0.5` — a **spatially uniform** state. On a uniform state
every stencil is the identity, so *any* two operators agree exactly:

| initial state | fused vs canonical | legacy `C∘S` vs canonical |
|---|---|---|
| uniform (old test IC) | 0.0 | **0.0** |
| non-uniform (shear, `u = A sin(ky) e_x`) | 0.0 | **2.31e-3** |

The published "5.55e-17 PASS" therefore asserted nothing; a *wrong* operator
passes it. This is the root cause of F1 surviving review.

### F3 — MAJOR (physics/BC): the wall treatment leaked mass

`aa_fused_step` applied bounce-back on *solid* nodes **after** streaming (full-way
style), which does not conserve mass in a closed box. The repository already
characterised this (`tests/test_aa_fused_push_pull.py::
test_legacy_kernel_leaks_mass_with_walls`, drift > 1e-8). Consequence: **every
wall-bounded number in the published report — including the cavity table and the
93.9 % "mass difference" — came from a kernel with a known mass leak**, so the
"BC ordering" explanation in §5.2 of that report is not established. The 93.9 %
mass difference is a conservation defect, not an ordering effect.

### F4 — MAJOR (honest negative): smooth single-mode flows cannot discriminate the orderings

For the exact shear-wave solution `u = A sin(ky) exp(-nu k^2 t)` (`Re`-free
anchor; `ny=64`, `A=0.02`, `nu=0.05`, `tau=0.65`, 4000 steps) **all three**
operators reproduce the decay rate to the same 0.073 % and the final amplitude to
0.214 %:

| scheme | `nu_eff/nu - 1` | final amplitude rel. error |
|---|---|---|
| canonical collide-then-stream | +0.073 % | 0.214 % |
| AA fused pull (fixed) | +0.073 % | 0.214 % |
| legacy `C∘S` | +0.073 % | 0.214 % |

The ordering defect lives in the **higher harmonics / strong local
non-equilibrium**, so a smooth benchmark cannot be used to certify the pattern.
Recorded as a negative result: no accuracy claim about the AA pattern should rest
on single-mode flows.

### F5 — UNRESOLVED (not run): 4-arm cavity ablation

Whether the published cavity FAIL (`L2_u = 0.247`) is grid coarseness or a
kernel/BC defect can only be answered by an ablation that separates the two
effects. It is implemented and **submitted to the scheduler, not finished**:

* arms: `fused+halfwayBB`, `streamThenCollide+halfwayBB`, `fused+legacyBC`,
  `audited(legacy module)`; `L=64` (150 000 steps) and `L=128` (200 000 steps)
  vs Ghia et al. (1982), Re=100;
* job **cj3609** (5 shards), `run_compute`, submitted 2026-11-15 09:27, status
  `queued` at write time (58 jobs in flight, position ≈ 39).

**No cavity number is quoted in this document.** The previously published
L2 values are withdrawn as accuracy evidence (F3), not replaced by new ones.

## 3. Fix

`src/tensorlbm/aa_fused_stream_collide.py` rewritten around the correct algebra:

* `aa_fused_stream_collide(f, tau)` — AA fused pull, `f_i + inc_i` pulled from the
  source, no post-collision array. Bit-identical to the canonical operator.
* `separate_stream_collide` / `separate_step` — materialised canonical oracle.
* `aa_fused_step` — fixed step: fused pull + **halfway bounce-back** (stationary
  or moving wall, `lid_row` preserved for API compatibility); mass conserving to
  machine precision in a closed box.
* `stream_then_collide_legacy` — the retired `C∘S` operator, kept **only** as a
  negative control so the 2026-11-13 numbers remain reproducible.

### Verification (all run locally, results in the repo)

| check | result |
|---|---|
| fused vs canonical, non-uniform, 1 step | `torch.equal` (max diff **0.0**) |
| fused vs canonical, non-uniform, 200 steps | `torch.equal` |
| fused vs independent kernel `fused_push_pull_step` | < 1e-14 |
| fused vs materialised step with walls + moving lid (1 and 50 steps) | `torch.equal` |
| mass drift, closed walled box, 500 steps (perturbed rest state) | **< 1e-12** (was > 1e-8) |
| shear-wave viscosity vs analytic | 0.073 % (< 1 % gate) |
| retired `C∘S` still measurably different (negative control) | 2.31e-3 per step |
| `pytest tests/test_aa_fused_stream_collide.py` | **13 passed** in 14.5 s |
| `pytest tests/test_aa_fused_push_pull.py` | **11 passed** in 44 s |

## 4. Artifacts

| path | content |
|---|---|
| `benchmarks/xval/aa_fused_stream_collide_ordering_audit.py` | audit driver (E1/E2 ordering, E3 shear wave, E4 sharded cavity) |
| `artifacts/xval_aa_pattern/ordering_audit.json` | raw E1–E3 numbers |
| `artifacts/xval_aa_pattern/fig1_ordering_equivalence.png` | degenerate-test evidence + divergence trace |
| `artifacts/xval_aa_pattern/fig2_shear_wave_viscosity.png` | viscous decay vs analytic; `nu_eff` errors |
| `tests/gen_aa_fused_ordering_figures.py` | figure generator (also plots cavity arms when present) |
| `tests/test_aa_fused_stream_collide.py` | new non-degenerate gate suite |
| `tests/test_aa_fused_push_pull.py` | defect characterisation retargeted to the retired operator + re-validation of the fixed step |

Reproduce:

```
cd TensorLBM
python benchmarks/xval/aa_fused_stream_collide_ordering_audit.py --skip-cavity   # ~19 s
python benchmarks/xval/aa_fused_stream_collide_ordering_audit.py --arm-index 0    # one cavity arm
python tests/gen_aa_fused_ordering_figures.py
pytest tests/test_aa_fused_stream_collide.py tests/test_aa_fused_push_pull.py -q
```

## 5. Limitations / remaining work

1. **Cavity ablation not finished** (F5, job cj3609). Until it lands, the anchor's
   `< 15 %` acceptance is **not** demonstrated for this module; the only
   *validated* quantitative accuracy statements are the exact-solution ones
   (shear-wave viscosity 0.073 %) and the walled-box mass conservation.
2. The audit covers D2Q9/BGK only; nothing here transfers automatically to MRT or
   D3Q19/27 variants of the pattern.
3. `stream_then_collide_legacy` is intentionally retained; it must not be used as
   a solver. Its mass leak is pinned by a failing-if-fixed test.
4. `draw`: the ordering defect is O(2e-3) per step in smooth flows — a
   `< 3 %` end-to-end gate can pass *despite* it (F4). Structural equivalence
   tests (`torch.equal` against the canonical operator) are the right gate, not
   profile errors.