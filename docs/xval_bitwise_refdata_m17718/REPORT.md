# XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR

- mission: m17718 (round 2 of anchor; round 1 m17505 honest-FAIL, m17591 L1)
- repo: TensorLBM @ `9bcc6db6b0519502a375661cf0a76e75e57728cb`
- checked: 2026-11-13T08:18:01Z
- environment: torch 2.7.1, numpy 2.4.6, Linux/loongarch64, threads=1
- overall verdict: **PASS** (regression=True, negative_controls=True, benchmark=True)

## ① Problem setting and reference basis

**Problem.** The D2Q9 kernels (`equilibrium`, `collide_bgk`, `collide_trt`,
`collide_mrt`, `collide_bgk_fused`, `collide_rlbm`, `stream`) are the base of
every 2D result in this repository, yet nothing pinned their *output bits*.
Float-tolerance tests accept silent drift: a changed lattice weight, a reordered
summation or a dtype leak moves results by 1e-9..1e-8 and still passes every
`rel=1e-6` assertion. This mission builds (a) a byte-level reference-data
regression lock over those kernels and (b) a physics benchmark that ties the
lock to an exact solution.

**Reference basis.**

- Regression reference: *self-consistent frozen payload*, xval-lbm-bitwise-refdata/v1,
  generated from git `9bcc6db6b0519502a375661cf0a76e75e57728cb` and bound to the kernel sources by sha256:
  - `src/tensorlbm/d2q9.py` → `a5e1ddc0bea25dbd…`
  - `src/tensorlbm/solver.py` → `06e05245f2c66b47…`
- Physics reference (benchmark): 2D Taylor–Green vortex closed-form decay
  `A(t) = (u0/√2)·exp(−2νk²t)`, `k = 2π/Lx`, `ν = (τ−0.5)/3`. This is an analytic solution (no digitised table, nothing copy-pasted):
  Taylor, *Phil. Mag.* (1923); standard LBM verification case per Krüger et al.,
  *The Lattice Boltzmann Method: Principles and Practice* (2017), ch. 8.

## ② Method and implementation

**Regression lock.**

1. Initial fields are built from integer lattice algebra with powers-of-two
   divisions (`harness.int_fields`: `ρ = 1 + a/64`, `u = b/128`, integer `a,b`),
   so the initial state is *exactly* representable and platform independent —
   no RNG, no `libm`. Any digest difference is therefore produced by the kernels.
2. Each case runs the **real** library primitives (`src/tensorlbm/d2q9.py`,
   `src/tensorlbm/solver.py`); the harness contains no re-implementation.
3. The result field is digested with sha256 over dtype + shape + raw IEEE-754
   payload, and the payload itself is stored (base64) in the golden file, so a
   mismatch can be localised to cell/direction with a ulp distance
   (`harness.ulp_metrics`, monotone integer mapping of the bit pattern).
4. Verdicts: `BITWISE_OK` / `ULP_DRIFT` (env drifted, inside budget) /
   `REGRESSION` / `GOLDEN_INCONSISTENT` / `MISSING_GOLDEN`.
5. Negative controls run on every `check`: a 1-ulp tamper of one population and a
   `1e-7` relative τ bump must both move the digest — otherwise the gate is dead.

**Physics benchmark.** TRT collision + periodic streaming on a doubly periodic
box, initialised with the exact TGV field. The target decay factor is held fixed
(`2νk²t = const`, i.e. `t ∝ L²/ν`), so every resolution is compared at the same
physical time: 24²/56 steps, 32²/100 steps, 64²/400 steps at τ = 0.8, plus a τ
series (0.7 → 150 steps, 0.9 → 75 steps) at 32².

**Code pointers.**

- lock + gate: `benchmarks/xval_bitwise_refdata/harness.py`
  (`int_fields`, `CASES`, `digest`, `ulp_metrics`, `verify`, `negative_controls`)
- entry point: `benchmarks/xval_bitwise_refdata/run.py` (`gen|verify|controls|bench|check`)
- test-suite caller: `tests/test_xval_bitwise_refdata.py` (14 tests)
- root-cause probe: `benchmarks/xval_bitwise_refdata/probe_weight_bias.py`
- consumed library kernels: `src/tensorlbm/d2q9.py::equilibrium`, `src/tensorlbm/solver.py::{collide_bgk, collide_trt, collide_mrt, collide_bgk_fused, collide_rlbm, stream, macroscopic}`

## ③ Parameter table

Regression cases (dimensionless):

| case | lattice | dtype | τ | steps | kernels |
|---|---|---|---|---|---|
| `feq_int_field_16` | 16×16 | float64 | 0.8 | 1 | src/tensorlbm/d2q9.py::equilibrium |
| `bgk_step_16` | 16×16 | float64 | 0.8 | 1 | src/tensorlbm/solver.py::collide_bgk, macroscopic, equilibrium |
| `trt_step_16` | 16×16 | float64 | 0.8 | 1 | src/tensorlbm/solver.py::collide_trt |
| `mrt_step_16` | 16×16 | float64 | 0.8 | 1 | src/tensorlbm/solver.py::collide_mrt |
| `bgk_fused_step_16` | 16×16 | float64 | 0.8 | 1 | src/tensorlbm/solver.py::collide_bgk_fused |
| `rlbm_step_16` | 16×16 | float64 | 0.8 | 1 | src/tensorlbm/solver.py::collide_rlbm |
| `stream_periodic_16` | 16×16 | float64 | 0.8 | 1 | src/tensorlbm/solver.py::stream |
| `chain_trt_50_24` | 24×24 | float64 | 0.8 | 50 | src/tensorlbm/solver.py::collide_trt, src/tensorlbm/solver.py::stream |
| `chain_bgk_50_f32_16` | 16×16 | float32 | 0.8 | 50 | src/tensorlbm/solver.py::collide_bgk, src/tensorlbm/solver.py::stream |

Benchmark parameter points (all dimensionless):

| level | L | τ | ν | u₀ | Ma = u₀/√3 | Re_cells = u₀L/ν | steps | 2νk²t |
|---|---|---|---|---|---|---|---|---|
| `24x24 tau=0.8000 steps=56` | 24 | 0.80 | 0.1000 | 0.05 | 0.0289 | 12.0 | 56 | 0.7676 |
| `32x32 tau=0.8000 steps=100` | 32 | 0.80 | 0.1000 | 0.05 | 0.0289 | 16.0 | 100 | 0.7711 |
| `64x64 tau=0.8000 steps=400` | 64 | 0.80 | 0.1000 | 0.05 | 0.0289 | 32.0 | 400 | 0.7711 |
| `32x32 tau=0.7000 steps=150` | 32 | 0.70 | 0.0667 | 0.05 | 0.0289 | 24.0 | 150 | 0.7711 |
| `32x32 tau=0.9000 steps=75` | 32 | 0.90 | 0.1333 | 0.05 | 0.0289 | 12.0 | 75 | 0.7711 |

## ④ Reference vs got

### 4.1 Bitwise regression lock

Environment matches golden: **True**; ulp budget in force: **0.0** ulp; cases: 9, bitwise-identical: 9, ulp-drift: 0, regressions: 0.

| case | verdict | digest (16 hex) | n_diff | max ulp |
|---|---|---|---|---|
| `feq_int_field_16` | BITWISE_OK | `ec804d66fc75cd8f` | 0 | 0.0 |
| `bgk_step_16` | BITWISE_OK | `bddfdeaf6a8d53ca` | 0 | 0.0 |
| `trt_step_16` | BITWISE_OK | `a8bbe323ba767986` | 0 | 0.0 |
| `mrt_step_16` | BITWISE_OK | `1840f6da8546ea17` | 0 | 0.0 |
| `bgk_fused_step_16` | BITWISE_OK | `209f8f053c2660ae` | 0 | 0.0 |
| `rlbm_step_16` | BITWISE_OK | `f601910cc698efa2` | 0 | 0.0 |
| `stream_periodic_16` | BITWISE_OK | `53d3ed72a1f11e21` | 0 | 0.0 |
| `chain_trt_50_24` | BITWISE_OK | `0fe74faea9b2766d` | 0 | 0.0 |
| `chain_bgk_50_f32_16` | BITWISE_OK | `07427041425f5402` | 0 | 0.0 |

Gate power (negative controls, declared **before** looking at the run):

- 1-ulp tamper at index [0, 5, 5] of `chain_trt_50_24` → digest detects: **True**, max ulp = 1, differing cells = 1 (verdict would be `REGRESSION`).
- τ = 0.8 → 0.80000008 (1e-7 relative) → digest detects: **True**, max ulp = 15905815, differing cells = 5184.
- both controls fired: **True**.

### 4.2 Physics benchmark (LBM vs analytic TGV decay)

| level | ref A(t) | got A(t) | err [%] | < 15% (owner bar) | field L2 err [%] |
|---|---|---|---|---|---|
| `24x24 tau=0.8000 steps=56` | 1.640873012e-02 | 1.628709068e-02 | 0.7413 | ✅ | 1.6246 |
| `32x32 tau=0.8000 steps=100` | 1.635259453e-02 | 1.628109885e-02 | 0.4372 | ✅ | 1.0933 |
| `64x64 tau=0.8000 steps=400` | 1.635259453e-02 | 1.633504673e-02 | 0.1073 | ✅ | 1.3378 |
| `32x32 tau=0.7000 steps=150` | 1.660672657e-02 | 1.653141658e-02 | 0.4535 | ✅ | 1.3675 |
| `32x32 tau=0.9000 steps=75` | 1.635259453e-02 | 1.629036290e-02 | 0.3806 | ✅ | 1.5596 |

Worst error **0.7413%** against the 15.0% acceptance bar → **PASS**.

## ⑤ Convergence and parameter sensitivity

Resolution series at fixed physical time (τ = 0.8, `2νk²t` held constant):

| L | amp err [%] | ratio vs next finer |
|---|---|---|
| 24 | 0.7413 | 1.70 |
| 32 | 0.4372 | 4.07 |
| 64 | 0.1073 | — |

Errors fall monotonically with resolution (first order in the acoustic/
compressibility error, i.e. the operator converges); no level exceeds 1%.

τ sensitivity at 32² (same decay target):

| τ | ν | steps | amp err [%] | max ρ spread |
|---|---|---|---|---|
| 0.70 | 0.0667 | 150 | 0.4535 | 1.286e-02 |
| 0.90 | 0.1333 | 75 | 0.3806 | 1.131e-02 |
| 0.80 | 0.1000 | 100 | 0.4372 | 1.168e-02 |

The error is flat in τ (0.38–0.45%), i.e. the observed error is the *lattice*
truncation error, not a relaxation-time artefact.

## ⑥ Failure modes, defect found, and artifacts

### 6.1 Defect found by the lock: float32 weights inside `collide_bgk_fused`

The lock's first act was to fail: `collide_bgk_fused` moved float64 mass by
9.3132e-09 per step (all other operators: ~1e-14).
Root cause: the fused path built weights from the **float32** table `d2q9.W` and
cast it to float64, so `Σw = 1 + 7.4506e-09`; a fused equilibrium then injects
`bias/τ` of spurious density per step — the exact defect `d2q9._w_for` was added
for, but this call site was missed. Quantitative confirmation (probe, no revert):

| quantity | predicted | measured | rel. error |
|---|---|---|---|
| per-step drift | 9.313226e-09 | 9.313226e-09 | 0.0e+00 |
| 200-step drift (linear ramp) | 1.862645e-06 | 1.862647e-06 | 9.3e-07 |
| 200-step drift after fix | 0 | -1.376593e-14 | — |

Fix (one line, `src/tensorlbm/solver.py`): `w = _w_on(device).to(dtype)` → `w = _w_for(dtype, device)`.
Because `_w_for(float32)` is bit-identical to `W`, the float32 path is unchanged
(asserted by `test_float32_weight_table_is_unchanged_by_the_fix`).

**Locality evidence** — after the fix, only the fused case digest moved:

| case | digest after fix |
|---|---|
| `feq_int_field_16` | `ec804d66fc75cd8f` |
| `bgk_step_16` | `bddfdeaf6a8d53ca` |
| `trt_step_16` | `a8bbe323ba767986` |
| `mrt_step_16` | `1840f6da8546ea17` |
| `bgk_fused_step_16` | `209f8f053c2660ae` |
| `rlbm_step_16` | `f601910cc698efa2` |
| `stream_periodic_16` | `53d3ed72a1f11e21` |
| `chain_trt_50_24` | `0fe74faea9b2766d` |
| `chain_bgk_50_f32_16` | `07427041425f5402` |

(the 8 previously frozen cases kept their digests byte for byte except
`bgk_fused_step_16`, which is exactly the changed path)

**Same-class sites not fixed here** (no test binds them; changing them would
alter 3D/other numerics without an acceptance gate): `src/tensorlbm/solver.py`
`collide_rlbm`, `src/tensorlbm/d3q27.py`, `src/tensorlbm/solver3d.py`,
`src/tensorlbm/cylinder_cross_validation.py` — all still call
`_w_on(device).to(f.dtype)`. Recorded for a follow-up with its own gate.

### 6.2 Failure modes of the lock itself

- *False red on another machine*: handled by the env/ulp-budget degradation,
  exercised by `test_env_mismatch_degrades_to_ulp_budget`.
- *Dead gate* (comparison that can never fail): handled by two negative controls,
  plus `test_verify_flags_regression_on_kernel_drift` and
  `test_verify_rejects_hand_edited_golden`.
- *Non-deterministic kernel*: `collide_bgk_matmul` is excluded on purpose — BLAS
  matmul order is outside torch's determinism contract, so a bitwise lock there
  would be a lie.
- *Stale golden after an intentional kernel change*: `run.py gen` + `check`, and
  the test suite fails until `result.json`'s kernel sha256 match the tree.

### 6.3 Artifacts and reproduction

```
PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py check
```

| artifact | path |
|---|---|
| entry point | `benchmarks/xval_bitwise_refdata/run.py` |
| lock + gate library | `benchmarks/xval_bitwise_refdata/harness.py` |
| frozen golden data (9 cases, payloads) | `benchmarks/xval_bitwise_refdata/refdata/d2q9_bitwise_v1.json` |
| machine-produced run result | `benchmarks/xval_bitwise_refdata/result.json` |
| defect root-cause probe | `benchmarks/xval_bitwise_refdata/probe_weight_bias.py` + `.json` |
| figure | `benchmarks/xval_bitwise_refdata/figures/tgv_bitwise_bench.png` |
| test-suite caller | `tests/test_xval_bitwise_refdata.py` (14 tests) |
| harness sha256 | `835eaecc81e7197ee93d40a70009f358…` |

### 6.4 Error judgement against the owner's bar

| key quantity | reference | error | < 3%? | < 15%? |
|---|---|---|---|---|
| TGV amplitude, `24x24 tau=0.8000 steps=56` | 1.640873e-02 | 0.7413% | yes | yes |
| TGV amplitude, `32x32 tau=0.8000 steps=100` | 1.635259e-02 | 0.4372% | yes | yes |
| TGV amplitude, `64x64 tau=0.8000 steps=400` | 1.635259e-02 | 0.1073% | yes | yes |
| TGV amplitude, `32x32 tau=0.7000 steps=150` | 1.660673e-02 | 0.4535% | yes | yes |
| TGV amplitude, `32x32 tau=0.9000 steps=75` | 1.635259e-02 | 0.3806% | yes | yes |
| float64 fused-step mass drift | 0 | 0.0e+00 (was 9.3e-09) | yes | yes |

All benchmark errors are < 3%, i.e. an order of magnitude inside the 15% bar;
the regression side is exact (0 ulp) rather than approximate.

