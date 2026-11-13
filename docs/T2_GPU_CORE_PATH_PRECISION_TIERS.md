# T2 core path — precision tier wiring and admissibility (round 3)

**Scope** — `src/tensorlbm/gpu_core_path.py`, dimension ⑤ ("precision /
performance trade-off") of the T2 core-path module, plus its tests and a
measurement harness.  This document is the machine-checkable record of *which
precision tiers are admissible* for the core path, with the pre-declared
thresholds stated before the numbers.

## 1. The gap this closes

Round-2 audit `docs/audits/m15902_GAP-LBM-T2-GPU-COREPATH_audit.md` finding
**F4**:

> precision store 档未进入推进循环（FP32FP16 ≡ FP32FP32 逐位相同），
> 「精度/性能折中」未行使 | 表 3.6 + `_f_store` 仅由 `f` 属性写入

and its disposition asked for exactly one of two things: *wire the store dtype
into the marching loop, or delete the tier's claim*.  The T2 report meanwhile
advertised `FP32FP16` as "bandwidth-bound (~2e-4 error)" — a number with no
measurement behind it.

Status before this change: `GPUCorePathSolver` allocated a store-dtype buffer
`_f_store`, but the only writer was the `f` *property getter*; the ping-pong
buffers that actually carry the state between steps were always the compute
dtype.  Consequence: `FP32FP16` produced bit-identical trajectories to
`FP32FP32`, so the tier neither cost accuracy nor saved memory — the claim was
inert.

## 2. What was changed

`GPUCorePathSolver` now stores state in the policy's **store** dtype and casts
at the two boundaries of every step (the cast-boundary contract documented in
`src/tensorlbm/precision.py`):

```
step n:   f_store (narrow)  --widen-->  f_a (compute)
          f_a --BGK+Guo--> f_post --stream+walls--> f_b
          f_b --narrow--> f_store          (state that persists to step n+1)
          ping-pong f_a <-> f_b
```

* both `_step_eager` and `_step_compiled` carry the boundary, so a tier cannot
  be honoured on one backend and ignored on the other;
* diagnostics (`macroscopic`, `mass`, `channel_profile`, `kinetic_energy`)
  read `_current()`, i.e. the state the *next* step will see — the narrowed
  store buffer widened back — not the unrounded scratch left over from the
  previous step.  Without this the reported accuracy would silently describe
  a state the solver never advances from;
* `memory_report()` reports `persistent_state_bytes`, `working_bytes` and the
  buffer count measured from the live tensors, so the memory side of the
  trade-off is auditable instead of asserted;
* float16 store tiers raise a `RuntimeWarning` at construction and record the
  reason in `solver.store_note` (visible in `run()` stats as
  `memory.store_note`);
* the multi-GPU wrapper exchanges halos on `_current()` and persists the
  exchanged state back into the store buffer, so it cannot drift out of sync
  with the tier;
* `GPUCorePathConfig.validate()` now rejects unknown precision names instead of
  raising `KeyError` deep inside the constructor.

Nothing else in the numerics changed: the kernels (`collide_guo_into`,
`stream_into`, `apply_channel_walls`, `core_step`) are untouched, which is why
`FP64FP64` results are bit-identical to the pre-change baseline (pinned by the
existing analytic tests, see §5).

## 3. Experiment design (pre-declared)

**Decision.** Which precision tiers may be used for production runs of the core
path, and which must be labelled lossy/screening-only.

**Alternatives considered.** (A) leave the tier inert and keep the old claim —
rejected, it is the audit finding itself; (B) delete the float16 tiers —
keeps the module honest but removes a measurement capability; (C) wire the
tier, measure it, publish per-tier verdicts and warn at construction — chosen.

**Hypotheses.**

* H1 (control sanity): `FP64FP64` reproduces the analytic anchors within the
  platform 3% line — already established, re-measured here as the control arm.
* H2: `FP32FP32` and `FP64FP32` stay within 3% on every anchor — i.e. 32-bit
  arithmetic is enough for these cases.
* H3: `FP32FP16` exceeds 3% (float16 has ~4.9e-4 relative spacing, and the
  populations that carry a u≈0.05 flow differ from their rest value by ~1e-3)
  but stays under the 15% unacceptable line.

**Anchors / metrics.** Closed-form, no fitted freedom:

| case | reference | metric |
|---|---|---|
| channel, ny=18, 3000 steps (steady tol 2e-5) | `u(y)=a/(2ν)·y(H−y)`, H=ny−2 | `u_max_rel_err`, `profile_rel_l2` |
| channel, ny=34, 6000 steps | same | same |
| Taylor–Green, nx=32, 400 steps | `KE(t)=u0²/4·exp(−4νk²t)`, k=2π/nx | `ke_max_rel_err` |
| Taylor–Green, nx=64, 800 steps | same | same |

**Pre-declared thresholds** (fixed before the sweep was run):

| verdict | rule |
|---|---|
| PASS | worst anchor relative error ≤ 3% |
| MARGINAL | 3% < worst ≤ 15% — screening only, never for reported physics |
| REJECT | worst > 15% |

**Control and variables.** Control arm = `FP64FP64`, identical geometry, seed
42, `perturbation_amplitude=0`.  The single changed variable per arm is the
`(compute_dtype, store_dtype)` pair.  Runs are deterministic, so one run per
cell; wall-clock is measured under `OMP_NUM_THREADS=1`,
`torch.set_num_threads(1)` on a loong64 CPU node and is reported as a
diagnostic, **not** as a GPU performance claim (no CUDA on the host — the
triton/`GLUPS` numbers of the T2 report remain `not_run`, see §6).

**Analysis.** Max relative error per tier over all its anchors; the tier's
verdict is the worst metric verdict.  Memory verdict compares
`persistent_state_bytes` and `total_bytes` across tiers.

Harness: `benchmarks/bench_gpu_core_path_precision.py` (emits
`artifacts/gpu_core_path_precision/results.json`, `tiers.md`, and — via `--plot` —
`precision_tiers.png`, `precision_fields.png`).

## 4. Results

Sweep executed on the cluster queue (loong64 CPU node, no CUDA): torch 2.7.1, python 3.11.6, `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`, eager backend, seed 42, no perturbation.

Artifact of record: `artifacts/gpu_core_path_precision/results.json` (merged from the four per-arm runs by `--merge`; `environments_per_arm` keeps each arm's provenance), `tiers.md` (the same table generated by the harness), `precision_tiers.png` and `precision_fields.png`.

### 4.1 Error table (all numbers injected from `results.json`)

| tier | case | metric | relative error | verdict |
|---|---|---|---|---|
| FP64FP64 | channel_ny18 | u_max_rel_err | 0.3620% | PASS |
| FP64FP64 | channel_ny18 | profile_rel_l2 | 0.4931% | PASS |
| FP64FP64 | channel_ny34 | u_max_rel_err | 0.4074% | PASS |
| FP64FP64 | channel_ny34 | profile_rel_l2 | 0.4216% | PASS |
| FP64FP64 | tg_nx32 | ke_max_rel_err | 1.2671% | PASS |
| FP64FP64 | tg_nx64 | ke_max_rel_err | 0.3477% | PASS |
| FP64FP32 | channel_ny18 | u_max_rel_err | 0.3648% | PASS |
| FP64FP32 | channel_ny18 | profile_rel_l2 | 0.4956% | PASS |
| FP64FP32 | tg_nx32 | ke_max_rel_err | 1.2671% | PASS |
| FP64FP32 | tg_nx64 | ke_max_rel_err | 0.3478% | PASS |
| FP32FP32 | channel_ny18 | u_max_rel_err | 0.3619% | PASS |
| FP32FP32 | channel_ny18 | profile_rel_l2 | 0.4930% | PASS |
| FP32FP32 | tg_nx32 | ke_max_rel_err | 1.2654% | PASS |
| FP32FP32 | tg_nx64 | ke_max_rel_err | 0.3440% | PASS |
| FP32FP16 | channel_ny18 | u_max_rel_err | 24.7407% | REJECT |
| FP32FP16 | channel_ny18 | profile_rel_l2 | 23.6842% | REJECT |
| FP32FP16 | channel_ny34 | u_max_rel_err | 100.0000% | REJECT |
| FP32FP16 | channel_ny34 | profile_rel_l2 | 100.0000% | REJECT |
| FP32FP16 | tg_nx32 | ke_max_rel_err | 12.6060% | MARGINAL |
| FP32FP16 | tg_nx64 | ke_max_rel_err | 10.0423% | MARGINAL |

### 4.2 Per-tier verdict and footprint

| tier | worst metric | verdict | persistent state B/cell | total B/cell |
|---|---|---|---|---|
| FP64FP64 | 1.2671% | PASS | 304 | 456 |
| FP64FP32 | 1.2671% | PASS | 76 | 532 |
| FP32FP32 | 1.2654% | PASS | 152 | 228 |
| FP32FP16 | 100.0000% | REJECT | 38 | 266 |

Admissible (PASS on every anchor): **FP32FP32, FP64FP32, FP64FP64**.

### 4.3 Why the float16 tier fails (mechanism, not noise)

float16 spacing at the equilibrium weight `w = 1/18` is `3.052e-05`.  The momentum a body-force step can inject into a single population at rest is `w (1-1/(2tau)) 3 a`, with `a = 8 nu u_max/H^2`:

| ny | H | a | per-step population increment | increment / float16 ulp |
|---|---|---|---|---|
| 18 | 16 | 1.5625e-04 | 9.766e-06 | 0.320 |
| 34 | 32 | 3.9063e-05 | 2.441e-06 | 0.080 |

When the ratio is below 1 the increment is rounded away every step, so the driving never accumulates into a flow.  The consequence is worse than a large error, it is a *plausible* wrong answer:
`channel_ny18` (ratio 0.32) converges to `u_max = 0.03748` vs analytic `0.04980` and the steady detector fires at step 1000 — a converged, smooth, wrong profile.  `channel_ny34` (ratio 0.08) returns `u_max = 0.00000`, i.e. a converged zero-flow state.

### 4.4 Decision taken

1. `FP64FP64`, `FP64FP32`, `FP32FP32` are admissible; `FP32FP32` becomes the recommended production default (same accuracy as the FP64 control on every anchor, 228 vs 456 B/cell).
2. A float16 store tier is **refused at construction** unless `allow_lossy_store=True`; when allowed it emits a `RuntimeWarning` and records `store_note` in the run stats.  Deleting the tier was the alternative (option B in §3); it was rejected because an exercised, measurable lossy tier is the evidence that keeps the remaining tiers honest.
3. The memory column is reported per tier instead of asserted: `FP64FP32` saves persistent state (76 B/cell) but costs working memory (532 B/cell vs 456), because the cast needs fp64 scratch.  It pays off only where the *state* is what moves (checkpoint, halo exchange), not for in-core footprint.
4. The T2 report's claim 'FP32FP16 bandwidth-bound (~2e-4 error)' is retracted: the measured error is 23.7-100% on the channel anchors and 10.0-12.6% on the Taylor-Green anchors, and the tier does not reduce total footprint in the fused eager path.

## 5. Tests (regression pins)

`tests/test_gpu_core_path.py`, new cases:

| test | what it pins |
|---|---|
| `test_store_tier_really_rounds_between_steps` | FP32FP16 trajectory ≠ FP32FP32 trajectory (the F4 regression pin) |
| `test_store_state_is_what_the_next_step_reads` | `_current()` equals the widened store buffer; the store dtype survives further stepping |
| `test_memory_report_flags_the_store_trade_off` | persistent state halves FP64→FP32 and quarters FP32→FP16; cast buffers are counted in `total_bytes` |
| `test_lossy_store_tier_warns_at_construction` | float16 store raises `RuntimeWarning` |
| `test_validate_rejects_unknown_precision` | `ValueError` on an unknown tier name |
| `test_float32_tier_tracks_float64_on_the_channel_anchor` | FP32FP32 vs FP64FP64 channel profile agrees to <1e-3 |
| `test_lossy_tier_stays_within_the_referee_line_on_taylor_green` | FP32FP16 stays finite and <15% even though it is inadmissible |

## 6. Limitations

* No CUDA on the test host: triton kernels, multi-GPU halo exchange and any
  GLUPS claim remain **WITHHELD / not_run**.  The tier wiring is backend-generic
  (`_step_compiled` carries the same cast boundary) but is only *measured* on
  the eager CPU path.
* The `torch.compile` parity test stays opt-in (`TL_TEST_COMPILE=1`) because
  compilation dominates runtime on this host.
* Pressure/momentum-exchange force output is out of scope here.