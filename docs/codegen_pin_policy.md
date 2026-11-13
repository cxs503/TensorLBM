# Codegen pinning: pin the register budget before you tune

Status: implemented (`src/tensorlbm/codegen_pin.py`, `tools/codegen_pin_audit.py`,
`tests/test_codegen_pin.py`). Applies to the Triton kernels in
`triton_fused*.py`, `_voxel_kernels.py` and `backends/triton_backend.py`.

## 1. Why this exists

Upstream practice (`AminZar69/CudaPhaseField`): a CUDA build pins
`-maxrregcount` **before** tuning anything. The reason is not that one register
cap is intrinsically best — it is that *an unpinned register budget makes the
measurements meaningless*:

* `ptxas` chooses the register allocation per compiled kernel. Any edit to the
  kernel body — even one that should be performance-neutral — can move the
  allocation, which moves occupancy, which moves throughput.
* So a "2 % speed-up" measured across two revisions may be compiler noise, and
  an "optimisation" that loses 5 % may actually be a win masked by a lower
  occupancy. Tuning without a pin degenerates into chasing ptxas.
* Two kernels benchmarked in the same table with different register budgets are
  not comparable either, and the difference is invisible in the report.

Pinning first fixes the confounder. Only then does a throughput delta carry
information.

## 2. Triton equivalent of `maxrregcount`

Triton exposes three **launch-time codegen knobs**. They are not tile-shape
parameters (`BLOCK_X`, `block`), which belong to the algorithm and change with
the case; the codegen knobs decide how the *same* algorithm is compiled:

| knob | effect | CUDA equivalent |
|---|---|---|
| `num_warps` | threads per CTA (`32 * num_warps`); co-determines registers per thread for a target occupancy | `__launch_bounds__` / block size |
| `num_stages` | software-pipelining depth; trades registers for latency hiding | `#pragma unroll` depth |
| `maxnreg` | hard cap on registers per thread handed to ptxas | `-maxrregcount` |

`maxnreg` is the direct analogue. Note that it is *not* available on every
Triton build/backend; `codegen_pin.maxnreg_supported()` probes for it and the
launch path omits it when absent (see §5, "honest degradation").

## 3. What the audit found before this change

`tools/codegen_pin_audit.py` (static AST audit, no GPU required) over the five
Triton modules:

| quantity | value |
|---|---|
| `@triton.jit` kernels | 13 |
| launch sites | 12 |
| launch sites that forwarded a `maxnreg` argument | 5 (only after this change) |
| launch sites routing through `codegen_pin.launch_kwargs` | 5 (only after this change) |
| declared pins that fix a `maxnreg` value | 0 of 5 |
| distinct declared pin fingerprints | 4 |

The pre-existing state:

* every launch site took `num_warps`/`num_stages` from a symbol, but the
  *wrapper defaults* disagreed across modules and even inside one module:
  `triton_fused` 2/2, `triton_fused_obstacle_les` 2/2,
  `triton_fused_obstacle_xfar_les` **1**/2, `triton_obstacle_force_reduction`
  4/2, `_voxel_kernels` 4 (stages left at the backend default) with
  `link_q_min_t_binned` at 2 — four different register budgets, no record of
  why, and nothing preventing a silent change.
* **no kernel pinned `maxnreg` at all**, so the register budget was 100 %
  ptxas's choice in every benchmark TensorLBM has ever published.

## 4. The discipline

1. **One declared table.** `tensorlbm.codegen_pin.DECLARED_PINS` holds exactly
   one `KernelPin` per benchmarked kernel: `num_warps`, `num_stages`, `maxnreg`,
   the owning wrapper, and a note. It is the only place a codegen decision is
   written down.
2. **Launch through the helper.** Launch sites compose their kwargs with
   `codegen_pin.launch_kwargs(...)` instead of literal
   `num_warps=..., num_stages=...`. A pinned `maxnreg` therefore takes effect
   everywhere at once, and adding one is a one-line change to the table.
3. **Drift is a failure.** `audit_codegen()` re-derives every wrapper default
   from the source (AST, module constants resolved by import) and diffs it
   against the table. `tests/test_codegen_pin.py::test_declared_pins_match_kernel_sources`
   fails if somebody changes a default without re-declaring the pin, i.e.
   changing the register budget becomes a visible, reviewed event rather than an
   accident. `tools/codegen_pin_audit.py` exits 1 on drift, so a CI job can gate
   on it.
4. **Compare only within a pin.** Every resolution carries a 16-hex
   fingerprint (SHA-256 over the knob triple + pin-scheme version).
   `assert_comparable(a, b)` raises unless both records are `pinned=True` and
   their fingerprints match.
5. **Baselines cannot be contaminated.** `PinnedRecorder` writes JSONL and
   refuses (`CodegenPinViolation`) to append an unpinned measurement, so a
   tuning sweep cannot leak into a pinned baseline file.
6. **Tuning is explicit.** Deviating from the table requires
   `resolve(..., allow_tuning=True)`. The result is marked `pinned=False` and
   gets a distinct `-T<hash>` fingerprint, so a tuned run is structurally
   distinguishable from a pinned one.

### How to pin a register cap

```python
# src/tensorlbm/codegen_pin.py
declare(KernelPin(
    kernel="triton_fused_obstacle_xfar_les",
    num_warps=1, num_stages=2, maxnreg=32,     # <- the new cap
    source="tensorlbm.triton_fused_obstacle:triton_fused_obstacle_xfar_les",
    notes="occupancy-limited at 1 warp; cap tested on sm_86",
))
```

Then regenerate the manifest (`python tools/codegen_pin_audit.py`) and update
the test expectation. Nothing else changes: the launch site already forwards it.

### How to run a tuning sweep without polluting the baseline

```python
recorder = PinnedRecorder("artifacts/tuning_warps.jsonl", require_pin=False)
for warps in (1, 2, 4, 8):
    launch = resolve("triton_fused_obstacle_xfar_les",
                     overrides={"num_warps": warps}, allow_tuning=True)
    recorder.record(launch, metric="mlups", value=measured)
# promote the winner by editing DECLARED_PINS, then re-measure *pinned*
```

## 5. Honest degradation (no GPU on the build node)

The audit host is `loongarch64`, torch 2.7.1 with `cuda_available=False`,
triton 3.4.0 without the `nvidia` backend, `nvcc` absent. The manifest records
that verbatim (`maxnreg_supported: false`, `nvcc_version: null`) instead of
guessing. Consequences:

* no register count (`n_regs`) and no MLUPS number is claimed anywhere in this
  change — none could be measured here;
* `launch_kwargs` drops `maxnreg` when the backend cannot accept it rather than
  passing an unsupported kwarg (which would break the launch) or silently
  claiming the cap was applied;
* the "cap is actually honoured" path is covered by unit tests with the support
  probe monkeypatched (`test_launch_kwargs_forward_maxnreg_when_supported`,
  `test_launch_kwargs_uses_declared_maxnreg_when_caller_leaves_it_open`). It has
  **not** been executed against a real CUDA backend — that remains a GPU-host
  verification item.

## 6. Remaining work (explicitly out of scope for this increment)

* Route the six remaining launch sites (`triton_fused.py` ×3,
  `_voxel_kernels.py` ×4 — `triton_fused.py`'s sites all share one wrapper, so
  three sites; see the manifest's `launch_sites[].pinned_via == null`) through
  `launch_kwargs` as well.
* On a GPU host: record `n_regs`/`n_spills` per kernel, fill in `maxnreg`, then
  and only then run an occupancy/throughput sweep and promote the winner into
  `DECLARED_PINS`.
* Feed the pinned fingerprints into the benchmark result records
  (`benchmark_observability`) so published MLUPS tables carry the pin they were
  measured under.