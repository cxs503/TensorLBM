# m4733 — `tests/backends/` coverage extension (backend / kernel paths)

Scope: `TensorLBM`, branch `agent/m4733`, off `05d728b6` (`main`).
Mission type: **backend 扩面** — add positive + negative unit tests for backend /
kernel paths that `tests/backends/` did not cover. No production behaviour was
changed (one stale docstring was corrected, see §5).

## 1. What was uncovered before

`tests/backends/` held only two files: `test_backend_contracts.py` (framework-free
R1 dataclasses) and `test_torch_backend.py` (R1 cold-path `TorchBackend` facade).
Measured with `coverage.py` restricted to that directory:

| module | lines | before | after |
|---|---|---|---|
| `backends/__init__.py` (dispatch registry) | 37 | 37.8% | **100%** |
| `backends/torch_5op_step.py` (SUBOFF 5-op kernel chain) | 13 | 0.0% | **100%** |
| `backends/triton_backend.py` (Triton step + guards) | 58 | 0.0% | **74.1%** |
| `backends/paddle_backend.py` | 263 | 0.0% | **35.0%** |
| `backends/mindspore_backend.py` | 244 | 0.0% | **36.1%** |
| `backends/torch_backend.py` (unchanged by this mission) | 237 | 44.7% | 44.7% |
| **total** | 899 | **18%** | **47%** |

Remaining uncovered lines are the framework-gated bodies (`paddle.*` / `ms.*`
calls inside each op). PaddlePaddle and MindSpore are **not installed** on this
host and no fake framework stub was invented, so those numerics stay
**WITHHELD**; what is covered is the surface that decides whether a family can be
used at all (see §3).

## 2. New test files

### `tests/backends/test_torch_5op_step.py` (16 tests)

The default `"torch_5op"` SUBOFF step (`stream → Ladd force → 6-face far-field
BC → mass correction`). The reference implementation is the pre-refactor inline
chain rebuilt from the same primitives, so equality is a bitwise contract:

* `test_step_matches_inline_production_chain_bitwise[1|7|10]` — `torch.equal` on
  the post-step distribution **and** all three force scalars, on-cadence and off.
* `test_far_field_bc_overwrites_inlet_and_lateral_planes` — inlet/`y±`/`z±`
  planes equal the uniform `u_in` equilibrium; outlet is zero-gradient
  (`f[..., -1] == f[..., -2]`).
* `test_forces_are_zero_without_obstacle`, `test_streamwise_drag_is_positive…`,
  `test_mass_correction_fires_only_on_period_multiples`,
  `test_mass_correction_retargets_the_requested_mass`,
  `test_do_mass_correction_false_skips_the_rescale`,
  `test_step_does_not_mutate_its_input`, `test_step_is_deterministic`,
  `test_step_returns_four_tensors_with_stable_shapes_and_dtype`,
  `test_ten_step_loop_is_finite_and_mass_bounded`, `test_non_square_domain_supported`.
* Negative: `test_zero_mass_period_raises_instead_of_silently_streaming`
  (pins `ZeroDivisionError`), `test_zero_mass_period_is_harmless_when_correction_is_disabled`.

### `tests/backends/test_backend_dispatch.py` (20 tests)

`tensorlbm.backends` registry: `set_backend` / `get_ops` / `using_backend` and the
import-time `TENSORLBM_BACKEND` default.

* Positive: every declared backend is switchable; `get_ops()` returns the
  matching family module **without the framework installed** (both family modules
  import lazily); one cached module per name; `using_backend` restores the
  previous backend, `None` is a no-op switch.
* Negative: unknown names (`"tensorflow"`, `""`, `"TORCH"`, `" torch"`,
  non-strings) raise and **do not mutate** global state; an unresolvable active
  backend raises `No ops module…`; an exception inside `using_backend` still
  restores the old backend; an invalid `using_backend` name never enters the body.
* Subprocess probes (fresh interpreter per probe): default is `torch`; env var
  selects `paddle`; an invalid env var is accepted at import time but fails at
  `get_ops()` — i.e. the documented asymmetry is pinned.

### `tests/backends/test_triton_backend_guards.py` (14 tests)

Triton step cold path on a CPU host. The numerics need CUDA (WITHHELD), the
guards do not:

* `is_available()` must equal exactly `torch.cuda.is_available() and _Q_PAD > 0`;
  `_PROD_Q == D3Q19.q == 19`; declared `__all__`.
* `TritonStepState`: Q=19 production layout, ping-pong halves are **distinct**
  allocations (aliasing silently doubles drag), lazy `feq_vec_buf`, zeroed
  force/scratch buffers, `__slots__` (no `__dict__`), non-cubic shapes.
* Negative guards: CPU tensor → `RuntimeError`; wrong Q → `ValueError`; obstacle
  mask shape mismatch; `tau_eff` shape mismatch; **guard order** (Q before mask).
* `test_valid_shapes_pass_the_meta_data_guards` — with correct meta-data the
  wrapper's own guards fall through (only the absent CUDA device stops the call).

### `tests/backends/test_backend_family_parity.py` (20 tests)

Enforces the invariant the package docstring states but no test checked:
*"New functions added to one backend must be added to all three."*

* Public op surface of `paddle_backend` / `mindspore_backend` must equal
  `torch_backend`'s, with documented exceptions only
  (`TORCH_ONLY = {build_torch_lattice_constants}`,
  `MINDSPORE_ONLY = {ms_train_step}`; the latter is required by
  `value_and_grad`). Extra *or* missing ops fail.
* Call shapes (parameter name, kind, default; annotations ignored) must match.
* `BACKEND_NAME` declared per family.
* Negative: with the framework forced absent through `sys.modules`, every
  framework-touching entry point (`zeros`, `tensor`, `from_numpy`, `rand`,
  `build_flow_transformer`) raises an `ImportError` that names the install
  command — i.e. the failure is diagnosable, not a bare `AttributeError`.
* `pi()` stays callable without the framework; `float32_dtype()` raises.

## 3. Negative self-verification (tests must be able to go red)

Harness: `artifacts/m4733/negative_control.py` (log:
`artifacts/m4733/negative_control.log`). Each defect is injected into the source,
the target test file is run, the file is restored (`git diff --stat` verified
clean). All seven injections turned the suite red:

| # | injected defect | command | red output |
|---|---|---|---|
| I1 | force reduction moved **after** the far-field BC | `pytest tests/backends/test_torch_5op_step.py -q` | 6 failed, 10 passed |
| I2 | mass-correction cadence `step % p == 0` → `== 1` | same | 3 failed, 13 passed |
| I3 | `is_available()` hard-wired `True` | `pytest tests/backends/test_triton_backend_guards.py -q` | 2 failed, 12 passed |
| I4 | Q guard disabled (`if Q_in != _PROD_Q` → `if False`) | same | 2 failed, 12 passed |
| I5 | `set_backend` stops validating names | `pytest tests/backends/test_backend_dispatch.py -q` | 7 failed, 13 passed |
| I6 | `paddle_backend` grows an op torch lacks (surface drift) | `pytest tests/backends/test_backend_family_parity.py -q` | 1 failed, 19 passed |
| I7 | Paddle install hint degraded (`pip install paddle`) | same | 6 failed, 14 passed |

## 4. How to reproduce

```bash
# target suite (all new files), green
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/backends -q

# wider smoke around the dispatch layer
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
    tests/backends tests/test_backends.py tests/test_service_backends.py -q

# CI lint gate (Makefile / ci.yml scope)
ruff check src tests examples benchmarks
ruff format --check src tests examples benchmarks

# coverage + figures + metrics (writes artifacts/m4733/*.json, *.png)
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/backends \
    --cov=tensorlbm.backends --cov-report=json:artifacts/m4733/cov_after.json -q
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 artifacts/m4733/make_figures.py

# negative controls (source is restored automatically)
python3 artifacts/m4733/negative_control.py
```

Measured on this host: `tests/backends` → **90 passed** (was 20);
`tests/backends + tests/test_backends.py + tests/test_service_backends.py` →
**139 passed, 5 skipped**; `ruff check src tests examples benchmarks` → all
checks passed; `ruff format --check src tests examples benchmarks` → 1457 files
already formatted.

## 5. Findings / limitations (honest)

1. **Stale docstring fixed.** `torch_5op_step.py` documented dispatch through
   `tensorlbm.backends.get_step_impl()`, which does not exist anywhere in the
   tree. The docstring now names the real entry point
   (`torch_5op_step.step_torch_5op`). No behaviour change.
2. **`mass_period=0` is an unguarded modulo.** `step_torch_5op` raises
   `ZeroDivisionError` (short-circuits to harmless when
   `do_mass_correction=False`). Pinned by a test rather than changed: adding a
   guard would alter runner-visible behaviour outside this mission's scope.
3. **`step_torch_5op` silently broadcasts a wrong-rank obstacle mask.** A 2-D
   mask on a 3-D field does not raise; it broadcasts. Not pinned by a test
   (pinning laxity would block a future fix) — recommended follow-up: validate
   `solid.shape == f.shape[1:]` like `triton_suboff_step` already does.
4. **Paddle / MindSpore bodies remain WITHHELD.** Neither framework is
   installed, so ops that call `paddle.*` / `ms.*` cannot be executed here; only
   the import guard, the surface parity and the pure helpers are covered. Their
   remaining ~65% of lines needs a host with the frameworks (CI matrix item).
5. **Triton numerics remain WITHHELD.** `is_available()` is `False` on this CPU
   host; the step kernel body (`triton_backend.py:282-334`) needs a CUDA device.
   Guard/meta-data coverage (74.1%) is what a CPU CI can honestly claim.