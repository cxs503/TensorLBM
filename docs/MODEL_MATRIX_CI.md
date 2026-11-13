# Model-matrix CI plus coverage (TensorLBM) — XVAL vs CFD-GO/TCLB

Anchor: `XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE` · repo `TensorLBM` · date 2026-11-13

## 1. Problem and reference source

TensorLBM ships ~25 collision operators spread over `solver3d`, `turbulence`,
`advanced_collision`, `cumulant`, `central_moment_d2q9`, ... . Nothing enumerated
which models exist, which verification case each one owes, or how much of the
model set is actually machine-verified. A change to one operator could leave the
others silently untested.

Upstream practice being cross-validated (from the OSS inventory, tags and method
text in `oss_solutions`, repo `CFD-GO/TCLB`):

| tag | practice (as inventoried) |
|---|---|
| `model-matrix-ci-plus-coverage` | "CI 以模型为矩阵项：CPU 流水线对 16 个模型做 configure->compile->tests.sh->gcov 覆盖；GPU/HIP 流水线只做编译守卫；另一条流水线跑 tests/solid 独立单测" |
| `model-dir-as-build-and-ci-unit` | one directory per model `models/<class>/<lattice>_<method>/` with `Dynamics.R` + `Dynamics.c.Rt` + `conf.mk`; make target = model = CI matrix key |
| `instruction-dsl-tests-sh` | instruction DSL (`run/fail/exists/sha1/csvconcatenate`) + `tools/csvdiff` for *threshold* comparison |

## 2. Method and implementation (what was built)

Python-native transfer of the three practices; the registry entry replaces the
model directory, the case spec replaces `tests/external/*.test`.

| file | role |
|---|---|
| `src/tensorlbm/model_matrix/registry.py` | `ModelSpec` per model: CI key, lattice, **pointer to the real production operator** (`source = module:qualname`, machine-checked in tests), tau-argument mapping, fixed params (LES constant), declared verification surface, `gate`/`guard` enforcement |
| `src/tensorlbm/model_matrix/cases.py` | lattice-generic analytic cases (formula references only, no stored expected values) |
| `src/tensorlbm/model_matrix/coverage.py` | matrix driver, coverage accounting, ref-vs-got + coverage tables, figures |
| `src/tensorlbm/model_matrix/__main__.py` | entry point `python -m tensorlbm.model_matrix {list,guard,run}`; **exit code is the gate** |
| `src/tensorlbm/lbm_step_correct.py` | **integration caller**: `collide_fn` may be a registry key string, resolved at the single point where the shared stepper invokes the operator |
| `scripts/model_matrix_integration.py` | integration run: real stepper driven by key strings, bit-identical vs the legacy raw-function call |
| `tests/test_model_matrix.py` | 39 invariant tests (incl. two negative controls) |
| `.github/workflows/model-matrix-ci.yml`, `Makefile:model-matrix` | CI wiring |

Registry contents (9 matrix keys): `d3q19/{bgk,trt,rlbm,mrt,mrt-smag}`,
`d3q27/{kbc,kbc-les,cascaded,cascaded-les}`; the four LES operators are
`guard` (constructible + equilibrium-consistency only), the five molecular ones
are `gate`.

Cases:

* `feq_fixed_point` (gate) — `C(f_eq)=f_eq` and exact conservation of rho, u.
* `shear_decay` (gate) — periodic shear wave `u_x=U0 cos(ky)`, `U0=0.01`;
  fitted decay rate vs analytic `nu=(tau-1/2)/3`, acceptance 3 %.
* `tg_decay` (advisory) — Taylor-Green vortex `U0=0.1` (Ma 0.17); energy decays
  as `exp(-4 nu k^2 t)`; recorded, not gated (carries the O(Ma²) compressibility
  error).

Acceptance numbers are `owner` standard 3 %; all cells are additionally required
to conserve mass and to show `fit_r2 ≥ 0.999` (a decay that is not exponential
is not a viscosity measurement).

## 3. Parameters (complete, dimensionless)

| parameter | CI run | values |
|---|---|---|
| grid | 16×16×1 periodic (D3Q19 & D3Q27), quasi-2D | convergence sweep adds 24, 32 |
| relaxation time tau | 0.6, 0.8 | viscosity `nu=(tau-0.5)/3` = 0.03333, 0.10000 |
| shear-wave wavenumber | `k=2π/16=0.3927` | sweep: 0.2618, 0.1963 |
| shear amplitude | `U0=0.01`, Ma=`U0/c_s`=0.0173 | — |
| TG amplitude | `U0=0.1`, Ma=0.173 | Mach sweep 0.035/0.087/0.173 |
| LES constant | `C_s=0.1` (LES models), 0.0 (pure operators) | — |
| TRT magic parameter | `Λ=3/16` | — |
| KBC | `beta=0.99`, `C_s=0`/`0.1` | — |
| MRT rates | `s_e=1.19, s_eps=1.4, s_q=1.2` | — |
| working precision | float64 | — |
| decay window | amplitude/E above 25 % of its initial value, ≥3 samples | — |

## 4. Reference vs got (`artifacts/model_matrix_ci/run/BENCHMARK_REF_VS_GOT.md`)

Gate cells, tau=0.6 / 0.8, res=16 (`x` = outside tolerance):

| model | nu_ref (tau=.6) | nu_fit | err % | verdict | nu_ref (tau=.8) | nu_fit | err % | verdict |
|---|---|---|---|---|---|---|---|---|
| d3q19/bgk | 0.033333 | 0.033869 | +1.61 | OK | 0.10000 | 0.101057 | +1.06 | OK |
| d3q19/mrt | 0.033333 | 0.033730 | +1.19 | OK | 0.10000 | 0.100434 | +0.43 | OK |
| d3q19/rlbm | 0.033333 | 0.033730 | +1.19 | OK | 0.10000 | 0.100434 | +0.43 | OK |
| d3q19/trt | 0.033333 | 0.033239 | −0.28 | OK | 0.10000 | 0.100041 | +0.04 | OK |
| d3q27/kbc | 0.033333 | 0.033733 | +1.20 | OK | 0.10000 | 0.100439 | +0.44 | OK |
| d3q27/cascaded | 0.033333 | 0.033724 | +1.19 | OK | 0.10000 | 0.100434 | +0.43 | OK |

`feq_fixed_point` all nine models: max relative deviation `≤ 4.9e-16` (float64
round-off), rho/u conservation errors `0.0`.

Coverage: **21/21 declared cells executed = 100 %**, gate-model pass rate 100 %,
0 missing cells, verdict PASS (`artifacts/model_matrix_ci/run/COVERAGE.md`).
Guard pipeline: 9/9 `GUARD_PASS` (`artifacts/model_matrix_ci/guard.json`).

Advisory Taylor-Green (res=16, tau=0.6): nu error +1.14…+1.18 %,
energy-series relative L2 ≈ 4.6 %; recorded, not gated.

**Is the error < 3 %? Yes** for every gate cell at CI resolution (worst 1.61 %).

## 5. Convergence / parameter sensitivity (≥2 axes, `artifacts/model_matrix_ci/sweep_sensitivity.json`)

Resolution sweep, `shear_decay`, signed nu error %:

| model | res=16 | res=24 | res=32 | order |
|---|---|---|---|---|
| d3q19/bgk | +1.61 | +0.71 | +0.38 | ~2nd order in k |
| d3q27/kbc | +1.20 | +0.51 | +0.30 | ~2nd order in k |
| d3q19/trt | −0.28 | −0.12 | −0.06 | also 2nd order, **8× smaller in magnitude** |

tau sweep (0.6 vs 0.8 at fixed res): error falls from +1.61 % to +1.06 % (BGK),
+1.20 % to +0.44 % (KBC) — i.e. the residual is a discretization (`k²`) effect,
not a mis-set viscosity; a wrong `nu` would scale with `nu`, not with `k²`.

Taylor-Green Mach sweep (res=24, tau=0.6): nu error +0.54/+0.49/+0.32 % for
Ma = 0.035/0.087/0.173 (BGK). Interpretation: the residual is *not* dominated by
compressibility at these Mach numbers; cascaded/KBC produce slightly smaller
values (+0.44/+0.39/+0.24 %), consistent with their better Galilean invariance,
but the spread is below the 3 % acceptance and hence below measurement interest
for this case — which is exactly why it is advisory.

Physics note surfaced by the matrix: TRT with `Λ=3/16` reproduces the shear
viscosity an order of magnitude more accurately than BGK/MRT/regularized at the
same resolution (magic-parameter cancellation of the `k²` error, Ginzburg 2008).
That is a usable conclusion for production model selection, produced by the
matrix rather than hand-argued.

## 6. Differences vs the OSS practice, and transfer trade-offs

| dimension | CFD-GO/TCLB | this implementation | trade-off |
|---|---|---|---|
| matrix key | model **directory name** (`models/<class>/<lat>_<method>`) | registry key `d3q19/mrt` with a `source` pointer to the production callable | no file-per-model boilerplate; provenance is machine-checked instead of implied by the directory |
| what is covered | `gcov` **line coverage** per compiled model | **case coverage** per model: declared vs executed vs inapplicable vs MISSING + analytic error per cell | stronger (physics, not lines) but does not measure code reachability; the two are complementary |
| pipeline split | CPU pipeline (configure→compile→tests→gcov) + GPU/HIP **compile-guard only** | `gate` (5 models, analytic) + `guard` (4 LES models, construct + f_eq consistency) | same asymmetry, made explicit in the registry (`enforcement`) instead of in the CI YAML |
| test description | instruction DSL `tests.sh` + `csvdiff` thresholds | declarative `CaseSpec` (reference formula, tolerance, enforcement) + machine comparison | threshold comparison is preserved; the DSL's `exists/sha1` guards are not ported (no per-model source tree to hash) |
| failure semantics | model fails its own tests | honest `FAIL`/`ERROR` cell keeps the root cause; `MISSING` is a first-class status; gate exit code = non-zero | upstream hides a never-run model less explicitly (a model absent from the pipeline simply has no badge) |
| compile step | real template compilation | none (Python): construction + one-step run is the closest analogue | cannot catch compile-time-only regressions; covered instead by the import/provenance test |

Not ported: `tests/solid`-style separate unit-test pipeline; gcov line coverage;
GPU/HIP backend matrices (this host has no GPU).

## 7. Failure modes and artifacts

- A crashing model is recorded as `ERROR` (exception text in the record) and
  fails the gate — never as a pass.
- A case whose decay is not exponential (`fit_r2 < 0.999`) fails even if the
  fitted slope happens to be close to the reference.
- LES operators are excluded from the molecular-viscosity case **by
  declaration** (`inapplicable_reason`), and the exclusion is visible in the
  report as `n/a`, never as a green cell.
- Negative controls in tests: a 33 % wrong reference viscosity is rejected; a
  declared-but-unexecuted cell reports `MISSING` and `verdict=FAIL`.
- Artifacts: `artifacts/model_matrix_ci/{run/{model_matrix.json,matrix.csv,COVERAGE.md,BENCHMARK_REF_VS_GOT.md,figures/*.png},integration.json,guard.json,sweep_sensitivity.json}`.

## 8. Reproduce

```bash
PYTHONPATH=src python -m tensorlbm.model_matrix list
PYTHONPATH=src python -m tensorlbm.model_matrix guard --out artifacts/model_matrix_ci
PYTHONPATH=src python -m tensorlbm.model_matrix run --out artifacts/model_matrix_ci/run \
    --tau 0.6,0.8 --res 16 --min-coverage 0.85
PYTHONPATH=src python scripts/model_matrix_integration.py --out artifacts/model_matrix_ci/integration.json
PYTHONPATH=src pytest -q tests/test_model_matrix.py     # 39 passed in ~17 s
```
