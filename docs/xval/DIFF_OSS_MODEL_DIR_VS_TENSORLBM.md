# Diff list: TensorLBM model units vs CFD-GO/TCLB model directories

Round 1 XVAL, anchor `XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT`.
Full narrative + numbers: `XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT.md`.
TCLB facts below were read from the local mirror `/tmp/TCLB` (file paths given);
nothing here is quoted from memory or from a secondary description.

## 1. What "a model" is

| aspect | TCLB (OSS) | TensorLBM (this work) |
|---|---|---|
| unit of identity | directory containing `conf.mk` (plus `Dynamics.R`, `Dynamics.c.Rt`, `Description.md`) | directory containing `model.json` (schema `tensorlbm.model_unit/v1`) |
| per-model build config | `conf.mk` with R/Make variables `ADJOINT`, `TEST`, `OPT` | `build` block: `dtype`, `fingerprint_steps` |
| physics declaration | `Dynamics.R` (populations, quantities) + `Dynamics.c.Rt` (kernels) | `composition`: lattice / collision / turbulence / forcing / boundaries / physics_modules, validated through `tensorlbm.models.contracts.ModelComposition` |
| acceptance gate | model test directory `tests/external/<MODEL>` run by `tools/tests.sh`; gate = process exit status | `ci.gates`: named numeric metrics with declared maxima; a gated metric that the kernel does not produce is a failure |
| forcing is part of identity | yes — Guo forcing is its own model dir (`models/electrokinetic/d2q9_npe_guo/`) | yes — `composition.forcing = ["edm"]` is declared and cross-checked against the kernel |

## 2. Discovery

| aspect | TCLB | TensorLBM |
|---|---|---|
| mechanism | `get.models()` in `src/models.R`: `git ls-files \| grep 'conf.mk$'` ∪ `find models -name 'conf.mk'` | `discover_units()`: `iterdir()` over `models/units`, every sub-directory must carry `model.json` |
| malformed layout | `get.model.names()` stops with "Something is wrong with model paths" | `ManifestError`: a directory without a manifest, or `unit_id != directory name`, is rejected |
| CI model list | **hand-written** `strategy.matrix.model` (16 entries) in `.github/workflows/cpu_test.yml` | **derived** from the filesystem; there is no list that can be forgotten |

This is the one place where this work is deliberately *stricter* than the OSS
reference: TCLB discovers models for the build but enumerates them by hand for
CI.

## 3. Build and CI execution

| aspect | TCLB | TensorLBM |
|---|---|---|
| build artefact | generated `Dynamics.c` + compiled binary at `CLB/<model>/main` (`tools/tests.sh` asserts it exists) | `build_id` = sha256(manifest) ⊕ sha256(field after `fingerprint_steps`) + `build/<unit>.json` |
| build step | `make <model>` (`src/makefile.main.Rt` generates `all: <models>`; per-model target) | `build_unit()`: manifest validation → kernel instantiation → two identical runs must give the same fingerprint |
| CI step | `.github/actions/compile` (`make $model`) then `.github/actions/test` (`tools/tests.sh $model`), coverage flagged per model | `run_ci()`: run the unit's benchmark, compare against the analytic reference, apply the declared gates |
| failure isolation | per GitHub job | per unit record; `all` exits non-zero if any unit fails, `--only`/`--allow-failures` for triage |
| missing test | silently passes: `No tests for model $MODEL. Exiting with no error.` (exit 0) | impossible: `ci.gates` must be non-empty, and a gate whose metric is absent fails |
| entry point | `make <model>` + `tools/tests.sh <model>` | `python -m tensorlbm.models.ci_unit {list,build,ci,all}`, `tools/model_unit_ci.py`, `make model-units`, `tensorlbm.api.run_model_unit_ci` |

## 4. Benchmarks and reference values

| aspect | TCLB | TensorLBM |
|---|---|---|
| benchmark source | model-local test scripts in `tests/external/<MODEL>` (`.test` files: `run` / `fail` / `csvconcatenate` lines) | two kernels: `shear_wave_decay` (periodic) and `poiseuille_channel` (half-way walls + EDM force) |
| reference value | whatever the model's test script compares against (often a stored CSV / a previous run) | closed-form analytic, recomputed from the unit's own parameters (`ν=(τ-½)/3`; `u(y)=g/(2ν)(y+½)(ny-½-y)`), so a wrong parameter cannot silently "match" |
| tolerance | inside the test script (per model, in code) | declared in the manifest (per gate, data), machine-applied |
| measured quantities | whatever the script prints | `nu_rel_err`, `profile_l2_rel_err`, `umax_rel_err`, plus invariants `amp0_rel_err`, `mass_rel_drift` |

## 5. Parameter / boundary-condition differences

* **Forcing.** TCLB keeps forcing in the model directory. This work uses the
  exact-difference (EDM) equilibrium shift — `f += feq(ρ, u+F/ρ) − feq(ρ, u)` —
  so the same force term works under BGK, MRT and TRT without a per-operator
  branch. A Guo-style source `S_i = (1−1/(2τ)) w_i [(c_i−u)/c_s² + (c_i·u)c_i/c_s⁴]·F`
  applied without the equilibrium half-shift reproduces only
  `(1−1/(2τ)) ≈ 0.2308` of the analytic peak velocity (measured: 0.2308) — this
  was the first implementation and it is why the EDM form is used.
* **Walls.** Half-way bounce-back with the wall plane at `y=−½` and `y=ny−½`,
  i.e. channel height exactly `ny` (no free-slip, no interpolated BFL). TCLB
  model directories select boundary behaviour through their `Dynamics.R` /
  runtime `conf.xml`, so no single TCLB configuration is the "same" boundary;
  the comparison here is therefore at the level of *mechanism*, not per-BC.
* **Precision.** All units run `float64`. `tensorlbm.d2q9.W` is a float32
  constant with a documented ~7.45e-9/step mass leak at large τ, so the unit
  harness builds the exact double weights itself rather than inheriting them.
* **Steady state.** Channel units are time-marched to steady state with the
  step count scaled as `ny²` (the viscous time), measurement averaged over the
  last quarter; a transient-dominated measurement would make the gate measure
  the warm-up, not the scheme.

## 6. Porting trade-offs (what was adopted, rejected, and why)

1. **Adopted — directory-as-unit discovery.** The single most valuable property
   of the TCLB design: the model collection cannot drift from the filesystem.
   Ported with an extra strictness (every directory must be a unit).
2. **Adopted — per-unit build configuration and per-unit gate.** `conf.mk` →
   `model.json`; R variables → validated JSON fields. JSON was chosen over XML
   (`conf.xml`) because a strict key check is trivial in JSON and unknown
   attributes in an XML parser are the classic way a gate silently disappears.
3. **Adopted — per-model CI granularity with a per-model artefact.** Replaced
   the GitHub matrix by in-process discovery; same isolation, no list.
4. **Rejected — per-model code generation.** TCLB generates `Dynamics.c` from
   `Dynamics.c.Rt` with `Rscript` and compiles one binary per model. TensorLBM
   kernels are PyTorch functions composed at call time: there is no source to
   generate. Substituted by "validate + instantiate + determinism fingerprint +
   `build_id`", which gives the same *traceability* property (a verdict is bound
   to an identity) but not the same *compile-time isolation*.
5. **Known weaker point.** Because kernels are shared, a bug in
   `poiseuille_channel` fails two units at once instead of one model, and
   nothing is caught at "compile time". Partially compensated by having two
   kernels × three collision operators and by the cross-unit comparison the CI
   artefact now records (same kernel, different composition ⇒ the ratio of
   errors is itself evidence).
6. **Not ported — `tests/external` submodule test format.** In the local mirror
   `tests/external` is a git submodule that is not checked out, so the `.test`
   file grammar could only be read at the level of `tools/tests.sh` (`run`,
   `fail`, `csvconcatenate`, exit-code semantics). Per-file comparison is
   therefore **WITHHELD**, not claimed.