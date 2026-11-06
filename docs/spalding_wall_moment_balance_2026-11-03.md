# Spalding exchange wall model — source moment balance (validation ladder rung 1)

Date: 2026-11-03 · scope: `src/tensorlbm/spalding_wall_model.py` · gate:
`tests/test_spalding_wall_moment_balance.py` · generator:
`scripts/verify_spalding_wall_moments.py` · record + figure:
`artifacts/spalding_wall_moments_2026-11-03/`

## 1. Why this rung

`docs/mature-lbm-gap-analysis-2026-08-01.md` makes rung 1 of the mandatory
validation ladder: *"Manufactured wall-law states, plus zero-mass/exact-momentum
force-source moments on D3Q19 and D3Q27"*. Before this work:

* `apply_spalding_exchange_wall_model` had **no D3Q27 test at all**
  (`tests/test_spalding_wall_model.py` built only D3Q19 states), and
* the two moment identities that justify calling the population assimilation
  "mass-conservative with an exact momentum source" were **never asserted** —
  the module docstring claimed them, no test gated them.

A wall-modelled hull force is only as trustworthy as the momentum bookkeeping
behind it, so the identities were measured before and after a one-line fix.

## 2. Claim under test

The assimilation is

```
f' = f_eq(rho, u_target) + s (f - f_eq(rho, u)) ,     u_target = (u·n) n + u1 t
```

with `n` the outward wall normal, `t` the unit tangential direction of the
exchange-location velocity, `u1` the Spalding velocity at the first off-wall
distance, and `s = nonequilibrium_scale`. Because

* `Σ_i W_i = 1` and `Σ_i W_i c_i = 0` (lattice symmetry) give
  `Σ_i f_eq(rho,u) = rho` and `Σ_i c_i f_eq(rho,u) = rho u`, and
* `rho, u` are recovered from `f` itself, so the non-equilibrium part
  `f - f_eq` carries zero mass and zero momentum,

the source `δf = f' - f` must satisfy

| identity | analytic value |
|---|---|
| zeroth moment `Σ_i δf_i` | 0 (exactly) |
| first moment `Σ_i c_i δf_i` | `rho (u_target - u)` (exactly) |

for **every** `s` and `activation`, and must scale linearly with `activation`.

## 3. Case / parameters

| item | value |
|---|---|
| lattice | D3Q19 and D3Q27 (`q ∈ {19,27}`) |
| grid | `(q, 5, 7, 9)`, single boundary cell at `(2, 3, 4)` |
| wall topology | flat wall below the fluid node, links with `c_y < 0` flagged |
| wall normal | `(0, 1, 0)` (outward into the fluid) |
| state | equilibrium at `rho = 1`, `u = (0.06, 0, 0)` + `N(0, 0.02²)` non-equilibrium noise |
| `nu` | `1e-4` |
| `exchange_distance` | `2.0` cells (`y1 = 0.5`, `y2 = 2.0`) |
| clock / flux | `nonequilibrium_scale ∈ {0, 0.5, 1}`, `activation ∈ {0.25, 0.5, 0.75, 1}` |
| dtype | float64 throughout |

## 4. Finding and fix

The shared lattice constants (`d3q19.W`, `d3q27.W`) are **float32**, so their
float64 cast sums to `1 + 1.49e-8` (D3Q19) and `1 + 7.45e-9` (D3Q27). The mass
source of the assimilation is *algebraically* `rho · (ΣW − 1)` per boundary
node **per step** — i.e. the quantization bias was a **systematic spurious mass
source**, not a rounding artefact. On a hull-sized run (10⁵ boundary nodes ×
10⁶ steps) that is `~1.5e3` lattice mass-units of spurious injection.

Fix (`spalding_wall_model._lattice`): renormalize the cast weights
(`w / w.sum()`) before use. One line + one helper, no change to shared lattice
constants, no change to the assimilation algebra.

### Measured effect (from `moments.json`)

| lattice | weights | `ΣW − 1` | \|mass source\| | momentum-source rel. error |
|---|---|---|---|---|
| D3Q19 | raw float32 | `+1.49e-8` | `8.23e-9` | `2.79e-9` |
| D3Q19 | renormalized | `0` | **`4.07e-11`** | `2.79e-9` |
| D3Q27 | raw float32 | `+7.45e-9` | `4.43e-9` | `6.69e-9` |
| D3Q27 | renormalized | `−2.2e-16` | **`9.02e-17`** | **`1.81e-16`** |

D3Q27 reaches machine precision for both moments. D3Q19 lowers the mass source
by two orders of magnitude but the momentum source keeps a `2.8e-9` relative
floor because of a second, independent quantization: after renormalization
`Σ_i W_i c_{i,x}² − 1/3 = −2.48e-9`. Renormalizing the *sum* cannot repair a
*distribution* mismatch; that needs float64 lattice constants (open item, §7).

The non-equilibrium scale therefore leaks into the source momentum at
`rho |S2 − I/3| |u| ≈ 2.9e-10` on D3Q19 and exactly `0` on D3Q27. The gate
encodes that floor explicitly (`QUANTIZATION_FLOOR`) rather than hiding it in a
loose tolerance.

## 5. Gate and negative controls

`tests/test_spalding_wall_moment_balance.py`, 14 cases, all passing:

1. zero zeroth moment (per node and domain total), D3Q19 + D3Q27;
2. first moment equals `rho (u_target − u)` rebuilt from public primitives only
   (`sample_wall_exchange_velocity` → `solve_spalding_friction_velocity` →
   `spalding_u_plus_from_y_plus`);
3. `nonequilibrium_scale` independence of both moments;
4. linearity of the momentum source in `activation` (mass stays zero);
5. manufactured Spalding profile is a **fixed point**: with `y1 = 0.5`,
   `y2 = 2.5`, the assimilation leaves `f` unchanged to `1e-7 u1` while
   reporting `u_tau` back to `2e-6` relative.

Negative controls (pytest plugins placed on `PYTHONPATH` patching the module
inside the test process, run out-of-tree; harness definitions in §9.1, counts
re-measured on this branch):

| mutation | result | caught by |
|---|---|---|
| A: drop weight renormalization (`_normalized_weights` → identity) | 12/14 fail | mass + momentum identities |
| B: equilibrium missing the `−1.5 u²` term | 14/14 fail | every case |
| C: assimilation is a strict no-op (`δf = 0`) | 10/14 fail | momentum source + activation linearity |

Mutation C is the informative one: the `zero zeroth moment` and `manufactured
fixed point` families pass on an untouched field *because* a disabled model has
trivially zero source and is trivially a fixed point. The gate stays
load-bearing (the first-moment cases catch C), but the zero-mass identity must
never be quoted alone as evidence that the wall model is active.

Cross-check of the friction-velocity inversion (`inversion_sweep`, 35 states,
`u_tau ∈ [1e-4, 2e-2]`, `y2 ∈ {0.5, 1, 2, 3, 8}`, `y⁺ ∈ [0.5, 1600]`): maximum
relative error of the recovered `u_tau` is `1.13e-11`.

## 6. Error judgement vs benchmark

Benchmarks here are analytic (the identities are exact by construction), so the
acceptance question is "is the measured residual explainable and below the
physical error budget?":

| key quantity | benchmark | measured | <3%? |
|---|---|---|---|
| mass source, D3Q27 | 0 | `9.0e-17` (rel. 9e-17) | yes |
| mass source, D3Q19 | 0 | `4.1e-11` (rel. 4e-11) | yes |
| momentum source, D3Q27 | `rho Δu` | rel. `1.8e-16` | yes |
| momentum source, D3Q19 | `rho Δu` | rel. `2.8e-9` | yes |
| fixed-point residual | 0 | `≤ 2.5e-8 · u1` | yes |
| `u_tau` inversion | manufactured `u_tau` | rel. `1.1e-11` | yes |

These are pre-requisite gates only: they say the wall model's *bookkeeping* is
sound, **not** that any SUBOFF force is validated (rungs 2-6 remain open).

## 7. Notes, limits, traps

* **Limitation (open item).** D3Q19 source-moment exactness is capped at
  `~2.8e-9` relative by float32 lattice weights. Fixing it needs float64 weight
  tables (or quantized-rational weights) in `d3q19.py`/`d3q27.py`; that touches
  every solver path and was out of scope for a light-load audit slot. Recorded
  as a recommendation, not silently accepted.
* **Scope.** The identities constrain the population source only. They do not
  constrain the *traction* closure (`tau_w = u_tau²` reported in
  `SpaldingWallDiagnostics.shear_force`), which is a model assumption that still
  needs rung 2-3 (Couette/Poiseuille stress, turbulent channel `Re_tau`).
* **Trap (cost 30 min).** In this workspace `python3 script.py` resolved
  `tensorlbm` to **another worktree** (`work/m708/src`) through the co-installed
  editable path, while `pytest` correctly used the local `src/` (verified by a
  plugin printing `spalding_wall_model.__file__`). Any audit script must pin
  `sys.path` to its own `src/` and assert the resolved module path — otherwise it
  silently validates someone else's tree. `scripts/verify_spalding_wall_moments.py`
  now fails closed on this.
* **Trap.** The exchange sample sits `y2 − y1` cells outward from the *node
  centre*, not `exchange_distance` cells: with `y1 = 0.5`, `y2 = 2.0` the sample
  lands at `y = 4.5` (half-index). The fixed-point test uses
  `exchange_distance = 2.5` so the sample lands on a node and the manufactured
  profile needs no interpolation.
## 8. Re-land provenance (m4703, 2026-11-05) and reproduction

The original delivery lived on mission **m4641**; its PR was closed by the owner
on 2026-11-05 and the work is re-landed here from the current `main`
(`05d728b6`). Two pieces of the original commit were deliberately **dropped**,
and this section is the record:

* `.github/workflows/ci.yml` — the original hunk replaced
  `actions/setup-python@v5` with a runner-local venv because the self-hosted
  runner could not unpack the action (`tar: Cannot change mode ... EPERM`).
  `main` has since migrated every job to `runs-on: ci-loong64` and no longer
  invokes `actions/setup-python` at all (PR #679), so re-landing the hunk would
  resurrect a dead configuration and conflict with current CI.
* `docs/m4641/ci_red_fix_evidence.md` — that document is the evidence for the
  same obsolete CI change (it analyses `Set up job` failures of PR #658), so it
  cannot be restated as a live claim about today's CI.

What is landed here is the physics/feature part only: the weight
renormalization fix, the 14-case moment-balance gate, the audit script and this
document (5 files, single feature point).

**Reproduction on the landed tree** (2026-11-05):

```
python scripts/verify_spalding_wall_moments.py     # rewrites artifacts/spalding_wall_moments_2026-11-03/
python -m pytest tests/test_spalding_wall_moment_balance.py -q   # 14 passed
```

The re-run reproduces every number in §4 and §6 to the shown digits
(`moments.json` is committed alongside this document). Figure
`artifacts/spalding_wall_moments_2026-11-03/spalding_wall_moments.png` has three
panels: (a) measured source residuals before/after the fix against the analytic
identity, (b) `u_tau` inversion error vs manufactured `y+` with the 3%
acceptance line, (c) the Spalding `u+(y+)` actually used by the model against
the viscous-sublayer and log-law asymptotes. Panel (c) was corrected in the
re-land: the original version plotted the same function twice and labelled one
copy "inverse", which demonstrated nothing.

**No regression surface**: the only source change is the local renormalization
inside `spalding_wall_model._lattice`; the shared float32 lattice tables and the
assimilation algebra are untouched, so no other solver path changes.

## 9. Second re-land (m4792, 2026-11-05): scope and the carried CI-red fixes

PR **#669** (the m4703 re-land above) was also closed by the owner without
merge, so m4792 re-lands the same physics from the same `main` (`05d728b6`).
The content of §1-§8 is unchanged and re-verified on the new branch: the audit
script reproduces every number in §4 and §6 to the shown digits, the 14-case
gate passes, and the figure is regenerated.

Two things were added relative to the m4703 branch, both recorded here because
they widen the diff beyond the single wall-model feature:

1. **Pre-existing red CI tests carried along (3 files).** On this `main` the
   repository-wide pytest job is red for two files that never touch the Spalding
   wall model (verified: `grep -l spalding` matches neither), so *any* new PR
   would inherit the redness and could not go green:
   * `tests/test_regress_wall_multi_gpu.py` — two cases asserted the *bug*
     (`pytest.raises(ImportError)` for `wss_from_fneq_3d`, which raised because
     `wall_shear` imported `equilibrium` while `d3q19` exports `equilibrium3d`).
     The import has since been fixed, so "DID NOT RAISE" was guaranteed. Both
     cases are rewritten as positive regression guards on the real path: a
     deterministic xy-shear state must return a finite field, solid cells masked
     out, `sigma_xy > 0` in the fluid, matching the closed form
     `-(1 - 1/(2 tau)) A sum_q c_x^2 c_y^2` to `6e-15` relative; an equilibrium
     state must return zero.
   * `tests/test_control_volume_force.py` — passed a 0-d tensor to
     `pytest.approx()`; pytest >= 8 iterates the argument and torch raises
     "iteration over a 0-d tensor". Compare `float(...)` instead.
   * `docs/reports/REGRESSION_REPORT_wall_multi_gpu.md` — pointer note that the
     defect no longer reproduces and the two cases are now guards.
   These are the same fixes mission **m4669** produced for PR #669; the overlap
   is declared rather than hidden. Branch-vs-branch conflict risk is limited to
   those three paths and resolves to identical content.
2. **`.github/workflows/ci.yml` and `docs/m4641/ci_red_fix_evidence.md` stay
   dropped** for the reasons in §8 (the ci-loong64 migration made the
   `setup-python` workaround dead configuration).

Final PR scope: 9 files — 1 source fix (25 lines), 1 new gate (14 cases),
1 audit script, 1 document, 2 committed artifacts (record + figure), and the
3-file carried test-gate repair. Verified on the branch: `pytest
tests/test_spalding_wall_moment_balance.py tests/test_spalding_wall_model.py
tests/test_control_volume_force.py tests/test_regress_wall_multi_gpu.py` →
47 passed, 13 xfailed; `ruff check` / `ruff format --check` clean on every
touched Python file; `tools/check_repository_layout.py` OK.

### 9.1 Mutation-control harness (as re-run on this branch)

```python
# mut_a.py — A: drop the weight renormalization
import tensorlbm.spalding_wall_model as m
m._normalized_weights = lambda w: w

# mut_b.py — B: equilibrium with the wrong (missing) -1.5 u^2 term
import tensorlbm.d3q19 as d19, tensorlbm.d3q27 as d27
for mod in (d19, d27):
    name = "equilibrium3d" if hasattr(mod, "equilibrium3d") else "equilibrium27"
    orig = getattr(mod, name)
    setattr(mod, name, lambda C, W, rho, ux, uy, uz, _o=orig:
            _o(C, W, rho, ux, uy, uz) + 1.5 * (ux*ux + uy*uy + uz*uz))

# mut_c.py — C: assimilation is a no-op, diagnostics kept
import tensorlbm.spalding_wall_model as m
_orig = m.apply_spalding_exchange_wall_model
m.apply_spalding_exchange_wall_model = lambda f, *a, **k: (f.clone(), _orig(f, *a, **k)[1])
```

Run as `PYTHONPATH=src:/tmp python -m pytest
tests/test_spalding_wall_moment_balance.py -q -p mut_a` (likewise `mut_b`,
`mut_c`). Unmutated baseline is `14 passed`.

### 9.2 Full fast-suite run on this branch (honest record)

`python -m pytest tests -q -m "not slow" -n 24 --timeout=1200` on the wetted
branch: **6539 passed, 17 failed, 157 skipped, 111 xfailed, 11 xpassed** in
35m52s. None of the 17 failures touches this PR's surface (the wall model, its
gate, the control-volume force test and the wall-shear regression test all
pass):

| failing file | count | reading |
|---|---|---|
| `tests/test_cut_cell_bgk_2d.py` | 12 | checkpoint/cut-cell suite (IndexError family) — no import path to `spalding_wall_model` |
| `tests/test_multi_device_sdaa_common.py`, `tests/test_suboff_wallfn_fullgrid_runner.py::TestSdaaSmokeTest` | 4 | SDAA device absent on this host |
| `tests/test_autograd_calib.py` | 1 | calibration trend under parallel xdist |

They are reported as **pre-existing/environmental, not fixed here**: this
mission did not re-run them serially on a clean `main` to prove it (the serial
re-run exceeded the 120 s shell budget on a host at load ~400), so the claim is
limited to "outside this PR's dependency surface", which was checked by import
graph, not by a clean-main A/B run — an open item for whoever owns those files.
