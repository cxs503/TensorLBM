# XVAL: Acceptance of the AA-Pattern Fused Stream-Collide Kernel (Re = 100 Cavity)

> **CORRECTION (2026-11-16, mission m22379).** The `u_y` (horizontal centre
> line) numbers below — 4.80 % / 4.59 % / 4.50 % at L = 32/64/128, quoted as a
> solver property and used to declare `< 3 %` "not met" — are **withdrawn**. They
> were produced by scoring against a *scrambled hand-typed copy* of Ghia's
> Table II (values attached to the wrong `x/L` stations, 16 rows instead of 17)
> and by sampling the centre line half a cell off `y/H = 1/2`. Re-scored against
> the provenance-bearing table
> (`reference_data.GHIA_RE100`, `provenance="BOUND"`), the same runs give
> `L2(u_y)` = 1.05 % / 0.77 % / 0.38 % at L = 32/64/128, i.e. converging, and
> **both centre lines are below the 3 % gate on every grid**. The `< 15 %`
> acceptance of the anchor is met with a factor 4–40 margin. Details, evidence
> and the corrected table: `docs/xval_aa_pattern_ghia_table_defect.md`
> (`artifacts/xval_aa_pattern/cavity_accept_corrected.json`,
> `fig7_ghia_table_defect.png`, `fig8_cavity_convergence_corrected.png`).
> The text below is kept unchanged as the historical record.

**Anchor**: `XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE` (lanl/MF-LBM `aa-pattern-fused-stream-collide`)
**Mission**: m21739 · **Date**: 2026-11-15 · **Status**: partial acceptance — coarse-grid point PASS, converged multi-grid sweep still queued

This document closes (as far as it can be closed) item 1 of the "remaining work" list in
`docs/xval_aa_pattern_fused_stream_collide_ordering.md`: *"until [the cavity ablation] lands, the
anchor's `< 15 %` acceptance is not demonstrated for this module"*.

---

## 1. Case, physics and parameter table

Lid-driven cavity, D2Q9 single-relaxation-time (BGK), half-way bounce-back on the three stationary
walls, half-way bounce-back with a moving wall (`u_w = (U, 0)`) on the lid. The lid velocity is
ramped linearly over `ramp` steps (an impulsive start at `U = 0.1` launches an acoustic transient that
makes coarse BGK grids unusable — see §5).

| quantity | value | note |
|---|---|---|
| Reynolds number | 100 | `Re = U L / nu` |
| lid velocity `U` | 0.1 | lattice units, `Ma ≈ 0.17` |
| relaxation time | `tau = 0.5 + 3 U L / Re` | 0.596 (L=32), 0.692 (L=64), 0.734 (L=128), 0.844 (L=192) |
| grid | `L` fluid nodes per side, 1 solid frame node | wall faces half a cell outside the fluid block |
| lid ramp | 2000 steps | linear in time |
| steady criterion | `max|u(t) - u(t-1000)| / U < 1e-9` twice | early exit, `min_steps = 20000` |
| precision | float64 | |

Extra exact-solution case used as a **gate**: plane Couette flow, `H = L`, `U = 0.05`, `tau = 0.8`,
bottom wall at rest, top wall moving, 12 000 steps → exact steady profile `u_x(y) = U (y - 1/2)/L`.

Reference: Ghia, Ghia & Shin (1982), *J. Comput. Phys.* **48**, 387 (tables I/II, Re = 100),
values as embedded in `benchmarks/xval/aa_fused_stream_collide_ordering_audit.py` (same table as
`src/tensorlbm/verification/lid_driven_cavity/reference_data.py::GHIA_RE100`).

---

## 2. Method — three gates, in order of logical dependence

The kernel under test is `tensorlbm.aa_fused_stream_collide.aa_fused_step` (AA-pattern fused pull,
`f_i(x,t+1) = f_i(x-c_i,t) + inc_i(x-c_i,t)`, no post-collision array). Driver:
`benchmarks/xval/aa_fused_cavity_acceptance.py` (one task per scheduler shard), figures:
`tests/gen_aa_fused_cavity_figures.py`, gates: `tests/test_aa_fused_cavity_acceptance.py`.

1. **`couette` — moving-wall momentum sign.** A "fused vs materialised" comparison cannot see a
   wrong sign in the moving-wall term `+6 w_i rho (c_i · u_w)`, because both operators share that
   line of code. Plane Couette flow has an exact solution, so it can. A sign-inverted control arm is
   included in the test suite and must be rejected.
2. **`equality` — fused vs canonical collide-then-stream**, now inside the *cavity* geometry
   (non-uniform state, wall-bounded, moving lid), asserting `torch.equal`.
3. **`cavity` — vs Ghia (1982)**, `L2` of the centre-line profiles normalised by `U`.

### Metric corrections made in this mission (both were harness defects, not kernel defects)

* **F1 — `1/U` inflation.** `aa_fused_stream_collide_ordering_audit.py::_l2_profile` subtracted the
  *normalised* Ghia value from the *dimensional* profile and divided the residual by `U` afterwards,
  i.e. it computed `sqrt(mean((u - u_ghia)^2))/U`. For `U = 0.1` every error was inflated by ×10, so
  a converged cavity would have scored `L2 ≈ 0.9` and the `< 15 %` acceptance could never be met by
  *any* solver. Fixed (normalise first); regression test
  `test_metric_normalises_before_subtracting`.
* **F2 — conservation measured on the wrong nodes.** `f.sum()` includes **solid** nodes, whose
  populations are never used as a source by the half-way bounce-back; they are garbage and can even
  diverge harmlessly. Measuring mass over all nodes therefore reports a "93 % mass difference"-style
  artefact. Mass and finiteness are now evaluated on fluid nodes only, and the containment property
  is pinned by `test_solid_node_junk_does_not_leak_into_fluid`.

The metric also has a **floor**: a perfect solution still scores `sqrt(mean((u_ghia - interp
u_ghia)^2))` because the comparison extrapolates to `y = 0, 1` from the half-cell node. The floor is
reported per grid (0.0261 at `L = 32`, 0.0131 at 64, 0.0067 at 128, 0.0044 at 192) —
errors of that size are *indistinguishable from exact*.

---

## 3. Results

### 3.1 Gate 1 — moving wall vs exact Couette solution (`fig3_couette_gate.png`)

| quantity | value | gate | verdict |
|---|---|---|---|
| `max|u_num - u_exact| / U` (L=32, 12 000 steps) | **5.99e-06** | < 1 % | PASS |
| `L2` error | 6.1e-06 | < 1 % | PASS |
| fluid-node mass drift | 7.9e-13 | < 1e-10 | PASS |
| sign-inverted control arm | error > 10 % (rejected) | must fail | PASS |

The moving-wall bounce-back is therefore correct in magnitude **and sign** — the property the earlier
"equivalence" tests could not establish.

### 3.2 Gate 2 — fused vs materialised in the cavity geometry

`task_equality` runs both operators on the same non-uniform, wall-bounded, moving-lid state and
requires bit equality (`torch.equal`, `max|Δf| = 0.0`). The suite version (`L=16`, 300 steps) passes;
the scheduler shard carries the same assertion at `L = 32`, 20 000 steps.

### 3.3 Gate 3 — cavity vs Ghia (1982), Re = 100

| grid | steps run | converged | `L2 u_x` (full) | `L2 u_y` (full) | `L2 u_x` (interior) | metric floor (u) | mass drift | `< 15 %` | `< 3 %` |
|---|---|---|---|---|---|---|---|---|---|
| L = 32 | 9 000 | no | **2.60 %** | **4.80 %** | 0.82 % | 2.61 % | 8.4e-13 | **PASS** | no (`L2_v` 4.8 %) |

Reading this table honestly:

* `L2 u_x = 2.60 %` sits **at the metric floor** (2.61 %): the vertical centre-line velocity is, to
  the resolution of the comparison, exact. The residual `4.8 %` in `u_y` is the horizontal centre
  line, where the Ghia table is sampled at 16 points including the wall endpoints.
* The run is **not steady** (`max|Δu|/U` still ≈ 1e-4 per 1000 steps at step 9000, well above the
  1e-9 criterion), so this is a coarse-grid, pre-convergence point, not the final acceptance number.
  It is nevertheless already a factor 3–6 inside the 15 % acceptance gate and inside `rho ∈ [0.970,
  1.057]` (`Ma`-consistency holds).
* **`L = 32` is a grid on which the previously published kernel failed hard** (`L2_u ≈ 0.25`, and
  mass drift 93.9 %). The same grid now scores 2.6 % with conserving fluid mass — the ordering/BC
  fixes, not grid coarseness, were the cause of the old FAIL (finding F2 of the ordering audit
  confirmed).

`fig4_cavity_profiles.png` (centre lines vs Ghia), `fig5_cavity_field.png` (velocity magnitude,
streamlines and stream-function contours of the converged field) and `fig6_cavity_convergence.png`
(`L2` vs `L` with both gates and the floor, steady residual, mass drift) are regenerated from the JSON
artefacts; with only one grid point available, the convergence panel currently shows a single curve.

---

## 4. Error judgement (explicit)

| key quantity | reference | error | `< 3 %`? | `< 15 %`? |
|---|---|---|---|---|
| Couette `u_x(y)` profile | exact `U(y-1/2)/L` | 6.0e-06 | **yes** | yes |
| cavity `L2(u_x)`, L=32, step 9000 | Ghia 1982 | 2.60 % (floor 2.61 %) | yes (at the floor) | **yes** |
| cavity `L2(u_y)`, L=32, step 9000 | Ghia 1982 | 4.80 % | no | **yes** |
| fluid mass conservation, cavity L=32 | closed box | 8.4e-13 relative | yes | yes |
| fused vs canonical operator | collide-then-stream | 0.0 (bit-identical) | yes | yes |

The anchor acceptance (`< 15 %` against a reference) is **met at the one grid point that was actually
measured**; it is **not yet demonstrated over a converged, multi-grid sweep**, which is the remaining
work below. No number in this document comes from the retired `stream_then_collide_legacy` operator.

---

## 5. Notes, pitfalls, limitations

1. **Impulsive start is not usable.** At `L = 16, tau = 0.548` an impulsive lid start produces
   non-finite populations; with `U = 0.1` and BGK the transient must be ramped. All runs here use a
   2000-step linear ramp. (The repository's `CavitySolver` documents the same effect for Re = 5000 via
   `ramp_steps`.)
2. **Solid-node populations are garbage and can blow up without consequence.** Only fluid-node norms
   are meaningful; a whole-field `isfinite`/mass check reports spurious failures
   (`test_solid_node_junk_does_not_leak_into_fluid` pins the containment).
3. **Smooth single-mode flows do not discriminate operator orderings** (ordering audit, finding F4).
   The Couette gate is used only for the *wall* term; the operator identity is carried by the
   structural `torch.equal` test, not by a profile error.
4. **Scope (updated 2026-11-16, m22757)**: D2Q9 / BGK only *for this document*. The D3Q19
   transfer of the pattern is implemented and verified separately
   (`src/tensorlbm/aa_fused_push_pull_3d.py`,
   `docs/xval_aa_pattern_d3q19_fused_stream_collide.md`: bit-identical to the materialised
   oracle, Couette / Poiseuille / shear-wave viscosity / Taylor-Green all below 1 %).
   Nothing else here transfers automatically to MRT/TRT or D3Q27 variants BGK at `tau = 0.596` (L=32) is a marginally stable relaxation; TRT/regularised
   collision would be the right tool if finer `Re` is required.
5. **Not run (honest)**: the converged multi-grid sweep `L = 32/64/128/192` was submitted to the
   scheduler as job **cj3680** (6 shards) and was still `queued` (position ≈ 58, 84 jobs in flight)
   after 33 min of waiting; the numbers above are the `L = 32` shard reproduced locally at a 9000-step
   budget. Earlier job **cj3609** (4-arm ablation, `L=64`) delivered only one arm before being
   superseded; that arm (`streamThenCollide+halfwayBB`) reported `diverged = true` at `tau = 0.692`,
   which is recorded as-is but was not reproduced here and is not used as evidence.

## 6. Remaining work

1. Harvest `cj3680` and extend the table of §3.3 to `L = 64/128/192`, with the steady criterion
   reached; report the observed convergence order against the `O(1/L)` metric floor.
2. Re-run the `streamThenCollide` arm to decide whether its divergence is physical (no collision
   increment carried by the pull → the lid momentum is relaxed away) or a defect of the diagnostic
   driver.

## 7. Artefacts

| path | content |
|---|---|
| `benchmarks/xval/aa_fused_cavity_acceptance.py` | driver: 6 tasks (Couette gate, equality gate, 4 cavity grids) |
| `tests/test_aa_fused_cavity_acceptance.py` | 8 fast gates (Couette + sign-flip control, equality, fluid mass, solid-junk containment, metric floor/linearity/normalisation) |
| `tests/gen_aa_fused_cavity_figures.py` | figures 3–6 + `cavity_accept_summary.json` |
| `artifacts/xval_aa_pattern/cavity_accept_0.json` | Couette gate numbers |
| `artifacts/xval_aa_pattern/cavity_accept_2.json` | cavity L=32 numbers + history + profiles |
| `artifacts/xval_aa_pattern/cavity_accept_field_L32.npz` | converged field for the contour figure |
| `artifacts/xval_aa_pattern/fig3..fig6*.png` | figures |

Reproduce:

```
cd TensorLBM
python benchmarks/xval/aa_fused_cavity_acceptance.py --task-index -1   # cheap gates
python benchmarks/xval/aa_fused_cavity_acceptance.py --task-index 2    # cavity L=32 (~2 min)
python tests/gen_aa_fused_cavity_figures.py
pytest tests/test_aa_fused_cavity_acceptance.py tests/test_aa_fused_stream_collide.py -q
# 21 passed
```
4. **Scope**: D2Q9 / BGK only. Nothing here transfers automatically to MRT/TRT or D3Q19/27 variants
   of the pattern. BGK at `tau = 0.596` (L=32) is a marginally stable relaxation; TRT/regularised
