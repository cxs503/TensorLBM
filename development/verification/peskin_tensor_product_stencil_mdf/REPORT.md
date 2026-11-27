# XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF — TensorLBM vs haimingz/vivsim

**Anchor** `XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF` · round 1 · owner FEAT-ANCHOR-FEEDER ·
repo `TensorLBM` · branch `agent/m18131` · date 2026-11-13

**OSS reference** `haimingz/vivsim` @ `fb84b12f02e18f3b9dfdf738d6a1a318d81539f9` (MIT),
files `vivsim/ib/{kernels,stencil,mdf,geometry}.py` + `vivsim/ib3d/stencil.py`,
vendored under `oss_ref/` with `PROVENANCE.json` (sha256 upstream vs vendored,
`bodies_identical_ex_imports=true` for all five files). JAX is absent in this
environment, so the upstream bodies run behind a `numpy` shim (`oss_ref/np_jax_shim.py`);
the shim is validated against TensorLBM's independent brute-force loop
(`ibm_velocity_interpolate` / `ibm_force_spread`): agreement **2.2e-16 / 0.0**.

---

## 1. Question, alternatives, predeclared thresholds

| item | content |
|---|---|
| decision | ship a diff list + porting recommendation; if the operator deviates >15 %, port the OSS scheme and verify it |
| H0 | TensorLBM's Peskin tensor-product stencil + multi-pass direct forcing reproduces the OSS stencil + MDF within 15 % |
| H1 | the TensorLBM **pointwise** fixed point `f = 2ρ(U_t − u_int)` omits the round trip `A = interp∘spread` (row sum ≈0.4 at 1-cell spacing) and deviates >15 % |
| A1 | `tensorlbm.ibm_force_transfer.direct_forcing` (Guo-corrected pointwise multi-pass) — **pre-existing** |
| A2 | OSS `multi_direct_forcing` on `get_ib_stencil` (vendored, executed) |
| A3 | `direct_forcing_mdf` — **added by this mission** (round-trip consistent) |
| thresholds (predeclared, `predeclared.json`) | stencil/kernel identity <1e-12 abs; force & interface-slip <15 % rel at h=0.9, n=10 |

All iteration counts, marker spacings and run lengths were fixed in
`predeclared.json` **before** the run; nothing was re-tuned after seeing numbers.

## 2. Method

Four levels, one driver (`run_xval.py`), all outputs in `artifacts/xval_peskin_tp_mdf.json`:

* **L1 kernel/stencil identity** — compare the 1-D delta factors and the
  `(marker, node) → weight` maps after decoding both stencils to physical node
  coordinates (TensorLBM stores `(ny,nx)` with x last, OSS `(nx,ny)` with x first —
  the axis convention is decoded explicitly, not assumed).
* **L2 operator** — same marker set, same Eulerian field `u=(0.05,0)`, same kernel;
  measure the converged grid force and the **achieved no-slip**
  `max_k |U_t − (u_int + ½·interp(spread F))| / U∞` (the Guo half-force interface
  condition both codes target), swept over marker spacing `h = 0.5…3.0` cells and
  iteration count `n = 1…400`.
* **L3 port** — `direct_forcing_mdf(stencil, u_grid, u_target, rho_grid=None, n_mdf=10)`
  added to `src/tensorlbm/ibm_force_transfer.py` (same return contract as
  `direct_forcing`, so it is a drop-in replacement), verified against A2.
* **L4 end-to-end** — cylinder at Re=20 in one LBM, arms A1/A2/A3, cylinder drag.
  **Reported as partial / honest FAIL for the absolute column**, see §6.

## 3. Parameter table

| quantity | value |
|---|---|
| grid (L1/L2) | 64×64 periodic, float64 |
| markers | circle, R = 8, centre (32.13, 31.87), h = 0.5/0.7/0.9/1.2/2.0/3.0 → M = 101/72/56/42/25/17 |
| kernel | Peskin 4-point, support 2, tensor-product stencil (16 points/marker in 2-D, 64 in 3-D) |
| Eulerian field | `u = (0.05, 0)`, `U_target = 0` (steady-body limit), Ro→0, Ma = 0.05/c_s ≈ 0.087, Re = U·M_cell/ν undefined (no viscosity enters the operator level) |
| dimensionless groups | Ma ≈ 0.087 (operator levels); end-to-end Re = U D/ν = 20, Ma = 0.1, blockage D/H = 1/20 |
| iterations | n = 1,2,3,5,10,20,40 (+ 100/200/400 in the convergence study) |
| 3-D control | 40 Fibonacci sphere markers, R = 5, grid 32³ |

## 4. Results (all numbers machine-written into `artifacts/xval_peskin_tp_mdf.json`)

### 4.1 L1 — kernel and stencil are identical (PASS)

| check | result | verdict |
|---|---|---|
| Peskin 4-pt kernel, 2001 samples in [−2.5, 2.5] | max abs diff **0.0** | ✅ |
| 2-D tensor-product stencil, 44 markers × 16 pts | max abs weight diff **0.0**, non-zero node keys identical, partition-unity error 0.0 | ✅ |
| 3-D tensor-product stencil, 60 sphere markers × 64 pts | max abs weight diff **1.4e-17**, keys identical | ✅ |

So the "Peskin tensor-product stencil" part of the anchor is **already exact**;
the whole XVAL signal comes from the forcing scheme.

### 4.2 L2 — the forcing schemes are *not* equivalent (the finding)

Grid force = Σ_x of the spread force on a circle of radius 8 in uniform flow
(the force that must be injected to hold the interface at rest);
`A_row` = mean row sum of the round-trip operator `interp∘spread`.

| h [cells] | M | A_row | OSS MDF (n=40) | TL pointwise (n=∞) | rel | TL MDF port (n=20) | rel |
|---|---|---|---|---|---|---|---|
| 0.5 | 101 | 0.762 | −13.2469 | −10.1000 | **23.8 %** ❌ | −13.2469 | 1.8e-7 ✅ |
| 0.7 | 72 | 0.544 | −13.2478 | −7.2000 | **45.7 %** ❌ | −13.2476 | 1.3e-5 ✅ |
| **0.9** | 56 | **0.423** | **−13.2495** | **−5.6000** | **57.7 %** ❌ | **−13.1942 (n=10)** | **4.2e-3** ✅ |
| 1.2 | 42 | 0.317 | −13.2469 | −4.2000 | **68.3 %** ❌ | −13.2403 | 5.0e-4 ✅ |
| 2.0 | 25 | 0.189 | −13.2082 | −2.5000 | **81.1 %** ❌ | −13.0093 | 1.5e-2 ✅ |
| 3.0 | 17 | 0.145 | −11.7622 | −1.7000 | **85.5 %** ❌ | −11.2437 | 4.4e-2 ✅ |

Achieved no-slip at the markers (`iface_slip_rel`, lower = better, n = 10):

| h | OSS MDF | TL pointwise | TL MDF port |
|---|---|---|---|
| 0.5 | 1.2 % ✅ | **24.5 %** ❌ | 0.43 % ✅ |
| 0.9 | **1.5 %** ✅ | **58.2 %** ❌ | **1.0 %** ✅ |
| 2.0 | 1.0 % ✅ | 81.3 % ❌ | 1.7 % ✅ |
| 3.0 | 0.4 % ✅ | 85.8 % ❌ | 4.6 % ✅ |

**Verdict H1 accepted, H0 rejected**: at the marker density every IBM code
actually uses (h ≈ 0.7–1.2 cells) TensorLBM's `direct_forcing` injects
**46–68 % too little** marker force and leaves **46–69 % of the target velocity
deficit** at the interface, i.e. it violates no-slip by far more than the 15 % bar,
while the OSS MDF and the new port stay inside 1.5 %.

Mechanism (measured, not asserted): `direct_forcing` iterates
`u_phys = u_int + ½f/ρ` with `u_int` frozen, fixed point `f = 2ρ(U_t − u_int)`.
The marker actually feels `interp(spread f) = A f`, and `A` has row sum 0.42 at
h = 0.9 — so the force booked is only `A` times the force applied, and the
operator compensates exactly by the measured amplification
`1/A_row`: 1.31/1.84/**2.37**/3.15/5.28/6.92 at h = 0.5/0.7/**0.9**/1.2/2.0/3.0
(measured force ratios 1.31/1.84/**2.37**/3.15/5.28/6.92 — the two columns agree
to 3 digits). OSS MDF solves `U_t = u_int + ½A f` by re-interpolating the
just-spread force inside the iteration; TensorLBM's port does the same.

### 4.3 Convergence (⑤)

* OSS MDF contraction per iteration at h = 0.9: 0.62 → 0.39 → 0.24 → 0.092 (n=5)
  → 8.6e-3 (n=10) → 1e-4 (n=20); the `ds` weights are a **preconditioner only** —
  converged force with `ds = 1` vs the true `ds` differs by ≤2e-3 (h=3.0) and
  ≤3.4e-5 for h ≤ 1.2 (`ds_preconditioner_only`).
* Long-iteration study at h = 0.9: OSS force −13.24951 (40) → −13.25005 (100) →
  −13.25055 (200) → −13.25114 (400), i.e. still drifting by 1e-5 per 200 sweeps
  (the `A^{-1}` fixed point has slow long-wavelength modes along the circle); the
  port tracks it to **4.6e-6 … 6.6e-6** at every n, so the two formulations share
  one fixed point and only differ in the preconditioner.
* Independent arm (A1) is itself convergent but to a *different* fixed point
  (−5.6 at h = 0.9, `f_n = 2 r_0 (1 − 0.5^{n+1})`): the default
  `n_correctors = 2` in `direct_forcing` sits at **87.5 %** of that (wrong)
  value, so both the iteration budget and the formulation are involved.
* Sensitivity to marker density (second sensitivity axis): the OSS force is
  **h-independent to 0.5 %** for h ≤ 2 (−13.21…−13.25, as physics requires: the
  momentum removed is set by the body, not by the marker count) and drops 11 %
  only at h = 3 (too few markers to resolve the circle). The A1 force is a
  monotone function of h (−1.7 … −10.1, 6× spread) — a diagnostic that A1 is
  measuring the discretisation, not the body.

## 5. Diff list — TensorLBM vs OSS (做法 / 参数 / 边界条件) and porting tradeoffs

| aspect | OSS haimingz/vivsim | TensorLBM (before) | action |
|---|---|---|---|
| 1-D kernel | `kernel_peskin_4pt`, `kernel_peskin_3pt`, **`kernel_cosine_4pt`** | `ibm_delta_4pt`, `ibm_delta_hat` (no cosine) | identical for Peskin-4pt (0.0 diff); cosine kernel **absent** in TensorLBM — documented gap, not ported (no caller) |
| kernel docstring | — | `ibm.py` calls the Peskin kernel "Peskin 4-point (**cosine**)" | mislabel; the formulas are the Peskin branches, `kernel_cosine_4pt` is a different function (max weight diff 5.1e-3) |
| stencil | tensor product, offsets `floor(p)+arange(-1,3)`, precomputed `(M,4^d)` | same offsets, same product, `(M, support^d)` | **identical** (0.0 / 1.4e-17) |
| index layout | `idx = x*ny + y` on `(nx,ny)` | `idx` on `(ny,nx)`, x last | equivalent after axis decode; do not mix |
| kernel outside support | `jnp.where` masks NaN from `sqrt(<0)` | `clamp(min=0)` then `where` | numerically identical (NaN masked in both) |
| forcing | **MDF**: iterate with re-interpolation → fixed point `2ρA^{-1}r₀` | **pointwise**: fixed point `2ρr₀` | **ported** (`direct_forcing_mdf`) |
| marker spacing weights | `get_ds` (half adjacent segments) used as MDF preconditioner | none | not required for correctness (≤2e-3); optional speed-up |
| Guo factor | implicit 0.5 in the MDF velocity update | explicit `1−1/(2τ)` in `guo_force_2d` | keep TensorLBM's explicit form; the port uses the same ½ pairing |
| scatter semantics | `.at[].add` (unbuffered, duplicates accumulate) | `index_add_` | equivalent |
| dtype | JAX default float32 | float64 | both arms run float64 here (comparison limit) |

**Porting tradeoff (recommendation)**
1. **Keep** both: `direct_forcing` stays the Uhlmann-style pointwise scheme
   (correct for its own fixed point, cheap — no extra spread/interpolate per
   sweep) and `direct_forcing_mdf` is the interface-exact scheme. Do **not**
   silently switch defaults: existing callers (`ibm_openwater`,
   `fsi_viv_ibm`, `ibm_moving_body`) were calibrated against `direct_forcing`
   plus their own Guo/EDM pairing, and the correct factor (1 vs 2) depends on
   whether the force is applied with EDM (`body_source_2d`, no Guo factor ⇒
   `f = ρA^{-1}r₀`) or Guo (`f = 2ρA^{-1}r₀`). The port implements the Guo
   pairing, matching OSS.
2. **Do** call `direct_forcing_mdf` where the interface condition is the point
   (moving/rotating bodies, thin sheets, force/torque accounting), because the
   pointwise scheme under-reports the force by `1/A_row` (2.4× at h = 0.9, 5.3×
   at h = 2) and that error is *spacing-dependent*, i.e. it silently degrades
   with mesh refinement of the Lagrangian side.
3. Thin-sheet cases are the dangerous ones: the `slip_friction` model in
   `ibm_openwater.marker_forcing` was calibrated *with* the pointwise factor; a
   switch to MDF would multiply its friction torque by `1/A_row` unless the
   closure is re-fitted. Flagged, not changed.

## 6. Honest limits / failures

* **L4 end-to-end is an honest FAIL on the absolute column.** Cylinder Re = 20,
  D = 4, 30D×20D, U = 0.1, MRT, markers h = 1.0, 800 steps (t = 20 D/U):

  | arm | Cd (2nd half) | interface slip | rel vs OSS arm |
  |---|---|---|---|
  | A1 pointwise | 21.30 | 18.0 % | 16.6 % |
  | A2 OSS MDF | 25.55 | 1.1 % | — |
  | A3 TL MDF port | 25.55 | 1.0 % | 0.04 % |

  A2/A3 agree to 0.04 % and A1 is 16.6 % low — the same ordering as the operator
  level, so the arm-to-arm comparison is meaningful. But **all three arms
  over-predict Cd by ~12×** against `CYLINDER_CD_2D[20] = 2.05`
  (`src/tensorlbm/ibm_openwater.py:130`), and the 2600-step probe converges to
  Cd = 25.13 with 1.1 % slip. So the *driver* (not the operator) is at fault:
  the shared interior penalisation (`interior_mask_force` applied as a Guo
  source, i.e. only ~19 % of the deficit removed per step at τ = 0.62) plus a
  D = 4 body leave the solid interior permeable and inflate the momentum sink.
  `fail_stage = L4_end_to_end`, `failure_reason = absolute Cd 12× the
  literature value for all arms`. Reported as **WITHHELD** for absolute
  accuracy; the relative column is informational only. Fixing it needs a proper
  solid-interior treatment (bounce-back interior or iterated mask forcing) —
  a separate mission, deliberately not patched here.
* Operator level is a **frozen-field test**: it measures the force the scheme
  books for one velocity field, not the coupled steady state. In the coupled
  run the pointwise slip fell from 58 % to 18 % because the deficit is
  re-forced every step (temporal accumulation partially compensates). That is
  exactly why the operator level alone cannot settle the practical question —
  and why the L4 driver bug above matters for the follow-up mission.
* float32 (upstream JAX default) not reproduced; `ibm_delta_4pt` NaN masking
  path is exercised but the two implementations mask differently
  (`where` vs `clamp`), results identical.
* Only 2-D forcing was ported; the OSS 3-D `ib3d/stencil.py` stencil is verified
  identical, but `vivsim/ib3d` has no MDF module to compare against.

## 7. Artefacts

| path | content |
|---|---|
| `xval/XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF/predeclared.json` | decision, alternatives, thresholds, stopping rule, confounders |
| `.../oss_ref/` | vendored OSS files + `np_jax_shim.py` + `PROVENANCE.json` (sha256, body-identity flags, OSS conventions) |
| `.../run_xval.py` | the four-level driver |
| `.../artifacts/xval_peskin_tp_mdf.json` | every number in §4 (L1, L2 sweep, convergence, verdict) |
| `.../artifacts/l4_end_to_end_console.log` | verbatim L4 console transcription + the caveat |
| `.../figures/f1_kernel_stencil.png` | 1-D kernels + stencil identity |
| `.../figures/f2_mdf_convergence.png` | iteration convergence + round-trip row sum vs h |
| `.../figures/f3_interface_residual.png` | achieved no-slip vs marker spacing, 15 % bar |
| `src/tensorlbm/ibm_force_transfer.py` | **new** `direct_forcing_mdf` (+`__all__`) |
| `tests/test_xval_peskin_tensor_product_stencil_mdf.py` | 5 regression tests (kernel/stencil identity, port vs OSS, defect characterisation) |

Reproduce: `PYTHONPATH=src python3 xval/XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF/run_xval.py --quick`
(10 s) and `PYTHONPATH=src pytest -q tests/test_xval_peskin_tensor_product_stencil_mdf.py`
(5.4 s, 5 passed).

## 8. Error verdict vs the owner bars

| quantity | reference | got | error | <15 %? | <3 %? |
|---|---|---|---|---|---|
| Peskin 4-pt kernel | OSS `kernel_peskin_4pt` | TensorLBM `ibm_delta_4pt` | 0.0 | ✅ | ✅ |
| 2-D stencil weights | OSS `get_ib_stencil` | `build_stencil` | 0.0 | ✅ | ✅ |
| 3-D stencil weights | OSS `ib3d` stencil | `build_stencil` | 1.4e-17 | ✅ | ✅ |
| marker force @ h=0.9 | OSS MDF −13.2495 | **TL pointwise −5.600** | **57.7 %** | ❌ | ❌ |
| marker force @ h=0.9 | OSS MDF −13.2495 | **TL MDF port −13.194 (n=10)** | **0.42 %** | ✅ | ✅ |
| no-slip slip @ h=0.9 | target 0 | **TL pointwise 58.2 %** | — | ❌ | ❌ |
| no-slip slip @ h=0.9 | OSS 1.5 % | TL MDF port 1.0 % | — | ✅ | ✅ |
| cylinder Cd (relative) | OSS arm 25.55 | TL MDF port 25.55 | 0.04 % | ✅ | ✅ |
| cylinder Cd (absolute) | literature 2.05 | 21.3 / 25.6 | **~1100 %** | ❌ | ❌ (driver bug, §6) |