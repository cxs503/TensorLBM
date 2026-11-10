# IN62-N06 — LBM divergence-free synthetic inflow (DFSEM / digital filter)

## Development-section self-preservation law and spurious-pressure control

| field | value |
|---|---|
| campaign node | IN62-N06 (TensorLBM) |
| generated | 2026-11-10T01:05:41 |
| code revision | `65c355f4fba76f711244c6eb01970ca0747688e7` |
| python / torch | 3.11.6 / 2.7.1 |
| platform | Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 |
| scope | inlet quality + development section only (no hull resistance) |

## 1. Case, physics and parameters

**Physics.** A uniform stream `U∞` enters a plane channel through a synthetic-turbulence inlet; the fluctuation field is imposed on a thin inlet slab and then convected and dissipated in a straight development section (periodic in *y*, *z*, zero-gradient outlet). The questions are (i) how faithfully the inlet reproduces a prescribed Reynolds-stress tensor `R_ij = <u_i' u_j'>`, (ii) whether the inlet is solenoidal — `∇·u'` is the acoustic source of the pressure equation, so a compressive inlet radiates spurious pressure waves that corrupt every downstream quantity — and (iii) how far downstream a prescribed `R_ij` survives (the *development length* `L_dev`).

**Target inlet Reynolds stresses** (lattice units, `u_lb = 0.05` ≈ 2 % Mach):

| R_uu | R_vv | R_ww | R_uv | R_uw | R_vw | Tu=√(2k/3)/U∞ |
|---|---|---|---|---|---|---|
| 4.000e-04 | 2.000e-04 | 2.000e-04 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 32.66% |

`R_vv = R_ww = 0.5 R_uu` is the axisymmetric (rod-vortex) state left by a spanwise-homogeneous shear; the isotropic case used for the self-preservation and decay-law experiments is `R_ij = k δ_ij`.

**Numerics of the development section** (D3Q19, BGK):

| parameter | value |
|---|---|
| nx | 96 |
| ny | 32 |
| nz | 32 |
| tau | 0.505 |
| u_mean | 0.05 |
| injection_cells | 2 |
| ramp_cells | 4 |
| n_steps | 4000 |
| n_warmup | 1500 |
| sample_every | 10 |
| ν = (τ-1/2)/3 | 1.667e-03 |
| u'_rms at inlet | 2.018e-02 |
| statistics samples | 250 |

**Inlet generators.** DFSEM = eddy kernels built from the curl of a stream function (divergence-free by construction), with an `A = chol(R)·C^{-1/2}` stress-restoration layer and an a-posteriori Helmholtz projection; SEM = classical synthetic eddy method (independent scalar kernels, *not* divergence-free — the negative control); DFM = Klein digital filter with an optional Lund transformation.

## 2. Method

1. **Inlet-plane stage** (`plane`): every generator is sampled 64 times on a 48×48 plane; the metrics are the ensemble Reynolds stress error `max_ij |R̂_ij − R_ij*| / R_uu*` and the scale-free divergence diagnostic `D = ‖∇·u'‖_rms / ‖∂u'_t/∂x_t‖_rms` (zero for a solenoidal inlet, O(1) for a compressive one, independent of the amplitude).
2. **A-posteriori projection**: the stress-restoration map mixes the streamwise component into the transverse pair and therefore injects a compressive part; `P = I − k kᵀ/|k|²` is applied in spectral space plane-by-plane, then the (project → restore) pair is iterated `n_stress_iterations` times, the last step a projection. Both residuals are recorded: a frozen in-plane field cannot be simultaneously exactly divergence-free and carry an arbitrary `(R_vv, R_ww, R_vw)` set, so the trade-off is reported as a Pareto front rather than hidden.
3. **Development-section stage** (`duct`): the fluctuation sequence is re-imposed on a 2-cell slab with a cosine ramp (the ramp suppresses the spurious pulse that a hard velocity imposition emits), convected by D3Q19-BGK, and `R_ij(x)` plus the pressure variance are accumulated in slabs over the statistically steady window. All three inlet families share the same random seed (common random numbers).
4. **Self-preservation**: `L_dev` is the first slab beyond which both the normalised anisotropy shape `R_ij/R_uu` and the local decay exponent `d ln k / d ln x` stay within 5 % of their far-field values for *all* remaining slabs (moving-window averages, replicate-padded).
5. **Decay law**: periodic isotropic turbulence is integrated from a divergence-free `E(k) ∝ k^{-5/3}` initial condition and `k(t)` is fitted with `A (t+t₀)^{-n}` (two-stage scan of the virtual origin); `n` is compared with the Comte-Bellot & Corrsin (1971) coasting-decay exponent.
6. **Machine gates**: thresholds were declared in the script before the production run (`GATES`); the run only reports them.

## 3. Results

### 3.1 Inlet plane — stress recovery and divergence

| case | stress error | D = ‖∇·u'‖/‖∂_t u_t‖ | u'_rms | v'_rms | w'_rms |
|---|---|---|---|---|---|
| dfsem_projected | 0.48% | 7.926e-07 | 2.0048e-02 | 1.4113e-02 | 1.4167e-02 |
| dfsem_unprojected | 0.48% | 6.794e-02 | 2.0048e-02 | 1.4142e-02 | 1.4142e-02 |
| sem_projected | 3.13% | 1.013e-06 | 2.0005e-02 | 1.4579e-02 | 1.3791e-02 |
| sem_unprojected | 0.41% | 9.654e-01 | 2.0004e-02 | 1.4200e-02 | 1.4149e-02 |
| dfm_lund | 0.00% | 1.025e+00 | 2.0000e-02 | 1.4142e-02 | 1.4142e-02 |
| dfm_raw | 36.51% | 1.026e+00 | 1.9382e-02 | 1.8602e-02 | 1.8077e-02 |

### 3.2 Projection / stress-restoration Pareto front (shear-carrying target)

| n_stress_iterations | stress error | D |
|---|---|---|
| 0 | 8.97% | 8.752e-07 |
| 1 | 9.26% | 8.716e-07 |
| 2 | 9.36% | 8.886e-07 |
| 4 | 9.53% | 8.810e-07 |
| 8 | 10.01% | 9.131e-07 |

### 3.3 Development section — inlet recovery, decay and L_dev

| case | inlet R_ij error | R_uu(2–5) | R_vv/R_uu | k(5)/k(2) | L_dev [cells] | p'_rms(2–8) |
|---|---|---|---|---|---|---|
| dfsem | 9.30% | 4.006e-04 | 0.476 | 0.8786 | 80.5 | 6.453e-03 |
| sem | 3.76% | 4.004e-04 | 0.500 | 1.1323 | 71.5 | 1.569e-02 |
| dfm | 8.96% | 3.888e-04 | 0.502 | 0.9997 | 31.5 | 3.092e-02 |
| dfsem_iso | 9.58% | 4.004e-04 | 0.952 | 0.9127 | nan | 6.725e-03 |

`L_dev` above is the campaign definition: the shortest distance beyond which the anisotropy shape `R_ij/R_uu` stays within 5 % of its far-field value (shape-only criterion).

**L_dev tolerance sweep** (same data, same estimator; the strict variant additionally requires the local decay exponent `d ln k/d ln x` to be stationary):

| case | samples | shape-noise floor | L_dev(5 %) | L_dev(10 %) | L_dev(15 %) | L_dev(20 %) | L_dev(25 %) | L_dev(30 %) |
|---|---|---|---|---|---|---|---|---|
| dfm | 250 | 0.114 | 31.5 | 23.5 | 18.5 | 15.5 | 13.5 | 11.5 |
| dfsem | 250 | 0.025 | 80.5 | 75.5 | 67.5 | 59.5 | 48.5 | 40.5 |
| dfsem_iso | 250 | 0.027 | nan | 74.5 | 68.5 | 62.5 | 58.5 | 54.5 |
| sem | 250 | 0.010 | 71.5 | 12.5 | 7.5 | 5.5 | 3.5 | 2.5 |

The shape-only criterion reaches 5 % only far downstream (`L_dev` comparable to or larger than the domain), and the strict rate criterion is never satisfied inside 96 cells — the decay rate relaxes only logarithmically, so a 5 % rate tolerance needs a domain several times longer. Both numbers are reported; neither is extrapolated.

Inlet recovery error is `max_ij |R_ij(2..5) − R_ij*| / R_uu*`; it is the quantity a downstream user sees *after* the ramp, so it is larger than the plane error by construction (viscous dissipation inside the ramp + finite-sample error).

### 3.4 Isotropic decay vs Comte-Bellot & Corrsin (1971)

| resolution | diverged | n | n_range (t₀ conditioning) | Re_λ (initial) | K(0) |
|---|---|---|---|---|---|
| n64 | False | 1.899 | [1.892, 1.907] | 143.0 | 1.250e-03 |
| n48 | False | 1.922 | [1.908, 1.936] | 121.8 | 1.250e-03 |

**Window dependence of the exponent** (`n` fitted on the last 30/50/70/100 % of the series; a drifting `n` means the run has not reached a self-similar decay):

| resolution | n(30 %) | n(50 %) | n(70 %) | n(100 %) |
|---|---|---|---|---|
| n64 | 1.912 | 1.920 | 1.934 | 1.891 |
| n48 | 2.774 | 2.177 | 2.048 | 1.901 |

## 4. Machine gates (thresholds predeclared before the run)

| gate | quantity | measured | threshold | verdict |
|---|---|---|---|---|
| G1 | plane stress error, projected DFSEM | 4.761e-03 | ≤ 0.05 | PASS |
| G2a | plane D, projected DFSEM | 7.926e-07 | ≤ 0.001 | PASS |
| G2b | divergence reduction (unprojected / projected) | 8.571e+04 | ≥ 30.0 | PASS |
| G3 | negative-control D (max of unprojected DFSEM / SEM) | 9.654e-01 | ≥ 0.05 | PASS |
| G4a | digital filter + Lund stress error | 6.786e-07 | ≤ 1e-06 | PASS |
| G4b | Lund gain (raw / Lund) | 5.380e+05 | ≥ 10.0 | PASS |
| G5 | development-section inlet R_ij error | 9.300e-02 | ≤ 0.15 | PASS |
| G6 | L_dev, shape-only 5 % (campaign definition, DFSEM) | 8.050e+01 | ≤ 0.75 nx = 72 cells | **FAIL** |
| G6-iso | L_dev, shape-only 5 % (isotropic inlet) | nan | ≤ 72 cells | **FAIL** |
| G6r | L_dev, shape + rate 5 % (strict) | nan | ≤ 72 cells | **FAIL** |
| G7 | spurious pressure ratio SEM / DFSEM | 2.431e+00 | ≥ 2.0 | PASS |
| G8 | decay exponent vs CBC n = 1.30 | 1.899e+00 | rel. error ≤ 0.15 | **FAIL** |
| G9 | kernel equivalence with library | 0.000e+00 | ≤ 1e-06 | PASS |
| G10 | bit-for-bit reproducibility | True | identical | PASS |

**10/14 gates pass.**

## 5. Pitfalls found while building this (all fixed, regression-tested)

1. **Nyquist de-aliasing of the Helmholtz projector.** With the raw wavenumber array the transverse Nyquist plane (`k = ±N/2`) breaks the Hermiticity of the spectral correction (the cross term `k_y k_z` flips sign because the plane is its own negative), leaving `D ≈ 0.10` on white noise and an unrepresentable correction in the one-sided `rfft2` form. De-aliasing the Nyquist wavenumber (equivalent to the central-difference symbol `sin k = 0` there) and using a full complex transform makes the projection exact to round-off (`D = 3e-16` in float64, `2e-7` in float32).
2. **`L_dev` mask scanned past the admitted window.** `ok[i:].all()` included the slabs that the criterion itself marks inadmissible (`x ≥ x_stop`), so `L_dev` was *always* `nan`. Fixed to `ok[i:x_stop]`.
3. **Moving-window endpoints.** Zero padding of the slab average and an even window width faked a decay-rate slope at both ends; replicate padding plus an odd width plus a `w//2` endpoint margin was needed before an exactly stationary profile gave `L_dev = x_start + 0.5`.
4. **Virtual-origin bias in the decay fit.** A 240-point scan over `t₀` biased `n` by 1.7 % even on an exact power law; two refinement passes bring it to 1.6e-6.
5. **Forcing ramp is not optional.** A hard velocity imposition on the inlet slab emits a pressure pulse that annihilates the injected fluctuations within a few cells; the cosine ramp is part of the experiment, not a cosmetic choice.

**Limitations.** (i) A frozen in-plane inlet field cannot be simultaneously exactly divergence-free and carry an arbitrary transverse stress triple — the Pareto front is the honest statement, and the reported `D` is for the projected member. (ii) the decay runs start at `Re_λ ≈ 120–140` but the initial spectrum is cut off at `k = 4` rad/cell, i.e. the inertial range spans less than one decade, and the late-time decay is viscously dominated (`n` rises again when only the last 30 % of the series is fitted). The CBC comparison is therefore an order-of-magnitude check, not a spectrum-level match. (iii) The development section is a straight duct with periodic transverse boundaries: wall blocking, streamwise pressure gradient and curvature effects are outside this node's scope. (iv) Statistics use 250–560 samples, so `R_ij(x)` carries a few-percent standard error that is reported per slab (`R_stderr_profile`).

## 6. Error assessment against a reference

| quantity | reference | measured error | < 3 % ? |
|---|---|---|---|
| plane R_ij recovery (projected DFSEM) | prescribed R_ij* | 0.48% | yes |
| digital filter + Lund R_ij | prescribed R_ij* | 6.786e-07 | yes |
| in-plane divergence after projection | 0 (solenoidal) | 7.926e-07 (absolute, scale-free) | yes (absolute) |
| development-section inlet R_ij (x = 2–5) | prescribed R_ij* | 9.30% | **no** — viscous decay inside the forcing ramp plus finite-sample error |
| decay exponent n | CBC (1971) n = 1.30 | 46.07% | **no** — the measured exponent is 1.90 ± 0.01 at both resolutions, i.e. 46 % above CBC; honest FAIL (see §3.4, §5) |
| L_dev | definition (95 % self-preservation) | — | n/a — no experimental benchmark exists; the value is a definition, and only its resolution dependence is diagnostic |

The >3 % items are reported as such and are **not** rounded away: the inlet *plane* is where the 3 % standard is met, the downstream duct cannot meet it because the forced slab necessarily dissipates energy before the first measuring station.

## 7. Figures

- `artifacts/in62_n06_inlet_quality/figures/fig1_inlet_plane_quality.png` — inlet-plane Reynolds stresses (normalised by the target `R_uu`) and the measured divergence diagnostic for the four inlet variants
- `artifacts/in62_n06_inlet_quality/figures/fig2_development_section.png` — development section: `R_ij(x)` relaxation and the rms pressure fluctuation (spurious-acoustic negative control) for DFSEM / SEM / DFM
- `artifacts/in62_n06_inlet_quality/figures/fig3_isotropic_decay.png` — isotropic decay against the Comte-Bellot & Corrsin (1971) exponent, and the energy spectra at the first and last sample
- `artifacts/in62_n06_inlet_quality/figures/fig4_inlet_field_clouds.png` — inlet-plane field clouds: projected `u'`, `v'`, `w'` and the divergence source term, unprojected versus projected

## 8. Artifacts and audit chain

| artifact | sha256[:16] |
|---|---|
| `artifacts/in62_n06_inlet_quality/meta.json` | `0ca12014a39261cd` |
| `artifacts/in62_n06_inlet_quality/report.json` | `b1cd7641f9f5ff00` |
| `artifacts/in62_n06_inlet_quality/results_decay.json` | `2b7dc98e071a9717` |
| `artifacts/in62_n06_inlet_quality/results_duct.json` | `ed29e54387d77744` |
| `artifacts/in62_n06_inlet_quality/results_duct_dfm.json` | `3bbc3db4053916a4` |
| `artifacts/in62_n06_inlet_quality/results_duct_dfsem.json` | `0bdd6816416c9b55` |
| `artifacts/in62_n06_inlet_quality/results_duct_dfsem_iso.json` | `cb9180adaaecf29f` |
| `artifacts/in62_n06_inlet_quality/results_duct_sem.json` | `9d2d43f0da703858` |
| `artifacts/in62_n06_inlet_quality/results_ldev.json` | `49f7651a63946435` |
| `artifacts/in62_n06_inlet_quality/results_plane.json` | `e58b2d690a3f6322` |

Reproduce with:

```bash
PYTHONPATH=src python experiments/in62_n06_inlet_quality/run_inlet_quality.py \
    --stage all --out artifacts/in62_n06_inlet_quality --threads 8
PYTHONPATH=src python experiments/in62_n06_inlet_quality/make_report_md.py \
    --run-dir artifacts/in62_n06_inlet_quality --out docs/IN62_N06_INLET_QUALITY.md
```

Module-level tests: `PYTHONPATH=src pytest tests/test_inlet_quality_in62_n06.py -q`.

