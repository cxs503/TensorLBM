# CASE-LBM-WOMERSLEY-PULSATILE — pulsatile channel flow vs the exact Womersley solution

**Anchor**: `CASE-LBM-WOMERSLEY-PULSATILE` — ref. Womersley, J. R. (1955),
*Method for the calculation of velocity, rate of flow and viscous drag in
arteries when the pressure gradient is known*, J. Physiol. **127**, 553–563
(DOI 10.1113/jphysiol.1955.sp005276). Acceptance: centreline amplitude ratio
and phase lag vs the analytic solution at `α = 5` (whole-period phase average),
**< 5 %**.

**Status**: gate **PASS** at every tested resolution; measured error is ~100×
smaller than the tolerance (0.05 % amplitude, 0.06° phase at `H = 32`).

* module: `src/tensorlbm/verification/womersley_pulsatile.py`
* tests: `tests/test_womersley_pulsatile_lbm.py` (15 tests, ≈13 s)
* study driver: `scripts/run_womersley_pulsatile_case.py`
* artifacts: `artifacts/womersley_pulsatile_20261112/` (`results.json`,
  `table_study.md`, figures)

---

## 1. Case definition (physics, geometry, parameters)

Fully developed flow in a plane channel, walls at `y = ±H/2`, driven by a
harmonic body force (equivalently a streamwise pressure gradient)
`-∂p/∂x = ρ a₀ cos(ωt)`.  The unsteady Stokes problem

```
ν Û'' − iω Û = −a₀ ,      Û(±H/2) = 0
```

has the closed-form solution with `κ = (H/2)·sqrt(iω/ν)`, `ŷ = 2y/H`:

```
Û(ŷ) = (a₀ / iω) · (1 − cosh(κŷ)/cosh κ)
Ŵ(ŷ) = Û / ⟨Û⟩ = (1 − cosh κŷ/cosh κ) / (1 − tanh κ/κ)      (unit bulk flux)
Ŵ(0) = (1 − sech κ) / (1 − tanh κ/κ)                        (centreline phasor)
```

Normalising by the **cross-sectionally averaged** velocity is deliberate: that
is the quantity an inlet BC or a flow meter supplies, and it removes the
forcing-amplitude calibration from the comparison (the run's own bulk phasor
rescales the analytic profile, so a mis-set `a₀` cannot fake agreement).

Why a channel and not the pipe solution already in
`tensorlbm/turbulence/womersley_exact.py`: the pipe solution is a Bessel
function `J₀/J₁`, the channel solution a cosh. Gating a Cartesian D2Q9 channel
run against the Bessel form would be a category error; the two share only the
high-`α` asymptotes, which this case checks as well.

### Reference parameter table (gate case)

| quantity | symbol | value | note |
|---|---|---|---|
| Womersley number (full gap) | `α = H sqrt(ω/ν)` | 5.0 | requested / effective shown in tables |
| channel rows | `H = ny` | 32 | walls at `±H/2`, half-way bounce-back |
| viscosity | `ν` | 0.05 | `τ = 3ν + ½ = 0.65` |
| angular frequency | `ω = ν(α/H)²` | 1.221e-3 | period 5147 steps |
| Stokes layer | `δ = sqrt(2ν/ω)` | 9.05 nodes | `δ/H = 0.283` |
| bulk amplitude | `u_bulk` | 0.03 | `Ma = u/c_s = 0.076` |
| forcing amplitude | `a₀` | 8.44e-5 | from the exact linear response |
| analytic centreline | `|Ŵ(0)|`, `∠Ŵ(0)` | 1.4603, −5.559° | amp ratio / lag vs flow rate |
| skip / measure | | 2 / 2 periods | transient-independence checked (see §3.3) |

---

## 2. Method

1. **Analytic side** — closed form above, with an overflow-safe
   `exp`-product evaluation of `cosh(κŷ)/cosh κ`, plus four independent
   references so the analytic side is not self-certified:
   * flux identity `⟨Ŵ⟩ = 1` by quadrature (walls included, `Ŵ(±1) = 0`);
   * quasi-steady limit `Ŵ → (3/2)(1 − ŷ²)` with measured `O(α²)` order;
   * a **finite-difference solution of the same ODE** (complex tridiagonal
     Thomas solve, no cosh algebra) agreeing with the closed form to < 2e-3;
   * classical high-`α` asymptotes: `|Ŵ(0)| → 1`, centreline lag → 0, wall
     shear `dŴ/dŷ|_wall = κ tanh κ/(1 − tanh κ/κ) → κ`, i.e. **+45°** lead.
2. **LBM side** — D2Q9/BGK, periodic in `x` (field is `x`-uniform, `nx = 1`),
   pre-streaming half-way bounce-back with ghost rows (no-slip planes exactly at
   `±H/2`), Guo (2002) forcing with shifted equilibrium velocity
   `u = (Σ cᵢ fᵢ + a/2)/ρ` and zero-sum source
   `Sᵢ = (1 − 1/(2τ)) wᵢ [3(cᵢ − u)·a + 9(cᵢ·u)(cᵢ·a)]`.
3. **Calibration before physics** — steady Poiseuille with a constant force:
   `u_c = a₀H²/(8ν)` measures both the forcing bookkeeping (an `a/2` error is a
   factor here) and the wall position (a wall one half-cell off is a
   percent-level profile error). Only after that passes is the pulsatile case
   trusted.
4. **Measurement** — sample `u_x(y,t)` every step after the skip window, bin by
   phase over whole periods (24 bins), least-squares phasor per row and at the
   centreline; report amplitude ratio, phase difference, per-row complex error
   and the phase-averaged-waveform L2 error (interior rows, `|ŷ| ≤ 0.9`).

---

## 3. Results

### 3.1 Gate case and grid convergence (`α = 5`)

<!-- injected from artifacts/womersley_pulsatile_20261112/table_study.md -->

| ny (=H) | δ [nodes] | period [steps] | amp err [%] | phase err [°] | profile max [%] | profile L2 [%] | gate |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 16 | 4.53 | 1287 | 0.2223 | +0.2543 | 1.712 | 0.418 | pass |
| 24 | 6.79 | 2895 | 0.0974 | +0.1131 | 1.364 | 0.296 | pass |
| 32 | 9.05 | 5147 | 0.0545 | +0.0636 | 0.725 | 0.262 | pass |
| 48 | 13.58 | 11581 | 0.0242 | +0.0283 | 0.603 | 0.232 | pass |

Errors fall roughly as `N⁻²` (amplitude error ratios 2.28, 1.79, 2.25 between
successive resolutions). Figure: `docs/figures/womersley_pulsatile_convergence.png`.

### 3.2 Womersley-number sweep (Stokes layer held at ≈6.4–17 nodes)

<!-- same source -->

| α | ny | period [steps] | amp ratio LBM/ana | amp err [%] | lag LBM [°] | lag exact [°] | phase err [°] | Ma |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 12 | 18096 | 0.99260 | 0.7397 | 0.218 | 0.239 | +0.0208 | 0.077 |
| 2.000 | 12 | 4524 | 0.99269 | 0.7308 | 0.870 | 0.953 | +0.0829 | 0.077 |
| 5.000 | 23 | 2659 | 1.00055 | 0.0547 | 5.488 | 5.559 | +0.0712 | 0.076 |
| 9.999 | 45 | 2545 | 1.00062 | 0.0617 | 10.544 | 10.545 | +0.0012 | 0.063 |

The three regimes are all captured: quasi-steady (`α ≲ 2`, lag ≤ 1°), the
intermediate regime where the centreline **lags** the flow rate (`α = 5`,
5.56°), and the boundary-layer regime (`α = 10`, lag ≈ 10.5° (the maximum lag of the channel solution is near `α ≈ 8`)). Figure:
`docs/figures/womersley_pulsatile_alpha_sweep.png`.

### 3.3 Controls (is the agreement real?)

| run | skip periods | init | centreline lag [°] | amp err [%] | phase err [°] |
|---|---:|---|---:|---:|---:|
| seeded (exact IC), skip 1 | 1 | exact | 5.4461 | 0.0974 | +0.1132 |
| seeded (exact IC), skip 3 | 3 | exact | 5.4462 | 0.0974 | +0.1131 |
| **rest start, skip 4** | 4 | rest | 5.4462 | 0.0974 | +0.1131 |

(analytic lag 5.5593° ⇒ measured lag is 0.113° behind it, i.e. the reported
phase error.)  The rest-start run never sees the analytic solution except as
the comparison target, and it reproduces the seeded phasor to 1e-4° — so the
agreement is the physical periodic state, not the initial condition being
measured back. The seed only shortens the transient; skip 1 vs 3 differ by
0.0001°, well below the gate.

### 3.4 Steady Poiseuille calibration

`ny = 16`, `ν = 0.05`, `a₀ = 1e-4`, 25600 steps (converged: doubling the steps
changes `u_c` by < 1e-4 %): `u_c` LBM 0.06353000 vs `a₀H²/8ν` 0.064 →
**0.734 % low**, profile max 1.012 %, density drift 1e-9 %. This is the known
`O(h²)` half-way bounce-back wall error and it is stable under time stepping;
it fixes the systematic floor of the pulsatile case too (see §5).

### 3.5 Figures

| figure | content |
|---|---|
| `docs/figures/womersley_pulsatile_profile_alpha5.png` | amplitude ratio and phase lag across the channel, LBM vs exact (`α = 5`, `H = 32`) |
| `docs/figures/womersley_pulsatile_centreline_waveform.png` | whole-period centreline waveform (LBM vs exact) + per-row complex error |
| `docs/figures/womersley_pulsatile_inst_profiles.png` | instantaneous profiles at four phases: markers LBM, lines exact |
| `docs/figures/womersley_pulsatile_convergence.png` | grid convergence with an `N⁻²` reference slope; same data vs `α` |
| `docs/figures/womersley_pulsatile_alpha_sweep.png` | centreline amplitude and lag vs `α`, LBM vs exact |

---

## 4. Reproduce

```bash
python -m pytest tests/test_womersley_pulsatile_lbm.py -q      # 15 tests, ~13 s
python scripts/run_womersley_pulsatile_case.py                 # full study, ~2 min, single core
```

Both are single-core and memory-light (largest array `9 × 48 × 1` doubles); the
study needs no cluster.

---

## 5. Notes, traps and limits

1. **Bin-phase mean (caught in this work).** Phase-averaging over an *even*
   number of measured periods puts samples of one bin at `φ` and `φ + 2π`,
   whose arithmetic mean is `φ + π`, not `φ`. Using that mean as the design
   phase negates every profile phasor while the separately fitted centreline
   phasor stays correct — the symptom was amplitude ratios of 0.99 with lags of
   ±180°. Fixed by using the bin centre `(b + ½)·2π/N`; the regression is
   covered by comparing the binned profile phasor with the full-series
   centreline phasor (test `test_pulsatile_gate_alpha5_meets_anchor`).
2. **Rest start must be equilibrium, not zero populations.** Initialising `f`
   to zeros gives `ρ = 0`, a first-step division by zero and a NaN run. The
   rest-start control in §3.3 uses `f = f^eq(ρ=1, u=0)`.
3. **`α` must be recomputed from the lattice.** `ω = ν(α/H)²` with `H` = number
   of *fluid rows* (not `ny − 1`): the ghost-row bounce-back puts the walls at
   `±H/2`. The module reports `alpha_effective` and the gate refuses to pass if
   it is not 5 (`test_gate_rejects_off_target_alpha`).
4. **Cancellation near the wall.** `1 − cosh(κŷ)/cosh κ` cancels for
   `|ŷ| → 1`; analytic profiles are trusted for `|ŷ| ≤ 0.98` and the L2 gate
   uses `|ŷ| ≤ 0.9`.
5. **What the residual error is.** At `H = 32` the centreline amplitude is 0.05 %
   low and the whole interior profile is off by at most 0.73 %. The steady
   calibration shows the same sign and magnitude floor (0.73 % at `H = 16`), so
   the residual is dominated by the half-way bounce-back wall position rather
   than by the unsteady term; `Ma = 0.076` bounds the compressibility error at
   `O(Ma²) ≈ 0.6 %`, consistent with the measured `H⁻²` trend.
6. **Scope.** BGK only, single-phase, `1 ≤ α ≤ 10`, `ν = 0.05`, `Ma ≤ 0.08`.
   Nothing here is a turbulence or non-Newtonian result; the pulsatile *pipe*
   (Bessel) case is verified separately in
   `docs/womersley_exact_verification_20261108.md`.
7. **Numerical trap avoided by design.** The comparison is normalised by the
   *measured* bulk phasor, so a wrong `a₀` cannot masquerade as agreement; the
   forcing strength is instead certified by the steady Poiseuille calibration
   (§3.4), which would show a factor-level error.

---

## 6. Error judgement against the acceptance criteria

| key quantity | reference | LBM (`H = 32`) | error | < 3 %? |
|---|---|---:|---:|---|
| centreline amplitude ratio | `|Ŵ(0)| = 1.46034` (exact) | 1.45955 | **0.054 %** | **yes** |
| centreline phase lag vs flow rate | 5.5588° (exact) | 5.4952° | **0.064° (0.0011 rad)** | **yes** |
| per-row amplitude, `|ŷ| ≤ 0.9` | exact phasor | — | max 0.73 %, rms 0.40 % | **yes** |
| phase-averaged waveform L2 | exact waveform | — | 0.26 % of `u_c/√2` | **yes** |
| steady `u_c` (calibration) | `a₀H²/(8ν)` | — | 0.73 % (`H = 16`) | **yes** |

Anchor tolerance (5 % / 5°) is met with ≈100× margin at `H = 32`, and it is
also met by the coarsest run (`H = 16`: 0.22 % / 0.25°). No key quantity
exceeds 3 %; the largest single number reported anywhere in this document is
the wall-adjacent per-row error, which is excluded by construction (analytic
cancellation, test §5.4) and is not a gate quantity.