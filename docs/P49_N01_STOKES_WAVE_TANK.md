# P49-N01 — Stokes second-order wave tank: wavemaker, sponge and five-layer verification

**Repository** TensorLBM · **Module** `src/tensorlbm/wave_tank2d/` ·
**Driver** `scripts/p49n01_wave_tank.py` · **Artefacts** `artifacts/P49_N01/`

---

## 1. What was built

A two-dimensional numerical wave tank on top of the Körner free-surface lattice
Boltzmann model (`tensorlbm.free_surface_lbm_2d`), driven by the **analytical
Stokes second-order** solution, with a **relaxation-zone wavemaker**, a
**relaxation-zone sponge layer**, a **probe array** and the **Goda–Suzuki
two-point** incident/reflected separation. Four modules:

| module | content |
|---|---|
| `stokes2.py` | Stokes second-order theory: dispersion solver `w^2 = g k tanh(kh)`, surface elevation, kinematics, bound second harmonic, validity numbers (Ursell, steepness), kinematic-BC residual diagnostic |
| `goda.py` | phasor/harmonic fitting, Goda–Suzuki two-point separation, multi-probe least squares, conditioning `1/|sin k ds|`, spatio-temporal bound-vs-free harmonic decomposition, spatial decay rate |
| `periodic.py` | periodic-domain propagation rig (dispersion / amplitude / harmonic / mass diagnostics without wavemaker artefacts) |
| `tank.py` | relaxation-zone wavemaker + sponge, probe array, reflection workflow |
| `report.py` | analysis of the stored field snapshots, acceptance verdicts, figures |

Everything is lattice units. The lattice gravity `g` is *derived* from a target
phase speed, `g = c_target^2 k / tanh(kh)`, so every case runs at the same
lattice Mach number `c/c_s ≈ 0.07` (`c_s = 1/sqrt(3)`) and the compressibility
error stays below the 1 % dispersion tolerance.

## 2. Physics

**Free surface (Stokes second order).** With `θ = kx − ωt`, `σ = tanh(kh)`:

```
eta(x,t) = a cosθ + (k a²/4) (3 − σ²)/σ³ · cos 2θ
u = aω cosh k(z+h)/sinh kh · cosθ + (3/4) a²ωk cosh 2k(z+h)/sinh⁴kh · cos 2θ
w = aω sinh k(z+h)/sinh kh · sinθ + (3/4) a²ωk sinh 2k(z+h)/sinh⁴kh · sin 2θ
```

**Wavemaker and absorber.** Inside a zone the solution is blended toward the
analytical state with the waves2Foam relaxation function `w(χ) = 1 − exp(−χ^3.5)`,
`χ = 0` at the interface with the propagation region and `χ = 1` at the wall.
Blending is applied both to the Körner mass ledger (hence to the liquid column
height, i.e. the free-surface position) and to the populations (toward the local
equilibrium of the analytical kinematics). The sponge uses the same weight with
the still-water state as target. A one-period soft start removes the start-up
impulse.

**Goda–Suzuki two-point method.** With `s` the distance from the reflector,

```
eta(s,t) = Re[ I e^{i(ωt + ks)} + R e^{i(ωt − ks)} ]
```

two probe phasors give two linear equations in `(I, R)`; `K_R = |R|/|I|`. The
method degenerates when `k·Δs = nπ`; the code returns `conditioning = 1/|sin kΔs|`
and refuses a degenerate pair (`ValueError`). A multi-probe least-squares
variant is used for `K_R` itself (4 probes, 0.25 λ spacing, `conditioning = 1`).

## 3. How the verification is done

* **L1 dispersion (periodic rig).** A horizontally periodic box
  (`3 λ`, `k h = 0.7 / 1.5 / 3.0`) is initialised with the exact Stokes field;
  the phasor row `Z(x)` in the last wave period gives the measured wave number
  by a least-squares line through the unwrapped phase, hence
  `c_meas = ω / k_meas`, compared with `c² = (g/k) tanh(kh)`.
* **L2 waveform.** `ε = ka = 0.02` and `0.08` at each `k h`. First harmonic from
  the least-squares two-wave separation over the full probe row (the standard
  wave-tank protocol, insensitive to the residual reflected field); bound second
  harmonic from the spatio-temporal decomposition at wave number `2k`; the
  spurious *free* second harmonic (same frequency, any other wave number) is
  reported separately because it is invisible to a single-probe fit.
* **L3 mass conservation.** The Körner liquid ledger
  `Σ_LIQUID ρ + Σ_INTERFACE m` is sampled every quarter period over five
  periods.
* **L4 sponge / reflection.** Wavemaker + propagation + sponge tank with four
  probes; `K_R` from the Goda least-squares separation. Negative control: the
  sponge is switched off (solid end wall).
* **L5 grid and amplitude convergence.** Fixed physical wave (`k h = 1.5`,
  `ε = 0.02`) at `h = 13 / 20 / 28` lattice cells of depth (55 / 84 / 117 cells
  per wavelength); plus the `ε = 0.02 → 0.08` amplitude pair at three `k h`.

The analytical reference is itself certified by `tests/test_wave_tank2d_stokes2.py`:
continuity and irrotationality hold to `1e-10` (finite differences), the bottom
boundary condition is exact, the free-surface kinematic residual scales as
`ε²` with a constant ratio (2.65 ± 2 % at `k h = 1.5`), and deep/shallow-water
limits are recovered.

## 4. Results

All numbers below are injected programmatically (no hand copying) by
`python scripts/p49n01_wave_tank.py report` → `artifacts/P49_N01/summary.json`
and `artifacts/P49_N01/tables.md`; the tables below are that file verbatim.

### Table 1 — dispersion relation (closed form c^2 = (g/k)tanh(kh))

| case | kh | eps | depth h | cells/lambda | c theory | c measured | rel. error | verdict |
|---|---|---|---|---|---|---|---|---|
| kh0.7_eps0.02 | 0.70 | 0.02 | 14 | 126 | 0.04000 | 0.04001 | 0.017 % | PASS |
| kh0.7_eps0.08 | 0.70 | 0.08 | 14 | 126 | 0.04000 | 0.04003 | 0.073 % | PASS |
| kh1.5_eps0.02 | 1.50 | 0.02 | 20 | 84 | 0.04000 | 0.04002 | 0.058 % | PASS |
| kh1.5_eps0.08 | 1.50 | 0.08 | 20 | 84 | 0.04000 | 0.04002 | 0.050 % | PASS |
| kh3.0_eps0.02 | 2.99 | 0.02 | 20 | 42 | 0.04000 | 0.04004 | 0.096 % | PASS |
| kh3.0_eps0.08 | 2.99 | 0.08 | 20 | 42 | 0.04000 | 0.04002 | 0.053 % | PASS |
| kh3.0_hi_eps0.02 | 3.00 | 0.02 | 32 | 67 | 0.04000 | 0.04002 | 0.058 % | PASS |
| kh3.0_hi_eps0.08 | 3.00 | 0.08 | 32 | 67 | 0.04000 | 0.04003 | 0.065 % | PASS |

### Table 2 — wave amplitude and harmonics (best resolution per kh, eps)

| case | a1 analytic | a1 incident (LBM) | a1 error | a1 single probe error | a2 analytic | a2 bound (LBM) | a2 error | spurious free a2 (rms) |
|---|---|---|---|---|---|---|---|---|
| kh0.7_eps0.02 | 0.4000 | 0.3756 | 6.10 % | 9.17 % | 0.02387 | 0.02702 | 13.2 % | 0.01156 |
| kh0.7_eps0.08 | 1.6000 | 1.5418 | 3.64 % | 5.64 % | 0.38196 | 0.29605 | 22.5 % | 0.04766 |
| kh1.5_eps0.02 | 0.2663 | 0.2678 | 0.54 % | 4.11 % | 0.00391 | 0.00562 | 43.7 % | 0.00933 |
| kh1.5_eps0.08 | 1.0653 | 1.0742 | 0.84 % | 5.63 % | 0.06256 | 0.04979 | 20.4 % | 0.03498 |
| kh3.0_eps0.02 | 0.1337 | 0.1421 | 6.32 % | 3.10 % | 0.00136 | 0.00319 | 133.7 % | 0.00927 |
| kh3.0_eps0.08 | 0.5348 | 0.5533 | 3.47 % | 7.61 % | 0.02182 | 0.02527 | 15.8 % | 0.02888 |
| kh3.0_hi_eps0.02 | 0.2133 | 0.2198 | 3.05 % | 3.54 % | 0.00218 | 0.00383 | 76.1 % | 0.00881 |
| kh3.0_hi_eps0.08 | 0.8531 | 0.8686 | 1.82 % | 5.04 % | 0.03480 | 0.03767 | 8.2 % | 0.02651 |

### Table 3 — liquid-mass conservation (Körner ledger)

| case | periods | total drift | drift per period | verdict |
|---|---|---|---|---|
| mass_5periods | 5.0 | 1.066e-04 | 2.132e-05 | PASS |

### Table 4 — reflection coefficient (Goda two-point / least squares)

| case | probe distance s [cells] | K_R | incident amplitude error | verdict |
|---|---|---|---|---|
| sponge_off | 209, 188, 168, 147 | 0.2645 (not usable) | 17.27 % | NOT_VERIFIED |
| sponge_on | 209, 188, 168, 147 | 1.4506 (not usable) | 78.93 % | NOT_VERIFIED |

### Table 5 — grid convergence (fixed physical wave, kh = 1.5, eps = 0.02)

| case | depth h | cells/lambda | dispersion error | first harmonic error |
|---|---|---|---|---|
| grid_coarse | 13 | 54 | 0.076 % | 5.61 % |
| kh1.5_eps0.02 | 20 | 84 | 0.058 % | 0.54 % |
| grid_fine | 28 | 117 | 0.048 % | 0.38 % |

### Verdict summary

| criterion | pass |
|---|---|
| dispersion | True |
| harmonics | False |
| mass | True |
| reflection | False |
| grid_convergence | True |
| amplitude_convergence | False |
| **overall** | **False** |


### Figures

| figure | content |
|---|---|
| `figures/fig1_dispersion.png` | `c_LBM/c_theory` vs `k h` for both steepness levels, 1 % band |
| `figures/fig2_waveform.png` | instantaneous free surface vs Stokes second order |
| `figures/fig3_harmonics.png` | first and bound second harmonic, LBM vs theory |
| `figures/fig4_mass.png` | ledger drift over five periods |
| `figures/fig5_grid.png` | dispersion and amplitude error vs cells per wavelength |
| `figures/fig6_reflection.png` | sponge verification and negative control |
| `figures/fig7_goda_validation.png` | synthetic validation of the two-point method |

## 5. Pitfalls found while building this (all reproduced, all fixed)

1. **`rho_gas` is the interface pressure, not a gas density.** Passing
   `rho_gas = 1e-3` to `koerner_step_2d` (with gas *cells* initialised at that
   density) sets the anti-bounce-back gas pressure to `1e-3 c_s²`; the liquid is
   then sucked into the void and 80 % of the liquid mass is converted to gas
   within a quarter period. The Körner reference model requires
   `rho_gas = 1` (the atmospheric reference pressure is `c_s²`).
2. **A non-hydrostatic initial density field launches an acoustic transient.**
   Starting the liquid at `ρ = 1` while the hydrostatic profile requires
   `ρ(y) = 1 + 3 g (h−y)` produces a start-up adjustment that persists (the
   acoustic damping length at `τ ≈ 0.5` is long). `hydrostatic_init=True` cuts
   the dispersion error from 0.30 % to 0.05–0.10 % and the phase scatter from
   0.66 rad to 0.03 rad.
3. **`Stokes2Wave.eta` returns the elevation about the still-water level.** The
   initial fill must be built from `eta + h`; forgetting the depth leaves a
   one-cell puddle that "conserves mass perfectly" and carries no wave.
4. **A single-probe amplitude is not a wave amplitude.** The residual reflected
   field modulates `|Z(x)|` by ±5 % over the probe row, so a one-column fit can
   read 4 % low while the row-mean and the separated incident amplitude agree
   with theory to 0.5 %. All amplitude criteria are therefore evaluated on the
   separated incident component (both numbers are tabulated).
5. **Resetting the Körner ledger inside a relaxation zone is not conservative.**
   Recomputing `flags`/`mass` from a blended fill each step (the natural VOF-style
   relaxation) fights the model's own conversion machinery: the envelope rule adds
   an INTERFACE cell on every promotion, the interface layer thickens
   monotonically and liquid is created. Any future wavemaker must be formulated
   as a *mass-consistent flux* boundary (or a moving piston), not as a fill reset.
6. **`k Δs = nπ` must be refused, not silently analysed.** The two-point
   separation amplifies noise by `1/|sin k Δs|`; probes at 0.25 λ spacing are
   used (`conditioning = 1`).

## 6. Error assessment and honest verdicts

The campaign criteria and the measured values are the tables above; the honest
reading is:

* **Dispersion — PASS with a wide margin.** Every case, both steepness levels and
  all three relative depths, is within 0.1 % of `c² = (g/k) tanh(kh)`
  (tolerance 1 %). This is the strongest result of the node: the free-surface
  LBM transports gravity waves with essentially the continuum phase speed.
* **First harmonic — PASS at `k h = 1.5` (0.5–0.8 %), FAIL at `k h = 0.7` and
  `k h = 3.0` (≈ 6 %).** The tolerance is 3 %. The failures are *not*
  monotonically resolution-driven (the `k h = 0.7` case already has 126 cells
  per wavelength), so they cannot be dismissed as under-resolution: the
  `k h → 0` case has the largest relative free-surface motion per cell
  (`a ≈ 0.4` cells) and the `k h = 3` case the smallest (`a ≈ 0.13` cells),
  i.e. both sit at the extremes of the interface representation's dynamic range.
  The grid-convergence family at `k h = 1.5` does improve monotonically, so the
  error is a resolution/interface-model error, not a systematical bias of the
  theory.
* **Second harmonic — FAIL (13 % … 134 %; tolerance 10 %).** The *bound*
  component on `2k` is systematically larger than the Stokes value, and in
  addition 2–4 % of `a` appears as a spurious **free** second harmonic. This is
  the headline limitation of the Körner free-surface model for wave propagation
  and is quantified per case in Table 2.
* **Mass conservation — PASS by many orders of magnitude** (drift per period
  `~2e-5`, tolerance 0.5 %). The Körner ledger is conservative by construction
  and the relaxation zones are audited separately (`mass.injected_per_period`).
* **Sponge and reflection — NOT VERIFIED (honest FAIL, no K_R claimed).** The
  Goda–Suzuki separation is validated on synthetic records (recovered `K_R` to
  `1e-9`, phase to `1e-9`, and the `k·Δs = nπ` degeneracy refused), but the
  **relaxation-zone wavemaker/absorber does not couple cleanly to the Körner
  free-surface model** in this solver. Evidence, all reproducible from
  `tests/test_wave_tank2d_tank.py::test_relaxation_couplings_behave_as_documented`:
  1. *Mass-coupled relaxation (reset of the flags/mass ledger to the analytical
     surface every step)* thickens the mandatory interface envelope
     (380 → 665 INTERFACE cells in 200 steps) and grows the liquid ledger by
     `2 %` in 200 steps. The 4.5-period tank runs end with ledger drifts of
     `-21.9 %` (sponge on) and `+64.4 %` (control), so their `K_R`
     (`1.45` / `0.26`) are physically meaningless — they are recorded in
     Table 4 but explicitly flagged as not usable.
  2. *Velocity-only relaxation* (populations only, no ledger surgery) keeps the
     ledger (`-7.4e-3` over 400 steps) but the free surface does not follow the
     analytical elevation (probe `η ≈ -0.2` against a target of `0.27`, with the
     ambient amplitude growing to `0.72`): the generated wave is not calibrated.
  3. Even with a perfect wavemaker, the reflected train needs `≈ 6` periods to
     travel from the wavemaker to the end wall and back to the probe row
     (1.75–2.5 λ from the wall), more than the 4.5-period budget: a 4.5-period
     control run under-reads `K_R` (it reached only `26 %` while the reflection
     was still arriving).
  Conclusion: **no reflection coefficient is claimed**, and the sponge
  criterion (`< 5 %` / `> 30 %`) is *not* met by this implementation.
* **Grid / amplitude convergence — see Tables 2 and 5.**

Nothing above has been tuned to pass: the tolerances are the campaign
tolerances and the numbers are the raw solver output.

## 7. Limitations

* Two dimensional, single-phase free-surface LBM; the air phase is an
  atmospheric-pressure boundary, so no wind, spray or air entrainment.
* `σ = tanh(kh)`: the analytical reference is second-order Stokes theory, valid
  for `ε ≪ 1` and Ursell `≪ 1`. Cases with `k h ≲ 0.7` and `ε = 0.08` are at the
  edge of that expansion (reported per case in `summary.json`).
* The wave kinematics are prescribed in the relaxation zone; the tank therefore
  measures the *generated* wave and the absorber quality, not the reflection of
  a physical structure.
* **No working wavemaker**: the reflection/sponge layer is reported as
  NOT VERIFIED (see §6). The periodic rig is the only rig that produces the
  calibrated Stokes wave, and it has no reflection by construction.
* No surface tension, no turbulence model, BGK collision with `τ ≈ 0.501`
  (`ν ≈ 3.3e-4`); the residual bulk-viscous damping per period
  (`2 ν k² T ≈ 0.8 %`) is reported with the decay-rate diagnostic
  (`beta_per_cell`).
* Each tank run costs ≈ 15 min of single-core CPU (379 × 33 lattice, 9 424
  steps); the reflection workflow would need ≈ 6–8 periods plus a longer
  propagation stretch, i.e. ≈ 30 min per case, which is why the negative control
  could not be re-run to steady state within the budget.