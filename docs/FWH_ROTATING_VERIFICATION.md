# FW-H rotating-source module — independent verification and repair

Anchor: `NEW:FWH-ROTATING-UNVERIFIED` (TensorLBM) · harness
`benchmarks/verified/fwh_rotating/run.py` · tests
`tests/test_fwh_rotating_verified.py` (22 passed) · module
`src/tensorlbm/fwh_rotating.py` (sha256 `2d79d4a00ff6d2bc…`,
892 lines)

## 1  Case introduction (why this module needed auditing)

`tensorlbm/fwh_rotating.py` implements the Ffowcs Williams–Hawkings surface
integral for rotating sources (propeller / rotor / fan noise) with a Newton
retarded-time solver, a Doppler factor, thickness (monopole) and loading
(dipole) terms, SPL spectra, BPF harmonic extraction and a directivity /
acoustic-power utility. Its docstring advertises three *analytical validation
anchors* — "Gardner rotating monopole", "Curle dipole", "Blake BPF harmonics".

**The capability had no external evidence at all**: `grep -rn fwh_rotating
--include=*.py --include=*.md .` outside the file itself returned nothing — no
test, no benchmark, no case-pool entry, no doc. A module whose only evidence is
its own docstring is *unverified by construction*: reference and implementation
are the same author, same file, same assumptions. That is exactly the class of
debt this seat exists to clear.

Physics set-up (lattice-free, tiny cost): a compact point monopole of
volume-injection rate Q(t)=Q₀sin(ωt) on a circle of radius R at angular speed ω,
and a compact dipole (point force) F(t)=F₀sin(ωt) — both with the observer in
the acoustic far field, c₀ = 340 m/s (air) and ρ₀ = 1000 kg/m³.

| parameter | value |
|---|---|
| rotation radius R | 0.5 – 2.0 m |
| tip Mach M = ωR/c₀ | 0.05 – 0.60 |
| observer distance | 3 – 1000 m |
| monopole strength Q₀ | 1e-3 m³/s |
| dipole force F₀ | 5 – 10 N |
| tone frequency | 400 – 1000 Hz |
| SPL reference pressure | 1 µPa |

## 2  Method

Nine gates, each comparing the library to something the library does not
produce (`run.py`, `--tag pre|post`):

1. **G1 retarded time** — `solve_retarded_time` (Newton) versus an independent
   bracketed Brent root of the same equation over 216 geometry/time cases.
2. **G2 Doppler** — `compute_doppler_factor` versus a central difference of the
   retarded-time map, using the identity dτ/dt = 1/(1−Mᵣ).
3. **G3 monopole amplitude law** — the *exponent-free* falsification: as R→0 a
   rotating monopole must become a stationary monopole with the finite field
   ρ₀Q̇(τ)/(4πr). This test needs no opinion about the Doppler exponent.
4. **G4 reference audit** — the far-field closed form itself is audited against
   a second numerical route (Green's-function derivative). The two routes differ
   by one power of (1−Mᵣ), which this harness deliberately does **not**
   adjudicate → verdict WITHHELD rather than a fabricated pass.
5. **G5 Curle dipole** — amplitude F₀ω/(4πc₀r), cos θ directivity, and the r/c₀
   phase delay.
6. **G6 SPL calibration** — a 1 Pa bin-centred tone must read 20log₁₀(A/p_ref)
   for every window (coherent-gain correction).
7. **G7 BPF harmonics** — a synthetic Σ(A₁/n)sin(2πn·BPF t) must be recovered.
8. **G8 non-dimensional definitions** — ω, tip speed, tip Mach, BPF.
9. **G9 density normalisation** — for a fixed kinematic source the thickness
   branch must scale linearly in ρ₀ (loading excluded: a prescribed force
   radiates a density-independent dipole field).

Gate tolerances were fixed before the fixes were applied, and the pre-fix run
was recorded (`artifacts/fwh_rotating/result_pre_fix.json`) so the delta is
measurable rather than asserted.

## 3  Results

### 3.1  Pre-fix versus post-fix (all gates)

| gate | pre-fix err [%] | pre verdict | post-fix err [%] | post verdict |
|---|---|---|---|---|
| `G1_retarded_time_vs_brent` | 8.269e-11 | PASS | 8.269e-11 | PASS |
| `G2_doppler_vs_dtaudt` | 1.304e-08 | PASS | 1.304e-08 | PASS |
| `G3_gardner_monopole_vs_exact` | 100 | FAIL | 2.961e-07 | PASS |
| `G4_farfield_reference_audit` | 41.46 | WITHHELD | 41.46 | WITHHELD |
| `G5_curle_dipole_amplitude_and_directivity` | 0.006147 | PASS | 0.006147 | PASS |
| `G6_spl_tone_calibration` | 6.021 | FAIL | 1.624e-10 | PASS |
| `G7_bpf_harmonic_extraction` | 100 | FAIL | 1.786e-07 | PASS |
| `G8_nondimensional_definitions` | 0 | PASS | 0 | PASS |

Pre-fix: **4/9 PASS**, failures
`G3_gardner_monopole_vs_exact`, `G6_spl_tone_calibration`, `G7_bpf_harmonic_extraction`, `G9_pressure_linear_in_rho0`.
Post-fix: **8/9 PASS**, withheld
`G4_farfield_reference_audit`.

### 3.2  Post-fix gate table

| gate | what it pins | value | reference | err [%] | tol [%] | verdict |
|---|---|---|---|---|---|---|
| `G1_retarded_time_vs_brent` | 216 cases, subsonic tip Mach 0.1/0.3/0.6 | 8.26908e-13 | 0 | 8.269e-11 | 0.0001 | **PASS** |
| `G2_doppler_vs_dtaudt` | library 1/(1-M_r) vs finite-difference dtau/dt of the retarded-time map | 1.30356e-10 | 0 | 1.304e-08 | 0.01 | **PASS** |
| `G3_gardner_monopole_vs_exact` | library 'exact solution' claim falsified by the R->0 limit (the stationary limit returns 0 while the exact field is finite) | 1 | 1 | 2.961e-07 | 3 | **PASS** |
| `G4_farfield_reference_audit` | route A vs route B differ by a (1-M_r) power (ratio 0.7763 at 1000 m) | 0.41458 | 0 | 41.46 | 0.5 | **WITHHELD** |
| `G5_curle_dipole_amplitude_and_directivity` | F0*w/(4 pi c0 r) amplitude with cos(theta) directivity | 6.1471e-05 | 0 | 0.006147 | 1 | **PASS** |
| `G6_spl_tone_calibration` | 1 Pa peak tone at a bin centre | 120 | 120 | 1.624e-10 | 0.1 | **PASS** |
| `G7_bpf_harmonic_extraction` | signal = sum_n (A1/n) sin(2 pi n BPF t) | 1.78557e-09 | 0 | 1.786e-07 | 5 | **PASS** |
| `G8_nondimensional_definitions` | omega, tip speed, tip Mach, BPF = n_blades*rpm/60 | 0 | 0 | 0 | 1e-09 | **PASS** |
| `G9_thickness_linear_in_rho0` | thickness branch with loading excluded | 4 | 4 | 0 | 2 | **PASS** |

### 3.3  What was wrong

* **G3 / `gardner_monopole_analytical`** — the "exact solution"
  p = ρ₀Q₀ω²R/(4πc₀r)·cos(ωτ−φ)/(1−Mᵣ)² carries a spurious **ωR/c₀** factor:
  the radiated pressure was low by the tip Mach number and, decisively, **zero
  for a source at rest** (measured ratio to the exact stationary field: 0 at
  R→0; 1.0 to machine precision after the fix). The amplitude is now
  ρ₀Q̇(τ)/(4πr(1−Mᵣ)), i.e. the leading-order term of the Green's-function
  solution of □²p = ρ₀∂/∂t[Q δ³(x−y(t))].
* **G6 / `compute_spl_spectrum`** — windowed spectra were not corrected for the
  window coherent gain, so every Hann-windowed tone read **−6.02 dB** low
  (hamming −5.35 dB). After correction the bias is < 1e-9 dB for all windows.
* **G7 / `_extract_bpf_harmonics`** — the band peak was selected with
  `np.argmax(|f−f_target| < bandwidth)`, i.e. `argmax` over a *boolean* array =
  the **first** bin of the band, not the closest one. With a ±0.5·BPF band this
  systematically returned the wrong bin (100 % amplitude error in the synthetic
  test); now `argmin(|f−f_target|)`, error 1.8e-7 %.
* **G9 / `compute_rotating_source_fwh`** — the thickness source strength already
  contained ρ₀ while the acoustic relation multiplies by ρ₀ again, so the
  radiated pressure scaled like **ρ₀²** (measured ratio 16.0 for a 4× density
  change; now 4.00 ± 0).

### 3.4  Structural findings that are *not* fixed (declared, not hidden)

* **The thickness branch contributes ~nothing.** With `include_loading=False`
  the pressure amplitude is ~1e-10…1e-12 Pa — the eight blade elements' phases
  (sin(ωτ+θ)) cancel. In practice 100 % of the radiated pressure comes from the
  loading term. The thickness term is therefore *numerically inert*; a real
  thickness (monopole) prediction needs the FW-H surface integral with element
  areas and surface normal velocities, not a phase-modulated per-element sum.
* **No absolute calibration.** The element source strengths are blade-element
  heuristics (thrust/torque split over `n_blades × n_segments`), so the SPL
  levels are not comparable with measurements. The module is a machinery
  skeleton, not a validated acoustic predictor.
* **`blake_harmonic_spectrum` is a definition, not physics** — it returns
  A₁/n^α. The "α ≈ 1 for compact sources" claim has no anchor in the code and
  is not verifiable as written; it is documented as a shape function only.
* **G4 WITHHELD** — the Doppler exponent of the far field for a prescribed
  volume-injection rate ((1−Mᵣ)⁻¹ vs (1−Mᵣ)⁻²) is *not* adjudicated here; it
  needs a direct time-domain solution of the 3D wave equation for a moving
  source. Numbers: route A / route B ratio
  0.7763
  at r = 1000 R. Downstream users must not quote either exponent as "verified".

## 4  Figures

![Fig 1 — per-gate error before/after the fix (log scale; G4 is WITHHELD, its bar is the route-A vs route-B disagreement, not an error of the library)](artifacts/fwh_rotating/fig1_gate_errors.png)

Fig 1 — per-gate error before/after the fix (log scale; G4 is WITHHELD, its bar is the route-A vs route-B disagreement, not an error of the library)

![Fig 2 — G3: the exact stationary monopole field versus the library before/after the fix (the old amplitude law carried a spurious ω²R/c₀ factor and vanished for a source at rest)](artifacts/fwh_rotating/fig2_monopole_limit.png)

Fig 2 — G3: the exact stationary monopole field versus the library before/after the fix (the old amplitude law carried a spurious ω²R/c₀ factor and vanished for a source at rest)

![Fig 3 — G6: SPL bias of a 1 Pa, 1 kHz tone. Windowed spectra were 20log₁₀(cg) dB low (hann −6.02 dB); after the coherent-gain correction all windows agree with the analytic value](artifacts/fwh_rotating/fig3_spl_bias.png)

Fig 3 — G6: SPL bias of a 1 Pa, 1 kHz tone. Windowed spectra were 20log₁₀(cg) dB low (hann −6.02 dB); after the coherent-gain correction all windows agree with the analytic value

![Fig 4 — G7: prescribed A₁/n harmonic series versus the extracted amplitudes after fixing the bin selection (the pre-fix argmax-over-boolean picked the lowest bin of each band)](artifacts/fwh_rotating/fig4_bpf_harmonics.png)

Fig 4 — G7: prescribed A₁/n harmonic series versus the extracted amplitudes after fixing the bin selection (the pre-fix argmax-over-boolean picked the lowest bin of each band)

## 5  Error judgement against the acceptance rule

Acceptance rule of this seat: a quantity is accepted when its error against an
experimental/analytical reference is **< 3 %**. Per gate, versus its own
reference, post-fix:

| gate | reference | error | < 3 %? |
|---|---|---|---|
| `G1_retarded_time_vs_brent` | independent reference (see §2) | 8.269e-11 % | **yes** |
| `G2_doppler_vs_dtaudt` | independent reference (see §2) | 1.304e-08 % | **yes** |
| `G3_gardner_monopole_vs_exact` | independent reference (see §2) | 2.961e-07 % | **yes** |
| `G4_farfield_reference_audit` | independent reference (see §2) | 41.46 % | **n/a (withheld)** |
| `G5_curle_dipole_amplitude_and_directivity` | independent reference (see §2) | 0.006147 % | **yes** |
| `G6_spl_tone_calibration` | independent reference (see §2) | 1.624e-10 % | **yes** |
| `G7_bpf_harmonic_extraction` | independent reference (see §2) | 1.786e-07 % | **yes** |
| `G8_nondimensional_definitions` | independent reference (see §2) | 0 % | **yes** |
| `G9_thickness_linear_in_rho0` | independent reference (see §2) | 0 % | **yes** |

G1 retarded time (8.27e-11 %),
G2 Doppler (1.3e-08 %),
G3 monopole limit (2.96e-07 %),
G5 Curle dipole (0.00615 %),
G6 SPL (1.62e-10 %),
G7 BPF harmonics (1.79e-07 %),
G8 definitions (0 %),
G9 density scaling (0 %)
are all far inside 3 %. **G4 is not accepted and not rejected — it is
WITHHELD** because the reference itself is un-adjudicated; that is reported as
such rather than papered over.

## 6  Notes, limits, reproduction

* Cost: the whole harness runs in ~50 s on one core (all analytic; no lattice).
* Reproduce: `python benchmarks/verified/fwh_rotating/run.py artifacts/fwh_rotating --tag post`
  then `python benchmarks/verified/fwh_rotating/report.py`;
  regression: `python -m pytest tests/test_fwh_rotating_verified.py -q` (22 passed).
* The pre-fix record `result_pre_fix.json` was produced by running the same
  harness against the untouched module (module sha256 in the JSON).
* Scope limit: this verifies *machinery* (retarded time, Doppler, calibrations,
  definitions, density scaling) plus the amplitude law of the compact monopole.
  It does **not** validate propeller-noise levels — see §3.4 for the three
  structural reasons.
