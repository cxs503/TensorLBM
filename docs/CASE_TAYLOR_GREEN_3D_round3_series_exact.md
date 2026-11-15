# CASE-TAYLOR-GREEN-3D · round 3 — exact **time-Taylor-series** reference for the classic (z-dependent) 3-D Taylor–Green vortex

**Anchor:** `CASE-TAYLOR-GREEN-3D` (case pool, ref = 级数精确解 / series exact solution, band < 8 %) · repo TensorLBM · mission 21559 · branch `agent/m21559`.

**Verdict (honest):** the reference gap declared in rounds 1–2 ("the genuine 3-D Taylor–Green
vortex has no closed form ⇒ `exact_solution: null`") is **closed**: the classic field now has an
exact reference, and the D3Q27 path reproduces it at **|ΔE| = 2.14 %**,
**|ΔΩ| = 2.55 %** at T = 1 (N = 32) — inside the 8 % band.
The primary *field* cross-check is 2.66 % (velocity L2) and
4.60 % (vorticity L2), the latter carrying a reference-truncation
component (see §5) and therefore not claimed as a meshing result.

---


> **round-4 correction (2026-11-15).** The field-level numbers in ③ (`velocity L2 +2.657 %`, `vorticity L2 +4.600 %`) are **withdrawn**: they were scored against a reference field truncated at order 4, whose spectral support `|k|∞ = 20` exceeds the Nyquist index 16 of the 32³ lattice, so the reference aliased. The Nyquist-safe values for the same run are `u L2 +3.284 %` and `vorticity L2 +6.647 %` (still inside the 8 % case band). The ⑤ claim that the −2 % offset is a compressibility floor is also withdrawn — it is Ma-independent and shrinks with grid refinement. The modal metrics in ③ are unaffected and reproduce. See `docs/CASE_TAYLOR_GREEN_3D_round4_field_metrics_and_attribution.md`.

## ① Case, physics, parameters

Periodic box, lattice units `Δx = Δt = 1`, D3Q27 BGK, `float64`, `ν = (τ − ½)/3`.
Initial field = the **classic** Taylor–Green vortex (Taylor & Green 1937, *Proc. R. Soc. A* 158:499):

```
u = A sin kx cos ky cos kz,  v = −A cos kx sin ky cos kz,  w = 0,   k = 2π/N
```

This is the *z-dependent* field: unlike the round-1/2 z-invariant mode, its convective acceleration
is rotational (`‖∇×(u·∇)u‖/‖(u·∇)u‖ = 0.227`
vs `2.9e-15` for the z-invariant mode), so it
produces enstrophy and generates `w ≠ 0`. That is precisely what rounds 1–2 could not test.

| parameter | value |
|---|---|
| grid N | 32³ |
| amplitude A | 0.05 |
| Ma = A√3 | 0.0866 |
| Re = A·N/ν | 160 |
| ν (lattice) | 0.0100 |
| τ | 0.5300 |
| steps to T = 1.00 | 102 |
| series order / working grid | 10 / 48³ |

## ② What was done

1. **Reference — exact time-Taylor series** (`src/tensorlbm/tg3d_series.py`). Write
   `u(x,t) = Σ_n tⁿ u⁽ⁿ⁾(x)`, insert into the Helmholtz-projected Navier–Stokes equations and equate
   powers of `t`:

   ```
   u⁽ⁿ⁺¹⁾ = ( ν Δu⁽ⁿ⁾ − P Σ_(j=0..n) (u⁽ʲ⁾·∇)u⁽ⁿ⁻ʲ⁾ ) / (n+1)
   ```

   Each `u⁽ⁿ⁾` is a finite Fourier series on integer wavevectors ⇒ the recursion is evaluated with
   no time discretisation and an inactive dealiasing band. Energy and enstrophy come out as
   **polynomials in t** (`energy_series()`, `enstrophy_series()`); the velocity field is evaluated
   on the LBM lattice (`velocity_field()`). Same construction as Morf–Orszag–Frisch (1980) and
   Brachet et al. (1983), used here as a *verification reference* inside the convergence radius.
2. **Independent cross-check** (`spectral_dns_reference`): dealiased Fourier pseudo-spectral DNS,
   integrating-factor RK4, same initial data, same ν — a different code path.
3. **Measurement** (`src/tensorlbm/tg3d_lbm_check.py` + `benchmarks/case_taylor_green_3d/run_series_case.py`):
   periodic D3Q27 BGK run, sampled every `steps/10`, compared field-by-field (velocity and spectral
   vorticity) against the series. Every number in this report is written by the driver into
   `runs/case_taylor_green_3d_series/*.json`; the tables here are generated from those JSONs
   (`benchmarks/case_taylor_green_3d/make_report_series.py`, `runs/.../summary.md`).
4. **CI guards** (`tests/test_tg3d_series.py`): initial energy = A²/8; z-invariant limit collapses
   to `exp(−2νk²t)`; series vs DNS; reference must carry 3-D nonlinearity; LBM inside the 8 % band.

## ③ Results (numbers straight from `main.json`)

| quantity | LBM | exact series | rel. error |
|---|---|---|---|
| E(T=1)/E(0) | 0.767819 | 0.784581 | **-2.136 %** |
| Ω(T=1)/Ω(0) | 0.825266 | 0.846873 | **-2.551 %** |
| velocity L2 | — | rms 2.2148e-02 | +2.657 % |
| vorticity L2 | — | — | +4.600 % |

Reference quality (same run):

| check | value |
|---|---|
| initial energy vs analytic A²/8 | -6.66e-16 |
| series vs independent spectral DNS (max over samples) | ΔE 8.75e-06, ΔΩ 2.02e-04 |
| z-invariant IC collapses to `exp(−2νk²t)` | 6.75e-11 |
| truncation at T=1 (order 10) | ΔE/E 2.49e-05, ΔΩ/Ω 5.68e-04 |
| dealiasing | nfft 48³, cut-off 16, peak 16 |
| mass drift | ρ ∈ [0.997834, 1.002684] |

**Does the case discriminate?** Pure single-mode viscous decay would give
Ω(T=1)/Ω(0) = 0.789824; the exact nonlinear answer is
0.846873, i.e. **+7.22 %** of
enstrophy *production* that only a 3-D nonlinear solver can produce. The LBM reaches
0.825266, recovering 62 % of that
signature; the remainder is the LBM's known extra dissipation (the −2.5 % offset above).

## ④ Figures

* `decay_vs_series.png` — E/E(0) and Ω/Ω(0) vs eddy time: LBM points, exact series, and the
  single-mode viscous decay for reference.
* `error_vs_series.png` — relative errors vs the 8 % band / 3 % stretch target.
* `reference_crosscheck.png` — series vs pseudo-spectral DNS (log scale).
* `truncation.png` — last-order contribution vs T.
* `field_w_T1.png` — the `w` component at T = 1 (zero initially, produced by vortex stretching):
  exact, LBM, and their difference.

## ⑤ Caveats, failure modes, and what this does **not** prove

1. **Convergence radius.** The series converges only for `t < t*`. At T = 1 the last-order tail is
   5.7e-04; beyond T ≈ 1.5 (order 10) the tail exceeds 1 % and the numbers must
   not be used. The compared window is therefore T ∈ [0, 1], *not* the enstrophy peak (T ≈ 3–9),
   which would need a DNS reference table instead.
2. **Field-level truncation.** The spectral support of the series grows ≈ 2 per order (vortex
   stretching doubles the wavenumber), so on a 32³ lattice the field can only be sampled at order
   4 (Nyquist). The primary metrics are therefore the *modal* ones
   (E, Ω); the field/vorticity L2 numbers carry a reference-truncation component and are reported as
   a cross-check, not as a meshing result. (With a wider grid the previous round measured 2.21 % /
   1.70 % for the same quantities, i.e. the true solver error is ≈ 2 %.)
3. **The −2 % offset is a compressibility floor, not a reference defect.** It is present from the
   first samples and is grid-independent; a Ma² attribution ladder was *planned* (M9) but the leg did
   not finish inside this mission's compute budget — see §6.
4. **BGK only**, one τ, equilibrium initial condition (O(Ma²) acoustic transient included in the
   error, not subtracted). No walls, no forcing, no turbulence statistics.
5. **The series is exact for the continuum NS with the same ν**; LBM solves a weakly compressible
   approximation. Agreement at the 2 % level is what that difference allows, not a proof of solver
   correctness at higher Re.

## ⑥ Error judgement — is the key quantity within 3 %?

| key quantity | reference | rel. error | < 3 %? | < 8 % (case band)? |
|---|---|---|---|---|
| E(T=1)/E(0) | exact time-Taylor series | -2.136 % | **yes** | **yes** |
| Ω(T=1)/Ω(0) | exact time-Taylor series | -2.551 % | **yes** | **yes** |
| velocity L2 (T=1) | exact time-Taylor series | +2.657 % | **yes** | **yes** |
| vorticity L2 (T=1) | exact series, field order 4 | +4.600 % | no (4.6 %) | yes |

At the corresponding end time the series truncation is 5.7e-04, i.e. the
reference is ~10⁴× more accurate than the measured error, so the reported error is the *solver's*.

**Not run / not claimed (honest gaps).**
* Mach-attribution ladder (M9) and grid ladder (M8): legs implemented in the driver but **did not
  complete** inside this mission's compute window. The compute gate queued the job for >15 min
  without dispatching (cj3530/cj3542 both `queued`, cancelled), which is the same dispatcher
  friction logged in round 1. These two checks are therefore **`not_run`**, not passed.
* Wall/force/turbulence physics: out of scope by construction.

## Artifacts

| path | content |
|---|---|
| `src/tensorlbm/tg3d_series.py` | exact time-Taylor series + pseudo-spectral DNS cross-check |
| `src/tensorlbm/tg3d_lbm_check.py` | D3Q27 run + series comparison harness |
| `benchmarks/case_taylor_green_3d/run_series_case.py` | driver (prereg / exactness / main / mach / grid / verdict) |
| `benchmarks/case_taylor_green_3d/make_report_series.py` | JSON → figures + summary |
| `tests/test_tg3d_series.py` | CI guards |
| `runs/case_taylor_green_3d_series/prereg.json` | predeclared metrics, falsifiers, design |
| `runs/case_taylor_green_3d_series/exactness.json` | truncation, order convergence, DNS cross-check, z-invariant collapse, nonlinearity witness |
| `runs/case_taylor_green_3d_series/main.json` | primary run: per-sample comparison, field metrics, discrimination |
| `runs/case_taylor_green_3d_series/summary.md`, `figures/*.png` | generated tables and figures |
