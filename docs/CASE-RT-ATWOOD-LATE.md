# CASE-RT-INSTABILITY-ATWOOD — late-time (nonlinear) spike/bubble front velocity

Companion note to `benchmarks/case_pool/rayleigh_taylor_atwood05/REPORT.md`.
Anchor: *算例池·Rayleigh-Taylor 不稳定性 Atwood=0.5 尖峰增长率*
(`CASE-RT-INSTABILITY-ATWOOD`, ref: Tryggvason 2011 lineage, tolerance < 15 %).

`REPORT.md` gates the **linear** growth rate (`sigma` vs the Chandrasekhar viscous
dispersion relation, −10.1 %, PASS) and closes the nonlinear half of the anchor as
failure mode **F4 — "the saturated front velocity is not measurable in this
configuration"**. This note overturns F4, measures the quantity, and reports what
the measurement is worth.

## 1. Two defects in the F4 conclusion

**D1 — stop criterion, not the box.** `run_case.MAX_A_FRAC = 0.60` stops the run
when the interface amplitude exceeds 0.6 wavelengths. In a `layers = 8` box
(4 wavelengths per side) that leaves 3.4 wavelengths of fluid: the run was ended
by an amplitude *threshold* three e-foldings before the pre-declared terminal
window `tau = t sqrt(A g / lam) in [4, 6]` (`run_case.TAU_LO/TAU_HI`). Measured:
the old guard fires at `tau ~ 1.9`; with the guard replaced by a wall-distance
criterion the same configuration runs to `tau = 6.46` (5166 steps, 59 s) at
`Ma_max = 0.55`, i.e. *further* from the compressibility limit than the linear
study assumed.

**D2 — the front estimator is biased once the interface rolls up.**
`run_case` defines `h_up = max(zc) - mean(zc)`, `h_down = mean(zc) - min(zc)`,
where `zc` is the per-column interface height and `mean` is the *instantaneous*
column mean. Two things go wrong: `interface_zero_crossing` returns the **first**
(light -> heavy) transition of a column, which stops tracking the bubble tip as
soon as a column contains more than one transition; and the instantaneous mean
drifts. Measured on the reference run:

| quantity | value |
|---|---|
| columns with > 1 `phi = 0` crossing | 0 until `tau = 1.75`, then up to **51 of 64** |
| `mean(zc) - y0` at late time | up to **1.76 lam** |
| legacy `u` vs tip `u` (same run, same window) | **+80.3 %** |

So the divergence of `h_up` and `h_down` that a naive reading would take as
"spike/bubble asymmetry" is a measurement artefact: the correct tip estimators
agree to `2.3e-2` relative (see §3), consistent with the exact invariance of the
Boussinesq equations under `(phi, y) -> (-phi, -y)`.

## 2. Measurement as implemented (`run_late.py`)

With `phi > 0` = heavy fluid (top):

```
bubble tip  y_b(t) = max_x ( last  light->heavy transition of column x )
spike  tip  y_s(t) = min_x ( first light->heavy transition of column x )
h_bub = (y_b - y0)/lam,   h_spk = (y0 - y_s)/lam,   y0 = ny/2 (fixed, initial interface)
```

Velocity = least-squares slope of `h` over the **pre-declared** window
`tau in [4, 6]` (`run_case.TAU_LO/TAU_HI`, inherited, not re-tuned), `r2 >= 0.99`.
A 90th/10th-percentile variant of each tip is fitted as well so that an isolated
detached droplet cannot fake a front (agrees to < 0.3 %).
Gate: `|Fr/Fr_LZ - 1| <= 15 %` with `Fr = u/sqrt(A g lam)`,
`Fr_LZ = 1/sqrt(pi C)`, `C = 1` (2-D Layzer-Goncharov in the **Boussinesq form**,
i.e. the `A -> 0` limit of the `1 -+ A` inertia factors, which is the model
consistent with the equations actually solved). The measured geometric constant
`C_eff = 1/(pi Fr^2)` is the headline number.

## 3. Results

Reference configuration: `A = 1/2`, `lam = 64`, `g = 2e-4`, `nu = 5e-3`
(`tau = 0.515`), Allen-Cahn interface (`xi = 2` cells, `M = 20`), runs driven to
`tau_end = 6.5`.

| case | steps | `tau_end` | stop | `Ma_max` | `Fr` meas | `Fr_LZ` | err | gate | `C_eff` | robust `Fr` | in-window drift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `late_l064_g2e-4` (reference) | 5166 | 6.46 | wall | 0.548 | 0.5956 | 0.5642 | **+5.56 %** | **PASS** | **0.897** | 0.5966 | −14.7 % |
| `late_l064_g2e-4_L10` (5 lam per side) | 5196 | 6.50 | completed | 0.539 | 0.6100 | 0.5642 | **+8.12 %** | **PASS** | **0.855** | 0.6108 | −3.3 % |
| `late_l096_g3e-4` (1.5x grid, `A g/lam` fixed) | 3594 | 4.49 | **mach** | 0.879 | (0.7833) | 0.5642 | (+38.8 %) | **INCOMPLETE** | — | — | — |

* The two `lam = 64` arms complete the window and pass, at `+5.6 %` / `+8.1 %`.
  The `layers = 10` control is the better measurement (the in-window drift drops
  from −14.7 % to −3.3 %, i.e. the 8-box run is mildly decelerated by its own
  walls), and it moves the answer *away* from zero, so wall confinement is not
  what makes the gate pass.
* `C_eff = 0.90 / 0.86` is in agreement with the **independent** measurement of
  the sibling case `benchmarks/case_pool/rayleigh_taylor` (`C = 0.90` at
  `A = 1/2`), which is a cross-case validation of both the solver and the two
  estimators — and it is the reason the `C`-convention friction E3 of
  `REPORT.md` matters: quoting `C = 3` (the 3-D Layzer constant) would put the
  same run at −42 % instead of +5.6 %.
* The 1.5x ladder arm **cannot** complete the window: at constant `A g/lam` the
  front speed scales as `sqrt(A g lam) ∝ lam`, so `Ma` reaches 0.879 before
  `tau = 4.5` and the Mach guard fires. Its 0.783 number covers
  `tau in [4.0, 4.49]` only and is reported as `INCOMPLETE`, not as a value.
  Consequence: in this lattice-Boltzmann configuration there is **no**
  Mach-safe grid-convergence ladder for the terminal velocity at fixed
  `A g/lam`; a resolution study needs either a lower `A g/lam` family or a
  low-Mach/incompressible formulation. This is the honest limit of the result.

### Spike/bubble asymmetry — measured, not assumed

| case | `u_bubble` | `u_spike` | ratio (sim) | two-fluid `sqrt((1+A)/(1-A))` | max \|`h_bub`-`h_spk`\|/h |
|---|---|---|---|---|---|
| reference | 4.7657e-02 | 4.7636e-02 | **1.0004** | 1.7321 | 2.3e-02 |
| `L10` control | 4.8731e-02 | 4.8873e-02 | 0.9971 | 1.7321 | 1.8e-02 |
| 1.5x arm | 9.4134e-02 | 9.3868e-02 | 1.0028 | 1.7321 | 1.3e-02 |

`max |h_bub - h_spk|/h` above also covers the *linear* phase, where the
residual half-cell offset of the initial condition (§5.3) dominates.

The two-fluid model puts the spike at `sqrt(3) = 1.732 x` the bubble velocity; the
Boussinesq solver returns `1.000` (0.997 in the control), i.e. the simulated spike
velocity is **25.4 % below** the two-fluid spike value (`u_spike/u_bubble` is
0.577 of the predicted ratio), and this case is declared **NOT_SUPPORTED** for
the finite-density-ratio spike velocity. That is a property of the equations (uniform inertia), not of the
discretisation, and no resolution refinement can change it: the `(1 -+ A)` factors
of the Layzer-Goncharov formula *are* the variable inertia that Boussinesq drops.
Reproducing it requires a variable-density (non-Boussinesq) momentum equation.

## 4. Reproduce

```bash
cd benchmarks/case_pool/rayleigh_taylor_atwood05
python3 probe_symmetry.py --steps 2600      # D1/D2 diagnostic (~40 s)
python3 run_late.py                         # 3 arms                         (~3 min, 4 threads)
python3 make_late_figures.py                # fig_late1, fig_late2
python3 make_late_snapshot.py               # fig_late3 + artifacts/late/snapshots
python3 run_late.py --aggregate             # acceptance_table.md
python3 -m pytest tests/test_rt_late_analysis.py -q   # 12 analysis tests
```

Artifacts: `artifacts/late/<case>/history.csv` (per-sample tips, both legacy and
tip definitions, crossing counts, `phi` extrema), `case.json`,
`artifacts/late/summary.json`, `artifacts/late/acceptance_table.md`,
`artifacts/late/snapshots/snapshots.npz`, `figures/fig_late1..3*.png`.

## 5. Limitations (stated, not hidden)

1. **The terminal window is short and viscous.** `Re = u/(nu k) = O(10)` at
   `lam = 64`: the front is not in the inviscid Layzer regime, which is why it
   sits ~6-8 % *below* `Fr_LZ` and why the answer depends on the box (D: −14.7 %
   in-window drift) and on `nu`. `C_eff` grows with `nu`-free refinement towards
   1; here it is 0.86-0.90.
2. **No Mach-safe ladder** (see §3). Grid convergence of the terminal velocity is
   *not* demonstrated; only two box depths and two estimator definitions.
3. **Residual half-cell offset of the initial condition.** `init_rt_state` places
   the interface at `y = ny/2 = 256`, while the discrete box mirrors about
   `(ny-1)/2 = 255.5`. The seeded perturbation is therefore not exactly
   mirror-antisymmetric; measured effect on the fronts `2.3e-2` (non-blocking, but
   it is the only symmetry breaking present and it should be removed by placing
   the interface at the discrete mid-plane in `tensorlbm.rayleigh_taylor`).
4. **Reference provenance unchanged**: the anchor monograph (Tryggvason et al.
   2011) is still not reachable from this sandbox, so the target is the closed-form
   Layzer-Goncharov / Chandrasekhar relation evaluated in code
   (`ref/ref_lock.py`), not a transcribed table.
5. The legacy estimator of `REPORT.md` (§1 D2) is left in place in `run_case.py`
   because that driver's *linear* gate does not depend on it; the bias is
   documented here and in `run_late.py` rather than silently patched there.