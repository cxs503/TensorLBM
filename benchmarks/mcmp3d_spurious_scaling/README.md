# 3-D MCMP spurious-current scaling law — pre-registered protocol

Gap: `GAP-PP-3-3DMCMP-SPURIOUS-SCALING` (paper track `JOUR-MCMP-3D`, Sec. 5).

## Question

For the 3-D pseudo-potential (Shan-Chen single-component, D3Q19) model a static
droplet at rest is the exact null solution `u = 0`.  Everything the solver
reports instead is the **spurious current**.  The quantity that sets the
resolution floor of any 3-D capillary simulation is

    u_max(Δρ, n, forcing) = max |u|   at steady state,

where `Δρ = ρ_l − ρ_v` is the coexistence density difference of the model EOS.
This benchmark measures the closed form

    u_max = A (Δρ)^p                (power-law hypothesis, fitted per grid and pooled)

and asks three machine-checkable questions:

| # | criterion (machine gate) | threshold |
|---|---|---|
| C1 | grid coverage of the `u_max` table | `≥ 3` resolutions × `≥ 3` density ratios |
| C2 | power-law exponent `p` | relative uncertainty `σ_p/|p| < 20 %` |
| C3 | cross-check of the exponent between force formats | `p` determined for BOTH `classic` (`velocity_shift`) and `EDM` (`kupershtokh`) at `≥ 2` density ratios, and the two exponents agree within their combined 2σ |

Three falsification controls are run alongside (a measurement that cannot fail
carries no information) and they *gate* the verdict:

* **Z — no-interface control.**  At the critical coupling (`G_eff = −4.0`,
  `Δρ ≈ 1.7e-3`, still a Maxwell pair) the model has no interface.  `u_max` must
  collapse by ≥ 1 order of magnitude below every main case, i.e. the law must
  have a vanishing intercept at `Δρ → 0` (power law, not an offset).
* **P — plateau control.**  The last three steady checks must agree on `u_max`
  to better than 5 % and every case must satisfy the steady criterion; otherwise
  the "steady" number is a transient in disguise.
* **M — mass control.**  `|ΔM|/M < 1e-9` per run (collide-and-stream on a
  periodic box conserves mass to round-off; a leak would corrupt `Δρ`).

One **exploratory** arm is run and reported but does *not* gate: the `form="psi2"`
force (pure discrete gradient of `ψ²`), hypothesised in the engine docstring to
be a well-balanced / curl-free control.  See `REPORT.md` §6 — measured, it is a
different physical model (thicker interface, collapsed Laplace pressure) and it
does not suppress `u_max`, so the hypothesis is recorded as rejected.

## Model and units

* Engine: `tensorlbm.pseudopotential_edm3d` (D3Q19, `float64`), pseudopotential
  `ψ = ρ0 (1 − e^{−ρ/ρ0})`, `ρ0 = 1`; BGK, `τ = 1.0`.
* Interaction force `form="difference"` (exact-difference spatial form; algebraically
  identical to the product form on the complete D3Q19 stencil, see
  `pseudopotential_edm3d` module docstring), forcing axis
  `kupershtokh` (= EDM, Kupershtokh 2004/2009) vs `velocity_shift` (= classic).
* Coexistence densities `(ρ_v, ρ_l)` from the Maxwell equal-area construction on
  `p(ρ) = c_s²ρ + (G_eff/2) c_s² ψ(ρ)²` (`maxwell_coexistence`).
* `G_eff < 0` attractive (library `G = −G_eff`).

## Pre-registered parameters (fixed before any production number was produced)

| parameter | value | rationale |
|---|---|---|
| grids `n` | `32, 48, 64` (³) | 3 resolutions is the acceptance floor |
| droplet radius | `R = n/8` | self-similar box (`R/L = 0.125`) |
| initial interface width | `W_init = 2.0` lattice units (constant) | the *equilibrium* interface thickness is set by the model (≈ constant in lattice units at fixed `G_eff`); the ladder therefore refines box/radius at fixed interface resolution, the standard spurious-current resolution ladder.  `W_init` only seeds the transient |
| couplings `G_eff` | `−4.4, −4.7, −5.0, −5.3, −5.6` | 5 Maxwell pairs spanning `Δρ ≈ 1.0 … 2.35` (the window in which the Maxwell solver converges); factor ≈ 2.4 lever arm on the abscissa |
| forcings | `kupershtokh` (EDM), `velocity_shift` (classic) | acceptance criterion C3 |
| `τ` | `1.0` | fixed viscosity so `p` is not confounded by `ν` |
| steady state | `|Δdp|/dp < 1e-3` over 3 consecutive checks, `check_every = 200`, cap `6000` steps | `dp` (Laplace pressure) is the slowest relaxing global observable |
| `u_max` reported | mean of `u_max` over the last 3 steady checks | de-noises the plateau |
| controls | `Z`: `G_eff=−4.0`, `kupershtokh`, all 3 grids; `P`,`M`: derived from every main run |
| exploratory | `form="psi2"`, `G_eff=−5.0`, all 3 grids | rejected hypothesis, not a gate |

Everything is deterministic (fixed initial tanh profile, no randomness), so a
single run per cell is the correct replication unit; the uncertainties quoted in
the report are **fit** uncertainties, not run-to-run noise.

## Fit

`ln u_max = ln A + p ln Δρ`, ordinary least squares

* per grid `n` (3 estimates of `p`),
* pooled fixed-effects model `ln u_max = Σ_n α_n 1[n] + p ln Δρ` (one exponent,
  grid-specific intercepts) — this is the reported headline exponent, because it
  uses the resolution axis as an independent replication of the slope;
* uncertainty: standard error of the slope from the OLS covariance
  `σ_p = s · sqrt[(XᵀX)^{-1}_{pp}]` with `s² = RSS/(m−k)`.

Secondary exponents (robustness, not acceptance):

* fit against the **measured** `Δρ = ρ_max − ρ_min` of the same run instead of
  the Maxwell `Δρ`;
* fit of `u_max` vs `R_eq` (resolution axis alone).

## Reproduce

```bash
# full protocol (≈30 min on the 2×loong64 cluster, 36 single-slot jobs)
python benchmarks/mcmp3d_spurious_scaling/run.py --out artifacts/mcmp3d_spurious_scaling --jobs 32
# fit + tables + figures + verdict (seconds)
python benchmarks/mcmp3d_spurious_scaling/fit_report.py --results artifacts/mcmp3d_spurious_scaling/results.json
```

Outputs: `results.json` (every number), `tables.md`, `figures/*.png`,
`verdict.json` (C1/C2/C3 evaluated, not typed by hand), `REPORT.md`.