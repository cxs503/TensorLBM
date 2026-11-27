# Prereg — XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL (mission 18128, round 1)

Written **before** any run of the driver (`xval_driver.py`); the file is committed
in the same commit as the driver but precedes it in execution order. Thresholds
below are predeclared and are **not** tuned after seeing results.

Anchor: `XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL`, benchmark object
`PorousMediaSimulation/openLBMPM` (mirror at `/tmp/openLBMPM`, master
`3d84189`, last push 2018-10-15), feature "Peng–Robinson pseudopotential".

## 1. Decision (what this study decides)

Two coupled decisions:

| id | decision | adopt if |
|---|---|---|
| **D1** | Is TensorLBM's Peng–Robinson pseudopotential **conformant** with the OSS implementation of the same "P-R EOS effective mass" (Yuan & Schäfer), i.e. the same ψ up to a documented constant? | E1 ≤ 1e-6 relative deviation **after** the documented mapping |
| **D2** | At the adopted operating point (density-ratio ≈ 10 tier), is the PR pseudopotential's LBM coexistence **within the owner 15 % band** of the analytic PR-Maxwell reference? | E2 < 0.15 relative error on ρ_l, ρ_v and the density ratio at the finest grid |

## 2. Competing alternatives (what the outcome distinguishes)

- **A0 — conformant + accurate**: adopt PR-ψ as cross-validated; cite OSS as the
  independent second implementation.
- **A1 — conformant + inaccurate**: the two implementations agree algebraically
  yet both are off the Maxwell reference at high density ratio ⇒ the defect is in
  the SCMP scheme/forcing/grid, **not** in the EOS-ψ formula; recommendation =
  grid refinement / forcing change, not a ψ rewrite.
- **A2 — non-conformant**: a parameter/sign/normalisation mismatch (e.g. the
  `c0` factor, the α(T) term, the clamp at ρ_cross) ⇒ a porting defect in
  TensorLBM to be fixed.
- **A3 — reference impossible** (honest-FAIL branch): the OSS P-R path is not
  runnable/not wired, so only formulation-level conformance can be decided;
  D2 must then fall back to the analytic PR-Maxwell reference alone.

## 3. Hypotheses

- **H1 (favours A0)**: the OSS ψ expression, evaluated with
  `constC0 = cs² = 1/3`, `constG = −1`, `α = 1`, and TensorLBM's
  `psi_eos_peng_robinson` with `G = +1` are **the same function** (relative
  deviation at float64 round-off), because both reconstruct
  `p_mech = ρcs² + (G cs²/2)ψ²`.
- **H0 for H1 (favours A2)**: they differ by a constant factor (the `c0`
  convention) ⇒ mapping needed and, if unrecorded, a real porting hazard.
- **H2 (favours A1 at high ratio)**: PR-SCMP LBM coexistence error grows with the
  Maxwell density ratio; the ratio-10 tier passes the 15 % band, the ratio-100
  tier does not (prior campaign `benchmarks/eos_maxwell/RESULTS.md` suggests
  this; this study re-measures it programmatically rather than citing numbers).

## 4. Outcome metrics and units

| id | metric | unit | reference |
|---|---|---|---|
| E1 | max over ρ-grid of \|ψ_LBM − ψ_OSS(mapped)\| / ψ_OSS(mapped) | – (rel.) | OSS source-evaluated ψ |
| E1b | ratio ψ_LBM/ψ_OSS as a function of ρ (constancy ⇒ pure normalisation) | – | – |
| E2 | relative error of ρ_liquid, ρ_vapour, ratio = ρ_l/ρ_v | – (rel.) | PR equal-area Maxwell construction |
| E2b | relative error of p_sat recovered from the LBM mechanical pressure at both plateaus | – (rel.) | Maxwell p_sat |
| E3 | max spurious Mach number, max\|u\|/(cs) of the held droplet | – | reported, not gated |
| E4 | isotropy moments Σw, Σw·c_x², A4, A6, A8 of the OSS interaction stencil (iso4 shell) | – | analytic isotropic values 1,1,1 |

Unit of analysis = one LBM run (planar or droplet) / one ψ-sweep evaluation.

## 5. Baseline and controls

- **E1 baseline**: OSS source-evaluated ψ (the second implementation).
- **E2 baseline**: analytic PR equal-area construction (`probe_lib.maxwell_at_tr`,
  area residual < 1e-10 by construction — the residual is reported).
- **Control for E2**: the same driver/config on the ψ_star (Carnahan–Starling)
  arm at the matched ratio tier is **out of scope** (cost); instead the
  grid-refinement arm L = 48/64/96 acts as the numerical control that separates
  "scheme defect" from "resolution defect".

## 6. Variables

- **Controlled**: EOS parameters a = 2/49, b = 2/21, R = 1, α = 1, G = +1,
  τ = 1.0, forcing = `edm` (default `velocity_shift` shim), float32 fields,
  torch thread count = 4, initial profile width = 4 (tanh), sample every 100 steps.
- **Changed**: grid size L ∈ {48, 64, 96}; reduced temperature
  T_r ∈ {0.8266 (ratio-10 tier), 0.75, 0.95}; geometry ∈ {planar slab, droplet}.
- **Confounders identified**: float32 reduction order (thread count),
  finite-hold drift (mass loss ⇒ measured "coexistence" is not exactly
  equilibrium), the ψ clamp at ρ_cross (kills the dense branch above
  ρ_cross ≈ 9.7), initial-profile width, and the number of interface cells
  (the dominant one at high density ratio).

## 7. Predeclared thresholds and stopping rule

- E1 pass: ≤ 1e-6 (float64 round-off scale; a true algebraic identity).
  Owner 15 % band additionally applies as the "unacceptable" line for E1.
- E2 pass: < 0.15 on each of ρ_l, ρ_v, ratio (owner standard).
- Stopping rule: if E2 fails at L = 96, refine **once** to L = 128 and stop.
  If a run collapses (NaN / |u| blow-up) the run is recorded, not retried with
  different parameters, and counted in the sensitivity table as "collapsed".

## 8. Repetitions

E1 is deterministic algebra (no repetitions needed). E2: one run per
(L, T_r, geometry) cell; the grid axis provides the replication in the sense that
matters here (systematic resolution trend). Marginal construct ⇒ a single
repeat of the primary cell at half the thread count is predeclared as the
determinism check (reported, not gated).

## 9. Environment and provenance

CPU-only (no GPU, no `numba` ⇒ the OSS code itself cannot be executed; recorded
as friction). torch 2.7.1, python 3.11.6. OSS source sha256 recorded per file in
`artifacts/oss_facts.json`; TensorLBM commit = HEAD of branch `agent/m18128`;
driver + probe sha256 recorded in `artifacts/results.json`.

## 10. Analysis plan (fixed in advance)

Aggregation = relative errors against the pre-computed reference per cell;
uncertainty = spread over the grid axis (no statistical CI claimed); outliers =
collapsed runs reported in a separate table, never silently dropped; comparison
method = plain relative error, no post-hoc tolerances. All numbers are
machine-written into `artifacts/*.json` and inlined into the report by the plot
script — nothing is transcribed by hand.

## 11. Interpretation limits (declared up front)

- Formulation-level conformance (E1) does **not** validate the OSS code's
  runtime behaviour — the OSS P-R kernel has no call site, so semantics beyond
  the formula cannot be cross-checked without a GPU.
- E2 measures the *whole* SCMP scheme (forcing, discretisation, clamp), so an
  E2 failure is not attributable to ψ alone.
- Both arms share the same literature source (Yuan & Schäfer) ⇒ E2 is not an
  independent-scheme test, only a thermodynamic consistency test.