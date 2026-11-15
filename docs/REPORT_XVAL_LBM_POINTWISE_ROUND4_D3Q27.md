# XVAL-LBM-POINTWISE-REFERENCE-REGRESSION — Round 4: D3Q27 whole step

**Anchor:** `XVAL-LBM-POINTWISE-REFERENCE-REGRESSION` (ledger status at start: `open`)
**Practice transplanted:** lssfau/lbmpy *pointwise reference regression* — build the
collision operator independently in the OSS reference implementation, evaluate it
*pointwise on the same PDF field* the in-repo kernel sees, and gate the agreement at a
predeclared tolerance.
**Status of this round: harness + gate delivered, measurement run PENDING** (job `cj3264`
was still queued when this report was written — see §6; no numbers are invented here).

## ① Case / scope

Rounds 1–3 covered D2Q9 (single-step live-lbmpy comparison, then 30-step frozen refdata)
and D3Q19 (30-step frozen refdata, 2 grids × 3 τ × 2 seeds). **D3Q27 had no independent
pointwise reference at all**, although it is the lattice on which the flagship 3-D path
runs (`collide_bgk27`, `collide_trt27`, `collide_mrt27`,
`collide_cumulant_d3q27`, `collide_cumulant_geier_d3q27`, `stream27_roll`).

| item | value |
|---|---|
| lattice | D3Q27 (27 directions, velocity tuples checked as a set in `phase_sanity`) |
| grids | 8³ and (6, 9, 13) periodic — the asymmetric grid kills axis-transposition aliasing |
| τ | 0.55, 0.7, 1.0 |
| steps | 30 (whole step `f^{n+1} = Stream(Collide(f^n))`) |
| seeds | 0, 1 |
| initial field | discrete 2nd-order Maxwellian + O(1 %) ρ/u perturbation + 1e-3 non-equilibrium noise; one float64 array handed to both sides |
| precision | fp64 both sides |

**Arms** (repo side, torch fp64): A1 `collide_bgk27`, A2 `collide_trt27`(Λ=3/16),
A3 `collide_mrt27` (documented defaults), A4 `collide_cumulant_geier_d3q27`,
A5 `collide_cumulant_d3q27` (**declared convention probe, not gated**).

**Reference collisions** (lbmpy 2.0, symbolic → `LbmCollisionRule.lambdify`):
`create_srt(compressible, discrete eq.)`, `create_trt`, a `MomentBasedLbMethod` over the
moment polynomials *reconstructed* from the repo's own `_M_D3Q27_DATA`,
`create_with_default_polynomial_cumulants` (Geier 2015) with continuous and with
discrete-Maxwellian equilibrium. Reference **streaming** is an independent numpy
`np.roll` pull-streaming operator.

## ② Method / toolchain

* `d3q27/refcore27.py` — numpy reference core (velocities/weights read from
  `tensorlbm.d3q27`, definition streaming, both equilibrium conventions, pointwise
  metric `eps_scaled_inf`, conservation drift, shear-wave helpers). No lbmpy, no torch.
* `d3q27/ref27.py` — lbmpy reference colliders + whole-step driver + MRT rate vector
  built from `collide_mrt27`'s documented defaults.
* `d3q27/arms27.py` — the five in-repo arms, each `Stream_repo(Collide_arm(·))`.
* `d3q27/predeclared.json` — **thresholds and arm classification fixed before the run**;
  the probe arm A5 is declared non-gated up front so it cannot be re-tuned into a pass.
* `d3q27/run_xval.py` — phases: sanity → gated code-equivalence sweep → positive
  controls (τ perturbation 1e-3, wrong equilibrium convention, direction
  mis-permutation, streaming omission) → shear-wave physics anchor → amplitude-scaling
  attribution for the cumulant convention gap → frozen bundle.
* `tests/test_xval_pointwise_d3q27_refdata.py` — **lbmpy-free** gate on the frozen
  bundle (sha256 + per-array sha256 + replay of each gated arm from the stored initial
  field, `eps_scaled_inf < 1e-10`), so the regression survives outside the dev container.
* `lbmpy_pointwise_ref.py` extended minimally: D3Q27 lattice branch, `cumulant` and
  `cumulant_discrete_eq` models, 27-term monomial basis `{1,x,x²}⊗{1,y,y²}⊗{1,z,z²}`.

Predeclared gates: code equivalence `1e-10` fp64; MRT basis identity `1e-12`; positive
controls detected at `>1e-6`; shear-wave decay error `<15 %` (owner caliber) with the
`3 %` target reported separately.

## ③ Results obtained so far (verified, reproducible)

| check | value | gate | verdict |
|---|---|---|---|
| D3Q27 velocity set == {−1,0,1}³ | 27/27 | exact | PASS (local) |
| direction permutation TensorLBM→lbmpy bijection | 27/27 | exact | PASS (local) |
| **MRT moment basis reconstructed from `_M_D3Q27_DATA`** | **1.04e-15** | ≤1e-12 | **PASS** |
| D3Q19 reconstruction path unchanged (regression of the shared helper) | 0.0 | ≤1e-12 | PASS |

The basis number is the only point where a wrong repo basis could hide, and it is
reconstructed symbolically (not transcribed): the 27 moment rows of the repo are solved
for in the tensor-product monomial basis and re-evaluated through `lbmpy.moments`.

The pointwise sweep, the positive controls, the shear-wave anchor and the cumulant
convention scaling are **NOT RUN** — the cluster job `cj3264`
(`run_xval.py`, 8 cores, stage_dir = this worktree) was still `queued` at report time.
No table of eps values is given because none was measured; per the honest-FAIL rule
this is recorded as WITHHELD, not as a pass.

## ④ Figures

None in this version: the figures are produced by `d3q27/make_figures.py` from
`summary.json`, which does not exist yet (run pending). Emitting a placeholder or a
hand-made curve would misrepresent the evidence state.

## ⑤ Notes / pitfalls found while building the harness

1. **Symbolic 27×27 inversion is the wrong tool.** Recovering the moment polynomials by
   `sympy.Matrix.inv()` on the 27-node Vandermonde is exact but needlessly slow; the
   float solve + residual assertion (`≤1e-10` elementwise) gives the same basis at
   1e-15 and is auditable. Snapping coefficients to "small rationals" *degrades* the
   result (9e-11, then 1.5e-3) — the reconstruction must stay float.
2. **Cumulant equilibrium convention is a real confounder.** lbmpy's default cumulant
   method uses the *continuous* Maxwellian; the repo uses the *discrete* 2nd-order one.
   Their order ≥3 equilibrium cumulants differ, so A4 is gated against the
   discrete-equilibrium cumulant reference. The continuous-equilibrium reference is
   still measured, and `phase_scaling` fits the log-log slope of the gap versus |u|
   so that an O(u³) convention artefact can be *identified* rather than guessed.
3. **A5 is not the Geier operator.** `collide_cumulant_d3q27` relaxes the cumulants of
   `f_neq` (its own docstring warns about this), so it is registered as a probe: its
   gap is a finding, never a pass/fail.
4. MRT rates must follow `collide_mrt27`'s tail order `[s_eps] + [s_pi]*7` with
   `s_pi = s_e` when unspecified; getting that tail wrong shows up as an O(1e-4)
   "defect" that is really a rate-vector typo.

## ⑥ Error judgement versus baseline

| quantity | baseline | error | <3 %? |
|---|---|---|---|
| MRT basis reconstruction | repo `_M_D3Q27_DATA` | 1.04e-15 (absolute, matrix entries) | yes |
| pointwise operator agreement | lbmpy 2.0 symbolic rule | **not measured (job cj3264 queued)** | **unknown** |
| shear-wave decay | analytic e^{−νk²t} | not measured | **unknown** |

The 15 % owner caliber and the 3 % target both apply to the physics anchor only; for the
code-equivalence arms the meaningful caliber is fp64 round-off (a 15 % band there would
be vacuous), which is why the predeclared gate is 1e-10 and is reported as such.

## Next step (for the following self-start)

Re-fetch `cj3264`; if `run_xval.py` returns 0, the report sections ③/④/⑥ are filled from
`summary.json` by `d3q27/make_figures.py` + a report generator (programmatic injection,
zero hand-copied numbers). Then commit the frozen bundle
`refdata/pointwise_d3q27_v1.npz` so `tests/test_xval_pointwise_d3q27_refdata.py` stops
skipping and becomes the fourth leg of the pointwise gate (D2Q9 single-step, D2Q9
stepwise, D3Q19, D3Q27).
