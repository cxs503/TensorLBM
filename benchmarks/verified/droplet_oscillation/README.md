# B24 droplet_oscillation — W2-B strict-standard re-verification

**Verdict (see result.json): RAW observed damped frequency ω_d vs Rayleigh, per-level
|err| ≤ 3% with strictly monotone convergence in R. The forbidden damping-restoration
ω₀=√(ω_d²+γ²) appears ONLY as a diagnostic column and is not part of any criterion.**

## What this benchmark measures

An m=2-capillary-mode droplet oscillation in SC94 single-component pseudopotential
LBM: the RAW damped frequency ω_d of the interface-extent observable RxRy(t)=R_x−R_y
(time-domain LSQ damped-sine fit) is compared against the Rayleigh prediction
ω_th=√(6σ/((ρ_l+ρ_v)R³)) with per-level R-matched surface tension σ_i(R) and band
densities from a static Laplace run of the same droplet (basis C, primary).

## Model (library chain only — grep-clean run.py)

- `tensorlbm.multiphase.collide_sc_single_component` (D2Q9 SCMP BGK) +
  `tensorlbm.solver.stream` (periodic pull); psi=1−e^(−ρ); physical G_eff=−5.0
  (library called with G=+5.0, backward-gather sign convention); τ=1.0; float32.
- Domain L=4R periodic; elliptic tanh initial droplet R(θ)=R0(1+0.05·cos2θ), W=4.
- Levels R = 128 / 160 / 224; each simulated ≥5.6 theory periods (criterion
  requires ≥4); sample every 50 steps.

## Pre-registered protocol (NOTES.md §3, locked before the formal scan)

- Primary observable: RxRy(t) (best σ_ω AND best r² of all tested observables at
  every level; Q_mask threshold-core quadrupole reported alongside).
- Primary window: skip 500 steps + 2.0 theory periods (basis C). Window rule from
  the pre-scan instability map (NOTES.md batch 8): 1p windows are initial-transient
  biased, ≥2.5p windows are contaminated at R≥224 by the late elongation
  instability; 2.0p is clean at all levels.
- Theory basis C (primary): σ_i(R)=dp·R_eq, ρ_in/ρ_out band means from the
  embedded static Laplace run (same R, same model family, conv 3e-5 + nucleation
  guard, tail-8 mean); R_eq from the primary window mean. Basis A (legacy small-R
  constants) and the vacuum (ρ_l-only) form reported as secondary columns.
- PASS: |err|≤3% at every level (basis C), |err| strictly decreasing in R,
  σ_ω/ω_d<1%, mass drift<2e-3.

## Artifacts

- `run.py` — standalone formal scan (static + oscillation + fits + gates).
- `result.json` — per-level full metrics: fitted ω_d, γ, σ_ω, r², all window
  variants (1/1.5/2/2.5/3/4p × RxRy/Q), theory bases C/A/vacuum, ω₀ diagnostic,
  static-protocol status (convergence, guard), raw errors.
- `hist_R{R}.npz` — per-oscillation-run time series (step, Q, R_eq, R_x, R_y,
  ρ_in, ρ_out, max_u, mass); `hist_static_R{R}.json` — static Laplace histories.
- `NOTES.md` — full campaign record: pre-scan table (P1–P15), theory-basis
  decision, observable/window selection on noise metrics, instability map,
  pre-registration, amendments, honest-disclosure list.

## Key physics facts established in the pre-scan (all in NOTES.md)

1. The raw-ω_d error is dominated by the damped-oscillator pull-down
   (ω_d=√(ω₀²−γ²)): the model eigenfrequency is Rayleigh-correct to −0.5±0.3%,
   the observed deficit shrinks with R as γ/ω₀ falls (∝R^−1/2), giving the
   monotone convergence — no correction applied, the RAW error is the criterion.
2. Legacy small-R theory constants (basis A) flatter the error by 1.1-1.4% at
   R≥64 (σ_i(R) rises, ρ_lv(R) falls); basis C (R-matched statics) is the
   stricter, physically-consistent primary.
3. A slow elongation instability (model-family long-time limitation, e-folding
   ~60k steps) ends the clean oscillation at ~3.1p/2.9p/1.9p for R=128/160/224
   and makes R=320 unmeasurable; static droplets dissolve at ~283k/400k steps
   (R=160/320) once float32 mass drift (~7.7e-9/step) crosses coexistence —
   the nucleation guard protects σ_i extraction and all fit windows sit far
   inside the clean regime.

## Reproduce

```bash
/nfs/wangxi/venvs/tensorlbm/bin/python run.py --radii 128,160,224 \
    --device cuda:0 --out <dir>        # ~1 h on one RTX 5090
```
