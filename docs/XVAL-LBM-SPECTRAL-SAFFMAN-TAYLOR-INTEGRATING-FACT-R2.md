# XVAL R2 — TensorLBM spectral Saffman–Taylor: **independent linear-stability anchor**

**Mission**: m21044 · **Layer**: cross-validation (XVAL), round 2 · **Repo**: TensorLBM
**Anchor (feature ledger)**: `XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT`
**Round 1** (m18378): `docs/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACTOR.md`,
module `src/tensorlbm/spectral_saffman_taylor.py`
**This round's artifacts** (all numbers below are injected from them, nothing hand-copied):
`benchmarks/cross_validation/spectral_saffman_taylor/round2/`
— `xval_linear_theory.py` (anchor + study driver), `artifacts/linear_theory.json`,
`artifacts/xval_r2_run.log`, `artifacts/fig_linear_theory_{sigma_vs_R,eigenmode,step_limit}.png`
**Tests**: `tests/test_spectral_saffman_taylor_linear_theory.py` (5 tests, pass)

---

## ① Why a round 2: the round-1 anchor was self-referential

Round 1 ported `ebinan92/Fingering_dynamics::spectrum_method` (IF + RK4 bootstrap + AB2 +
Crank–Nicolson) into `SpectralSaffmanTaylor2D` and validated it two ways:

1. **against the OSS code it was ported from** — relative L2 `2.965e-15` (round-off). This
   proves the *transcription* is faithful; it says nothing about whether the scheme solves
   the intended physics.
2. **against a constant `C = (σ + k²)/R`** whose value was itself the **mean of the same
   six measurements** (`C = 1.84245e-02`, "max deviation 2.28% → PASS"). This is an internal
   consistency check (is `σ` linear in `R`?), not a validation: it cannot detect a wrong
   coupling constant, and it cannot detect a wrong **sign**.

The ledger acceptance ("对标误差 <15%") therefore remained unmet: no independent
analytic/literature growth rate had ever been computed for this configuration.
Round 1 also *measured* the elliptic defect (normalised residual 1.5–2.0 for `R ≥ 0.05`) and
flagged the OSS closure as "a first-order-in-R approximation", but never quantified what that
does to the predicted growth rate.

## ② Method: first-principles anchor derived from the PDEs (no OSS code, no TensorLBM code)

Miscible Hele-Shaw / porous medium, `μ = exp(R c)`, `D = 1`:

```
dc/dt + u·∇c = ∇²c ,            u = (ψ_y, −ψ_x)
∇²ψ + R (c_x ψ_x + c_y ψ_y) = 0          (curl of Darcy — exact, no frozen coefficients)
```

Base state (exactly satisfies the elliptic equation for every `R`):
`c̄(ξ) = ½(1 − tanh(ξ/w))`, `ξ = x − t` (unit base flow along `+x`, i.e. `ψ₀ = y`),
perturbation `~ exp(i k y + σ t)`. Exact linearisation:

```
ψ̃'' − k²ψ̃ + R c̄' ψ̃' = −i k R c̃
σ c̃ = c̃'' − k² c̃ − i k c̄' ψ̃
```

Eliminating `ψ̃` (`A = D₂ + R diag(c̄') D₁ − k² I`): `σ c̃ = (D₂ − k²I) c̃ − k²R diag(c̄') A⁻¹ c̃`.
Discretised with 2nd-order finite differences in `ξ ∈ [−L, L]`, homogeneous Dirichlet data,
`N = 601…801` (`h = 0.67…1.33`) and the **leading eigenvalue** taken as the anchor `σ_theory`.

Sharp-interface limit, **derived by hand** from the delta-function jump conditions of the same
linearised system (`c̄' = −δ(ξ)`):

```
κ² = k² + σ ,      2κ² + 2kκ + kR = 0 ,      σ = κ² − k²      (admissible root Re κ > 0)
```

This analytic relation independently reproduces the classical Saffman–Taylor criterion: for
`μ = exp(Rc)` with `c = 1` upstream, the front is unstable **only for `R < 0`** (the fluid
*ahead* of the front must be the more viscous one).

Parameters (identical to the round-1 growth-rate study, so the comparison is like-for-like):
`N_y × N_x = 128 × 256`, `A = 2`, `Pe = 500`, `dt = 0.1`, `Δ = 1`, transverse mode `n = 4`
(`k = 2πn/Pe = 5.026548e-02`, `k² = 2.526619e-03`), front width `w = 4`, fit window `t ∈ [2, 8]`,
perturbation amplitude `1e-3` (linear regime).

### Validation of the anchor itself (two independent checks)

| Check | Result | Verdict |
|---|---|---|
| **V1** `R = 0` ⇒ `σ = −k²` (pure transverse diffusion) | `σ = −2.542040e-03` vs `−k² = −2.526619e-03`, rel. dev **0.61%** (residual = Dirichlet-box truncation `O((π/2L)²)`, `L = 400`) | ✓ |
| **V2** `w → 0` must converge to the hand-derived step relation | `R = −1`: `20.38% → 9.74% → 4.42%` for `w = 8 → 4 → 2` (monotone, `k w = 0.40 → 0.10`); `R = −2`: `27.78% → 18.13% → 11.31%`; `R = −0.5`: `19.53% → 7.57% → 2.58%` | ✓ convergence to the analytic limit |
| **Grid/domain independence** (`R = −1, w = 4`) | `N=401/L=400: 1.428294e-02`, `N=601/L=400: 1.425568e-02`, `N=601/L=800: 1.431248e-02` — spread **0.4%** | ✓ |

## ③ Results: growth rate, module (frozen closure) vs first principles

`σ_module` = re-measured here with the round-1 protocol (`single_mode_ic`, mode-amplitude fit);
`σ_theory` = exact curl-Darcy linearisation; `σ_step` = hand-derived sharp-step relation.

| R | σ_module | σ_theory (exact) | σ_step (analytic) | sign agree? | rel. dev. | <15%? |
|---|---|---|---|---|---|---|
| −2.0 | −3.809027e-02 | +3.083445e-02 | +3.766208e-02 | **no** | 223.5% | ✗ |
| −1.0 | −2.085114e-02 | +1.426268e-02 | +1.580117e-02 | **no** | 246.2% | ✗ |
| −0.5 | −1.177471e-02 | +5.110120e-03 | +5.528429e-03 | **no** | 330.4% | ✗ |
| 0.0 | −2.526619e-03 | −2.542040e-03 | — | yes | **0.6%** | ✓ |
| +0.5 | +6.772162e-03 | −2.589178e-03 | (no adm. root) | **no** | 361.6% | ✗ |
| +1.0 | +1.602481e-02 | −2.589838e-03 | (no adm. root) | **no** | 718.8% | ✗ |
| +2.0 | +3.421283e-02 | −2.590663e-03 | (no adm. root) | **no** | 1420.6% | ✗ |

**Mirror test** — replay the comparison with `R → −R` (the hypothesis that the frozen symbol
carries the first-order coupling with the opposite sign):

| R | σ_module | σ_theory(−R) | rel. dev. | <15%? |
|---|---|---|---|---|
| +0.5 | +6.772162e-03 | +5.110120e-03 | 32.5% | ✗ (near-neutral, relative error inflated) |
| +1.0 | +1.602481e-02 | +1.426268e-02 | **12.4%** | ✓ |
| +2.0 | +3.421283e-02 | +3.083445e-02 | **11.0%** | ✓ |
| −1.0 | −2.085114e-02 | −2.589838e-03 | 705.1% | ✗ (theory(+1) is near-neutral) |
| −2.0 | −3.809027e-02 | −2.590663e-03 | 1370.3% | ✗ |

Reading of the two tables: the OSS frozen-coefficient closure reproduces the *magnitude* of
the first-order coupling constant to 11–12 % at `|R| = 1–2`, but with the **sign of `R`
reversed**. At `|R| = 0.5` the exact `R > 0` branch is nearly neutral (`σ_theory ≈ −k²`), so
the relative error there is dominated by the tiny denominator and is not informative; the
robust statements are (i) the **sign** disagreement and (ii) the 11–12 % magnitude agreement
under mirroring.

Figures: `artifacts/fig_linear_theory_sigma_vs_R.png` (σ(R): module vs exact theory vs
sharp-step, plus the mirror panel), `fig_linear_theory_eigenmode.png` (leading eigenmode of the
exact operator at `R = ±1`), `fig_linear_theory_step_limit.png` (V2 convergence).

## ④ Where the sign flip comes from (analytic diagnosis, testable)

Linearising the implemented elliptic closure of the OSS symbol term by term:

* exact linearisation: `ψ̃'' − k²ψ̃ + R c̄' ψ̃' = −i k R c̃` ⇒ frozen form
  `ψ̂ = −i k R ĉ / (k² + k_m² − i R k_m c̄')` — note the **minus** sign of the stratification term;
* OSS/TensorLBM symbol: `ψ̂ = −i k R ĉ / (k² + k_m² + i R k_m c̄')` — **plus** sign;
* the implemented *equation* `∇²ψ + R(c_xψ_x + c_yψ_y) = −R c_y`, linearised about
  `(c̄(x), ψ₀ = y)`, gives a **doubled** vorticity source (`−2ikRc̃` instead of `−ikRc̃`),
  because the base-flow contribution `c_y ψ_y ⊃ i k c̃·1` is added on the right-hand side as
  well. This is consistent with the O(1) elliptic residual (1.5–2.0) measured in round 1.

Consequence: the closure is not the exact curl–Darcy equation **for either sign convention of
`R`** — so the sign flip is not merely a `μ = exp(±Rc)` documentation issue: the extra source
term survives under any convention. The regression test
`test_module_grows_where_first_principles_predict_decay` pins the discrepancy (module `σ > 0`
at `R = +1`, anchor `σ < 0`) so it cannot silently change.

## ⑤ Caveats / limits (what this report does and does not establish)

* **Quasi-steady base**: the anchor freezes the base front (`c̄` diffuses on `t_d ~ w²/D = 16`,
  comparable to the fit window `t ∈ [2,8]`). Base broadening moves `w` from 4 to ≈ 4.6 over the
  window; the V2 table shows `∂σ/∂w ≈ 6%/unit w` at `R = −1`, i.e. a ≤ 4% anchor uncertainty —
  small compared with the factor-of-2 sign disagreement.
* **Linear regime only**: no statement about finite-amplitude finger shapes or tip splitting.
* **Module-side resolution**: `w = 4` is spanned by ≈ 1 cell of the module grid
  (`Δx = A·Pe/N_x = 3.91`), so the measured `σ_module` itself carries a resolution error. That
  error is *not* what produces the sign flip (it would bias the magnitude, not the sign).
* **Near-neutral branch**: for `R > 0` the exact growth rate is within 2.5% of `−k²`, so
  relative errors on that branch are not a meaningful metric; the sign and the mirror table are.
* The anchor is a *linear-stability* anchor; it does not validate the nonlinear saturation, and
  it does not replace the round-1 round-off agreement with the OSS reference (which stands).

## ⑥ Error verdict (explicit)

| Key quantity | Benchmark | Error | <3%? |
|---|---|---|---|
| `σ(R=0)` (module) | exact `−k²` / anchor `−2.542040e-03` | 0.6% | ✓ (<3%) |
| anchor `σ(R=0)` | analytic `−k²` | 0.61% | ✓ (<3%) |
| `σ(R=−1)` (module) | exact curl-Darcy `+1.426268e-02` | sign flip, 246% | ✗ |
| `σ(R=+1)` (module) | exact curl-Darcy `−2.589838e-03` | sign flip, 719% | ✗ |
| `σ(R=+1)` (module) | exact curl-Darcy at `−R` (`+1.426268e-02`) | 12.4% | ✗ (not <3%) |
| `σ(R=+2)` (module) | exact curl-Darcy at `−R` (`+3.083445e-02`) | 11.0% | ✗ (not <3%) |
| anchor `σ(R=−1,w=2)` | hand-derived step relation | 4.42% | ✗ (but the limit is convergent: 4.42% at `k w = 0.10`) |

**Verdict (honest, not reconciled with round 1):** the round-1 statement "analytic-anchored
growth rate … PASS (<15%)" **does not survive** an independent anchor. The module's growth
rate is correct to 0.6% only in the trivial `R = 0` (pure diffusion) limit; for `R ≠ 0` its
**sign** is opposite to first-principles theory for the documented convention
(`μ = exp(Rc)`, `c = 1` upstream), and its magnitude matches only under an `R → −R` mirroring
(11–12% at `|R| = 1–2`). The owner precision standard (<15%) **fails** on every `R ≠ 0` row.
What still stands from round 1: the port is a faithful transcription of the OSS scheme
(3e-15), the integrating factor is exact for diffusion, and the closure defect is O(1) — the
last of which this report now quantifies as a sign-level error in the predicted stability.
Absolute finger-growth predictions must not be quoted from this module.

**Remediation (registered, not silently "fixed" in this mission):** (1) correct the
stratification term to `ψ̂ = −i k R ĉ / (k² + k_m² − i R k_m c̄')` and drop the duplicated
`−R c_y` source, then re-run this anchor study; or (2) implement the exact variable-coefficient
curl–Darcy solve (deferred in round 1) and validate it against the V2 step relation
(4.4% at `k w = 0.10`). Either route is a self-contained follow-up; the anchor harness in
`xval_linear_theory.py` is the acceptance gate for it.