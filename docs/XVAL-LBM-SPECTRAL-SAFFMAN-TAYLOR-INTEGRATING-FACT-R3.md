# XVAL R3 — spectral Saffman–Taylor: exact variable-coefficient curl–Darcy closure

**Mission**: m21247 · **Layer**: cross-validation (XVAL), round 3 · **Repo**: TensorLBM
**Anchor (feature ledger)**: `XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT`
**Round 1** m18378 (`…-FACTOR.md`, port fidelity) · **Round 2** m21044 (`…-FACT-R2.md`, honest FAIL:
frozen closure flips the sign of the growth rate). **This round** ships the remediation registered
there.
**Historical artifact references**: `xval_exact_darcy_r3.py` (study driver),
`artifacts/exact_darcy.json`, `artifacts/r3_parts.json` (chunked execution record),
`artifacts/fig_r3_*.png`, `write_r3_doc.py`, `publish_r3_report.py`, and
`tests/test_spectral_saffman_taylor_exact_darcy.py` (reported as 7 tests).
The listed round 3 files were absent from the pre-migration `main` tree.
The [benchmark index](../benchmarks/cross_validation/spectral_saffman_taylor/README.md) links the tracked evidence for rounds 1 and 2.

---

## ① Case

Miscible viscous fingering in a Hele-Shaw cell / porous medium, `μ = exp(R c)`, `D = 1`, unit base
flow along `+x`; OSS scheme `ebinan92/Fingering_dynamics::spectrum_method` (exponential integrating
factor + RK4 bootstrap + AB2 + CN corrector) as ported in round 1. Protocol identical to rounds 1–2
(like-for-like comparison):

| parameter | value |
|---|---|
| grid | `N_y × N_x = 128 × 256`, `Δx = A·Pe/N_x = 3.906` |
| domain / Peclet | `A = 2.0`, `Pe = 500` |
| transverse mode | `n = 4`, `k = 2πn/Pe = 5.026548e-02`, `k² = 2.526619e-03` |
| time step / fit window | `dt = 0.1`, σ fitted on `t ∈ (2.0, 8.0)` |
| front | `c̄ = ½(1−tanh((x−x₀)/w))`, `w = 4.0` (≈1.0 cell), perturbation `1e-3 cos(ky)/cosh` |
| `R` values | -2.0, -1.0, -0.5, +0.0, +0.5, +1.0, +2.0 |

## ② What was done

Round 2 showed the frozen-coefficient OSS symbol (`elliptic="oss_frozen"`) is not the curl of
Darcy's law: O(1) residual, wrong **sign** of `σ` for `R ≠ 0`. This round implements the registered
remediation as a second, selectable closure `elliptic="exact_varcoef"`:

```
lap ψ + R (c_x ψ_x + c_y ψ_y) = −R c_y        (exact, variable coefficients; −R c_y = unit base flow)
solved matrix-free: restarted GMRES(20), right preconditioner = Fourier 1/lap symbol,
warm start from the previous step's ψ, tol 1e-9, ≤60 iterations, 6 solves per time step
```

The time-integration scheme of the OSS port is untouched — only the elliptic closure changes.
The anchor is the **independent** first-principles linear-stability solver of round 2
(`xval_linear_theory.exact_linear_sigma`: finite-difference eigenvalues of
`ψ'' − k²ψ + R c̄' ψ' = −ikRĉ`), itself validated there against a hand-derived sharp-interface
relation; it is imported by the study, never fitted to module output.

Closure verification on the same initial field (normalised residual of
`lap ψ + R(c_xψ_x + c_yψ_y) + R c_y = 0`):

| R | residual `oss_frozen` | residual `exact_varcoef` | GMRES iters | GMRES rel. residual | max&#124;∇·u&#124;/max&#124;u&#124; |
|---|---|---|---|---|---|
| -1.00 | 1.767 | 2.4e-14 | 20 | 2.7e-14 | 9.7e-17 |
| -0.50 | 1.871 | 2.8e-14 | 20 | 3.0e-14 | 9.4e-17 |
| +0.50 | 1.871 | 2.8e-14 | 20 | 3.0e-14 | 1.1e-16 |
| +1.00 | 1.767 | 2.6e-14 | 20 | 2.8e-14 | 9.7e-17 |

## ③ Results — growth rate vs the first-principles anchor

| R | σ anchor | σ `exact_varcoef` | rel. err | sign | σ `oss_frozen` | rel. err | sign |
|---|---|---|---|---|---|---|---|
| -2.00 | +3.083445e-02 | +3.067893e-02 | 0.50% | yes | -3.809027e-02 | 223.5% | **no** |
| -1.00 | +1.426268e-02 | +1.554744e-02 | 9.01% | yes | -2.085114e-02 | 246.2% | **no** |
| -0.50 | +5.110120e-03 | +6.712994e-03 | 31.37% | yes | -1.177471e-02 | 330.4% | **no** |
| +0.00 | -2.530474e-03 | -2.526619e-03 | 0.15% | yes | -2.526619e-03 | 0.2% | yes |
| +0.50 | -2.542151e-03 | -1.172004e-02 | 361.03% | yes | +6.772162e-03 | 366.4% | **no** |
| +1.00 | -2.542233e-03 | -2.045464e-02 | 704.59% | yes | +1.602481e-02 | 730.3% | **no** |
| +2.00 | -2.542335e-03 | -3.560568e-02 | 1300.51% | yes | +3.421283e-02 | 1445.7% | **no** |

## ④ Figures

`fig_r3_sigma_vs_R.png` — (a) σ(R): exact closure vs anchor vs round-2 frozen closure;
(b) relative error with the owner 15 % line. `fig_r3_fieldmaps.png` — c field at `t = 1.5`
for both closures plus the GMRES streamfunction. `fig_r3_resolution.png` — refinement panel
(**empty: see ⑤, refinement not run**).

## ⑤ Notes / traps

* **The fix is the equation, not a constant.** The solver knows nothing about the sign convention of
  `R`; the corrected stability is a consequence of solving the relation the anchor linearises.
* **Sign is fixed everywhere**: 0 sign mismatches over 7 `R` values for the
  exact closure, versus 6 for the frozen one.
* **Unstable branch (`R < 0`) is quantitative**: `R = −2` → 0.50 %,
  `R = −1` → 9.01 %,
  `R = −0.5` → 31.37 % relative error. The
  0.5 % at `R = −2` and 9 % at `R = −1` are inside the owner 15 % bar; `R = −0.5` is not.
* **Stable / near-neutral branch (`R > 0`) is over-damped, and this is a new, open finding.** The
  anchor's `σ` there is diffusion-dominated (`≈ −k² = −2.54e-3`), while the module decays at
  −1.2e-2 … −3.6e-2 (4–14 × faster). Sign is right, magnitude is not. Candidate causes, in order of
  plausibility: (i) the module's base front is *static and diffusing* while the anchor is
  quasi-steady (base `w` grows 4 → ≈4.6 over the fit window; damping of a broadened front rises);
  (ii) sub-cell front resolution (`w ≈ 1` cell at this protocol); (iii) the AB2/CN splitting cost
  becomes dominant when the physical `σ` is tiny. **These are hypotheses, not established
  explanations** — round 4 should separate them by (a) re-imposing the analytic base each step,
  (b) grid refinement, (c) `dt` refinement.
* **Refinement not run**: the cluster job (`cj3287`, nodes n315/n330) that carried the `N_x = 512,
  1024` scan returned no artifacts (empty fetch), and the local <8-core/<10-min budget does not
  accommodate a 1024-point scan. The refinement rows are therefore *not_run*, not "passed".
* **Cost**: ≈ 0.6 s per time step at the base protocol (6 GMRES solves/step, 40–80 steps/run).
  `oss_frozen` remains the default (OSS transcription fidelity, relative L2 3e-15); pass
  `elliptic="exact_varcoef"` for physics-grade growth rates.
* **Linear regime only**: no statement about finite-amplitude fingers / tip splitting.

## ⑥ Error verdict (explicit, honest)

| quantity | benchmark | error | <3 %? | <15 %? |
|---|---|---|---|---|
| closure residual, `exact_varcoef` | exact curl–Darcy (→0) | 2.8e-14 | YES | YES |
| closure residual, `oss_frozen` | exact curl–Darcy (→0) | up to 1.87 (O(1)) | NO | NO |
| `σ(R=0)` | anchor `−k²` | 0.15 % | YES | YES |
| `σ(R=−2)` | anchor `3.0834e-02` | 0.50 % | YES | YES |
| `σ(R=−1)` | anchor `1.4263e-02` | 9.01 % | NO | YES |
| `σ(R=−0.5)` | anchor `5.1101e-03` | 31.37 % | NO | NO |
| sign of `σ`, all `R` | anchor sign | 0 mismatches | — | YES |
| `σ` on the stable branch (`R>0`) | anchor (near-neutral) | 361–1301 % | NO | NO |
| `σ` grid refinement | anchor | **not_run** | unknown | unknown |

**Verdict.** The blocking defect of round 2 — the *wrong sign* of the linear growth rate, which made
absolute finger-growth predictions unusable — is **fixed**: the exact variable-coefficient closure
reproduces the anchor sign for all 7 `R` values and the magnitude to
0.50 % (`R=−2`) and 9.01 %
(`R=−1`) on the unstable branch, i.e. inside the ledger acceptance (<15 %) where the front is
actually unstable. It is **not** met at `R = −0.5` (31 %) nor on the stable/near-neutral branch
(4–14 × over-damping, sign correct), and grid/`dt` refinement is *not run*. The ledger acceptance
`<15 %` is therefore **partially met** — met on the physically relevant unstable branch for
`|R| ≥ 1`, unmet elsewhere — and the remaining discrepancy is registered with falsifiable
hypotheses for round 4, not reconciled away.
