# Diff list: TensorLBM vs pylbm / lbmpy — linear interpolated bounce-back α-scheme

Anchor `XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME`. Numbers: `REPORT.md`, `artifacts/`.

| aspect | pylbm / lbmpy (OSS) | TensorLBM | impact here |
|---|---|---|---|
| family parameterisation | `α̂=−1+α(1/2+δ)`, `β=1−αδ`, `β̂=1−α/2`, `K̂=1−α̂` | same (added this mission, backward compatible to 5.6e-17) | none |
| BFL member | α=2 (δ≤1/2), 1/δ above | `alpha_bfl` | none |
| CLI member | α=4/(1+2δ) ⇒ α̂=1 | `alpha_cli` | none |
| δ measured from | first/last **fluid** node | same | none |
| low-branch second term | post-collision opposite pop `f̂_q̄(x_f,t)` | post-streaming same-direction pop `f_new(x_f,i)` | **suspected +0.5 cell wall offset (measured, not fixed)** |
| exactness condition | `Λ⁺Λ⁻=δ²/2`, wall momentum `Λ⁻F` | `Λ⁺Λ⁻=δ²/2` transplanted; wall-value term not implemented | exactness not reproduced |
| forcing | post-collision, isotropic weights, report `momentum−F/2` | same | none |
| correction source term | step-to-step change (`f(t)−f(t−1)`) | collision+source change (`f_pre−f_old`) | needs bit-level recheck |
| periodic wrap | x periodic, y walls | same + 2 outer rows reset (anti-wrap guard) | extra safety here |
| second OSS implementation | lbmpy `NoSlipLinearBouzidi` (same 3 branches) | — | independent confirmation of the algebra |

Porting tradeoff: keep the α-family + `K̂` in the production module (tested, no behaviour change by
default); do **not** port pylbm code (formulas/conventions only); the moving-wall/Dirichlet value term
(`−α e⁻_q(x_w)`) is left out (it vanishes for the vertical cut links of this benchmark, mandatory for
inclined/moving walls); the 15 % acceptance line is not relaxed for the measured failure.
