# XVAL — OSS `analytic-solution-relative-L2-harness` (Taylor-Green 2D)

Cross-validation of this repository's D2Q9 path against the error-evaluation practice of
`lbm-principles-practice/code` (MIT; Krüger et al., *The LBM: Principles and Practice*, Springer 2016),
read from `chapter13/matlab/{vortexdecay,taylorgreen,equilibrium}.m` and `chapter5|6/*.m`.

* reusable contract: `src/tensorlbm/verification/analytic_l2.py`
* report: `docs/XVAL_m17416_TGV2D_ANALYTIC_L2_HARNESS_20261113.md`
* tests: `tests/test_tgv2d_analytic_l2_xval.py`

## Run

```bash
python run.py --out result.json --csv table.csv   # full: ~2 min, --quick: ~20 s
python make_assets.py                             # figures/*.png
python make_doc.py                                # regenerate the report from result.json
python -m pytest ../../../tests/test_tgv2d_analytic_l2_xval.py -q   # 16 passed, ~6 s
```

## Two arms

| arm | implementation |
|---|---|
| `oss_port` | line-by-line NumPy translation of `vortexdecay.m` (`circshift` streaming, BGK, identical diagnostics) |
| `tensorlbm` | this repo: `solver.stream` + `collide_bgk` + `d2q9.equilibrium/macroscopic`, float64, diagnostics via `verification.analytic_l2` |

Same NX / tau / u_max / step count / sampling cadence / IC / coordinates ⇒ any difference is
implementation, so agreement measures whether the *convention* was transferred correctly.

## Headline numbers (see `result.json["verdict"]`)

* cross-arm agreement: ≤ 2.3e-9 over all four diagnostics (threshold 1e-8)
* velocity relative L2 vs analytic: 0.077% (NX=32) → 0.019% (NX=64) → 0.0048% (NX=128) — passes the 3% target
* density relative L2 under the OSS "fluctuation" normalisation: 11.3% / 5.9% / 2.9% — under the 15%
  limit, but shown (§5.4 of the report) to be an **IC-excited acoustic transient**
  (period `2π/(c_s 2k)`, measured within 1% of prediction), not discretisation error
* fixed-Ma refinement (u_max = 0.005): velocity order 2.04 → 1.61 (≈ second order)
* negative control: grading against a wrong-viscosity analytic solution inflates the velocity error
  to 22% (290× the honest value) ⇒ the gate can fail