# XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY (one-pager)

**Anchor** `XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY` — OSS row `oss_solutions` 274
(`lssfau/walberla` / `GAP-LBM-MCMP-POISEUILLE` / tag `no-mcmp-two-phase-only`).
Full study `xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/REPORT.md`; numbers
`results.json` (rendered, not hand-typed); driver `run_xval.py`; model
`two_phase_only.py`; tests `tests/test_xval_no_mcmp_two_phase_only.py`.

## Question
The OSS repository has no MCMP: only the phase-field layered initialisation
geometry and the variable-density equilibrium assembly can be borrowed, i.e.
"two-component" is downgraded to "two-phase layered". What does the downgrade
cost against the two-component layered Poiseuille analytic benchmark?

## Answer (21 predeclared arms)
* Two-phase-only model (one distribution, frozen phase indicator, local tau from
  C, Guo forcing): worst profile L2 error over the matched-density matrix
  **5.99%**, worst peak-normalised max error
  7.96%, worst per-layer flux error
  6.23% — inside the owner 15% band.
* Component-resolved MCMP arm on the same problem: L2 15.4% (ratio 2), 36.2% (ratio 4) — outside the band.
* The viscosity mixing rule is load-bearing: harmonic
  0.82% vs linear
  5.99% at ratio 10.
* A density contrast across the frozen interface leaves the band (L2
  17.35%) — the structural
  limit of the downgrade.
* Mass-drift check K6 (3.46e-04 vs predeclared
  1e-06) **FAILS**; reported as failed, threshold not relaxed afterwards.

## Disposition
Admit the no-MCMP two-phase-only route as a verification arm for
viscosity-stratified parallel channel flow; keep MCMP where component transport
or interface dynamics matters; do not run a density contrast through the
frozen-indicator model without a sharpening term.
