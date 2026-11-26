# Young Laplace Analytic — historical cross-validation

The archived original numerical JSON and field snapshot remain byte-identical. Three original report images were recovered directly from the original tracked report payload; the historical payload remains unchanged, including its original report paths. Report reproduction from these existing data is not a new solver replay.

- [Archive index](../../../development/verification/young_laplace_analytic/README.md)
- [Historical report](../../../development/verification/young_laplace_analytic/REPORT.md)
- [Preregistered protocol](../../../development/verification/young_laplace_analytic/predeclared.json)
- [Original summary](../../../development/verification/young_laplace_analytic/artifacts/summary.json)
- [Original field snapshot](../../../development/verification/young_laplace_analytic/artifacts/hero_fields.npz)
- [Original payload](../../../development/verification/young_laplace_analytic/artifacts/report_payload.json)
- [Analysis regression](../../../tests/test_xval_young_laplace_analytic.py)
- [fig_laplace.png](../../../development/verification/young_laplace_analytic/artifacts/fig_laplace.png)
- [fig_hero.png](../../../development/verification/young_laplace_analytic/artifacts/fig_hero.png)
- [fig_sensitivity.png](../../../development/verification/young_laplace_analytic/artifacts/fig_sensitivity.png)

## Compatibility

Run from the repository root and use a separate worktree for any execution that writes campaign results. This checked launch command is not scientific acceptance.

```sh
PYTHONPATH=src python development/verification/young_laplace_analytic/run_xval.py --help
```

[Migration and source-object ledger](../../../docs/architecture/migrations/young-laplace-two-pdf-xval-2026-11-26.json). Original path: `xval/XVAL-LBM-YOUNG-LAPLACE-ANALYTIC-PLUS-DROPLET-CALI` at `c06cb1c28ece2a024545e34c12a2a644a505ae3c`.
