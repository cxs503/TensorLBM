# Results artefacts — P46W3-N03

| file | content | produced by |
|---|---|---|
| `raw_cases.json` | every case record returned by the LBM kernels (drag ladder + shear-cell sweep + pure-fluid control), plus the design block that was actually executed | `run_lbm_calibration.py` |
| `analysis.json` | `eta_r` per measured point (mean/SD/SEM over seeds), hydrodynamic radius ladder, dilute fits in two volume conventions, Krieger-Dougherty fits, divergence exponents, acceptance verdicts, figure paths | `fit_calibration.py` |
| `figures/*.png` | `eta_r_vs_phi.png`, `dilute_limit.png`, `resolution_ladder.png`, `divergence.png`, `residuals.png` (matplotlib, generated from `analysis.json` only) | `fit_calibration.py` |

Provenance: design and pre-registered thresholds in
`../P46W3_N03_PREDECLARED_DESIGN.md` (with amendments AMENDMENT-1..3, all written
before the first case of this campaign produced a result); narrative report in
`../../../docs/reports/P46W3-N03_suspension_viscosity.md`, generated from
`analysis.json` by `../make_report.py` (no hand-transcribed numbers).

Re-generate the downstream artefacts from the raw records with

```bash
PYTHONPATH=src python benchmarks/suspension/fit_calibration.py
PYTHONPATH=src python benchmarks/suspension/make_report.py
```