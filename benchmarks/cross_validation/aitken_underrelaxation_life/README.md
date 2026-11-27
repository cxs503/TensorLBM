# Historical Aitken interface cross-validation

The source and original recorded data from `xval/aitken_underrelaxation_life/` are preserved under
`development/verification/aitken_underrelaxation_life/`. This is a historical scalar, linear interface study and independent
method transcription, not a direct external solver run or a current
scientific acceptance. The maintained tests import the relocated reference module.

Published reports (stable paths):
- [Original report](../../../docs/XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXED-POINT.md)
- [Independent audit report](../../../docs/AUDIT-XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE.md)
- [Original audit SQL](../../../development/audit/m17828/verdict_m17828.sql)

Original recorded inputs and outputs are preserved (42 original rows,
preregistration, scale-guard observations and CSVs). Two historical scripts
have repository-relative path and report navigation edits only, documented in
the [migration ledger](../../../docs/architecture/migrations/aitken-xval-archive-2026-11-27.json). The numerical algorithms and
original test assertions remain unchanged.

Five original figures named in the historical reports are absent from the
inspected Git objects. The figure-generation scripts are preserved; generated
figures would be new derivatives, not recovered originals. No new output was
generated. The original report generator needs all four figure inputs and
writes to the official report path; do not run it against the preserved files.
No full scientific replay was performed.

Original tracked files:

- [figures.py](../../../development/verification/aitken_underrelaxation_life/figures.py)
- [make_report.py](../../../development/verification/aitken_underrelaxation_life/make_report.py)
- [out/cases.csv](../../../development/verification/aitken_underrelaxation_life/out/cases.csv)
- [out/results.json](../../../development/verification/aitken_underrelaxation_life/out/results.json)
- [out/scale_guard.json](../../../development/verification/aitken_underrelaxation_life/out/scale_guard.json)
- [out/summary.csv](../../../development/verification/aitken_underrelaxation_life/out/summary.csv)
- [predeclared.json](../../../development/verification/aitken_underrelaxation_life/predeclared.json)
- [reference.py](../../../development/verification/aitken_underrelaxation_life/reference.py)
- [run_xval.py](../../../development/verification/aitken_underrelaxation_life/run_xval.py)
- [scale_guard_figures.py](../../../development/verification/aitken_underrelaxation_life/scale_guard_figures.py)
- [scale_guard_probe.py](../../../development/verification/aitken_underrelaxation_life/scale_guard_probe.py)

The existing compatibility regressions are [scale guard](../../../tests/test_fsi_aitken_guard_scale.py)
and [reference model](../../../tests/test_fsi_aitken_life_xval.py). Current scientific acceptance:
**not evaluated**.
