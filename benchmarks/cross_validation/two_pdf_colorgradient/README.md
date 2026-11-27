# Two Pdf Colorgradient — historical cross-validation

The two tracked result JSONs, protocol, report, reference library and SQL/source-inventory artifacts are preserved. The reported full results.json and figure file are absent from the original tracked snapshot; they are not manufactured. The historical report explicitly describes the limitations of the retained measurements.

- [Archive index](../../../development/verification/two_pdf_colorgradient/README.md)
- [Historical report](../../../development/verification/two_pdf_colorgradient/REPORT.md)
- [Preregistered protocol](../../../development/verification/two_pdf_colorgradient/predeclared.json)
- [Archived Laplace log](../../../development/verification/two_pdf_colorgradient/artifacts/laplace_log.json)
- [Archived resolution probe](../../../development/verification/two_pdf_colorgradient/artifacts/resolution_probe.json)
- [Reference library](../../../development/verification/two_pdf_colorgradient/cg_reference.py)
- [Source inventory](../../../development/verification/two_pdf_colorgradient/oss_solutions.json)
- [Regression tests](../../../tests/test_color_gradient_two_pdf.py)

## Compatibility

Run from the repository root and use a separate worktree for any execution that writes campaign results. This checked launch command is not scientific acceptance.

```sh
PYTHONPATH=src python development/verification/two_pdf_colorgradient/run_xval.py --help
```

[Migration and source-object ledger](../../../docs/architecture/migrations/young-laplace-two-pdf-xval-2026-11-26.json). Original path: `xval/XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT` at `c06cb1c28ece2a024545e34c12a2a644a505ae3c`.
