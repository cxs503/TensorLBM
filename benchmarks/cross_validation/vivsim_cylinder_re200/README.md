# Re=200 cylinder comparison with vivsim

This case compares mean drag and peak absolute lift from the final half of a
force record with the reference values 1.389 and 0.718 recorded in vivsim at
`fb84b12`. Both relative errors must be strictly below 15%. Strouhal number
is diagnostic.

The historical matched runs hit non-finite values and the comparison was
withheld. The frozen D48 and D64 values in the published report were
obtained with a different setup. The [audit result and raw evidence](../../../docs/board_audit/m17747/AUDIT_m17747.md)
preserve that distinction. The case has no current scientific verdict.

From the repository root, use the [runner](xval_vivsim_cylinder_re200.py) and
[report builder](xval_vivsim_re200_report.py) with explicit output paths:

```bash
PYTHONPATH=src python benchmarks/cross_validation/vivsim_cylinder_re200/xval_vivsim_cylinder_re200.py \
  --D 16 --u 0.05 --re 200 --conv-lengths 150 --strict \
  --out outputs/vivsim_re200/run.json --series outputs/vivsim_re200/run.npz
python benchmarks/cross_validation/vivsim_cylinder_re200/xval_vivsim_re200_report.py \
  outputs/vivsim_re200/run.json --out outputs/vivsim_re200/summary.json --figdir outputs/vivsim_re200
```

See the [published report](../../../docs/XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS.md)
for its numerical scope and historical settings, and the [case manifest](benchmark_case.json)
for the original reference file, recorded decisions and audit outputs. The
[original scripts](https://github.com/cxs503/TensorLBM/tree/32bf22f8074b9232d70fa5675089e7c1756dd9cd/benchmarks/xval)
are available at the pinned source revision. The [migration ledger](../../../docs/architecture/migrations/vivsim-cylinder-re200-2026-11-25.json)
records path mappings and checksums.
