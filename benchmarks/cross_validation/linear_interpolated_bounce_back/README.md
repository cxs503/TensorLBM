# Straight-wall interpolated bounce-back

This D2Q9 case compares halfway and two-point interpolated bounce-back against the analytic velocity profile of fully developed plane Poiseuille flow at prescribed wall offsets. The [historical report](evidence/historical/REPORT.md) documents the configuration, reference conventions and the failed 15% velocity-profile error gate. Its [reported measurements](evidence/historical/results.json) and [comparison table](evidence/historical/table.csv) preserve the original data.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/cross_validation/linear_interpolated_bounce_back/run_xval.py --quick --out outputs/linear-ibb-quick
python tools/benchmark_registry.py --case xval:linear_interpolated_bounce_back
```

The [CPU migration replay](evidence/2026-11-25-cpu-quick.json) records the four quick-sweep arms and their results. Both interpolated arms exceed the 15% threshold. A complete campaign and scientific acceptance remain pending. The [migration ledger](../../../docs/architecture/migrations/linear-interpolated-bounce-back-xval-2026-11-25.json) records the original paths and checksums.
