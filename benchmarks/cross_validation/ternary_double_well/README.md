# Ternary double-well free-energy cross-validation

The CPU runner compares the TensorLBM ternary chemical potential against a
local evaluation of the documented CudaPhaseField formulas. It also checks
the discrete operators against analytic fields. The comparison covers the
free-energy kernel on periodic grids. It does not execute the upstream CUDA
solver or validate the full multiphase flow, wetting or interface evolution.

From the repository root in an environment with PyTorch installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/cross_validation/ternary_double_well/run_xval.py --out outputs/verification/ternary-double-well
```

The [case manifest](benchmark_case.json) links the embedded inputs, measured
quantities, tolerances and [2026-11-25 CPU replay](evidence/2026-11-25-cpu.json).
All eight checks passed on both sides of the relocation with byte-identical
result JSON. The largest full-field relative chemical-potential error was
`2.3962e-16` against the local reference formula and the executable
`1e-10` tolerance.

The original [report](evidence/historical-report.md) and
[implementation comparison](evidence/DIFF_OSS_VS_TENSORLBM.md) preserve their
historical text and stated verdicts. The [original Git tree](https://github.com/cxs503/TensorLBM/tree/32bf22f8074b9232d70fa5675089e7c1756dd9cd/xval/XVAL-LBM-TERNARY-DOUBLE-WELL-WITH-HARMONIC-LAMBDA)
retains the old command and paths at the source revision. The
[migration ledger](../../../docs/architecture/migrations/ternary-double-well-xval-2026-11-25.json)
records the file checksums and replay environment.
