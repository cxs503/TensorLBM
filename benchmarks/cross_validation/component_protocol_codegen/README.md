# Component protocol code generation

The six D2Q9 Taylor–Green arms compare template-generated BGK and TRT
kernels with an independent NumPy reference and fit the viscosity from
energy decay. The [preregistration](predeclared.json) specifies the inputs
and thresholds.

The historical audit reports a worst viscosity error of about 1.571%.
The full preregistered verdict is **FAIL** because the divergence magnitude
exceeds its declared threshold. The analyzer's exit status reports a
selected anchor subset. Read the full verdict and failed criteria in the
generated `summary.json`.

From the repository root:

```bash
for arm in A1 A2 A3 A4 A5 A6; do
  python benchmarks/cross_validation/component_protocol_codegen/run_arm.py --arm "$arm"
done
python benchmarks/cross_validation/component_protocol_codegen/analyze.py
```

See the [published report](../../../docs/codegen_component_protocol.md),
[board audit](../../../docs/board_audit/M17824_XVAL_LBM_COMPONENT_PROTOCOL_BOARD_AUDIT.md),
and [case manifest](benchmark_case.json). The
[migration ledger](../../../docs/architecture/migrations/component-protocol-codegen-xval-2026-11-25.json)
records the original checksums. The
[original harness](https://github.com/cxs503/TensorLBM/tree/32bf22f8074b9232d70fa5675089e7c1756dd9cd/xval/component_protocol_codegen)
is available at its pinned source revision.
