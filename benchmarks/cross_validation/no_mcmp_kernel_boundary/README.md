# No-MCMP kernel-boundary study (archived gate evidence)

The [published report](../../../docs/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY-KERNEL-BOUNDARY.md)
describes six gate groups and nine subgates. The archived
[merged result](../../../xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/artifacts/kernel_boundary/results.json)
records 9/9 historical passes; the six gate chunks are listed in the
[case manifest](benchmark_case.json).

The existing [staged driver](../../../xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/run_kernel_boundary.py)
and [canonical gate source](../../../src/tensorlbm/xval/no_mcmp_two_phase.py)
remain at their current paths. Actual offline aggregation of the six real
tracked gate chunks in an isolated temporary copy reproduced the merged JSON
byte-for-byte. No solver was invoked by that aggregation. The original
figures are not tracked in this revision; no replacement was fabricated.

This is not a fresh scientific acceptance of the kernel study or the separate
[round-one layered Poiseuille study](../no_mcmp_layered_poiseuille/README.md).
See the [recovery and compatibility ledger](../../../docs/architecture/migrations/round1-two-phase-package-authority-2026-11-28.json).
