# Round-one layered Poiseuille (partial historical evidence)

This index points to the existing legacy study without moving its incomplete
scientific evidence. The original [one-pager](../../../docs/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY.md)
and [machine-rendered report](../../../xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/REPORT.md) remain at their published paths.

- [Canonical implementation](../../../src/tensorlbm/verification/two_phase_only.py)
  and [historical runner](../../../xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/run_xval.py)
- [Preregistration](../../../xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/predeclared.json) and
  [posthoc addendum](../../../xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/posthoc_addendum.json)
- [Actual tracked quick result](../../../xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/results.json)
  contains **2 arms**, reports a false preregistered aggregate verdict, and cannot
  satisfy the existing minimum of 15 arms.
- [Evidence recovery receipts](../../../docs/architecture/migrations/round1-two-phase-package-authority-2026-11-28.json) record the exact Git objects
  from main, original mission branch, independent audit branch and #2655.
  None of the inspected versions contains the full matrix.
- The [kernel-boundary case](../no_mcmp_kernel_boundary/README.md),
  [dam-break case](../no_mcmp_dam_break/benchmark_case.json), and
  [round-three evolving-interface case](../no_mcmp_phase_field_round3/benchmark_case.json)
  are distinct protocols, not substitutes for the missing full matrix.

The historical full matrix has not been recovered. A separate
[fresh 21-arm replay](evidence/replay-2026-11-28-ca2a9e50/README.md) was executed
on main at `ca2a9e50`: all arms converged, but the original aggregate gate failed
(K6 and K8). This is not historical recovery or a Verified promotion.
Original paths, thresholds, historical failure and compatibility alias remain unchanged.
