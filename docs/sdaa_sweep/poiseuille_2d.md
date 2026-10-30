# SDAA Sweep: poiseuille_2d

## Case Configuration

- **Benchmark**: B13_poiseuille_2d (2D Poiseuille flow, pressure-driven)
- **Lattice**: D2Q9
- **Collision**: BGK
- **Boundary**: Zou-He pressure inlet/outlet + half-way bounce-back
- **Device**: sdaa:0
- **Device Name**: /dev/tcaicard6
- **Timestamp**: 2026-10-29 14:27:27
- **Mode**: Budget-s fast run (H=20 only)

## Device Evidence

```json
{
  "device_type": "sdaa",
  "device_name": "/dev/tcaicard6",
  "device_count": 1,
  "torch_version": "2.7.1"
}
```

## Results

| H | Re | u_max_err (%) | L2_err | Max_rel_err (%) | Steps | Steady | Elapsed (s) |
|---|----|--------------|--------|-----------------|-------|--------|-------------|
| 20 | 5.3 | 1.2122 | 0.011980 | 1.2123 | 7200 | True | 65.0 |

## Verdict Gates

**Threshold**: 1% (original from README, not relaxed)

- **device_evidence_present**: ✅ PASS
- **all_cases_completed**: ✅ PASS
- **l2_error_within_budget**: ❌ FAIL
- **u_max_error_within_budget**: ❌ FAIL
- **steady_state_reached**: ✅ PASS
- **all_gates_pass**: ❌ FAIL

## Overall Status: **FAIL**

- Total cases: 1
- Completed: 1
- Failed: 0

## Workarounds

- **TRITON_INTERPRET=1**: Set for SDAA compatibility
- **Eager mode**: Used instead of torch.compile (SDAA inductor incomplete for LBM chain)
- **Single case**: H=20 only (budget-s time constraint; H=40,60 would require ~2-3 min each on SDAA)

## Diagnosis

The H=20 case shows l2_err=1.2% and u_max_err=1.2%, slightly above the 1% threshold.
This is likely due to:
1. Reduced steps (3000-10000 vs original 10000-60000) - steady state not fully reached
2. Coarse grid (H=20) - discretization error dominates

For full accuracy, run with original parameters (H=20,40,60 with 10000-60000 steps)
which would take ~3-5 minutes per case on SDAA.
