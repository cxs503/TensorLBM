# SDAA Sweep: wigley_hydrostatics

## Case Summary

| Field | Value |
|---|---|
| **Case** | `benchmarks/verified/wigley_hydrostatics` |
| **Scope** | Hydrostatic form coefficients of Wigley (1934) parabolic hull (geometry + static closure only; NOT seakeeping / wave-body motion) |
| **Method** | Voxel mask (`wigley_hull_mask`, bitwise-identical to library) integrated in memory-safe blocks |
| **Reference** | Closed-form Wigley (1934) parabolic hull integrals — Faltinsen, Lewis; independently re-derived with sympy |
| **Entry point** | `benchmarks/verified/wigley_hydrostatics/wigley_hydro_full.py` |
| **Sweep script** | `scripts/sdaa_sweep_wigley_hydrostatics.py` |
| **Artifact** | `artifacts/sdaa_wigley_hydrostatics.json` |

## Configuration

| Parameter | Value |
|---|---|
| **Device** | SDAA (`torch.sdaa`) |
| **Device name** | `/dev/tcaicard2` |
| **Device evidence** | `torch.sdaa.get_device_name(0) = '/dev/tcaicard2'` |
| **Grids** | L=640 (1922×195×122), L=960 (2882×291×182) |
| **Triton dependency** | None — pure PyTorch tensor ops |
| **TRITON_INTERPRET** | Not needed (no triton kernels) |
| **Wall clock** | 3.3 s |

## Analytic Reference Values

| Coefficient | Analytic |
|---|---|
| C_B | 4/9 ≈ 0.44444 |
| C_wp | 2/3 ≈ 0.66667 |
| C_M | 2/3 ≈ 0.66667 |
| C_P | 2/3 ≈ 0.66667 |
| KB/T | 5/8 = 0.625 |
| I_T/(B³L) | 4/105 ≈ 0.03810 |
| I_L/(L³B) | 1/30 ≈ 0.03333 |

## Results

### L = 640 (grid 1922×195×122)

| Coefficient | Computed | Analytic | Error (%) |
|---|---|---|---|
| C_B | 0.453218 | 0.444444 | +1.974 |
| C_wp | 0.665552 | 0.666667 | −0.167 |
| C_M | 0.681641 | 0.666667 | +2.246 |
| C_P | 0.664893 | 0.666667 | −0.266 |
| KB/T | 0.631275 | 0.625000 | +1.004 |
| I_T/(B³L) | 0.037822 | 0.038095 | −0.717 |
| I_L/(L³B) | 0.033331 | 0.033333 | −0.008 |

### L = 960 (grid 2882×291×182)

| Coefficient | Computed | Analytic | Error (%) |
|---|---|---|---|
| C_B | 0.450164 | 0.444444 | +1.287 |
| C_wp | 0.665896 | 0.666667 | −0.116 |
| C_M | 0.674826 | 0.666667 | +1.224 |
| C_P | 0.667082 | 0.666667 | +0.062 |
| KB/T | 0.629361 | 0.625000 | +0.698 |
| I_T/(B³L) | 0.037935 | 0.038095 | −0.420 |
| I_L/(L³B) | 0.033309 | 0.033333 | −0.073 |

## Gate Criteria (逐条判据)

| Coefficient | err L=640 (%) | err L=960 (%) | Span (%) | err ≤ 3%? | span ≤ 3%? | **PASS/FAIL** |
|---|---|---|---|---|---|---|
| C_B | 1.974 | 1.287 | 0.687 | ✅ | ✅ | **PASS** |
| C_wp | 0.167 | 0.116 | 0.052 | ✅ | ✅ | **PASS** |
| C_M | 2.246 | 1.224 | 1.022 | ✅ | ✅ | **PASS** |
| C_P | 0.266 | 0.062 | 0.328 | ✅ | ✅ | **PASS** |
| KB/T | 1.004 | 0.698 | 0.306 | ✅ | ✅ | **PASS** |
| I_T/(B³L) | 0.717 | 0.420 | 0.298 | ✅ | ✅ | **PASS** |
| I_L/(L³B) | 0.008 | 0.073 | 0.065 | ✅ | ✅ | **PASS** |

## Summary

| Metric | Value | Threshold | Status |
|---|---|---|---|
| max |err| | 2.246% | ≤ 3% | **PASS** |
| max span | 1.022% | ≤ 3% | **PASS** |
| **Overall** | | | **PASS** |

## Notes

- **No Triton workaround needed**: The benchmark uses pure PyTorch tensor operations (arange, broadcasting, boolean indexing). No `torch.compile` or custom Triton kernels are involved.
- **Voxel staircasing**: At low L (≤320), C_B error is large (+4% at L=320, +21% at L=160). The verified case documents this boundary condition and uses L=640/960 for the gate.
- **Wall clock**: 3.3 s total (1.2 s for L=640, 2.1 s for L=960), well within the 20-minute budget.
- **Device evidence**: All computations ran on SDAA device `/dev/tcaicard2`, confirmed via `torch.sdaa.get_device_name(0)`.
