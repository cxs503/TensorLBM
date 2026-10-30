# m467: Blasius Flat-Plate Benchmark Diagnostic

## Mission
Diagnose and attempt to fix the TensorLBM Blasius flat-plate benchmark case.
Previous result (m449): honest-FAIL with C_f error +27.34%

## Verdict
**HONEST-FAIL CONFIRMED**

The LBM with standard half-way bounce-back cannot achieve <3% C_f accuracy due to first-order near-wall resolution limitation.

## Key Findings

### Best Result
- **Run**: R6p (plate400_y1600, TRT collision)
- **C_f error**: +27.34%
- **First cell velocity error**: +27.05%
- **Edge velocity**: u_e/U = 1.07974

### Error Decomposition
1. Edge velocity effect: ~+21% (physical, finite domain height)
2. Near-wall resolution: ~+27% (numerical, half-way bounce-back limitation)
3. After edge correction: +9.23% (still above 3% gate)

### Grid Convergence
| Grid | C_f Error (%) |
|------|---------------|
| plate200 | +47.33 |
| plate400 | +27.38 |
| plate400_y1600 | +27.34 |

Error plateaus at ~27%, confirming BC formulation issue, not grid resolution.

## Root Cause
Standard half-way bounce-back BC has **first-order accuracy** near the wall. The first fluid node at y=0.5 overpredicts velocity by ~27%, directly causing C_f overprediction.

## Correction Methods Tested
1. **Edge velocity correction**: +27.34% → +9.23% (18.11 pp reduction)
2. **Profile fitting**: Not reliable (insufficient near-wall data)
3. **Wall shift**: No significant improvement

## Recommendations
To achieve <3% error:
1. Implement interpolated bounce-back (second-order accurate)
2. Use wall models/wall functions for high-Re flows
3. Use grid stretching with finer near-wall resolution

## Files
- `docs/m467_diagnostic_report.md`: Full diagnostic report
- `docs/m467_summary.json`: Machine-readable summary
- `docs/m467_report.json`: Platform API report
- `_m467_analysis.py`: Error analysis script
- `_m467_corrections.py`: Correction methods testing
- `results/R6p_plate400_y1600_trt.json`: Best case raw data

## Conclusion
This is an honest-FAIL case. The LBM method with standard bounce-back cannot achieve quantitative C_f accuracy. Algorithmic improvements (higher-order BCs or wall models) are required.
