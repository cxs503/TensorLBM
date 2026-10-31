# Dam-break Validation: Martin & Moyce (1952)

## Overview

This validation suite compares the 2D Color-Gradient LBM dam-break simulation
against the classic experimental data of Martin & Moyce (1952).

**Reference**: Martin, J.C. & Moyce, W.J. (1952). "An experimental study of the
collapse of liquid columns on a rigid horizontal plane". *Phil. Trans. R. Soc. A*
244, 312–324.

## Non-dimensionalisation

| Quantity | Formula | Description |
|----------|---------|-------------|
| $L$ | `dam_width` | Initial column width (lattice cells) |
| $t^*$ | $t \cdot \sqrt{2g/L}$ | Dimensionless time |
| $X^*$ | $x_f / L$ | Dimensionless front position |

Martin & Moyce use $a = L/2$ (half-width), $T = t\sqrt{g/a}$, $Z = x_{f,MM}/a$.
Conversion: $T = t^*$, $Z = 2X^* - 1$ ⟹ $X^*_{\text{ref}} = (Z + 1)/2$.

## Acceptance Criterion

**6-point mean absolute relative error ≤ 6%**

Six evaluation points: $T = 1.0, 1.5, 2.0, 2.5, 3.0, 3.5$.

## Files

| File | Description |
|------|-------------|
| `validate_mm.py` | Standalone validation runner (CG-LBM + comparison) |
| `postprocess.py` | Post-process existing run → report + figures |
| `validation_report.json` | Machine-readable verdict + metrics |
| `front_comparison.png` | Simulation curve vs MM experimental points |
| `phase_sequence.png` | 6-panel phase-fraction snapshot sequence |
| `mm_comparison.csv` | 6-point comparison table |
| `run_cg_100x50/` | Quick test run (unstable CG) |
| `run_cg_80x40/` | Quick test run (unstable CG) |
| `dam_break/mm_val_cg_1600/` | Main validation run (80×40, 1600 steps) |

## Results

| T | X*_sim | X*_ref | err% |
|---|--------|--------|------|
| 1.0 | 1.656 | 1.200 | 37.96% |
| 1.5 | 1.642 | 1.400 | 17.26% |
| 2.0 | 1.500 | 1.600 | 6.25% |
| 2.5 | 1.509 | 1.850 | 18.43% |
| 3.0 | 1.529 | 2.050 | 25.41% |
| 3.5 | 1.550 | 2.250 | 31.11% |

**Mean error: 22.74%** → **FAIL** (criterion: ≤6%)

## Discussion

The CG model at 80×40 resolution produces a front position that oscillates
rather than monotonically increasing, indicating numerical dispersion at this
resolution. The mean error of 22.74% is well above the 6% acceptance criterion.

**Root causes identified:**
1. **Low resolution**: 80×40 grid with dam_width=20 gives only 20 cells across
   the initial column, insufficient to resolve the sharp interface.
2. **CG model limitations**: The color-gradient recoloring step introduces
   numerical diffusion that smears the interface, causing the front detection
   to oscillate.
3. **Front detection sensitivity**: The threshold-based front finder picks up
   noise from the dispersed interface.

**Recommended improvements:**
- Increase resolution to ≥200×100 (dam_width≥50 cells)
- Consider SC model which shows better front tracking in existing benchmarks
- Implement sub-pixel front detection for improved accuracy

## Usage

```bash
# Run validation from scratch (requires GPU or patience on CPU)
PYTHONPATH=src python validation/dam_break/validate_mm.py \
    --nx 200 --ny 100 --dam-width 50 --g 1e-4 \
    --n-steps 8000 --output-dir validation/dam_break/run_cg_200x100

# Post-process existing run
PYTHONPATH=src python validation/dam_break/postprocess.py \
    --run-dir validation/dam_break/dam_break/mm_val_cg_1600 \
    --out-dir validation/dam_break
```
