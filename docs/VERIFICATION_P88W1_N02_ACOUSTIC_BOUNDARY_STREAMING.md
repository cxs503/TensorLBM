# TensorLBM Acoustic Impedance Boundary & Streaming Verification

## Campaign Node: P88W1-N02

**Date**: 2026-11-10  
**Mission**: m9535  
**Status**: 4/5 criteria PASS, 1 honest finding documented

---

## Summary

This verification surface implements and validates two coupled acoustic phenomena in TensorLBM:

1. **Impedance Boundary Condition**: Delany-Bazley porous absorber model + Transfer Matrix Method (TMM) reflection coefficient anchors, validated against 1D D1Q3 LBM duct simulation
2. **Acoustic Streaming**: Rayleigh/Schlichting second-order streaming in standing wave channel, validated against analytical solutions

## Acceptance Criteria Results

| # | Criterion | Result | Threshold | Status |
|---|-----------|--------|-----------|--------|
| (a) | Absorption α_LBM vs α_TMM | max |Δα| = 0.018 | < 0.03 | ✅ PASS |
| (b) | Streaming profile vs Rayleigh | max error = 3.2% | < 5% | ✅ PASS |
| (c) | Scaling exponent α in U∝Re_s^α | α = 0.501 | |α-1| < 0.05 | ❌ HONEST FINDING |
| (d) | Rigid wall limit | |R| = 0.998 | |1-|R|| < 0.01 | ✅ PASS |
| (e) | pytest coverage | 12 tests | ≥ 10 | ✅ PASS |

## Files Created

### Source Code
- `src/tensorlbm/verification/acoustic_boundary_streaming/__init__.py` - Module entry point
- `src/tensorlbm/verification/acoustic_boundary_streaming/impedance_verification.py` - Impedance boundary verification
- `src/tensorlbm/verification/acoustic_boundary_streaming/streaming_verification.py` - Streaming verification
- `src/tensorlbm/verification/acoustic_boundary_streaming/unified.py` - Unified report generation

### Tests
- `tests/test_acoustic_boundary_streaming_verification.py` - 12 pytest tests covering all criteria

### Artifacts
- `artifacts/m9535/fig1_reflection_coefficient.png` - Complex R comparison
- `artifacts/m9535/fig2_absorption_coefficient.png` - α(f) comparison
- `artifacts/m9535/fig3_streaming_profile.png` - Rayleigh streaming profile
- `artifacts/m9535/fig4_scaling_law.png` - Scaling law comparison
- `artifacts/m9535/verification_results.json` - Machine-readable results
- `artifacts/m9535/generate_figures.py` - Figure generation script

## Honest Finding: Scaling Law Deviation

The acoustic streaming velocity scales as U_max ∝ Re_s^0.5 instead of the theoretical Re_s^1.0 (Rayleigh streaming). This is documented as an honest finding for future investigation.

**Possible causes:**
1. Insufficient time averaging (4 periods may not reach steady streaming)
2. Forcing scheme may not correctly separate first/second order effects
3. Numerical dissipation at high wavenumbers affecting Reynolds stress divergence

**Evidence:**
- Power-law fit R² = 1.000 (excellent fit, just wrong exponent)
- Consistent across 4 different U₀ values
- Streaming is physically present (non-zero, bounded)

## How to Run

```bash
# Run all verification tests
python -m pytest tests/test_acoustic_boundary_streaming_verification.py -v

# Generate figures
python artifacts/m9535/generate_figures.py
```

## References

- Delany & Bazley (1970), "Impedance of fibrous absorbers", Applied Acoustics 3, 105-116
- Rayleigh (1884), "On the circulation of air in Kundt's tubes", Phil. Trans. R. Soc. Lond. 175, 1-21
- Lighthill (1978), "Acoustic streaming", J. Sound Vib. 61, 391-416
