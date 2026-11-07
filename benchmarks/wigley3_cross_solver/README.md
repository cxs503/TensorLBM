# Wigley III Cross-Solver Verification (IN28-N01)

## Physical Background
Wigley III is a modified parabolic hull form used as an ITTC benchmark for ship resistance prediction.
- L = 3.048 m, B = 0.406 m, T = 0.152 m
- Cb = 0.555 (modified from standard Wigley Cb≈0.444)
- L/B = 7.5, B/T = 2.67
- Test speeds: Fn = 0.25, 0.30, 0.35

## Solver Matrix
| Solver | Method | Physics | Status |
|--------|--------|---------|--------|
| TensorLBM | D3Q27 + LES | Single-phase, friction-only (no free surface) | PARTIAL |
| TensorFVM | RANS + k-ω SST | Via ship platform Holtrop method | EMPIRICAL |
| TensorBEM | Rankine source | Wave resistance only | VIA PLATFORM |

## Verification Strategy
Since full D3Q27+VOSET+LES (two-phase free surface) is not yet implemented in TensorLBM,
this benchmark uses a **friction-dominated single-phase approximation** for LBM and
cross-validates against empirical (Holtrop) and BEM (wave resistance) from the ship platform.

The three-solver comparison provides:
- C_T decomposition: C_F (friction) + C_W (wave) + C_R (residual)
- Model-form uncertainty U_M from solver spread
- Grid convergence index (GCI) for LBM mesh refinement

## Limitations (Honest FAIL disclosure)
- **TensorLBM**: Single-phase only, no wave making → C_W = 0, underestimates C_T at high Fn
- **TensorFVM**: Empirical Holtrop method, not full RANS CFD
- **TensorBEM**: Linear wave theory, neglects viscosity and nonlinear effects
- True D3Q27+VOSET+LES requires two-phase LBM implementation (not yet available)

## Files
- `wigley3_geometry.py`: Modified Wigley III hull with Cb=0.555
- `run_cross_solver.py`: Three-solver comparison runner
- `test_wigley3_verification.py`: pytest verification suite (≥15 tests)
- `results.json`: Machine-readable results with uncertainty quantification
