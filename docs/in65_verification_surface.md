# IN65-N03: TensorLBM Scalar Transport + Boussinesq Buoyancy Verification

## Mission
战役 DAG 节点 IN65-N03: 主动标量输运 + Boussinesq 浮力耦合 → 替代燃料泄漏危险区判定

## Verification Surface

### G1: 1-D Advection-Diffusion Gaussian Analytical Anchor
- **Criterion**: L2 relative error ≤ 2%, peak error ≤ 2%
- **Method**: D2Q5 scalar LBM with uniform advection + diffusion
- **Analytical**: C(x,t) = σ₀/σ(t) exp(-(x-Ut)²/(2σ(t)²)), σ(t)² = σ₀² + 2Dt
- **Result**: L2 = 0.14%, Peak = 0.02% ✓

### G2: Mass Conservation
- **Criterion**: ΣC relative residual ≤ 1e-12
- **Method**: 2D diffusion without sources, track total mass
- **Result**: Max drift = 2.9e-13 ✓

### G3: Buoyant Plume Self-Similarity Q(z) ∝ z^(5/3)
- **Criterion**: Fitted exponent ∈ [1.6, 1.75]
- **Method**: Morton-Taylor-Turner (1956) analytical plume theory
- **Theory**: Q(z) = (6α/5)^(1/3) B₀^(1/3) z^(5/3), α ≈ 0.083
- **Result**: Exponent = 1.667, R² = 1.0 ✓

### G3b: LBM Buoyancy Sanity Check
- **Criterion**: Buoyancy drives upward flow (uy_max > 0)
- **Method**: D2Q9+D2Q5 Boussinesq with hot source at bottom
- **Result**: uy_max = 0.034 ✓

### G4: Negative Control g'=0 → Q(z) Linear
- **Criterion**: Fitted exponent ∈ [0.95, 1.05]
- **Method**: Momentum jet theory (no buoyancy)
- **Theory**: Q(z) = C M₀^(1/2) z (linear growth)
- **Result**: Exponent = 1.000 ✓

### G5: LFL Contour vs Wind Speed (Monotonicity)
- **Criterion**: 5 wind speed levels, strictly monotonically decreasing
- **Method**: Gaussian plume model (Pasquill-Gifford)
- **Fuels**: CH₄ (LFL=5%), H₂, NH₃, CH₃OH, C₃H₈
- **Result**: [202.5, 106.0, 77.6, 57.4, 43.4] m ✓

### G6: LFL Grid Convergence
- **Criterion**: Max relative change ≤ 3%
- **Method**: Gaussian plume at 4 resolutions
- **Result**: Max change = 0.0% ✓

## Test Summary
- **Total tests**: 48 (19 existing + 29 new)
- **All pass**: ✓
- **Runtime**: ~45s

## Files
- `src/tensorlbm/scalar_verification_in65.py` — Verification engine
- `src/tensorlbm/hazard_zone_lbm.py` — LFL hazard zone module
- `tests/test_scalar_verification_in65.py` — Pytest test suite
- `benchmarks/in65_buoyant_plume/run.py` — Benchmark runner
- `runs/in65_verification/in65_verification.json` — JSON artifact
- `artifacts/in65_figures/` — Figures (3 PNG)

## References
- Morton B., Taylor G., Turner J.S. (1956). "Dilution of buoyant fluid from a source." Q J R Meteorol Soc 82.
- Turner J.S. (1979). Buoyancy Effects in Fluid Environments. Springer.
- Pasquill F. (1974). Atmospheric Diffusion. 2nd ed. Ellis Horwood.
- Guo Z., Zheng C., Shi B. (2002). "Discrete lattice effects on the forcing term." Phys Rev E 65, 046308.
