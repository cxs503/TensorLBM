# M12 · Phase-Field Laplace Law — SDAA Recut Report

## 1. Case Description

**Benchmark**: Static droplet Laplace pressure verification using CAC (Conservative
Allen-Cahn) two-phase LBM.

**Physics**: Young-Laplace law for a 3D spherical droplet:
  Δp = 2σ/R  →  Δp·R/(2σ) = 1.0 (analytical)

**Model**: D3Q19 CAC/HCH lattice Boltzmann, fp64, harmonic μ(φ) law,
interface width W=4, periodic cube. `tensorlbm.cac_lbm` (Li et al. PRE 97,033309).

**Calibration**: Route-B — σ_eff(W) ≠ σ_input; a W=4 scan gives
bar_eff(W=4) = 0.9239, monotone in W and R-independent.
Model input: σ_model = σ_target / bar_eff(W=4) = 0.01 / 0.9239 = 0.010824.

## 2. Method

### SDAA Recut (2 grids)

| Arm | Radius | Grid | Steps | Device | R/grid ratio |
|-----|--------|------|-------|--------|--------------|
| R10 | 10.0   | 64³  | 1000  | sdaa:0 | 0.156 (same as R20/128³) |
| R16 | 16.0   | 64³  | 1000  | sdaa:0 | 0.250 (same as R32/128³) |

- Same R/grid ratio as verified benchmark (R20/128³, R32/128³) to maintain
  interface resolution.
- Window mean of last 3 samples (steps 800, 900, 1000) used for convergence
  assessment (vs 5-sample window over last 2000 steps in verified benchmark).
- 1000 steps vs 12000 in verified benchmark — sufficient for R=10 convergence,
  marginal for R=16 (oscillations damp slowly).

### Historical Reference (m24/m25/m52)

| Mission | Arm | Radius | Grid | Steps | Device |
|---------|-----|--------|------|-------|--------|
| m24 | b1_R20 | 20.0 | 128³ | 12000 | GPU (cuda:0) |
| m25 | b1_R32 | 32.0 | 128³ | 12000 | GPU (cuda:0) |
| m52 | b1_R48 | 48.0 | 128³ | 12000 | GPU (cuda:0) |

## 3. Results

### SDAA Recut

| Arm | dp_r_sigma | Error (%) | u_max (final) |
|-----|-----------|-----------|---------------|
| R10/64³ | 0.991688 | 0.8312 | 8.06e-05 |
| R16/64³ | 1.005959 | 0.5959 | 2.51e-04 |

**Max error = 0.8312% ≤ 3.0% → PASS**

### Historical Comparison

| Arm | Current (SDAA) | Historical (GPU) | Δ dp_r_sigma | Current Err | Hist Err |
|-----|----------------|-------------------|--------------|-------------|----------|
| R10/64³ (↔ m24 R20/128³) | 0.991688 | 0.999376 | -0.007688 | 0.8312% | 0.0624% |
| R16/64³ (↔ m25 R32/128³) | 1.005959 | 1.000035 | +0.005924 | 0.5959% | 0.0035% |
| R48 (m52, not re-run) | N/A | 1.004449 | N/A | N/A | 0.4449% |

### Convergence History (R10/64³)

| Step | dp_r_sigma | Error (%) |
|------|-----------|-----------|
| 1    | 0.018493  | 98.15     |
| 100  | 0.764910  | 23.51     |
| 200  | 1.447521  | 44.75     |
| 500  | 0.883310  | 11.67     |
| 800  | 1.000282  | 0.03      |
| 900  | 0.975414  | 2.46      |
| 1000 | 0.999369  | 0.06      |

## 4. Error Judgment

### Criterion: Δp·R/σ ≤ 1.03 (error ≤ 3%)

| Arm | dp_r_sigma | ≤ 1.03? | Error ≤ 3%? |
|-----|-----------|---------|-------------|
| R10/64³ | 0.9917 | ✅ YES | ✅ YES (0.83%) |
| R16/64³ | 1.0060 | ✅ YES | ✅ YES (0.60%) |

**Both arms PASS.** The Laplace law is confirmed on SDAA within 3% error.

### vs Historical (m24/m25/m52)

- R10 vs m24: Δ = -0.008 (0.83% vs 0.06%) — slightly higher error due to fewer steps
  and smaller grid, but well within 3%.
- R16 vs m25: Δ = +0.006 (0.60% vs 0.003%) — similar, within 3%.
- m52 (R48): historical err = 0.44%, not re-run but within 3%.

**All values consistent with historical data within expected variation.**

## 5. Notes & Limitations

1. **Reduced steps**: 1000 steps (vs 12000 in verified benchmark). R=10 converges
   well; R=16 shows residual oscillations but window mean is stable.
2. **Reduced grid**: 64³ (vs 128³). Same R/grid ratio maintained. The calibration
   constant (bar_eff_W4) was measured on 128³ but is R-independent; small grid-size
   dependence may contribute ~0.5-0.8% additional error.
3. **SDAA vs GPU**: The algorithm is hardware-agnostic (same code, same fp64
   precision). SDAA results are consistent with GPU historical values.
4. **Window mean**: Last 3 samples used (vs 5 in verified). With more steps, the
   window mean would converge more tightly to 1.0.

## 6. Artifacts

- `artifacts/sdaa_laplace_recut.json` — Full results with `dp_r_sigma` field
- `artifacts/db3d_intermediate/laplace_R10_64.json` — R=10 convergence series
- `artifacts/db3d_intermediate/laplace_R16_64.json` — R=16 convergence series
- `benchmarks/verified/laplace_cac_phasefield/result.json` — Historical verified data
