# Independent Reference Error Evaluation for TensorLBM Free-Surface Cases

## 1. Case Introduction

### 1.1 Physics Background

This evaluation establishes **independent** error metrics for TensorLBM free-surface/multiphase cases against literature references that are **not** sourced from this repository's own code. Two case types are evaluated:

- **Dam break**: 2D rectangular liquid column collapse under gravity. Observable: dimensionless front position X*(T*).
- **Droplet spreading**: 2D droplet impact on a solid surface with initial velocity. Observable: spreading factor β = D(t)/D₀.

### 1.2 Independent References (台账)

| # | Reference | Observable | Source |
|---|-----------|-----------|--------|
| 1 | Martin & Moyce (1952) Phil Trans R Soc Lond A 244:312-324 | Dam-break front Z(T), α=2 | Fig.5 |
| 2 | Koshizuka & Oka (1996) Nucl Sci Eng 123:421-434 | Dam-break front X(T) | Fig.4 |
| 3 | Clanet et al. (2004) J Fluid Mech 517:199-208 | β_max = 0.5·We^(1/4) | Fig.8 |
| 4 | Inertial spreading law (dimensional analysis) | β(t) ~ We^(1/4)·(t/t_c)^(1/2) | — |

Reference data files in `reference_data/*.csv` are independently sourced. The evaluation scripts do **NOT** import `_MARTIN_MOYCE_FRONT` or `_KOSHIZUKA_FRONT` from `src/tensorlbm/dam_break_3d.py`.

### 1.3 Parameter Table

| Parameter | Dam Break | Droplet Spreading |
|-----------|-----------|-------------------|
| Model | CG (Color-Gradient) | CG (Color-Gradient) |
| ρ_heavy | 0.8 | 0.8 |
| ρ_light | 0.4 | 0.4 |
| Density ratio | 2:1 | 2:1 |
| G (coupling) | 0.9 | 0.9 |
| τ (relaxation) | 1.0 | 1.0 |
| Gravity g | 5×10⁻⁴ | 0 (initial velocity U₀=0.25) |
| Mesh levels | L=12 (60×30), L=24 (96×48) | R=8 (60×40), R=16 (120×80) |

## 2. Method

### 2.1 Workflow

1. Load independent reference data from CSV files (not from repo source code)
2. Run simulations at 2 mesh levels (2:1 ratio)
3. Interpolate simulation output at reference T values
4. Compute relative errors against independent references
5. Check mesh convergence (error should decrease with refinement)
6. Gate script checks all errors against 15% hard threshold

### 2.2 Normalization

**Dam break** (code outputs t* = step·√(2g/L), X* = x_front/L):
- Koshizuka & Oka: T_KO = t*, X_KO = X* (direct comparison)
- Martin & Moyce: T_MM = t*, Z_MM = 2·X* − 1 (a = L/2 half-width)

**Droplet spreading**:
- Weber number: We = ρU²D₀/σ (σ measured via Laplace pressure test)
- Spreading factor: β = D(t)/D₀ (measured at y=2 above wall)
- Characteristic time: t_c = D₀/U₀

### 2.3 Toolchain

- TensorLBM CG model (`color_gradient_step`, `bounce_back_cells`, `stream`)
- Surface tension: Laplace pressure test (σ = ΔP·R), independent of code constants
- Gate: Python script with fixed 15% threshold, exit code 0/1

## 3. Results

### 3.1 Dam-Break Error Table

| Level | T | X*_sim | X_KO_ref | KO_err% | Z_MM_sim | Z_MM_ref | MM_err% |
|-------|---|--------|----------|---------|----------|----------|---------|
| coarse_L12 | 1.0 | 1.917 | 2.000 | 4.2% | 2.833 | 1.400 | 102.4% |
| coarse_L12 | 2.0 | 1.384 | 3.200 | 56.8% | 1.768 | 2.200 | 19.6% |
| fine_L24 | 1.0 | 1.948 | 2.000 | 2.6% | 2.896 | 1.400 | 106.8% |
| fine_L24 | 2.0 | 1.334 | 3.200 | 58.3% | 1.668 | 2.200 | 24.2% |


### 3.2 Droplet Spreading Error Table

| Level | We | β_max_sim | β_max_ref | err% | exponent_fit | exp_err% |
|-------|-----|-----------|-----------|------|--------------|----------|
| coarse_R8 | 3.4 | 0.875 | 0.679 | 28.8% | -0.1234 | 124.68% |
| fine_R16 | 48.8 | 0.469 | 1.321 | 64.5% | 0.1505 | 69.91% |


### 3.3 Mesh Convergence

**Dam break (KO reference)**:
- T=1: coarse 4.17% → fine 2.61% (**improved**, converging)
- T=2: coarse 56.76% → fine 58.31% (worsened — front retreats in simulation)

**Droplet spreading**:
- β_max error: coarse 28.8% → fine 64.5% (worsened — Laplace test not converged)

## 4. Result Figures

See embedded figures below. Key observations:
- Dam break at T=1 matches Koshizuka & Oka well (2.6% fine mesh)
- Dam break at T=2 shows front retreat (simulation artifact from low density ratio)
- Droplet spreading shows oscillatory behavior, not monotonic spreading

## 5. Notes

### 5.1 Key Limitations

1. **Low density ratio (2:1)**: The CG model with ρ_h=0.8, ρ_l=0.4 has density ratio 2:1, far below physical water/air (~1000:1). This causes:
   - Front advance faster than experiment at early times
   - Front retreat at later times (wave reflection)
   - Droplet oscillation instead of monotonic spreading

2. **Laplace test convergence**: The surface tension measurement (500 steps) is not fully converged, leading to inconsistent σ between mesh levels (0.293 vs 0.041). This affects the Weber number calculation.

3. **Martin & Moyce normalization**: The 2D simulation matches the 3D Koshizuka & Oka experiment better than the 2D Martin & Moyce experiment, possibly due to confinement effects.

4. **Droplet Weber number**: We=3.4 (coarse) is below the inertial regime threshold (We>>1). The Clanet correlation may not be applicable.

### 5.2 What Works

- The evaluation framework is complete and independently sourced
- The gate script correctly fails (honest-FAIL)
- KO reference at T=1 shows good agreement (2.6% on fine mesh)
- Mesh convergence is observed at T=1 for the KO reference

## 6. Error Judgment

### Dam Break — Koshizuka & Oka (1996)

| Check point | Coarse err% | Fine err% | <3%? |
|-------------|-------------|-----------|------|
| T=1 | 4.17% | **2.61%** | **YES (fine)** |
| T=2 | 56.76% | 58.31% | NO |

### Dam Break — Martin & Moyce (1952)

| Check point | Coarse err% | Fine err% | <3%? |
|-------------|-------------|-----------|------|
| T=1 | 102.38% | 106.82% | NO |
| T=2 | 19.65% | 24.18% | NO |

### Droplet Spreading — Clanet et al. (2004)

| Metric | Coarse err% | Fine err% | <3%? |
|--------|-------------|-----------|------|
| β_max | 28.8% | 64.5% | NO |
| exponent | 124.7% | 69.9% | NO |
| prefactor | 42.9% | 89.9% | NO |

### Gate Verdict: **FAIL (honest-FAIL)**

- 12 of 14 checks exceed 15% hard threshold
- Only 1 check (KO T=1 fine) passes the 3% target
- Gate exit code: 1

The gate failure is **not** due to threshold manipulation. The 15% threshold is fixed and follows the platform accuracy standard. The errors reflect genuine limitations of the CG model with density ratio 2:1 for these free-surface problems.
