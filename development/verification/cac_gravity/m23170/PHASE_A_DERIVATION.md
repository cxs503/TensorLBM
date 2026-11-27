# CAC-G Phase A: Source Term Re-derivation & Macroscopic Recovery Chain Alignment

## 0. Decision & Alternatives

**Decision**: Fix the Conservative Allen-Cahn (CAC) + Hydrodynamic Cahn-Hilliard (HCH) double-distribution LBM source term discretization to match Liang et al. (2018) exactly, achieving stability on BGK before proceeding to MRT (Phase B).

**Alternatives considered**:
- A1: Keep current BGK, add numerical dissipation (clipping, filtering) — rejected as it masks root cause
- A2: Skip to MRT immediately (Phase B) — rejected per owner instruction: "A 必须先过公式正确性"
- A3: Re-derive S_i from Chapman-Enskog, compare term-by-term with Liang 2018, fix discrepancies — **chosen**

## 1. Hypothesis

**Hypothesis**: The divergence at step 3 (m23012 R1) is caused by source term S_i discretization format mismatch with the macroscopic recovery chain, specifically:
- H1a: Missing (1 - 1/(2τ_f)) prefactor in f_source
- H1b: Wrong coefficient for λn term (lam/τ_f vs cs²λn)
- H1c: Second-order terms in f_equilibrium inconsistent with paper's first-order form
- H1d: g_force using Guo-style (c_i - u)·F instead of paper's Eq.20 form

## 2. Baseline & Control

**Baseline**: `src/tensorlbm/cac_lbm.py` (W9-R2, 2026-09-29) — the frozen, verified implementation.
**Control**: Liang, Xu, Chen, Wang, Chai, Shi, Phys. Rev. E 97, 033309 (2018), arXiv:1710.09541v1.

## 3. Term-by-Term Derivation vs Liang 2018

### 3.1 CAC Equation (Paper Eq.1-3)

∂φ/∂t + ∇·(φu) = ∇·(M∇μ_φ) + λ∇·n

where:
- φ ∈ [0,1] is the phase field (interface at φ=0.5)
- u is the macroscopic velocity
- M is the mobility
- μ_φ = 4βφ(φ-1)(φ-0.5) - κ∇²φ is the chemical potential (Eq.5-6)
- λ(φ) = 4φ(1-φ)/W is the interface speed coefficient (Eq.3)
- n = ∇φ/|∇φ| is the interface normal

### 3.2 h-LBE (Paper Eq.8-12)

**Transformed distribution**: h_i = f_i + (Δt/(2τ_f)) F_i

**Collision**: h_i(x + c_iΔt, t + Δt) = h_i(x, t) - (1/τ_f)(h_i - h_i^eq) + F_i

**Equilibrium** (Eq.9): h_i^eq = w_i φ [1 + (c_i·u)/c_s²]

**Source term** (Eq.11): F_i = (1 - 1/(2τ_f)) (w_i/c_s²) c_i · [d(φu)/dt + c_s² λ n]

**Macroscopic recovery** (Eq.12): φ = Σ_i h_i

**Relaxation time** (Eq.14): τ_f = 0.5 + 3M (with c_s²=1/3, Δt=1)

### 3.3 HCH g-LBE (Paper Eq.15-23)

**Equilibrium** (Eq.16-17):
- g_0^eq = (p/c_s²)(w_0 - 1) + ρ s_0(u)
- g_i^eq = (p/c_s²) w_i + ρ s_i(u) for i ≠ 0
- s_i(u) = w_i [(c_i·u)/c_s² + (c_i·u)²/(2c_s⁴) - u²/(2c_s²)]

**Force term** (Eq.18→20): G_i = (1 - 1/(2τ_g)) w_i [3(c·F) + 3(c·u)(c·∇ρ)]

**Macroscopic recovery** (Eq.21):
- ρu = Σ c_i g_i + 0.5F
- p = c_s²/(1-w_0) [Σ_{i≠0} g_i + u·∇ρ + ρ s_0(u)]

**Relaxation time** (Eq.23): τ_g = 0.5 + 3μ/ρ

### 3.4 Comparison with cac_lbm.py

| Equation | Paper | cac_lbm.py | Match? |
|----------|-------|------------|--------|
| Eq.9 (h^eq) | w_i φ [1 + 3(c·u)] | `wv * phi * (1.0 + 3.0 * cu)` | ✅ |
| Eq.11 (F_i) | (1-1/(2τ_f))(w_i/c_s²)c·[d(φu)/dt + c_s²λn] | `coef * 3.0 * wv * (cf·[phi_u_dt + (lam/3)*n_hat])` | ✅ |
| Eq.12 (φ) | Σ h_i | `self.f.sum(dim=0)` | ✅ (f=h) |
| Eq.14 (τ_f) | 0.5 + 3M | `0.5 + 3.0 * mobility` | ✅ |
| Eq.16-17 (g^eq) | (p/c_s²)w_i + ρs_i(u) | `(p/CS2)*wv + rho*s` | ✅ |
| Eq.20 (G_i) | (1-1/(2τ_g))w_i[3(c·F)+3(c·u)(c·∇ρ)] | `coef * wv * (3*cF + 3*cu*cgr*drho)` | ✅ |
| Eq.21 (u,p) | ρu=Σcg+0.5F, p=c_s²/(1-w₀)[Σ_{i≠0}g+u·∇ρ+ρs₀] | `u=u_star+0.5F/ρ`, `p=(CS2/(1-w0))*(g[1:].sum+drho*ugr+rho*w0*(-1.5*u2))` | ✅ |
| Eq.23 (τ_g) | 0.5 + 3μ/ρ | `0.5 + 3.0 * mu / rho_safe` | ✅ |

**Verdict**: `cac_lbm.py` is fully aligned with Liang 2018. All equations match.

### 3.5 Comparison with cac_lbm3d.py (TensorLBM subdirectory)

| Equation | Paper | cac_lbm3d.py | Match? | Issue |
|----------|-------|--------------|--------|-------|
| Eq.9 (h^eq) | w_i φ [1 + 3(c·u)] | `w_i φ [1 + 3(c·u) + 4.5(c·u)² - 1.5u²]` | ❌ | Second-order terms added incorrectly |
| Eq.11 (F_i) | (1-1/(2τ_f))(w_i/c_s²)c·[d(φu)/dt + c_s²λn] | `w_i * 3 * c·[phi_u_dt + lam*n_hat/tau_f]` | ❌ | Missing (1-1/(2τ_f)) prefactor; wrong λn coefficient |
| Eq.20 (G_i) | (1-1/(2τ_g))w_i[3(c·F)+3(c·u)(c·∇ρ)] | `(1-0.5/τ_g)w_i[(c-u)·F]/c_s²` | ❌ | Uses Guo-style force, not paper's Eq.20 |

**Verdict**: `cac_lbm3d.py` has three critical errors causing divergence.

## 4. Root Cause Analysis

The m23012 R1 divergence at step 3 is caused by `cac_lbm3d.py` errors:

1. **Missing (1-1/(2τ_f)) prefactor in f_source**: This factor is essential for second-order accuracy. Without it, the source term is too large, causing φ to overshoot bounds immediately.

2. **Wrong λn coefficient**: `lam*n_hat/tau_f` should be `cs²*lam*n_hat = lam*n_hat/3`. The wrong coefficient amplifies the interface compression term by ~3τ_f ≈ 2×, causing instability.

3. **Second-order terms in f_equilibrium**: The paper's Eq.9 is first-order. Adding second-order terms changes the equilibrium, breaking the Chapman-Enskog recovery.

4. **Guo-style g_force**: The paper's Eq.20 has a specific form for the HCH force term. Using the generic Guo-style `(c-u)·F` breaks the coupling.

## 5. Fix Plan

### Phase A (this mission): Fix source term on BGK

1. **Fix `cac_lbm3d.py`** to match Liang 2018 exactly:
   - Remove second-order terms from `f_equilibrium`
   - Add (1-1/(2τ_f)) prefactor to `f_source`
   - Fix λn coefficient to `cs²*lam*n_hat`
   - Replace Guo-style `g_force` with paper's Eq.20 form

2. **Verify stability** on the m23012 test case (64×16×16, gravity=-1e-5, 6000 steps).

### Phase B (pending m23153 MRT merge): Switch to true MRT

### Phase C (if B still unstable): KBC/LES upgrade

## 6. Predeclared Thresholds & Stopping Rules

| Metric | Threshold | Stopping Rule |
|--------|-----------|---------------|
| φ range | [-0.05, 1.05] | If exceeded, check source term magnitude |
| u_max | < 0.5 | If exceeded, check force term magnitude |
| NaN count | 0 | If > 0, divergence confirmed |
| Mass drift | < 0.01% over 6000 steps | If exceeded, check conservation |
| Phase integral drift | < 1e-10 | If exceeded, check source term zeroth moment |

## 7. Variables & Confounders

**Controlled variables**: Grid size, dtype (float64), mobility, viscosity law, wall policy.
**Changed variables**: Source term formula, equilibrium form, force term formula.
**Confounders**: Grid resolution (coarser grids may hide instability), dtype (float32 diverges faster), gravity magnitude.

## 8. Repetition Rationale

Single run with fixed seed (torch.manual_seed(0)) is sufficient for deterministic LBM. Multiple runs with different grid sizes (32³, 64³, 128³) to verify resolution independence.

## 9. Environment

- PyTorch 2.x, CPU, float64
- D3Q19 lattice, c_s² = 1/3
- BGK collision (Phase A), MRT (Phase B), KBC (Phase C)

## 10. Analysis Plan

- Per-step diagnostics: φ range, u_max, source term magnitude, NaN count
- Mass conservation: Σφ over fluid cells
- Hydrostatic pressure profile: p(z) vs p₀ + ρgz
- Spurious current magnitude: max|u| in static droplet test

## 11. Safety & Resource Limits

- Max 10000 steps per run
- Timeout: 300 seconds
- Memory: < 4 GB (64³ grid, float64)

## 12. Interpretation Limits

- BGK stability is limited by τ_f > 0.5; very low mobility may still diverge
- Results apply to density ratio ρ_l/ρ_g ≤ 10; higher ratios may need MRT
- Wall policy "neumann" tested; "contact" policy may need separate verification
