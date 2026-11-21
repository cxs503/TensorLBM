# KBC Collision Cd Anomaly — Root Cause Analysis

## Summary

The entropic KBC collision operator produces abnormally high Cd (18.69/27.87 vs
Schiller-Naumann reference ~1.09) in sphere flow simulations.  This document
identifies two root causes and several contributing factors, confirmed by TDD
tests and a diagnostic runner on a 16³ grid.

## Root Causes

### 1. Admissibility Domain Expansion Bug (Primary)

**Location**: `entropic_kbc.py`, `solve_gamma_entropy()`, lines 247–248

```python
# Ensure gamma_init is within [lower, upper]   ← COMMENT IS WRONG
gamma_lower = torch.minimum(gamma_lower, gamma_init)
gamma_upper = torch.maximum(gamma_upper, gamma_init)
```

**Bug**: The comment says "Ensure gamma_init is within [lower, upper]", but the
code does the **opposite** — it expands `[lower, upper]` to include `gamma_init`.
When `gamma_init` is outside the natural admissibility (positivity) domain, this
places the bisection search in regions where populations are negative.

**Consequences** (confirmed by tests in `TestAdmissibilityDomainBug`):
1. **dH/dγ sign reversal**: In 12/64 cells, `dH/dγ < 0` at the expanded upper
   boundary (should be > 0).  This violates the bisection's sign assumption,
   causing convergence to a non-optimal γ.
2. **Negative populations**: 30/304 cells have `f* < 0` after collision (float64,
   seed=99, δ=0.02).
3. **H-theorem violation**: 6/64 cells have `H(f*) > H(f)`, with max excess
   0.131 (NOT a float32 precision issue — confirmed in float64).
4. **Extreme γ values**: In the 16³ sphere flow, γ ranges from −5.51 to +5.20
   at step 4, amplifying the shear mode unphysically.

**Fix**: Clamp `gamma_init` to the natural admissibility domain instead of
expanding the domain:
```python
gamma_init = torch.clamp(gamma_init, gamma_lower, gamma_upper)
```
Then use `[gamma_lower, gamma_upper]` (natural bounds) as the bisection domain.

### 2. Higher-Order Mode Retention (Secondary)

**Location**: `entropic_kbc.py`, `collide_kbc_d3q19()` / `collide_kbc_d3q27()`,
post-collision formula:

```python
return feq + gamma.unsqueeze(0) * s + h   # h is fully retained (γ_h = 1)
```

**Issue**: The higher-order non-equilibrium mode `h` is **fully retained** (not
relaxed).  In standard KBC, `h` should also be relaxed (typically with the BGK
rate `1 − 1/τ` or an entropy-optimal rate).

**Consequences** (confirmed by `TestHModeRetention` and diagnostic):
1. `h` does not decay over multiple collisions (test: ratio > 0.5 after 5 steps).
2. In sphere flow, boundary conditions (bounce-back, Zou-He) generate
   higher-order non-equilibrium at each step.  Since `h` is not relaxed, it
   accumulates near the obstacle, creating a persistent non-physical
   non-equilibrium "cloud".
3. The accumulated `h` distorts the entropy landscape, causing the γ-solve to
   find extreme values.
4. The contaminated populations near the obstacle contribute spurious momentum
   to the momentum-exchange force calculation, inflating Cd.

## Contributing Factors

### 3. Low τ (Near Stability Limit)

With the default sphere flow parameters (Re=100, u_in=0.05, radius=4.0):
- ν = 0.004, τ = 0.512 (very close to 0.5 stability limit)
- γ₀ = 1 − 1/τ = −0.953 (strong anti-relaxation)

This means the KBC collision barely relaxes the shear mode, and the entropy
solve can find even more extreme γ values.

### 4. Force Calculation Sensitivity

The momentum-exchange force `F_α = 2 Σ_{solid} Σ_i c_{iα} f_i` is computed
from post-stream populations at obstacle cells.  When KBC produces negative or
contaminated populations near the obstacle, these propagate through streaming
and contribute spurious momentum to the force, directly inflating Cd.

## Diagnostic Evidence

### 16³ Grid, 20 Steps (KBC vs BGK)

| Metric | KBC | BGK | Reference |
|--------|-----|-----|-----------|
| Final Cd | 22.06 | −3.03 | 1.09 |
| H-theorem violations (step 20) | 105 cells | — | 0 |
| Max H excess | 5.96e-8 | — | 0 |
| γ range (step 4) | [−5.51, 5.20] | — | [−1, 1] |
| h_norm (step 20) | 7.16e-3 | — | → 0 |
| Negative populations | 0 (float32) | 0 | 0 |

Note: Both KBC and BGK produce unrealistic Cd on this tiny grid/short run, but
KBC is significantly worse due to the identified bugs.

### Float64 Isolated Test (4³, δ=0.02, τ=0.6)

| Metric | Value |
|--------|-------|
| γ_init outside natural domain | 34/64 cells |
| dH/dγ sign reversal at expanded upper | 12/64 cells |
| H-theorem violations | 6/64 cells |
| Max H excess | 0.131 |
| Negative populations | 30/304 |
| γ range | [−1.78, 1.89] |

## Files

| File | Purpose |
|------|---------|
| `src/tensorlbm/entropic_kbc.py` | KBC collision (audited, NOT modified) |
| `src/tensorlbm/kbc_diagnostic.py` | **NEW** — diagnostic runner |
| `tests/test_kbc_diagnostic.py` | **NEW** — 20 TDD tests |
| `artifacts/kbc_diagnostic_16cube_20steps.json` | **NEW** — diagnostic artifact |

## Test Results

```
tests/test_kbc_diagnostic.py: 20 passed
tests/test_sphere_cross_validation.py: 13 passed (no regression)
```

## Recommended Fixes (Not Applied — Diagnostic Only)

1. **Fix admissibility domain**: Clamp `gamma_init` to natural bounds instead
   of expanding bounds.  This ensures the bisection always searches within the
   positivity domain.

2. **Relax higher-order mode**: Change post-collision to
   `f* = feq + γ·s + (1−1/τ)·h` or use an entropy-optimal γ_h for h.

3. **Increase τ**: Use a larger grid or lower Re to keep τ > 0.55, avoiding
   extreme γ₀ values.
