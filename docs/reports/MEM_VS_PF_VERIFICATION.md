# MEM vs P+F Worker Verification Results (SDAA 12-15)

**Date:** 2026-07-27
**Worker:** mem_vs_pf_worker.py (commit a556dba + Guo fix revert)

## Summary

| # | Case | SDAA | Cd_MEM (err%) | Cd_P | Cd_F | Cd_PF (err%) | Expected MEM | Expected P+F | Match? |
|---|------|------|---------------|------|------|--------------|-------------|-------------|--------|
| 1 | Cylinder D=48 Re=200 | 12 | 1.5353 (18.1%) | 1.2624 | 0.1630 | 1.4254 (9.6%) | ~88% | ~7.6% | P+F close; MEM differs¹ |
| 2 | Couette | 13 | 0.6666 (0.01%) | 0.0 | 0.6666 | 0.6666 (0.01%) | 0.01% | 0.01% | ✅ PASS |
| 3 | Poiseuille | 14 | 42.666 (0.00%) | 0.0 | 39.821 | 39.821 (6.67%) | 0.00% | 6.67% | ✅ PASS |
| 4 | SUBOFF L=80 Re=1000 | 15 | 0.0641 (52.7%) | 0.0061 | 0.0411 | 0.0472 (12.5%) | ~68% | ~6.3% | P+F high; MEM lower² |

## Bug Found & Fixed: Guo Velocity Shift (Bug 39 was a regression)

**Root cause:** The "Bug 39 fix" changed the Guo body-force collision from the correct
original Guo (2002) formulation to a broken shifted-equilibrium formulation.

- **Old (correct):** `u_phys = u + F/(2ρ)` used in BOTH equilibrium AND force term
- **"Fixed" (broken):** `u* = u + τF/ρ` in equilibrium, raw `u` in force term
- **Effect:** The broken version applies (2 - 1/(2τ))× the correct force.
  For τ=1 (Poiseuille), this gives 1.5× force → u_max=0.074 instead of 0.05.

**Fix applied:** Reverted `collide_bgk3d_guo()` to use `u_phys = u + F/(2ρ)` in both
the equilibrium and the force term (original Guo 2002 formulation).

**Verification:** Poiseuille u_max went from 0.074 (48% error) → 0.0493 (1.34% error).
Cd_MEM went from 64.0 (50% error) → 42.67 (0.00% error). ✅

## Discrepancy Notes

### 1. Cylinder MEM: 18.1% vs expected 88%
The verified result (force_methods_results_sdaa12.json) uses:
- **MRT+Smagorinsky** (Cs=0.05) collision, not BGK
- MEM computed **AFTER streaming** (solid cells corrupted by torch.roll wraparound)
- Sampling every step (4000 samples) vs every 50 steps (80 samples)

The mem_vs_pf_worker computes MEM **BEFORE streaming** (correct — solid cells have
proper bounced-back values). This gives a lower, more accurate MEM value.
The 88% error in the verified result is an artifact of post-streaming wraparound.

P+F is closer: 9.6% (BGK) vs 7.6% (MRT+Smag) — difference due to turbulence model.

### 2. SUBOFF: P+F 12.5% vs expected 6.3%, MEM 52.7% vs expected 68%
The verified P+F results use:
- **MRT+Smagorinsky** (Cs=0.05) collision
- **Quadratic extrapolation** for pressure (this worker uses 'none')
- Sampling every step (4000 samples) vs every 50 steps (80 samples)

Verified P+F: bb_fix_suboff_sdaa10.json → Cd=0.0436 (3.76% err, MRT+Smag, quad extrap)
             results_bbfix_retest/suboff_L80 → Cd=0.0444 (5.61% err, standard formula)

The MEM ~68% expected value likely comes from the force_methods_test_worker which
computes MEM after streaming (inflated by wraparound). The mem_vs_pf_worker's 52.7%
(before streaming) is more accurate.

## Parameters Verified

| Case | Parameter | Value | Expected | Match? |
|------|-----------|-------|----------|--------|
| Cylinder | u_in | 0.08 | 0.08 | ✅ |
| Cylinder | nx×ny×nz | 480×192×4 | 480×192×4 | ✅ |
| Cylinder | dpS | 0.6144 | 0.6144 | ✅ |
| Cylinder | from_cylinder | yes | yes | ✅ |
| SUBOFF | L | 80 | 80 | ✅ |
| SUBOFF | R | 4.6674 | 4.6674 | ✅ |
| SUBOFF | u_in | 0.06 | 0.06 | ✅ |
| SUBOFF | dpS | 4.223 | 4.223 | ✅ |
| SUBOFF | from_suboff | yes | yes | ✅ |
| Poiseuille | Guo fix | u+F/(2ρ) | correct | ✅ (reverted) |
| All | JSON float() | _float() | serialized | ✅ |

## Files Modified
- `mem_vs_pf_worker.py`: Fixed `collide_bgk3d_guo()` — reverted to original Guo (2002) formulation

## Files Created
- `mem_vs_pf_cylinder_sdaa12.json` — Cylinder results
- `mem_vs_pf_couette_sdaa13.json` — Couette results
- `mem_vs_pf_poiseuille_sdaa14.json` — Poiseuille results (with Guo fix)
- `mem_vs_pf_suboff_sdaa15.json` — SUBOFF results
- `log_mem_vs_pf_*.txt` — Run logs for all 4 cases
- `run_mem_vs_pf_all.sh` — Launcher script
- `MEM_VS_PF_VERIFICATION.md` — This summary
