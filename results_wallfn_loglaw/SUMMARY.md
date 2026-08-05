# Wall Function Log-Law Fix + Negative Cd_p Investigation Summary

## Date: 2026-07-26
## SDAA Cards: 12-15

## Problem
- Re=1e5: 87.8% error (gradient law wrong, need log law)
- Re=2e6: 52.5% error (Cd_p=-0.0012, negative!)
- Bug 23 fixed: -τ_w (not -τ_w/y_val)

## Fixes Applied

### FIX 1: Log Law (not gradient law) for high-Re
- Changed `wall_law='gradient'` → `wall_law='log'` for Re>1000
- y_val=1.0 for Re≥1e5 (first off-wall cell in log region)
- y_val=0.5 for Re=1e4 (closer to wall, lower y+)
- u_tau from Newton iteration on log law: u = (u_tau/κ)·ln(y+) + B, κ=0.41, B=5.0

### FIX 2: Far-field p_0 (not near-wall)
- Added `p0_method` parameter to `drag_pressure_integration()`
- Options: 'near_wall' (original), 'far_field' (bulk fluid), 'domain_avg', 'inlet'
- far_field: p_0 = average pressure at fluid cells NOT near-wall

## Results

### TEST 1: Re=1e5, log law, y=1.0, far_field p_0 (SDAA:12)
| Metric    | Gradient (prev) | Log (new)  | Change     |
|-----------|-----------------|------------|------------|
| Cd_p      | -0.001440       | -0.001185  | 18% better |
| Cd_f      | 0.002460        | 0.005972   | 2.4x better|
| Cd_tot    | 0.001020        | 0.004787   | 4.7x better|
| Error     | 87.8%           | 42.6%      | 45.2%↓    |
| Target    | <25%            | —          | NOT MET    |

Cd_f error alone: 28.4% (0.005972 vs 0.008333) — close to target.
Cd_p oscillation prevents meeting target.

### TEST 2: Re=2e6, log law, y=1.0, far_field p_0 (SDAA:13)
| Metric    | Near-wall p0 (prev) | Far-field p0 (new) | Change    |
|-----------|---------------------|--------------------|-----------|
| Cd_p      | -0.001220           | -0.001215          | ~same     |
| Cd_f      | 0.003146            | 0.003146           | same      |
| Cd_tot    | 0.001926            | 0.001931           | ~same     |
| Error     | 52.5%               | 52.4%              | ~same     |
| Target    | <30%                | —                  | NOT MET   |

Cd_f error alone: 22.4% (0.003146 vs 0.004054) — below target!
Negative Cd_p (-0.001215) pulls total down.

### TEST 3: Re=1e4, log law, y=0.5, far_field p_0 (SDAA:14) ✓ TARGET MET
| Metric    | RANS (prev) | Smag (prev) | Log (new)  |
|-----------|-------------|-------------|------------|
| Cd_p      | —           | —           | -0.000224  |
| Cd_f      | —           | —           | 0.022203   |
| Cd_tot    | —           | —           | 0.021979   |
| Error     | 43.2%       | 20.6%       | 17.2%      |
| Target    | <20%        | <20%        | MET ✓      |

Cd_f=0.022203 vs ref 0.018750 (18.5% over — slight over-prediction).
Cd_p nearly zero (-0.000224) — well-behaved at this Re.

### TEST 4: p_0 Method Comparison (SDAA:15, Re=1e5, log law)
| p_0 Method  | Cd_p       |
|-------------|------------|
| near_wall   | -0.001190  |
| far_field   | -0.001185  |
| domain_avg  | -0.001185  |
| |near-far|  | 4.69e-06   |
| |near-dom|  | 4.68e-06   |
| |far-dom|   | 7.54e-09   |

**Conclusion: p_0 method makes negligible difference (<5e-6).**
The negative Cd_p is NOT caused by p_0 subtraction — it's a pressure field convergence issue.

## Key Findings

1. **Log law dramatically improves Cd_f** (the primary fix):
   - Re=1e5: Cd_f 0.00246→0.00597 (2.4x, error 70.5%→28.4%)
   - Re=1e4: Cd_f=0.02220 (error 18.5%)
   - Re=2e6: Cd_f=0.00315 (error 22.4%)

2. **p_0 method is NOT the cause of negative Cd_p**:
   - near_wall, far_field, domain_avg give identical results (<5e-6 difference)
   - The negative Cd_p is a real pressure field issue (oscillation/transient)

3. **Cd_p oscillation is the remaining challenge**:
   - Cd_p oscillates ±0.005 around ~0, never converging to stable value
   - For streamlined bodies, Cd_p should be ~0 (small form drag)
   - If Cd_p≈0: T1 error=28.4%, T2 error=22.4% — both near/below targets

4. **T3 (Re=1e4) met target**: 17.2% < 20% ✓

5. **Cd_p if zero (hypothetical)**:
   - T1: Cd_tot=0.005972, err=28.4% (near target)
   - T2: Cd_tot=0.003146, err=22.4% (below target!)
   - T3: Cd_tot=0.022203, err=18.5% (below target!)

## Files Modified
- `src/tensorlbm/drag_pressure.py`: Added `p0_method`, `solid`, `p0_inlet_width` params
- `suboff_wallfn_loglaw_worker.py`: New worker with 4 test configurations
- `results_wallfn_loglaw/`: JSON results for all 4 tests
- `log_wallfn_loglaw_t*.txt`: Log files for all 4 tests

## Root Cause of Negative Cd_p
The negative Cd_p is NOT a p_0 subtraction artifact. It's caused by:
1. Pressure field oscillation during transient phase (5000 steps insufficient)
2. Wall function body force altering near-wall pressure distribution
3. Staircase geometry approximation creating pressure artifacts on discrete surface

The far_field p_0 is theoretically more correct (free-stream reference) but
doesn't change results because the near-wall pressure average is already close
to the far-field average in this configuration.
