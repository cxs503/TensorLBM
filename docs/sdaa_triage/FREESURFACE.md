# M12 · Free-Surface 3D (dambreak3d) — SDAA Triage Report

## 1. Case Description

**Benchmark**: 3D dam-break (Martin & Moyce 1952), free-surface VOF LBM (D3Q19).

**Geometry**: Water column `a × a × 2a` against the x=0 wall corner; gravity `-y`;
solid walls on all six faces; rest is gas.

**Parameters**:
| Parameter | Value |
|-----------|-------|
| a (column width) | 8 lattice units |
| g (gravity) | 1e-4 |
| τ (relaxation) | 0.8 |
| ρ_gas | 1.0 (density ratio 1:1) |
| Domain | (40, 20, 16) = 12,800 cells |
| Steps | CPU: 80, SDAA: 100 |
| Output interval | 10 steps |

**Note**: ρ_gas=0.1 (density ratio 10:1) triggers `TopologyTransactionError` at step 2
(fill outside [-0.1, 10]). This is a known code-level mass conservation defect in
`free_surface_topology_transaction.py`. ρ_gas=1.0 is the only stable configuration.

## 2. Method

1. **CPU baseline**: Run `free_surface_step` on CPU for 80 steps, tracking mass
   and height field every 10 steps.
2. **SDAA run**: Identical config on `sdaa:0` for 100 steps.
3. **L2 comparison**: Compute relative L2 norm of 2D height fields `h(x,z)` between
   CPU and SDAA at common steps.
4. **Mass drift**: Track `mass(t)/mass(0) - 1` and normalize to per-1e4-steps rate.

## 3. Results

### Mass Drift

| Device | Steps | Final Drift | Drift/1e4 steps |
|--------|-------|-------------|-----------------|
| CPU    | 80    | -0.0358%    | **-4.476%**     |
| SDAA   | 100   | -0.0350%    | -3.501%         |

### L2 Height Field Error (SDAA vs CPU)

| Step | L2 relative |
|------|-------------|
| 10   | 0.0000%     |
| 20   | 0.0000%     |
| 30   | 0.0000%     |
| 40   | 0.0000%     |
| 50   | 0.0000%     |
| 60   | 0.0000%     |
| 70   | 0.0000%     |
| 80   | 0.0000%     |

**Max L2 = 0.0000%** — Height fields are bit-identical between CPU and SDAA
(deterministic algorithm, same float32 precision).

### Mass Drift Time Series (CPU)

| Step | T (dimless) | Drift (%) |
|------|-------------|-----------|
| 10   | 0.0354      | -0.0167   |
| 20   | 0.0707      | -0.0391   |
| 30   | 0.1061      | -0.0427   |
| 40   | 0.1414      | -0.0268   |
| 50   | 0.1768      | -0.0251   |
| 60   | 0.2121      | -0.0355   |
| 70   | 0.2475      | -0.0393   |
| 80   | 0.2828      | -0.0358   |

## 4. Error Judgment

### Criterion: CPU baseline mass drift ≤ 1%/1e4 steps

**CPU drift = -4.476%/1e4 steps → EXCEEDS 1%/1e4 threshold → HONEST-FAIL**

The SDAA vs CPU L2 criterion (≤1%) is **invalid** because the CPU baseline itself
has unacceptable mass drift. The L2=0% result (bit-identical fields) is technically
correct but meaningless when the underlying simulation is non-conservative.

### Criterion: SDAA vs CPU L2 ≤ 1%

**L2 = 0.0000% → PASS** (but criterion invalid due to CPU baseline failure)

## 5. Root Cause Analysis

The mass conservation defect originates in `free_surface_topology_transaction.py`:
- The `paired_liquid_interface_debit=True` flag enables paired mass exchange, which
  fixes the I→G paired residual (0.515 → -3.8e-6) but leaves a `clamp(0, ρ_liquid)`
  residual that creates spurious mass.
- With ρ_gas=0.1 (density ratio 10:1), the fill field diverges outside [-0.1, 10]
  at step 2, triggering `TopologyTransactionError`.
- With ρ_gas=1.0 (density ratio 1:1), the simulation runs but accumulates ~0.04%
  mass drift in the first 20 steps, then oscillates around -0.035%.

Historical data (from `result.json`):
- a=16, ρ_gas=1.0, 1200 steps: drift = **-28.54%** (caliper_fix_remeasure)
- a=32, ρ_gas=1.0, 566 steps: drift = **-1.36%**

## 6. Action Required

**Priority: Fix CPU-side mass conservation before SDAA comparison is meaningful.**

Specific fixes needed:
1. Eliminate the `clamp(0, ρ_liquid)` mass residual in
   `free_surface_topology_transaction.py` (~line 631).
2. Support ρ_gas < 1.0 (density ratio > 1:1) without topology transaction failure.
3. Re-run this benchmark after fixes to verify drift < 0.1%/1e4 steps.

## 7. Artifacts

- `artifacts/sdaa_dambreak3d.json` — Full results with `mass_drift_pct` and `l2_error` fields
- `artifacts/db3d_intermediate/series_cpu.json` — CPU series data
- `artifacts/db3d_intermediate/series_sdaa_0.json` — SDAA series data
- `artifacts/db3d_intermediate/heights_cpu.npz` — CPU height fields
- `artifacts/db3d_intermediate/heights_sdaa_0.npz` — SDAA height fields
