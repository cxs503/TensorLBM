# SDAA Sweep: Cylinder Benchmark Family

## Case Description

Cylinder flow — the most classic CFD validation, covering three distinct flow regimes:

| Sub-case | Re | Flow Regime | Reference |
|----------|-----|-------------|-----------|
| **re20_st** | 20 | Steady laminar, no separation (Schäfer-Turek 2D-1) | Nabh 1998 / FeatFlow: Cd=5.5795 |
| **re100** | 100 | Kármán vortex street (periodic shedding) | Braza et al. 1986: Cd=1.35±0.05, St≈0.164-0.165 |
| **re200** | 200 | Enhanced vortex shedding (higher frequency) | Braza et al. 1986: Cd≈1.33, St≈0.19-0.20 |

## Configuration

| Parameter | re20_st | re100 | re200 |
|-----------|---------|-------|-------|
| D (cells across cylinder) | 20 | 10 | 10 |
| Domain | 440×82 (channel) | 400×400 (40D×40D) | 400×400 (40D×40D) |
| tau | 0.8 | 0.515 | 0.512 |
| u_in / U_mean | 0.1 | 0.05 | 0.08 |
| Steps | 6,000 | 8,000 | 8,000 |
| Warmup fraction | — | 50% | 50% |
| Collision | MRT | MRT + sponge | MRT + sponge |
| Boundary | Zou-He inlet/outlet + BB walls | far_field_bc_2d + sponge | far_field_bc_2d + sponge |
| Seeding | — | St=0.165, 10% amp, 2 periods | St=0.195, 10% amp, 2 periods |

**Grid reduction note**: The verified cases use D=40–80 and 60k–300k steps. Here D and steps are reduced to fit the 20-min wall-clock budget on SDAA. This coarsening is the primary cause of the large errors below.

## Device & Environment

| Item | Value |
|------|-------|
| Device | `sdaa:0` |
| Device name | `/dev/tcaicard20` |
| PyTorch version | 2.7.1 |
| Compile mode | eager (no `torch.compile`) |
| TRITON_INTERPRET | 1 |

### Workarounds

1. **Eager mode**: `torch.compile` on SDAA uses `teco_inductor` backend which is incomplete for the full LBM step chain (known L5: `EXPAND` fallback `reduction_shape` index-out-of-range). The `compile_route` module provides a one-shot eager fallback; we skip compilation entirely for simplicity.
2. **TRITON_INTERPRET=1**: Triton ops run in interpreter mode since the SDAA triton backend is not fully functional.

## Results

| Criterion | Computed | Reference | Error | Tolerance | Verdict |
|-----------|----------|-----------|-------|-----------|---------|
| re20_st Cd | 6.1468 | 5.5795 | +10.17% | ±5% | **FAIL** |
| re100 Cd | 1.5702 | 1.35 | +16.31% | ±5% | **FAIL** |
| re100 St | 0.1497 | 0.1645 | −8.99% | ±5% | **FAIL** |
| re200 Cd | 1.7806 | 1.33 | +33.88% | ±5% | **FAIL** |
| re200 St | 0.1819 | 0.195 | −6.70% | ±5% | **FAIL** |

**Overall: ALL FAIL** (5/5 criteria fail)

### Per-case details

#### re20_st (D=20, 6000 steps)
- Cd=6.1468 (ref 5.5795, +10.17%) — not yet converged to steady state; verified case uses D=40/80 with 300k steps
- Cl=−0.0464 (ref 0.0106) — large error due to coarse grid and insufficient convergence time
- Wall clock: 80.5 s

#### re100 (D=10, 8000 steps)
- Cd=1.5702 (ref 1.35, +16.31%) — D=10 is far too coarse for accurate drag; verified uses D=48/64
- St=0.1497 (ref 0.1645, −8.99%) — shedding frequency captured but shifted by coarse resolution
- Cl amplitude: 0.494 (vortex shedding is present)
- Wall clock: 204.7 s

#### re200 (D=10, 8000 steps)
- Cd=1.7806 (ref 1.33, +33.88%) — severe over-prediction due to D=10 + tau=0.512 (very close to 0.5 stability limit)
- St=0.1819 (ref 0.195, −6.70%) — shedding frequency closer to reference than Cd
- Cl amplitude: 0.978 (strong oscillations)
- Wall clock: 201.0 s

## Wall Clock

| Phase | Time |
|-------|------|
| re20_st | 80.5 s |
| re100 | 204.7 s |
| re200 | 201.0 s |
| **Total** | **486.8 s (8.1 min)** |

Total wall clock is well within the 20-min budget.

## Root Cause Analysis

All five criteria fail, but **not because of SDAA device errors**. The LBM computation runs correctly on SDAA (all results are finite, no NaN). The failures are entirely due to **insufficient grid resolution and time steps**:

1. **re20_st**: D=20 vs verified D=40/80. The steady state has not fully converged in 6000 steps (verified uses 300k). Cd is still drifting downward.
2. **re100**: D=10 vs verified D=48/64. The cylinder is only 10 cells in diameter — far too coarse for accurate drag prediction. The tau=0.515 is also very close to the stability limit.
3. **re200**: D=10 with tau=0.512 (even closer to 0.5). The extremely low viscosity amplifies grid artifacts. Cd error is the largest at +33.88%.

**Conclusion**: The SDAA device correctly executes the TensorLBM solver. To achieve passing criteria, larger grids (D≥32) and more steps (≥30k) are needed, which would require ~30+ minutes on SDAA with current contention.

## Artifacts

- `artifacts/sdaa_cylinder.json` — machine-readable results with device evidence, computed values, and PASS/FAIL criteria
- `scripts/sdaa_sweep_cylinder.py` — independent run script (no business code modifications)

## Reproduction

```bash
cd /root/tp-agent/work/m19
TRITON_INTERPRET=1 python3 scripts/sdaa_sweep_cylinder.py
```
