# Mission 20 Backfill Report — SDAA Sweep: cylinder_3d (3D Extruded Circular Cylinder, Re=40)

> **Mission ID**: 20 · **Repo**: TensorLBM · **Date**: 2026-10-29  
> **Verdict**: **honest-FAIL** (0/3 gates passed — step-budget limitation, not solver defect)  
> **Device**: SDAA (`sdaa:0`, `/dev/tcaicard0`, Torch-SDAA 3.1.1a3, Driver 3.1.0)

---

## 1. Purpose

SDAA (Sunway Deep-learning Accelerator Architecture) verification sweep for the `cylinder_3d` benchmark case in `benchmarks/verified/cylinder_3d`. The case simulates steady flow past a 3D extruded circular cylinder at Re=40 (infinite span, z-periodic) to validate the LBM solver's drag prediction against the classical numerical reference cluster.

**Primary objective**: Confirm that the TensorLBM solver, running on SDAA hardware in eager mode, reproduces the reference drag coefficient Cd≈1.50 within a 3% tolerance across two grid resolutions (D=20 and D=40), and that grid-convergence span is ≤3%.

**Secondary objectives**:
- Verify device placement (all tensors on `sdaa:0`)
- Record compile-mode behavior (eager fallback vs. teco-inductor)
- Document Triton dependency (or lack thereof)
- Produce machine-readable artifact (`sdaa_cylinder_3d.json`) for traceability

---

## 2. Method

### 2.1 Source Case & Reference

The source case is `benchmarks/verified/cylinder_3d`, which has established pass criteria on NVIDIA GPUs. The reference Cd value is drawn from the 2D free-stream numerical cluster:

| Source | Cd |
|---|---|
| Dennis & Chang 1970 | 1.522 |
| Fornberg 1985 | 1.498 |
| Takami & Keller 1969 | 1.48 |
| **Adopted reference** | **1.50** |

**Convention note**: The reference is the 2D numerical cluster, NOT Tritton 1959 (finite-span experiment).

### 2.2 Sweep Script

An independent script `scripts/sdaa_sweep_cylinder_3d.py` was written (no business code modified). It:
1. Probes SDAA device availability via `torch.sdaa.get_device_name(0)`
2. Places all tensors on `sdaa:0`
3. Runs two grid configurations with the LBM step chain: `collide_bgk3d` → `bounce_back_cells_3d` → `stream3d_roll` → `far_field_bc_3d`
4. Computes Cd via Ladd MEM (Momentum Exchange Method) restricted to surface solid cells
5. Evaluates 3 pass/fail gates

### 2.3 Configuration

The README recommends 30,000 steps for plateau convergence. On this SDAA host in eager mode, stepping rates are ~60 ms/step (D=20) and ~155 ms/step (D=40). The full recommendation would require ~108 min, far exceeding the 20-minute wall-clock budget. A reduced configuration was used:

| Grid | D | nx×ny×nz | Lateral | Blockage | Steps | Sample interval | τ |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 640×640×1 | 32D | 3.125% | 10,000 | 500 | 0.62 |
| 2 | 40 | 1280×1280×1 | 32D | 3.125% | 4,000 | 500 | 0.74 |

**Common parameters**: u_in = 0.08, Re = 40, nu = u_in·D/Re.

### 2.4 Compile Mode & Triton

- **Compile mode**: Eager (SDAA teco-inductor backend fails on the LBM step chain — same fallback as `run.py`). Bit-identical results.
- **Triton**: v3.4.0 installed but **not required**. The eager step chain uses pure PyTorch ops; no triton kernels are invoked. `TRITON_INTERPRET` not set.

---

## 3. Key Results

### 3.1 Per-Grid Drag Coefficient (Surface MEM — Primary Caliber)

| Grid | D | Steps | Cd_mem_surface | err vs Cd_ref=1.50 | Cd_mem_all (final) | Cd_mem_interior (final) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 10,000 | **1.5993** | **+6.62%** | 1.5812 | 0.0434 | 598 |
| 2 | 40 | 4,000 | **1.7437** | **+16.25%** | 1.6703 | 0.0673 | 622 |

**Grid span**: |1.7437 − 1.5993| / 1.50 × 100 = **9.62%**

### 3.2 Convergence History (Cd_mem_surface by sample)

**D=20** (sampled every 500 steps, 20 samples):

| Step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Cd | 1.941 | 1.779 | 1.698 | 1.649 | 1.619 | 1.598 | 1.583 | 1.573 | 1.565 | 1.557 |

| Step | 5500 | 6000 | 6500 | 7000 | 7500 | 8000 | 8500 | 9000 | 9500 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Cd | 1.552 | 1.548 | 1.545 | 1.543 | 1.541 | 1.540 | 1.540 | 1.538 | 1.538 | 1.538 |

**D=40** (sampled every 500 steps, 8 samples):

| Step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|
| Cd | 1.986 | 1.877 | 1.790 | 1.726 | 1.695 | 1.650 | 1.623 | 1.603 |

**Observation**: Cd is monotonically decreasing in both grids but has **not plateaued**. The D=20 last-sample Cd (1.538) is within 2.5% of reference — but the window average over all 20 samples (1.5993) is pulled up by the early transient. D=40 is even further from convergence at only 4,000 steps.

### 3.3 Mass Conservation

| Grid | Mass drift (%) |
|---|---|
| D=20 | 0.035% |
| D=40 | 0.017% |

Both grids maintain excellent mass conservation (< 0.04% drift), confirming numerical stability.

### 3.4 Gate Results

| Gate | Description | Value | Tolerance | Result |
|---|---|---|---|---|
| G1 | D=20 cd_mem_surface within 3% of 1.50 | Cd=1.5993, err=+6.62% | ≤3% | **FAIL** |
| G2 | D=40 cd_mem_surface within 3% of 1.50 | Cd=1.7437, err=+16.25% | ≤3% | **FAIL** |
| G3 | Grid span ≤ 3% | span=9.62% | ≤3% | **FAIL** |

**Overall verdict: FAIL** (0/3 gates passed)

### 3.5 Companion SubOFF Benchmarks (from same SDAA campaign)

| Benchmark | Collision | Re | Grid | Cd_total | Cf_ref | Error% | Verdict |
|---|---|---|---|---|---|---|---|
| B1: 4L+Cumulant | cumulant | 1,000 | 320×120×120 | 0.04179 | 0.04200 (Blasius) | **0.50%** | PASS |
| B2: RANS k-ε | rans_ke | 100,000 | 200×80×80 | 0.00662 | 0.00833 (ITTC-57) | **20.5%** | FAIL |
| B3: RANS k-ε | rans_ke | 10,000 | 200×80×80 | 0.01065 | 0.01875 (ITTC-57) | **43.2%** | FAIL |

---

## 4. Audit

### 4.1 Device Evidence

| Field | Value |
|---|---|
| Device requested | `sdaa:0` |
| Device type | `sdaa` |
| Device name | `/dev/tcaicard0` (via `torch.sdaa.get_device_name(0)`) |
| Probe tensor device | `sdaa:0` |
| Probe OK | `true` |
| Torch-SDAA | 3.1.1a3+gite3971b9 |
| SDAA Driver | 3.1.0 |
| TecoDNN | 3.1.0 |

### 4.2 Workaround / Bypass Record

| Item | Status | Detail |
|---|---|---|
| Compile mode | Eager fallback | SDAA teco-inductor fails on LBM step chain (5th backend defect, per README). `compile_route` auto-falls back to eager — bit-identical results. Same as original `run.py`. |
| Triton | Not required | Eager step chain uses pure PyTorch ops. No triton kernels in `collide_bgk3d`, `stream3d_roll`, `bounce_back_cells_3d`, `far_field_bc_3d`. `TRITON_INTERPRET` not set. |
| Step reduction | Applied | 10,000 / 4,000 steps (vs 30,000 recommended) to fit 20-min budget. Cd not converged — root cause of FAIL. |

### 4.3 Comparison with Verified Result

The original `result.json` (30,000–40,000 steps on NVIDIA) reports:
- D=20: Cd_mem_surface = 1.5244 (+1.62% vs 1.50) → **PASS**
- D=40: Cd_mem_surface = 1.4852 (−0.99% vs 1.50) → **PASS**

The discrepancy between the SDAA sweep and the verified result is **entirely due to step count**, not device or algorithm defects.

### 4.4 Total Resource Consumption

| Metric | Value |
|---|---|
| Total wall clock | 1220.09 s (20.33 min) |
| D=20 wall clock | 598.16 s |
| D=40 wall clock | 621.87 s |

---

## 5. Limitations & Root Cause

### 5.1 Root Cause of FAIL

The failure is **not** a solver or device defect — it is a **step-budget limitation**:

1. **Insufficient convergence**: The README states "30,000 steps to plateau". At 10,000 steps (D=20) and 4,000 steps (D=40), the Cd is still in the transient decay phase. The D=20 convergence history shows the Cd approaching ~1.538 at the last sample (within 2.5% of reference), but the averaging window includes the unconverged early transient.

2. **Wall-clock constraint**: The 20-minute budget allows only ~10,000 steps for D=20 and ~4,000 for D=40 on this SDAA host in eager mode. The full 30,000-step recommendation would require ~108 minutes.

3. **No device or algorithm defect**: Mass conservation is excellent (< 0.04% drift), no divergence, all tensors on `sdaa:0`. The Cd trajectory is physically correct (monotonic decay toward the reference). The solver, force computation, and boundary conditions are functioning correctly on SDAA.

### 5.2 Known Limitations

- **Eager-mode only**: SDAA teco-inductor compilation fails on the LBM step chain. Performance is ~60–155 ms/step vs. expected compiled-mode speedup. This is a known backend defect (5th occurrence per README).
- **Step budget**: The 20-min wall-clock budget is insufficient for convergence at both grid resolutions. A dedicated 2-hour run would likely yield PASS.
- **RANS model accuracy**: The companion SubOFF RANS benchmarks (B2, B3) show 20–43% error vs ITTC-1957, suggesting the k-epsilon closure needs calibration for external-flow hull forms at these Reynolds numbers.
- **Single-device sweep**: Only `sdaa:0` was used for cylinder_3d; multi-card SubOFF runs used `sdaa:20/21/22`.

### 5.3 Recommendations

1. **Full-convergence rerun**: Execute the original case entry with 30,000 steps per grid on SDAA (expected ~108 min). PASS is expected based on the convergence trajectory.
2. **teco-inductor backend fix**: Prioritize the 5th backend compilation defect to unlock compiled-mode performance on SDAA.
3. **RANS calibration**: Investigate k-epsilon closure coefficients for external-flow configurations at Re=10⁴–10⁵.

---

## Artifacts

| File | Description |
|---|---|
| `artifacts/sdaa_cylinder_3d.json` | Machine output: device evidence, per-grid Cd, convergence history, gate PASS/FAIL |
| `artifacts/suboff_bench1_cumulant_sdaa20.json` | SubOFF B1: 4L+Cumulant Re=1000 on sdaa:20 |
| `artifacts/suboff_bench2_rans_re1e5_sdaa21.json` | SubOFF B2: RANS k-ε Re=1e5 on sdaa:21 |
| `artifacts/suboff_bench3_rans_re1e4_sdaa22.json` | SubOFF B3: RANS k-ε Re=1e4 on sdaa:22 |
| `artifacts/suboff_best_rans_summary.json` | SubOFF combined summary table |
| `artifacts/kbc_diagnostic_16cube_20steps.json` | KBC diagnostic on 16-cube, 20 steps |
| `scripts/sdaa_sweep_cylinder_3d.py` | Independent sweep script (no business code modified) |

## Reproduce

```bash
cd /root/tp-agent/work/m20
PYTHONPATH=src:benchmarks python scripts/sdaa_sweep_cylinder_3d.py
# Output: artifacts/sdaa_cylinder_3d.json
# Wall clock: ~20 min on SDAA eager
```

For full convergence (PASS expected):
```bash
PYTHONPATH=src python benchmarks/verified/cylinder_3d/run.py single 20 --steps 30000 \
    --lateral 32 --nz 1 --device sdaa:0 --compile-mode eager
# ~30 min per grid on SDAA
```
