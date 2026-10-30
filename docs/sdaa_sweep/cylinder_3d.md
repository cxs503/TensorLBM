# SDAA Sweep — cylinder_3d (3D Extruded Circular Cylinder, Re=40)

**Case**: `benchmarks/verified/cylinder_3d` — 3D z-axis extruded circular cylinder, infinite span (z-periodic), Re=40, steady wake.

**Date**: 2026-10-29  
**Device**: SDAA (`/dev/tcaicard0`, Torch-SDAA 3.1.1a3)  
**Verdict**: **FAIL** (3/3 gates failed — Cd not converged within step budget)

---

## 1. Case Summary

| Item | Value |
|---|---|
| Lattice | D3Q19 |
| Collision | BGK |
| Geometry | Circular cylinder extruded along z, periodic z-boundaries |
| Re | 40 (below shedding threshold Re_c≈47, steady wake) |
| Force method | Ladd (1994) momentum-exchange, **surface-only** caliber (wall-adjacent solid cells) |
| Reference Cd | 1.50 (2D free-stream numerical cluster: Dennis&Chang 1.522, Fornberg 1.498, Takami&Keller 1.48) |
| Reference convention | 3D extruded + z-periodic ≡ 2D infinite span; compare to 2D numerical cluster, NOT Tritton 1959 (finite-span experiment ≈1.54) |

## 2. Configuration

The README recommends 30 000 steps with lateral=32, nz=1 for plateau convergence. On this SDAA host (eager mode), stepping rates are ~60 ms/step (D=20) and ~155 ms/step (D=40). The full 30 000-step recommendation would require ~30 min (D=20) + ~78 min (D=40) = ~108 min, far exceeding the 20-minute wall-clock budget.

**Feasible reduced configuration** (within 20 min):

| Grid | D | nx×ny×nz | Lateral | Blockage | Steps | Sample interval | τ |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 640×640×1 | 32D | 3.125% | 10 000 | 500 | 0.62 |
| 2 | 40 | 1280×1280×1 | 32D | 3.125% | 4 000 | 500 | 0.74 |

u_in = 0.08, Re = 40, nu = u_in·D/Re.

**Compile mode**: eager (SDAA teco-inductor backend fails on the LBM step chain — same fallback as `run.py`, recorded as `compile_mode_effective="eager (fallback)"` in the original `result.json`).

**Triton**: The eager step chain (`collide_bgk3d` → `bounce_back_cells_3d` → `stream3d_roll` → `far_field_bc_3d`) uses pure PyTorch ops — no triton kernels are invoked. `TRITON_INTERPRET` is not required. Triton 3.4.0 is installed but unused.

## 3. Results

### Per-grid Cd (surface MEM, primary caliber)

| Grid | D | Steps | Cd_mem_surface | err vs 1.50 | Cd_mem_all (final) | Cd_mem_interior (final) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 10 000 | 1.5993 | +6.62% | 1.5812 | 0.0434 | 598 |
| 2 | 40 | 4 000 | 1.7437 | +16.25% | 1.6703 | 0.0673 | 622 |

**Grid span**: |1.7437 − 1.5993| / 1.50 × 100 = **9.62%**

### Convergence history (Cd_mem_surface by sample)

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

The Cd is monotonically decreasing in both grids but has not plateaued. The D=20 last-sample Cd (1.538) is close to the reference (1.50, +2.5%), but the window average over all 20 samples (1.5993) is pulled up by the early transient. The D=40 grid is even further from convergence at only 4 000 steps.

### Mass conservation

| Grid | Mass drift (%) |
|---|---|
| D=20 | 0.035% |
| D=40 | 0.017% |

Both grids maintain excellent mass conservation (< 0.04% drift), confirming the solver is numerically stable.

## 4. Gate Results

| Gate | Description | Value | Tolerance | Result |
|---|---|---|---|---|
| G1 | D=20 cd_mem_surface within 3% of 1.50 | Cd=1.5993, err=+6.62% | ≤3% | **FAIL** |
| G2 | D=40 cd_mem_surface within 3% of 1.50 | Cd=1.7437, err=+16.25% | ≤3% | **FAIL** |
| G3 | Grid span ≤ 3% | span=9.62% | ≤3% | **FAIL** |

**Overall verdict: FAIL** (0/3 gates passed)

## 5. Root Cause Analysis

The failure is **not** a solver or device defect — it is a **step-budget limitation**:

1. **Insufficient convergence**: The README states "30 000 steps to plateau". At 10 000 steps (D=20) and 4 000 steps (D=40), the Cd is still in the transient decay phase. The D=20 convergence history shows the Cd approaching ~1.538 at the last sample, which is within 3% of the reference — but the averaging window includes the unconverged early transient.

2. **Wall-clock constraint**: The 20-minute budget allows only ~10 000 steps for D=20 and ~4 000 for D=40 on this SDAA host in eager mode. The full 30 000-step recommendation would require ~108 minutes.

3. **No device or algorithm defect**: Mass conservation is excellent (< 0.04% drift), no divergence, all tensors on sdaa:0. The Cd trajectory is physically correct (monotonic decay toward the reference). The solver, force computation, and boundary conditions are functioning correctly on SDAA.

4. **Comparison with verified result**: The original `result.json` (30 000–40 000 steps) reports Cd_mem_surface = 1.5244 (D=20, +1.62%) and 1.4852 (D=40, −0.99%), both within 3%. The discrepancy is entirely due to step count.

## 6. Device Evidence

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

## 7. Workaround / Bypass Record

| Item | Status | Detail |
|---|---|---|
| Compile mode | Eager fallback | SDAA teco-inductor fails on LBM step chain (5th backend defect, per README). `compile_route` auto-falls back to eager — bit-identical results. Same as original `run.py`. |
| Triton | Not required | Eager step chain uses pure PyTorch ops. No triton kernels in `collide_bgk3d`, `stream3d_roll`, `bounce_back_cells_3d`, `far_field_bc_3d`. `TRITON_INTERPRET` not set. |
| Step reduction | Applied | 10 000 / 4 000 steps (vs 30 000 recommended) to fit 20-min budget. Cd not converged — root cause of FAIL. |

## 8. Artifacts

| File | Description |
|---|---|
| `artifacts/sdaa_cylinder_3d.json` | Machine output: device evidence, per-grid Cd, convergence history, gate PASS/FAIL |
| `scripts/sdaa_sweep_cylinder_3d.py` | Independent sweep script (no business code modified) |

## 9. Reproduce

```bash
cd /root/tp-agent/work/m20
PYTHONPATH=src:benchmarks python scripts/sdaa_sweep_cylinder_3d.py
# Output: artifacts/sdaa_cylinder_3d.json
# Wall clock: ~20 min on SDAA eager
```

For full convergence (PASS expected), use the original case entry with 30 000 steps:
```bash
PYTHONPATH=src python benchmarks/verified/cylinder_3d/run.py single 20 --steps 30000 \
    --lateral 32 --nz 1 --device sdaa:0 --compile-mode eager
# ~30 min per grid on SDAA
```
