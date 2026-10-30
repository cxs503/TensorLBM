# SDAA Sweep — maneuvering_steady_turn

## Case

**Benchmark**: `benchmarks/verified/maneuvering_steady_turn` — steady turning
response of a KVLCC2-class large tanker (horizontal-plane linear sway-yaw
model).

**Library under test**: `tensorlbm.maneuvering` — `ManeuveringModel.steady_turn`
(exact 2×2 fixed-point solve), `simulate_turn` (semi-implicit Euler + RK4 time
integration), `analytic_transient` (exact modal transient).

**Status on NVIDIA side**: VERIFIED (2026-10-05). The steady-turn closure
reproduces the exact fixed point to <0.02 % on every level, with inter-level
span below 0.001 %.

## Configuration

| Parameter | Value | Note |
|---|---|---|
| Ship | KVLCC2-class tanker, L=320 m, B=58 m, D=20.8 m, Cb=0.81 | full scale |
| Speed | U = 7.97 m/s (~15.5 kn) | |
| Rudder angles | 10°, 20° | |
| dt levels | coarse = 1.0 s, fine = 0.25 s | |
| T_end | **300 s** | reduced from 600 s (verified benchmark) for sdaa wall-clock feasibility; still >7× slowest time constant (τ_max = 42.1 s) |
| Integrators | semi-implicit Euler, RK4 | |
| Tolerance | 3 % | same as verified benchmark |
| Derivatives | representative KVLCC2-class linear set (caller-supplied) | NOT from LBM flow or PMM experiment |

### Device setup

| Item | Value |
|---|---|
| Device type | `sdaa` |
| Device name | `/dev/tcaicard17` (via `torch.sdaa.get_device_name(0)`) |
| PyTorch version | 2.7.1 |
| Default dtype | `torch.float64` |
| Default device | `sdaa` (via `torch.set_default_device('sdaa')`) |
| Probe tensor device | `sdaa:0` |

### Workarounds

| # | What | Why | Impact |
|---|---|---|---|
| W1 | `torch.Tensor.numpy` monkey-patched to `.cpu().numpy()` for sdaa tensors | `analytic_transient` and cross-route verification call `.numpy()` on tensors created on sdaa; torch does not support `.numpy()` on non-CPU tensors | All `torch.linalg` operations (solve, eigvals, matmul) execute on sdaa. Only the final host-side numpy extraction is routed through CPU. No numerical results affected. |
| W2 | TRITON_INTERPRET not needed | This benchmark uses pure torch + numpy; no triton kernels | None |
| W3 | T_end reduced 600→300 s | sdaa kernel-launch overhead for 2×2 tensor ops makes full config exceed 120 s shell timeout | Still >7× slowest time constant. Steady-state window captures >3.5 e-foldings. All eight criteria remain testable. R_err slightly larger (~0.6 % vs 0.009 %) but well within 3 % tolerance. |

## Results

### Per-case results

| δ (deg) | Level | Integrator | dt (s) | n_steps | R_err (%) | β_err (%) | trans_r_err (%) | wall (s) |
|---|---|---|---|---|---|---|---|---|
| 10 | coarse | Euler | 1.0 | 300 | +0.639 | −0.771 | 0.509 | 1.63 |
| 10 | fine | Euler | 0.25 | 1200 | +0.663 | −0.800 | 0.124 | 7.01 |
| 20 | coarse | Euler | 1.0 | 300 | +0.599 | −0.745 | 0.509 | 1.69 |
| 20 | fine | Euler | 0.25 | 1200 | +0.621 | −0.772 | 0.124 | 7.13 |
| 10 | fine_rk4 | RK4 | 0.25 | 1200 | +0.672 | −0.810 | 5.8e-8 | 20.89 |
| 20 | fine_rk4 | RK4 | 0.25 | 1200 | +0.630 | −0.782 | 5.8e-8 | 21.35 |

### Two-level spans (Euler, per rudder)

| Rudder | R_span (%) | β_span (%) | trans_refinement_ratio |
|---|---|---|---|
| 10° | 0.028 | 0.034 | 4.09 |
| 20° | 0.028 | 0.034 | 4.09 |

### Independent route agreement

| Route | Agreement (%) |
|---|---|
| NumPy solve vs library | 2.79e-14 |
| Nomoto closed form vs library | 2.79e-14 |

### Modal analysis

| Eigenvalue | Time constant (s) |
|---|---|
| −0.02373 | 42.14 |
| −0.13998 | 7.14 |

Directionally stable: **yes** (both eigenvalues negative real).

## Criteria — PASS/FAIL

| ID | Criterion | Threshold | Measured | Result |
|---|---|---|---|---|
| C1 | Each level \|R_err\| ≤ 3 % | ≤ 3 % | max 0.672 % | **PASS** |
| C2 | Each level \|β_err\| ≤ 3 % | ≤ 3 % | max 0.810 % | **PASS** |
| C3 | Two-level span ≤ 3 % (R and β) | ≤ 3 % | max 0.028 % | **PASS** |
| C4 | Independent algebraic routes agree < 1e-6 % | < 1e-6 % | 2.79e-14 % | **PASS** |
| C5 | Euler transient r(t) err at fine ≤ 3 % | ≤ 3 % | 0.124 % | **PASS** |
| C6 | Transient refinement ratio > 3 | > 3 | 4.09 | **PASS** |
| C7 | RK4 transient r(t) err at fine ≤ 3 % | ≤ 3 % | 5.83e-8 % | **PASS** |
| C8 | Directionally stable | all Re(λ) < 0 | max λ = −0.0237 | **PASS** |

**Overall: PASS (8/8)**

## Wall clock

| Component | Time (s) |
|---|---|
| Total (including setup, all 6 cases, analysis) | 59.89 |

Well within the 20-minute budget.

## Notes

* The R_err values (~0.6 %) are larger than the verified benchmark's (~0.009 %)
  because T_end was reduced from 600 s to 300 s. The residual transient at the
  start of the steady-measurement window (t = 150 s, ~3.6 e-foldings of the
  slowest mode) contributes a small bias. This is a configuration reduction,
  not a device accuracy issue — the transient errors (C5, C7) and route
  agreement (C4) are identical to the CPU benchmark, confirming the sdaa
  tensor operations produce numerically identical results.
* The hydrodynamic derivatives are representative KVLCC2-class linear
  coefficients synthesised by the caller — not extracted from an LBM flow field
  and not from a PMM experiment. There is no free surface, no rudder/propeller
  slipstream, and no higher-order nonlinearity. End-to-end fluid-maneuvering
  coupling remains a structural blocker (free surface).

## Artifacts

* Script: `scripts/sdaa_sweep_maneuvering_steady_turn.py`
* Results JSON: `artifacts/sdaa_maneuvering_steady_turn.json`
