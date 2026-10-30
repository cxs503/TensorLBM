# SDAA Sweep: Womersley Oscillating Channel Flow (D2Q9)

## Case Description

Womersley 振荡流 2D 通道解析解 benchmark（`benchmarks/verified/womersley`）。
平面通道壁面 y=0,H，谐波振荡体力 a(t)=a0·cos(ωt) 驱动，解析解为 Womersley (1955) 复数相量形式。

## Configuration

| Parameter | Value |
|---|---|
| Device | `sdaa` (SDAA TECO) |
| Device name | `/dev/tcaicard23` |
| Lattice | D2Q9 |
| Collision | BGK (τ=0.8, ν=0.1) |
| Boundary | Pre-streaming half-way bounce-back (verified/poiseuille_2d pattern), x-periodic |
| Driving | `_apply_body_force_2d`, a(t)=a0·cos(ωt) |
| u_peak target | 0.03 (Ma=0.052) |
| nx | 16 |
| Grid H | 29, 59 (2× ratio; original verified scan uses 59, 119) |
| Alpha | 4.0, 8.0 |
| Measurement | 4 full periods, 8 phase instants |
| Compile mode | `default` → auto fallback to `eager` (teco_inductor incomplete) |
| TRITON_INTERPRET | 1 (precautionary; does not fix inductor backend gap) |

## Compile Bypass Record

`torch.compile` fails on SDAA with `InductorError: AssertionError` in teco
reduction-shape codegen.  The `compile_route` auto-fallback mechanism catches
this and runs every step eagerly (bitwise-identical to `--compile-mode eager`).
`TRITON_INTERPRET=1` is set as a precaution but does not resolve the inductor
backend gap — the failure is in teco's `EXPAND`/reduction lowering, not in
Triton kernel emission.

## Acceptance Criteria (fixed a priori, from README)

1. **C1**: Profile L2 error (fixed-scale ||du(t)|| / max_t' ||u_ref(t')||) ≤ 3% at each of 8 phase instants
2. **C2**: Max pointwise error / u_peak ≤ 3% (8 phases)
3. **C3**: Per-row complex-amplitude |U_num/U_ana − 1| ≤ 3% (|U_ana| ≥ 10% u_peak)
4. **C4**: Error decreases from H to ~2H for both α ∈ {4, 8}

## Results

| H | α | C1 L2(fixed) | C2 Pointwise/u_peak | C3 Complex amp | C1 | C2 | C3 |
|---|---|---|---|---|---|---|---|
| 29 | 4 | 0.419% | 0.415% | 2.934% | PASS | PASS | PASS |
| 59 | 4 | 0.107% | 0.106% | 0.467% | PASS | PASS | PASS |
| 29 | 8 | 1.555% | 1.742% | **4.890%** | PASS | PASS | **FAIL** |
| 59 | 8 | 0.390% | 0.424% | 2.899% | PASS | PASS | PASS |

| α | H=29 L2 | H=59 L2 | Converged? |
|---|---|---|---|
| 4 | 0.419% | 0.107% | **PASS** (4× reduction) |
| 8 | 1.555% | 0.390% | **PASS** (4× reduction) |

**C4: PASS** — both α lines show monotonic error decrease from H=29 to H=59.

## Verdict

**FAIL** — H=29, α=8 fails C3 (complex amplitude 4.890% > 3% threshold).

### Root Cause

The H=29 grid has insufficient near-wall resolution for α=8 (Womersley boundary
layer thickness ~ H/α ≈ 3.6 lattice units at α=8).  The original verified scan
uses H=59 and H=119, where H=59 α=8 complex amplitude is 2.900% (just under 3%)
and H=119 brings it to 0.450%.  At H=29 the first fluid row's phasor error
dominates the complex-amplitude metric.  This is a **genuine resolution
limitation**, not a device or numerics issue — the same H=29 α=8 case would
fail on any device.

The SDAA numerics are validated by:
- H=59 α=8 passing all criteria (2.899% < 3%)
- Convergence being monotonic and consistent with the original CUDA results
- Mass drift ≤ 0.01%, all values finite

## Wall Clock

| Case | Steps | Elapsed |
|---|---|---|
| H=29 α=4 | 9,086 | 85.0 s |
| H=59 α=4 | 37,587 | 299.8 s |
| H=29 α=8 | 6,386 | 25.6 s |
| H=59 α=8 | 26,474 | 114.4 s |
| **Total** | | **524.7 s (8.7 min)** |

Total wall clock ≤ 20 min ✓

## Artifacts

- `artifacts/sdaa_womersley.json` — full machine output with device evidence, per-grid metrics, phase tables, compile status, and PASS/FAIL verdicts
