# XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S — Round 3 (machine-gate evidence)

**Anchor:** `XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S`
**Mission:** m18744 (repair chain m18396 → m18470 → m18643 → m18744)
**Verdict:** **PASS** — pre-declared pointwise metric: D3Q19 main solver **0.31 %** (ny=10) /
**0.12 %** (ny=20); D2Q9 NEBB module at **machine precision** (6e-14). Both < 3 % and < 15 %.

## ① Case / problem + criterion source

2-D Poiseuille channel, periodic in *x*, no-slip top/bottom walls, constant body
force in *x*. Steady fully-developed profile (Batchelor 1967 §4.2, lattice units):
`u(y) = a/(2ν)·s·(H−s)`. Criterion (owner, unchanged since Round 1, per audit
m17963 F2): PASS if pointwise max relative error < 15 %; ecosystem stretch < 3 %.
This round's job (maturity gate): make the evidence **machine-recomputable** —
impl_files + entry + run_artifacts (L2), real main-solver caller (L3),
benchmark ref-vs-got + repro_cmd (L4).

## ② Method / flow / key settings / toolchain

* **Arm A (D2Q9, repo NEBB module):** `BodyForceChannel` — BGK + EDM forcing +
  force-corrected NEBB wall (rows 0/ny-1 are wall nodes), the OSS vivsim recipe
  implemented in-repo. Reference: `analytic_poiseuille(ny, gx, ν, 0)`.
* **Arm B (D3Q19, main solver):** `GPUCorePathSolver` — `collide_guo_into`
  (Guo/Zheng/Shi 2002) + `apply_channel_walls` (half-way bounce-back).
  Reference: `channel_analytic_velocity`.
* **OSS reference (haimingz/vivsim D2Q9):** board-audit rerun m17963,
  independently reproduced; in-repo sha256-pinned copy
  `xval/oss_vivsim_d2q9_reference_m17963.json`
  (`c62ff2bd…f2c197`). `jax` absent ⇒ OSS arm NOT_RUN here (honest scope).
* Settings: ν=0.2 (τ=1.1), u_max=0.05, nx=8, FP64, eager backend, CPU.
  Sensitivity: NY=10 vs NY=20. Toolchain: numpy 2.4.6, torch 2.7.1, matplotlib.

## ③ Parameter table (dimensionless)

| quantity | value | note |
|---|---|---|
| lattice | D2Q9 (arm A) / D3Q19 (arm B) | c_s² = 1/3 |
| ny (grids) | 10, 20 | H = ny−1 (A, offset-0) / ny−2 (B, half-way) |
| ν / τ | 0.2 / 1.1 | |
| u_max target | 0.05 | Ma = 0.087 (low-Mach) |
| Re = u_max·H/ν | 2.0 (ny=10, arm B) | creeping |
| steps | 200 000 cap (A, steady at 1 401/5 201), 8 000 (B) | |

## ④ Comparison table (ref vs got, error ✓/✗) — program-generated, zero hand-copy

| arm | pointwise err ny=10 | ny=20 | u_max err (ny=10) | <15 % | <3 % |
|---|---|---|---|---|---|
| **TensorLBM D2Q9 NEBB module** | **6.0e-14** | 2.6e-13 | 5.6e-14 | ✓ | ✓ |
| **TensorLBM D3Q19 main solver** | **0.308 %** | 0.121 % | 0.19 % | ✓ | ✓ |
| OSS vivsim D2Q9 (NOT_RUN, audit rerun) | 12.5 % | — | 5.0 % | ✓ | ✗ |

No-slip: first fluid row equals the analytic parabola value at that row within
1 % of u_max on **both** arms (`no_slip_d2q9=true`, `no_slip_d3q19=true`).

## ⑤ Convergence / sensitivity (≥ 2 points)

D3Q19: 0.308 % → 0.121 % (ny 10→20), monotonic. D2Q9: machine precision on
both grids (the discrete NEBB steady state sits exactly on the offset-0
parabola — a known property of this wall model, see module docstring).

## ⑥ Notes / failure modes / limits

* Round-1's 23.5 % was an artifact of a hand-written re-implementation that
  never called repo code (audit F1); Round-2's "0.19 %" used a mislabeled
  metric (max|Δ|/max|ref|). Round 3 uses the **true** max pointwise relative
  error — stricter — and still passes at 0.31 %.
* `runs/` is gitignored in this repo, so run artifacts live in
  `artifacts/m18744/` (not ignored) — same classifier bucket.
* 6 pre-existing failures in `tests/test_public_api_lazy_exports.py`
  (airfoil_benchmark breakage) fail **identically on the clean tree**
  (git stash A/B check) — not introduced by this delivery.
* Scope: single-phase, static geometry, CPU float64, BGK only.

## ⑦ Error judgement (是否 < 3 %, 逐项)

| key quantity | reference | error | < 3 %? |
|---|---|---|---|
| D3Q19 pointwise (ny=10) | analytic | 0.308 % | ✓ |
| D3Q19 pointwise (ny=20) | analytic | 0.121 % | ✓ |
| D3Q19 u_max (ny=10) | 0.049219 | 0.19 % | ✓ |
| D2Q9 pointwise (ny=10/20) | analytic | ~1e-13 | ✓ |
| OSS D2Q9 u_max | analytic | 5.0 % | ✗ (reference arm) |

**All deliverable-arm quantities < 3 %** — ecosystem stretch met.

## ⑧ Artifacts & repro

* `tools/nebb_bodyforce_xval.py` — CLI entry (integration surface `tools/`)
* `artifacts/m18744/xval_nebb_bodyforce_r3.json` — all numbers
* `artifacts/m18744/fig_nebb_bodyforce_r3_profile.png` — profiles vs analytic
* `artifacts/m18744/evidence_m18744.json` — machine-checkable bundle
  (impl_files/entry/run_artifacts/integration_callers/benchmark/repro_cmd/checks)
* `src/tensorlbm/__init__.py` — NEBB public API exported (9 symbols)

**repro_cmd:**
```
python3 tools/nebb_bodyforce_xval.py \
    --out artifacts/m18744/xval_nebb_bodyforce_r3.json \
    --figure artifacts/m18744/fig_nebb_bodyforce_r3_profile.png
```
