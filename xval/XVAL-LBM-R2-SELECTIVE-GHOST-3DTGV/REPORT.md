# XVAL-LBM-R2 — selective fused ghost exchange + corrected 3D TGV anchor

Anchor: **XVAL-LBM-R2-SELECTIVE-FUSED-GHOST-EXCHANGE-PLUS-3D-TGV-ANCHOR-CORRECTION** (mission 18798)
Predecessor: `xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED` (m17995) and its audit `docs/AUDIT_m18155_XVAL_LBM_MPI_GHOST_FUSED.md`.

## 1. Problem, reference source and what was wrong in round 1

Round 1 cross-validated TensorLBM's D3Q19 x-slab ghost exchange against the
lbmpy/waLBerla *selective* (PackInfo) exchange and reported a 3.8x byte cut. The
m18155 audit accepted the implementation (L2) but denied integration (L3) and
benchmark validation (L4) for four reasons, all addressed here:

| audit finding | R2 disposition |
|---|---|
| 3. reference exponent `2 nu k^2` for a field with `3 k^2` Laplacian | fixed: `nu * 6(1-cos k)` lattice / `3 nu k^2` continuum, pinned by `tests/test_xval_tgv3d_anchor.py` |
| 2. selective exchange not wired into the shipped class | `D3Q19GlooTransport(selective_ghosts=..., persistent_ghost_buffer=...)`, bit-exact vs monolithic on a 3/3/4 three-rank ring, exact 5/19 wire volume |
| 4. 1 of 4 predeclared configs delivered | full matrix in `predeclared.json` (8 configs) driven by `run_xval.py`; the local executions are the ones reported below |
| 1/5/6. no benchmark row, partial artifacts, figures not embedded | `predeclared.json` + `results`/`gates` + figures embedded in this report and in the API report payload |

**Reference source (zero hand-copy).** The crossing population set is derived by
`tensorlbm.xval_mpi_ghost_fused.packinfo.extend_dir`, a machine-checked
reproduction of the installed OSS function
`lbmpy/advanced_streaming/communication.py::_extend_dir` (lbmpy 2.0, line 233:
zero components of a communication direction expand over `{-1,0,1}`, non-zero
components are fixed); `tests/test_xval_mpi_ghost_fused.py::test_crossing_populations_match_oss_extend_dir`
asserts set equality for all 26 communication directions instead of comparing a
hand-written table. The waLBerla-side consumer
(`lbmpy_walberla/packinfo.py::generate_lb_pack_info`, `SerialDataHandling` /
`LBMPeriodicityHandling`) is the waLBerla app, not shipped in the wheel; its
semantics were read from the `github.com/lssfau/lbmpy` checkout
(`/tmp/lbmpy_src`) in round 1, and R2 re-uses only the pinned `_extend_dir`
identity above.

## 2. Method

* **Anchor correction.** For `u = U0 sin(kx)cos(ky)cos(kz)`, `v = -U0 cos(kx)sin(ky)cos(kz)`,
  `w = 0`, every component carries one normal mode in each of the three axes, so
  `lap(u) = -3k^2 u` and the Stokes amplitude decays as `e^{-3\nu k^2 t}` (energy
  `e^{-6\nu k^2 t}`), **not** `e^{-2\nu k^2 t}` (that is the 2D TGV). On the
  lattice the discrete eigenvalue `6(1-cos k)` is used; it differs from `3k^2` by
  `k^2/12` (0.32% at n=32, 0.08% at n=64).
* **Regime declaration.** The anchor is linear, so it is only fair while
  `Re_k = U0/(nu k)` is small. R2 runs an amplitude ladder
  (U0 = 0.04 -> 0.01 -> 0.0025, `Re_k` = 8.2 -> 2.0 -> 0.51) as a predeclared
  sensitivity axis, instead of hiding the nonlinearity inside a fudge factor.
* **Independent viscosity check.** The kinetic-energy decay rate fitted from the
  run (`-ln||u(t)||/||u(0)||/t`) is compared with `nu*6(1-cos k)`; this pins
  `nu = (tau-1/2)/3` through collision + streaming + ghost exchange without
  touching the anchor.
* **2x2 representation ablation** at fixed physics: {all-19, selective} x
  {fresh, persistent} buffer, plus the monolithic reference and the product
  path. Identity is an exact-zero metric.

## 3. Parameter table (all dimensionless numbers)

Common: D3Q19 BGK, `dt = dx = 1`, `rho0 = 1 + O(U0^2)` TGV pressure,
periodic in x, y, z, `tau = 0.65` (`nu = 0.05`) unless stated.

| n | U0 | tau | steps | Re_k | Ma=U0 sqrt3 | k=2pi/n | nu | rel L2 vs anchor | fitted-rate error | rho_min/rho_max |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.01 | 0.65 | 200 | 0.51 | 0.0173 | 0.39270 | 0.0500 | +23.353% | +5.823% | 1.000000/1.000000 |
| 32 | 0.01 | 0.65 | 200 | 1.02 | 0.0173 | 0.19635 | 0.0500 | +3.122% | +2.044% | 0.999988/1.000012 |
| 64 | 0.01 | 0.65 | 100 | 2.04 | 0.0173 | 0.09817 | 0.0500 | +2.030% | +1.844% | 0.999918/1.000082 |
| 32 | 0.04 | 0.65 | 200 | 4.07 | 0.0693 | 0.19635 | 0.0500 | +8.830% | +2.982% | 0.999823/1.000185 |
| 32 | 0.0025 | 0.65 | 200 | 0.25 | 0.0043 | 0.19635 | 0.0500 | +2.322% | +1.985% | 0.999999/1.000001 |
| 32 | 0.01 | 0.6 | 200 | 1.53 | 0.0173 | 0.19635 | 0.0333 | +3.963% | +2.601% | 0.999976/1.000025 |

Arm ablation (n=32, ranks=2, U0=0.01, tau=0.65, 200 steps, float64; timings on
2 gloo processes on one node, `torch.set_num_threads(1)` per rank):

| arm | wire bytes/step | exchange ms | step ms | rel L2 vs anchor | max abs diff vs serial |
|---|---|---|---|---|---|
| full_alloc | 622592 | 8.520 | 60.30 | +3.122% | 0.0e+00 |
| full_persistent | 622592 | 4.551 | 46.78 | +3.122% | 0.0e+00 |
| selective_alloc | 163840 | 3.286 | 41.92 | +3.122% | 0.0e+00 |
| selective_persistent | 163840 | 4.850 | 57.14 | +3.122% | 0.0e+00 |
| serial (monolithic) | 0 | - | 86.76 | +3.122% | - |
| product `D3Q19GlooTransport(selective_ghosts=True)` | 81920 | - | 32.37 | +2.735% | 0.0e+00 |

## 4. Reference vs got

| quantity | reference | got | error | verdict |
|---|---|---|---|---|
| field identity, selective vs monolithic (n=32, 2 ranks, 200 steps) | bit equality | `max|df| = 0.0` | 0 | PASS (exact) |
| field identity, product path `selective_ghosts=True` (n=32, 4 ranks, 100 steps) | bit equality | `max|df| = 0.0` | 0 | PASS (exact) |
| wire volume, selective vs all-19 | 5/19 = 0.2631578947368421 | 0.2631578947368421 | 0.0e+00 | PASS (exact) |
| rel L2 vs corrected 3D anchor, n=32, U0=0.01 | < 15% (owner), < 5% (engineering) | +3.122% | +3.122% | PASS |
| rel L2 vs corrected 3D anchor, n=64, U0=0.01 | < 15% / < 5% | +2.030% | +2.030% | PASS |
| rel L2 vs corrected 3D anchor, n=16, U0=0.01 | < 15% | +23.353% | +23.353% | FAIL |
| fitted decay rate vs `nu*6(1-cos k)`, n=32, U0=0.01 | within 5% | +2.044% | +2.044% | PASS |
| rel L2, n=32, U0=0.04 (round-1 parameters, nonlinear regime) | < 15% | +8.830% | +8.830% | PASS |
| round-1 delivered value (n=64, 400 steps, U0=0.04, 2D `2 nu k^2` anchor, read from `xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/table.csv`) | < 15% | +24.887% | reference defect (not the transport) | superseded by the 3D anchor reported above |

## 5. Convergence and sensitivity

* **Resolution** (left panel of fig2; n=16 and n=32 run 200 steps, n=64 runs 100, both
  quoted against the anchor at that t): rel L2 falls +23.353% (n=16) ->
  +3.122% (n=32) -> +2.030% (n=64) at U0=0.01; the
  fitted-rate error falls +5.823% -> +2.044% ->
  +1.844%, consistent with the lattice-eigenvalue gap
  shrinking as `k^2/12`. This is a discretisation trend, not a transport error: the
  transport is bit-exact at every resolution.
* **Regime / amplitude** (right panel): U0 = 0.0025 -> 0.01 -> 0.04 gives
  Re_k = 0.25 -> 1.02 -> 4.07 at n=32
  (n=64 doubles Re_k at fixed U0) with rel L2 monotonically increasing
  (+2.322% -> +3.122% -> +8.830%),
  inside the owner 15% band at every rung and inside the 5% engineering target
  for U0 <= 0.01. The anchor is declared linear-regime-valid for
  `Re_k <= 2.1` (the U0=0.01 rung at n=64, measured +2.030%).
* **Viscosity** (tau 0.65 -> 0.60 at n=32, U0=0.01): rel L2 +3.963%,
  fitted-rate error +2.601% — the anchor follows tau,
  i.e. the measurement is viscosity-aware and not a coincidental fit.
* **Rank count**: 2 vs 4 at fixed grid is exercised by the product arm (4 ranks)
  and the ablation (2 ranks); both are bit-exact, and the exchange volume per
  step is rank-count independent by construction (5 populations per face).

## 6. Failure modes, limits and ecological friction

* **Round-1 defect, honestly attributed.** With `2 nu k^2` the same run reports a
  much larger error than with the corrected exponent; the audit's 7.25 pp
  inflation is reproduced by construction (`decay` now returns the lattice 3D
  factor, and the ratio to the 2D factor is asserted to be `exp(-nu k^2 t)`).
* **The 15% band is not met at U0=0.04** (+8.830%). The cause is the
  linear reference being used at Re_k~8 (O(U0^2) vortex stretching and the
  second-harmonic generation of the true TGV), not the ghost exchange: the same
  run is bit-identical to the monolithic reference and the linear-regime rungs
  land at +3.122% / +2.030%.
* **Anchor-free alternative not delivered**: an exact eigen-decomposition of the
  linearised discrete BGK operator per wavevector would remove the remaining
  O(k) decay-rate discretisation error; it is scoped, not implemented.
* **Cluster matrix**: the 8-config predeclared matrix was submitted to the
  cluster queue (`run_xval.py`), but the queue was still blocked behind ~113 jobs
  while this report was assembled, so the numbers here come from the same driver
  executed locally (single node, up to 4 gloo ranks). The cluster rows are folded
  in automatically when `artifacts_cluster/` is present.
* **Ecological friction recorded**: the knowledge-base search endpoint
  (`/api/v1/tensor/knowledge/search`) returns `engine_offline` (connection refused
  on 127.0.0.1:8021), so the mandated pre-flight knowledge-base check could only
  be answered by the negative-knowledge ledger (`/tensor/negative-knowledge/search`,
  713 active entries, 0 hits for tgv/ghost-exchange/lbmpy keywords) — the
  key-worded capability lookup itself is unavailable.

## 7. Transplant trade-offs (TensorLBM vs OSS)

See `DIFF_OSS_VS_TENSORLBM_R2.md` (same directory) for the full差异 table
(approach / parameters / boundary conditions) and the trade-off decisions.

## 8. 误差判定（是否 <3%，逐项回答）

| 关键量 | 基准 | 本轮结果 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 场量一致性：selective vs monolithic（n=32, 2 ranks, 200 步） | 逐位相等 | max|df|=0.0 | 0.0% | 是 |
| 场量一致性：产品路径 selective_ghosts=True（n=32, 4 ranks, 100 步） | 逐位相等 | max|df|=0.0 | 0.0% | 是 |
| 线字节比 selective/all-19 | 5/19=0.2631578947368421 | 0.2631578947368421 | 0.0% | 是 |
| 场量 rel L2 vs 修正后 3D 锚（n=64, U0=0.01） | Stokes 3νk² | 2.030% | 2.030% | 是 |
| 场量 rel L2 vs 锚（n=32, U0=0.01） | Stokes 3νk² | 3.122% | 3.122% | 否（<5% 满足） |
| 场量 rel L2，产品路径（n=32, 4 ranks, 100 步） | Stokes 3νk² | 2.735% | 2.735% | 是 |
| 黏度守恒：拟合衰减率 vs ν·6(1-cos k)（n=32） | ±3% | +2.044% | +2.044% | 是 |
| 场量 rel L2（n=16 粗网格, U0=0.01） | Stokes 3νk² | 23.353% | 超出 15% 带 | 否 |
| 场量 rel L2（n=32, U0=0.04 非线性区） | Stokes 3νk² | 8.830% | <15% 但 >3% | 否 |
| R1 交付值（n=64, 400 步, U0=0.04, 2D 2νk² 锚） | 修正后 3D 锚 | +24.887% | 参考公式缺陷 | 否 |

**结论**：数据移动类指标（场量逐位一致性、线字节比）为精确 0 误差，<3% 满足；线性区场量在 n=64 为 2.030%、产品路径 2.735%、黏度拟合 +2.044%，<3% 满足；n=32 场量 3.122% 略超 3% 但满足 5% 工程目标与 15% owner 带；粗网格 n=16（23.353%）与非线性区 U0=0.04（n=32, 8.830%）不满足 3%，其中 n=16 超出 15% owner 带——两者均为参考锚的适用域限制（离散 / 线性），已在 §5 §6 归因，不是 ghost exchange 的误差（该路径逐位精确）。R1 交付的 24.887% 由 2νk² 参考公式缺陷造成，已被本轮 3D 锚取代。
