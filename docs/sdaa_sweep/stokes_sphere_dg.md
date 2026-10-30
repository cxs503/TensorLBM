# SDAA 验证案例扫荡 — `stokes_sphere_dg`

- 案例: `benchmarks/verified/stokes_sphere_dg`（TensorLBM benchmarks/verified，状态 verified）
- 产物: `artifacts/sdaa_stokes_sphere_dg.json`（机器生成，判据数值零手抄）
- 驱动脚本: `scripts/sdaa_sweep_stokes_sphere_dg.py`（独立驱动，业务代码 `src/tensorlbm/**` 未修改：`business_code_modified=False`）
- 仓库/分支/commit: `/root/tp-agent/work/m61` / `agent/m61` / `0d41766351ae`
- 生成时间: 2026-10-29T13:59:30+0000

## 1. 设备证据

```
torch.sdaa.get_device_name(0) = /dev/tcaicard24
torch.__version__            = 2.7.1
torch_sdaa.__version__       = 3.1.1a3+gite3971b9
torch.sdaa.device_count()    = 1
torch.cuda.is_available()    = False  (无 CPU/CUDA 回退路径)
fp64 probe 1.0+2^-40 (x4)    = 4.000000000003638 (期望 4.000000000003638) -> fp64_real=True
```
所有张量（分布函数 f、平衡态、力项、BFL link 数组、几何 mask）均创建/驻留在 `sdaa:0`；脚本没有任何 device fallback，未使用 CPU 结果。

## 2. 案例与判据口径（按原口径，未降判据）

被测物理量：周期盒子内单球 Stokes 阻力修正 `K = F_ledger / (6*pi*mu*a*U_sup)`。判据阈值全部从案例 `run.py` 导入，未改写：

| 门 | 阈值 | 来源 |
|---|---|---|
| MAIN \|err\| = \|K_sim/K_ref-1\| | ≤ 3% | `run.py:ERR_LIM` |
| S0 质量漂移 | ≤ 1e-06 | `run.py:S0_HALT` |
| S1 尾段滑窗摆动 | ≤ 1e-03 | `run.py:S1_LIM` |
| S2 尾段收敛 \|U20-U40\|/U20 | ≤ 1e-03 | `run.py:S2_LIM` |
| S3 双估计器 \|F_ME-F_ledger\|/F_ledger | ≤ 1e-02 | `run.py:S3_LIM` |

- 参考值 `K_ref = K_ref_bie_P5` 读自 `benchmarks/verified/stokes_sphere_dg/reference_table.json`（md5 `af32757f9398e3f85b2bd0022c7a8efd`，与案例锁定 md5 `af32757f9398e3f85b2bd0022c7a8efd` 一致: **True**）
- 步数表 / 采样间隔 / τ 均来自案例 `run.py`：`STEPS_BY_TIER`、`SAMPLE=100`、`TAU_MAIN=1.0`
- 判据窗口与案例一致：tail-20%（K_sim、S3）、tail-40%（S2）、S1 用案例自带 `sliding_window_swing`

## 3. 配置

| φ | R (cells) | L (cells) | N_solid | N_fluid | φ_disc | φ_ref | 步数(冻结表) | 步数(实跑) | τ | 采样 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 8 | 48 | 2109 | 108483 | 0.019070 | 0.019393 | 15973 | 15973 | 1.0 | 100 |

物理设置（与案例一致）：D3Q19、float64、BGK τ=1.0、Guo(2002) 体力碰撞 `feq(u*), u*=u+F/(2ρ)`、BFL 精确 ray-sphere q 的无滑移插值反弹、三向周期 `torch.roll` 流式、初值 `feq(ρ=1,u=(U_sup_target,0,0))`。

- 几何校验（Tier φ=0.02 R=8）：`φ_disc/φ_ref-1 = -1.663%`（门 ≤2% → PASS）、n_links=4530、wraparound=0、q∈[0.0456,1.0000]

## 4. 结果（机器数值，取自 artifacts JSON）

| φ | R | K_sim | K_ref | err | U_sup(tail20) | Ma | Re_a | 墙钟 (s) | 步/秒 |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 8 | 1.778634 | 1.831417 | -2.882% | 1.072922e-03 | 1.858e-03 | 0.05150 | 892.3 | 17.9 |

NVIDIA 侧归档交叉校验（机器读取 `case_phi*_R*_tau1p0.json`，非手抄）：

| φ | R | 归档 K_sim | 归档 err | SDAA K_sim | 相对差 |
|---|---|---|---|---|---|
| 0.02 | 8 | 1.778634 | -2.882% | 1.778634 | -1.443e-15 |

## 5. 判据逐条 PASS/FAIL

| tier | 判据 | 数值 | 阈值 | 结果 |
|---|---|---|---|---|
| 0.02:8 | MAIN \|err\| <= 3% | 2.8820e+00 | 3.0% | **PASS** |
| 0.02:8 | S0 mass drift <= 1e-6 | 1.3329e-07 | 1e-06 | **PASS** |
| 0.02:8 | S1 tail swing <= 1e-3 | 0.0000e+00 | 0.001 | **PASS** |
| 0.02:8 | S2 tail convergence <= 1e-3 | 5.7470e-04 | 0.001 | **PASS** |
| 0.02:8 | S3 dual estimator <= 1e-2 | 4.5598e-04 | 0.01 | **PASS** |
| 0.02:8 | GEOM Cb/porosity: \|phi_disc/phi_ref - 1\| <= 2% | -1.6628e+00 | 2.0% | **PASS** |
| 0.02:8 | GEOM BFL links: no wraparound, n_links == dense count | 0.0000e+00 | 0 | **PASS** |

执行判据 7 条，失败 0 条。

### 收敛门（R=8→12→16→24 阶梯）

| φ | 已跑 R | \|err\| 序列 | 4 点齐全 | 收敛门 |
|---|---|---|---|---|
| 0.02 | [8] | 2.88% | False | NOT_EVALUATED |

未评估原因（如实记录，不降判据）：full ladder R=8/12/16/24 needs the 40.1 h reference budget; not reachable inside a 20 min SDAA sweep — reported, not judged

## 6. 墙钟与预算

- 内核阶段合计: **892.3 s = 14.9 min**（预算 19.0 min，含几何/BFL 预计算与进程启动 ≈ 15.5 min）
- 覆盖: 1/12 tier（`complete=False`），步数上限=None
- 同 tier 的 NVIDIA 归档墙钟: 104.2 s —— 单卡 SDAA 约为其 8.6×（同 float64 步数，硬件代际差异，非精度差异）

## 7. 绕行 / 替代记录

- **triton**: no triton path to bypass: the case kernels used here (D3Q19 equilibrium/macroscopic, torch.roll streaming, sparse BFL bounce-back/ledger, Guo force delta) are pure torch ops; grep over src/tensorlbm/{bfl_common,solver3d,d3q19,boundaries3d,free_surface_common}.py finds no triton import. TRITON_INTERPRET=1 was therefore not needed.
- **SDAAGraph capture (attempted, rejected)**: wrapping the LBM step in torch.sdaa.SDAAGraph failed with 'sdaaMemcpyAsync: operation not permitted when stream is capturing' raised from at::sdaa::native::index (advanced indexing inside bfl_bounce_back_sparse / bfl_force_ledger_sparse). Fell back to plain eager kernel execution for every step; no numerics changed.
- **gauge constants**: the case runner benchmarks/verified/stokes_sphere_dg/run.py is imported as a module (its main() is not executed) to reuse the frozen step table / gate limits / tail-window function verbatim; the tier parameters come from reference_table.json (md5-pinned by the case).

**性能等价替代**（不改变所报数值，逐条披露；见 JSON `perf_substitutions`）：

- NoDynamics solid-node restore
  - 案例写法: `f.reshape(19,-1)[:, solid_idx] = f_solid_pre (gather + index_put)`
  - 本次写法: `f = torch.where(solid4, f_solid_pre, f) on a pre-collision clone`
  - 等价性: identical algebra: solid nodes are reset to their pre-collision values; only the write mechanism differs
  - 理由: SDAA index_put with this index shape costs ~25.5 ms/step vs ~0.4 ms for the masked select (measured on device)
- BFL momentum ledger F_ME
  - 案例写法: `bfl_force_ledger_sparse called every step`
  - 本次写法: `called on sampling steps only`
  - 等价性: the ledger does not modify f and F_ME is only read at sampling instants -> reported F_ME values are unchanged
  - 理由: ~72 ms/step of SDAA time saved; needed to fit the frozen A2 step count into the wall-clock budget

等价性实证：本次 SDAA 的 K_sim 与 NVIDIA 归档值相对差 -1.44e-15（float64 舍入级），说明替代写法未改变轨迹。

## 8. 结论

**overall_verdict = `PARTIAL_PASS`** — all 7 executed gates PASS, but coverage is 1/12 tiers and the convergence (R-ladder) gate is NOT_EVALUATED — not comparable to the case's 12/12 verdict

- 已执行判据全部 PASS：主门 |err| = -2.882%（阈值 ≤3%），S0/S1/S2/S3 全过。
- 未覆盖部分（如实说明）：12 个 tier 中只跑了 φ=0.02 / R=8 这一档；R=8→12→16→24 收敛门因此 **NOT_EVALUATED**，案例原始的 12/12 `verified` 结论不能在 20 min SDAA 预算内被完整复核。其余 11 档（含 L=226/R=24、72.6 万步档）需要数十小时，超出本次扫荡预算。

## 9. 复现

```bash
python scripts/sdaa_sweep_stokes_sphere_dg.py --tiers 0.02:8 --budget-min 19
python scripts/sdaa_sweep_stokes_sphere_dg.py --from-json --render-doc docs/sdaa_sweep/stokes_sphere_dg.md
```
