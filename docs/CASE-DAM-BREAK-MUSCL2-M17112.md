# Slot 8 (`dam_break`) — 方法选型与实现：二阶 MUSCL/HLL 替换一阶 Rusanov

任务锚点：`GAP-LBM-T3-BENCH`（九算例谱系自动化验证矩阵）· 第 2 轮（先调研→再结合）
任务号：m17112 · 仓：TensorLBM · 分支：agent/m17112

前轮 m17062 在本锚点 honest-FAIL 且未留 `fail_stage`/`failure_reason`。本轮先把矩阵
现状盘清（矩阵本体 `benchmarks/validation_matrix/` 已存在，11 case / 9 slot），发现缺口不
在"有没有矩阵"，而在**逐项判据是否有可救的物理根因**。本轮选定唯一一处可救且可证伪的缺口：
slot 8 溃坝 case `dam_break_ritter_ci` 的 `profile_l2_pct` 判据（9.82% / 6.29% vs 5% 门）。

## ① 调研

### ①-1 开源横向（cluster `~/oss-cache` + `oss_inventory`）

- `oss_inventory` 全表 33 仓 / 33 repo，focus 全为 LBM 域（Palabos/openLB/walberla/lbmpy/
  sailfish/FluidX3D/lettuce/…），**无浅水方程（SWE）有限体积实现**；`oss_solutions` 中本 slug
  `GAP-LBM-T3-BENCH` 此前 0 行。
- 向集群递交 `~/oss-cache` 的 ritter/dam-break/shallow-water 关键词横向 grep（作业 `cj1424`），
  递交时队列积压 566，本单窗口内未落盘（状态 `queued`，非失败）：**开源横向对该子问题无可用
  代码样本**，因此不把"从 OSS 抄结构"写成已完成的调研，改由方法谱系（教科书/文献）给出选型依据。
- 可迁移的**工程约定**（非本次算法本身）仍取自 OSS/平台既有工程实践：矩阵对"冻结工件重新判裁
  + 绝不升级状态"的裁判式设计（本仓 `benchmarks/validation_matrix/runner.py` 既有约定），
  本轮不改判据、只改求解器。

### ①-2 方法谱系（external_anchor）

| 环节 | 出处 | 本仓落点 |
|---|---|---|
| 解析锚（自相似解）`h=(2c0-ξ)²/9g`、`x_f=2c0t` | Ritter (1892) Z. Ver. Deutsch. Ing. 36, 947；实验对照 Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312 | `src/tensorlbm/shallow_water.py::ritter_depth/ritter_front` |
| HLL 通量（局部波速） | Toro (2009) *Riemann Solvers and Numerical Methods for Fluid Dynamics* §10.4，式 10.18 | `_hll_flux` |
| MUSCL + minmod 斜率限制 | Toro (2009) §13.4；van Leer (1979) J. Comput. Phys. 32, 101 | `_muscl_face_states`、`_minmod` |
| 二阶时间推进 SSP-RK2（Heun） | Gottlieb & Shu (1998) Math. Comp. 67, 73 | `swe_step_muscl2` |
| 干床/well-balanced（静湖精确保持） | Audusse et al. (2004) SIAM J. Sci. Comput. 25, 2050 | 静湖测试 `test_muscl2_lake_at_rest_is_exact` |
| 一阶全局波速 Rusanov（被替换对象） | Rusanov (1961)；本仓 2026-11 既有实现 | `_rusanov_step`（保留可选，作为基线） |

### ①-3 选型结论（已沉 `oss_solutions`，feature_slug=`GAP-LBM-T3-BENCH`）

选 **minmod-MUSCL 重构 + 局部波速 HLL 通量 + SSP-RK2（method-of-lines）**，不选：
(a) HLLC + 干床专门处理——自由度更多、本轮无收益证据；
(b) 只加密网格到 nx≈700 才能过 5% 门——相当于把 CI 档算力抬一个量级，且不修根因；
(c) 放宽判据——**明确禁止**（矩阵"判据不可为结果让路"纪律）。
判据保持不变（front 5%、profile L2 5%、mass 1%），只见求解器精度上升。

## ② 方法与实现

- 求解器新增 `ShallowWaterConfig.scheme ∈ {"rusanov","muscl2"}`，默认 `"rusanov"`（历史行为
  一字未改，向后兼容由测试锁定）。
- `muscl2` 单步：原始量/守恒量按单元 minmod 限幅重构面状态（边界面零斜率）→ 面通量 HLL
  （Einfeldt/Davis 局部波速 `s_L=min(u_L-c_L,u_R-c_R)`、`s_R=max(...)`，全干面通量置零）→
  半离散 RHS → SSP-RK2。干床正性保护：若任一 stage 出现 h<0，则重构幅值 α 由 1→0.5→0 降阶重试。
- 关键指针：`src/tensorlbm/shallow_water.py`（`_minmod` / `_hll_flux` / `_muscl_face_states` /
  `_muscl_rhs` / `swe_step_muscl2` / 分派 `swe_step`），case 侧
  `benchmarks/validation_matrix/cases/dam_break_ritter_ci/run.py --scheme muscl2`。

## ③ 参数表（无量纲）

| 量 | 值 | 说明 |
|---|---|---|
| h0 | 1.0 | 初始水深，长度单位 |
| g | 1.0 | 重力，速度单位 √(g h0)=1 |
| 域 | x∈[-4, 12] | 单位 h0；dam 在 x=0 |
| t_end | 5.0 | 单位 √(h0/g) |
| CFL | 0.4 | dt = CFL·dx/(2c0)（沿用一阶稳定性界，二阶同样满足） |
| nx（判裁档） | 200 / 400 | 矩阵 CI 档 |
| nx（诊断阶梯） | 800 / 1600 | 不入判裁，只看趋势 |
| 判据 | front 5% / profile L2 5% / mass 1% / span ≤5pp 且单调 | 矩阵 `matrix.json` 原判据，未改 |

## ④ 对照表（ref vs got，逐项判据）

参考 = Ritter (1892) 自相似解（`ritter_depth`/`ritter_front`，且与 `x_f=2c0t`、
`V_down=8c0³t/27g` 一致性在 `tests/test_shallow_water_ritter.py` 已锁）。

| nx | 方案 | front_err | profile L2 | mass drift | 判据 |
|---|---|---|---|---|---|
| 200 | 一阶 Rusanov（基线，冻结） | +4.512% | **9.825%** ✗ | 2.2e-14% | ✗ |
| 400 | 一阶 Rusanov（基线，冻结） | +2.562% | **6.292%** ✗ | 2.2e-14% | ✗ |
| 200 | MUSCL2+HLL（本轮） | **+0.119%** | **1.168%** ✓ | 2.2e-14% ✓ | ✓ |
| 400 | MUSCL2+HLL（本轮） | **+0.057%** | **0.830%** ✓ | 0.0% ✓ | ✓ |
| 800 | MUSCL2（诊断） | +0.028% | 0.795% | 1.8e-13% | — |
| 1600 | MUSCL2（诊断） | +0.014% | 0.829% | 1.3e-13% | — |

误差改善：profile L2 ×1/8.4（nx=200）、×1/7.6（nx=400）；front 误差 ×1/38。
基线数字来自 `git show HEAD:…/result.json`（**零手抄**，见
`artifacts/validation_matrix/dam_break_scheme_upgrade/baseline_vs_after.json`）。

## ⑤ 收敛性 / 参数敏感性

- 网格收敛：L2(nx) = 9.825, 6.292, 3.837（一阶，~O(Δx^0.8)）→ 1.168, 0.830, 0.795（二阶）；
  二阶在 nx=200→400 段观测阶 0.49，400→800 段降至 0.06 并在 ≈0.8% 触底（见 ⑥ 失败模式）。
- 判据一致性：判裁档 span = 1.168−0.830 = **0.338 pp ≤ 5 pp**，且单调下降（`require_monotone`）。
- CFL 敏感性：CFL=0.4 为判裁值；dt 随 CFL 线性变化而二阶格式稳定界更宽，未在判据内调参
  （避免"对着结果调参"）。
- 静湖检验：均匀水深 + 静止初值经 20 步后 `h` 逐位不变、`|hu|max = 0`（well-balanced 精确）。

## ⑥ 失败模式与工件

- **干床锋面尖点**：Ritter 解在 `x→x_f` 处 h→0 且 ∂h/∂x 有限但湿/干边界是接触型间断，
  数值解在锋面 1–2 格内被抹平，归一化 L2 的该项不随 Δx 消失（表现为二阶收敛在 ~0.8% 触底）。
  结论：CI 档 5% 门可达；若要 <0.3% 需专门干床锋面处理（HLLC-dry 或锋面追踪），本轮不做。
- **正性保护的代价**：α 降阶重试是保守兜底，本轮算例中从未触发（可用 `min(h)>=0` 断言
  在测试中守住）；触发时会局部降为一阶，不破坏守恒（通量形式）。
- **适用范围**：一维浅水（静压、无摩阻、平底）；不给垂向结构，也不替代三维自由面 LBM
  （后者由 `benchmarks/dam_break*` 独立跟踪，未在本 case 内声明）。
- 工件 / repro：
  - 判裁工件 `benchmarks/validation_matrix/cases/dam_break_ritter_ci/result.json`
  - 证据包 `artifacts/validation_matrix/dam_break_scheme_upgrade/{baseline_vs_after.json, figure_dam_break_scheme.png}`
  - 复现：`PYTHONPATH=src python benchmarks/validation_matrix/cases/dam_break_ritter_ci/run.py`
    → `PYTHONPATH=src python -m benchmarks.validation_matrix run --case dam_break_ritter_ci`
    → `PYTHONPATH=src python -m benchmarks.validation_matrix status --write`
  - 基线/门禁测试：`PYTHONPATH=src python -m pytest tests/test_shallow_water_muscl2.py tests/test_shallow_water_ritter.py tests/test_validation_matrix.py -q`

## 矩阵整单状态（诚实口径，未夸大）

`status --write` 生成：**passed 7 / failed 3 / pending 1**（此前 6/4/1）。九个 slot 均有
reference + repro_cmd + 机器判据 + CI 可跑；本轮只把 slot 8 由 failed 转 passed，其余原样：

- `channel_turb_re180`（failed，未决）：声明证据缺失 + 域尺寸/统计样本不足（既有 honest FAIL）。
- `cylinder_re100_dfg`（failed）：cl_amp 36% 超 5%（统计窗口议题，未本轮处理）。
- `rayleigh_taylor_rt`（failed）：伪势 SCMP 在可达参数面无增长窗（模型类缺口）。
- `ship_wave_michell`（pending/WITHHELD）：3D 自由面核 ~35h 超预算，主判据未取得。