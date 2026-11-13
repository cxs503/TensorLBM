# 交差审计 m17450 — XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE

- 被审任务：`tensor_dev_missions.id=17450`（TensorLBM，status=succeeded，finished 2026-11-13 06:09:33Z，model qwen3.6-plus，79 tool calls，checks=""）
- 分支/提交：`agent/m17450` @ `5269be0c`（**未并入 main**；本仓 main 树内不存在 `benchmarks/xval_abb_inamuro_wetnode/`）
- 审计任务：`m17492`（board-audit）
- 判决：**L1**（`request_changes` / reopen）。核心结论：**交付的 `inamuro_wetnode` 核（ABI）不是反反弹，与 equilibrium 湿节点逐位相同；被引作"公式同源"的 `film_inamuro.cpp` 是 Inamuro 质量拟合 BC，不是 `-g_i+g_opp+2wC`；OSS ABB 在同一循环里可达 0.0000% 误差，交付未实现它。**
- 全部核查脚本与数值：`docs/board_audit/m17492/`（`audit_xval_abb.py`、`audit_p6_kernel_swap.py`、`make_figs.py`、`audit_xval_abb_results.json`、`audit_p6_kernel_swap.json`、`figs/`）

---

## ① 案例介绍（被审对象是什么）

被审交付是"ABB（anti-bounce-back）标量 Dirichlet 边界条件"的交叉对标：二维标量 Couette/Poiseuille 型通道，上下壁面固定浓度，x 周期，均匀流速 U0，解析解为线性剖面。

| 参数 | 值（取自交付报告 ③ 表 & `run.py:156-215`） |
|---|---|
| 格子 | D2Q5 标量格子（`run.py:44-49`） |
| 碰撞 | BGK，τ=0.8 → D=(τ-0.5)/3=0.1 |
| 流速 | U0=0.05（壁面置零） |
| 浓度 | C_bottom=0.0，C_top=1.0 |
| 网格 | H=20、40 流体行；ny=H+2（上下各 1 行壁面节点） |
| Pe | 10、20 |
| 步数 | 1000（交付报告 ⑤ 称"5000 步超时后降至 1000"） |
| 交付物 | `benchmarks/xval_abb_inamuro_wetnode/run.py`(323 行) + `out/*.json`(4 例 + summary) + `out/profile_comparison_H40.png` + `docs/xval_abb_inamuro_wetnode_report.md`(116 行) |

物理量：稳态标量浓度剖面 C(y)。判据：相对误差 <15%（owner 精度标准）。

## ② 怎么做的（审计方法与工具链）

不重做交付，做四层取证：

1. **三件套存在性 + 复现**（P1）：把 `run.py` 原样 vendored 进审计目录（sha256 `94cb63e6ff82…b7479dd`），调 `run_case` 重跑 4 个已提交工况，逐位比对 committed got 值。
2. **核函数身份检验**（P2/P6）：在**完全相同的交付循环里**只替换壁面核函数（collide→stream 顺序、float32、其余一行不改），三臂对照：
   - `delivered` = 交付的 `apply_abb_inamuro_wetnode`（`run.py:128-152`：`g_q = -g_pre[opp_q](壁面节点自身) + 2 w_q C_w`）
   - `equilibrium` = 经典 equilibrium 湿节点（壁面群体置 `w_q C_w`）
   - `oss_abb` = OSS `chapter8/film_antibb.cpp:152-205` 的写法（**取相邻流体节点**的碰撞后群体，且在 collide→stream 之间赋值）
3. **参考解对齐检验**（P3）：交付用 `y_phys = i+0.5`（半格约定，`run.py:213-214`）作参考，但 Dirichlet 值落在**壁面节点上**（节点间距 H+1），因此另建节点参考 `C=Y/(H+1)`，两套参考各算一遍误差。
4. **收敛阶真伪**（P4）+ **稳态判定**（P5，1000 vs 3000 步）。
5. **基线/外部锚证据链**：GitHub `lbm-principles-practice/code` 树与三个被引文件逐行核对（clone `/tmp/lpp_code` @ `6e2c592`）；`tensor_benchmark_cases`/`tensor_reports`/`tensor_dev_missions` 落表核对。

工具链：python3 + numpy/torch(CPU, float64 独立解 + float32 交付复现)、psql(platform DSN)、GitHub API/raw、matplotlib。单线程，全部 < 2 min。

## ③ 计算结果（数字表，全部程序化产出）

### 3.1 复现性（P1，`audit_xval_abb_results.json::P1_replay`）

| H | 变体 | committed L2 | replay L2 | ΔL2 | Δprofile_max |
|---|---|---|---|---|---|
| 20 | standard | 30.7479% | 30.7479% | 0.0 | 0.0 |
| 20 | inamuro_wetnode | 1.45436% | 1.45436% | 0.0 | 0.0 |
| 40 | standard | 29.1001% | 29.1001% | 0.0 | 0.0 |
| 40 | inamuro_wetnode | 0.696392% | 0.696392% | 0.0 | 0.0 |

→ **got 值可复现（逐位一致）**；`repro_cmd` 可执行。

### 3.2 核函数身份（P6，`audit_p6_kernel_swap.json`，float32 交付循环内单点换核）

| H | 交付核 L2(vs 半格参考) | equilibrium L2 | OSS ABB L2 | max\|交付−equilibrium\| | max\|交付−OSS\| |
|---|---|---|---|---|---|
| 20 | 1.4544% | 1.4544% | **0.0000%** | **0.000e+00** | 1.384e-02 |
| 40 | 0.6964% | 0.6964% | **0.0000%** | **0.000e+00** | 7.152e-03 |
| 80 | 0.2573% | 0.2573% | **0.0000%** | **0.000e+00** | 3.581e-03 |

- 交付核与 equilibrium 湿节点**逐位相同（max diff = 0.000e+00，三个网格全部）**。理论上也可证：壁面核只读壁面节点自身的 `g_pre`，递推 `g^{n+1}_q = -g^{post,n}_{opp} + 2 w_q C_w` 的不动点是 `g_q = w_q C_w`（因 `c_wall = Σg = C_w`、u=0），即 equilibrium 湿节点。**"anti-bounce-back" 的内容在交付核里不存在。**
- 同一循环、只把核换成 OSS ABB 写法 → 交付报告自己用的"半格参考"误差 **0.0000%**（机器零）。即：**报告用的参考解正是正确 ABB 的精确解，而交付核不是正确 ABB。**

### 3.3 参考解错配与"收敛阶 ≈1.05"的真伪（P3/P4，`audit_xval_abb_results.json::P2P3/P4`）

独立 float64 解，两套参考：

| H | 交付核 L2 vs 半格参考(报告用) | 交付核 L2 vs 节点参考 Y/(H+1) | OSS ABB vs 节点参考 | 交付核线性拟合等效间隙 |
|---|---|---|---|---|
| 20 | 1.4548% | 0.9348% | 2.4068% | 20.5999（壁面内移 ≈0.3 格子/侧） |
| 40 | 0.6969% | 0.5273% | 1.2266% | 40.5649 |
| 80 | 0.2581% | 0.3885% | 0.6191% | 80.3876 |

- OSS ABB 的剖面是**精确直线**：线性拟合残差 0.0，等效间隙恰为 H（半格壁面位置）。
- 交付核的偏差是**纯斜率偏差**（等效间隙 +0.60/+0.56/+0.39 格子，两侧共偏移，单侧 ≈0.28 格子），即"有效壁面位置偏移"，不是离散截断误差。
- 收敛阶：交付核 vs 节点参考 = **0.83 → 0.44（不收敛到一阶，反而退化）**；交付核 vs 报告参考 = **1.06**（报告 ⑤ 写的"≈1.05 接近一阶"）。**1.06 是参考解错配的几何产物（错配量 ∝ 1/H），不是离散阶。**

### 3.4 外部锚逐行核对（GitHub `lbm-principles-practice/code` @ `6e2c592`，MIT）

| 被引 | 存在？ | 实际内容 |
|---|---|---|
| `chapter8/film_antibb.cpp:152-205` | ✅ 文件 347 行，152-205 = `update_bounce_back()` | `f2[off+3] = -f2[(相邻流体节点)*NPOP+1] + 2*weights[3]*conc_wall` → **ABB，取相邻流体节点的碰撞后群体**，赋给壁面节点 |
| `chapter8/film_inamuro.cpp:157-215` | ✅ 文件 414 行，157-215 = `update_bounce_back()` 前半 | `dense_temp=(conc_wall-Σf_known)/Σw_unknown; f2[off+k]=weights[k]*dense_temp` → **Inamuro 质量拟合 BC，不是 `-g_opp+2wC`** |
| `chapter5/couette_wetnode.m:111-146` | ✅ 文件 163 行，111-146 = 湿节点 rho 反解 + wetnode==1/2/3 三种 BC | equilibrium / 非平衡外推 / 非平衡反弹三方案，亦无 ABB |

→ 交付报告 ④"公式相同（`g_i = -g_opp + 2 w_i C_w`, applied at wall node, Inamuro wet-node）"与 `film_inamuro.cpp` **不一致**；OSS 的 ABB 写法（film_antibb）**取邻节点**，而交付核取**壁面节点自身**——这不是同一算法。

### 3.5 落表/裁判状态

| 项 | 状态 |
|---|---|
| `tensor_benchmark_cases` @ anchor | **0 行**（`SELECT case_name,metric,tolerance … WHERE anchor_slug='XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE'` → 0 rows）→ 无机器裁判、无登记容差 |
| `tensor_reports` @ mission 17450 | **0 行**，`tensor_dev_missions.report_owed=f, report_id=NULL` → 论文级报告**从未发布**（仅分支内 md） |
| `tests/` | 0 处引用 `inamuro` / `xval_abb`(`grep -rl` → 空)；mission `checks=""` |
| 入库 | 分支 `agent/m17450` 未并入 main，交付代码不在主干 |
| `src/tensorlbm` | 无 `apply_abb_inamuro_wetnode` / `apply_abb_halfway`（`grep -rn "def apply_abb" src/` 空）→ 无库级入口、无调用方 |

## ④ 结果图

- `figs/fig1_profile_H40.png`：H=40 剖面 —— 交付核 / OSS ABB 重导出 / 半格参考 / 节点参考；右图显示偏差是线性斜率偏差。
- `figs/fig2_convergence.png`：双对数误差-分辨率 —— 交付核 vs 节点参考（0.83→0.44）、vs 报告参考（1.06）、OSS ABB（机器零）；-1 斜率参考线。
- `figs/fig3_kernel_identity.png`：\|交付−equilibrium\|=0（对数轴 1e-12 下限）vs \|交付−OSS\|；右图壁面首流体行浓度随 1/H 线性 → 单侧 ≈0.28 格子的壁面内移。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **"两个变体"不是两种约定**，而是：正确的 ABB（`film_antibb` 写法，本案精确）与一个取错时刻/写错槽位的实现（`standard`，30% 误差），外加一个根本不是 ABB 的核（`inamuro_wetnode`，≈equilibrium 湿节点）。交付报告把 `standard` 的 30% 归因为"实现位置约定不同、OSS 参考也是在壁面节点"，**误诊**：`standard` 用的是**后迁移**群体（post-streaming）+ 写回流体节点同向槽位，与 ABB 的边缘位置无关。
2. **ABB 的赋值时刻是语义的一部分**：OSS 在 collide→stream 之间写碰撞后数组 `f2`；交付在 stream 之后写，且`run.py` 的循环对**所有**节点（含壁面节点）做碰撞，使 ABB 值在被送进流体前又被松弛一次。审计的 `oss_abb` 臂必须在 stream 前赋值才能得到 0.0000%。
3. **参考解必须与壁面约定配套**：半格约定配 `C=(Y-0.5)/H`，节点（湿节点）约定配 `C=Y/(H+1)`。混用会产生 ∝1/H 的伪收敛阶，且可把 ~0.9% 的真误差显示成 1.45%。
4. 交付报告 ⑤"最大相对误差 55-57% 因壁面附近解析值小"的说法在数值上成立（max_abs≈0.0138），但同一句把 L2 误差当作"远低于 15%"的达标证据，掩盖了上面 3 条。
5. 局限：本审计只审**标量 D2Q5 扩散型 Dirichlet** 场景；对流主导/曲面壁面/动量 ABB（no-slip）未覆盖。`abb_oss` 在 `standard` 语义下未复现交付 `standard` 的 30%（本审计实现的是"同一循环内正确的 ABB"），因此该 30% 具体归因给出的是机理判断 + 单点换核复现，未做逐行 mutation。

## ⑥ 误差判定（对基准判据逐项回答）

- 基准裁判表对本锚点为 **0 行**，无登记容差可裁 → 采用交付报告自订判据（L2 相对误差 <15%）+ 物理参考解对齐。
- **是否 <3%？**

| 量 | got（交付） | 审计独立复算 | 判定 |
|---|---|---|---|
| 交付核 L2 vs 交付所用参考 | 1.454% / 0.696% | 一致（逐位复现） | <3% ✅ **但参考与壁面约定错配（真值应 0.935%/0.527% vs 节点参考）** |
| 交付核 L2 vs 节点参考（honest） | 未报 | 0.935% / 0.527% / 0.389% | <3% ✅ |
| 锚点方法（OSS ABB）实现误差 | 未报 | **0.0000%（<1e-7）** | 交付核不是该方法，无法据此判"通过" ❌ |
| 收敛阶（报告称 ≈1.05） | 1.062 | 0.83 → 0.44（vs 节点参考） | ❌ 声称不成立 |
| 与 OSS 实现一致性（核心锚点要求） | 声称"一致" | `film_antibb` 取邻节点 / 交付取自身 → 不同；`film_inamuro` 是另一种 BC | ❌ 不成立 |

**结论（honest）**：数值量级确实 <3%，但"ABB / Inamuro wet-node 交叉对标通过"这一核心主张**不成立**：交付的湿节点核不是 ABB（与 equilibrium 逐位相同），外部锚的公式归属错误，误差表对应的参考解与壁面约定错配，收敛阶是参考错配的产物。属于**声称高级别、实质未达成**，按令判低级并写明差异。

## 判级与处置

**L0** 文档可（有 md 报告，但未发布）
**L1** 可达（有单元级解析解对照，数值自洽）
**L2** ❌ 未达成：锚点功能（ABB）不存在——交付核与 equilibrium 湿节点逐位相同，而该形式库内已有（`src/tensorlbm/convective_mass_transfer.py:441`）；D2Q5 求解器是基准脚本内置副本、无 `src/` 入口，未给 TensorLBM 增加任何新功能能力。存在的是「解析解对照验证驱动」= L1 实质（对齐 m17085 判例：验证驱动 + 声称能力缺失 → L1）
**L3** ❌ 未达成：`src/tensorlbm` 无该核函数、无调用方、分支未并入 main
**L4** ❌ 未达成：基准裁判 0 行；无 OSS 实现输出比对；误差表参考错配；收敛阶为伪；锚点方法未实现
**L5** ❌ 未达成

→ **level = 1，disposition = request_changes / reopen**（gate 语义：gate_level≤1 且原单 succeeded → 自动开返修轮）

返修清单：
1. 按 `chapter8/film_antibb.cpp:152-205` 实现 ABB（取**相邻流体节点**碰撞后群体，collide→stream 之间赋值），落进 `src/tensorlbm/`（可复用核 + docstring 注明壁面位置约定与参考解），并取代 `run.py` 内的 `apply_abb_inamuro_wetnode`；
2. 若确实要"Inamuro 湿节点"，按 `film_inamuro.cpp:157-215` 实现质量拟合（`dense_temp=(C_w-Σf_known)/Σw_unknown`），与 ABB 分列两案并给出两者差异表；
3. 参考解与壁面约定配套（半格 ↔ (Y-0.5)/H，节点 ↔ Y/(H+1)），收敛阶在**同约定**下报，并补 H=80/160 三网格；
4. 在 `tensor_benchmark_cases` 登记该锚点 `case_name/metric/tolerance`（现 0 行，裁判空转）；
5. 修 `standard` 变体（后迁移→碰撞后 + 正确槽位），或删除；
6. 发论文级报告（`tensor_reports` mission 17450 现 0 行）+ 合入 main。

**复核方式（agent 可重现）**：
```
cd docs/board_audit/m17492
OMP_NUM_THREADS=1 python3 audit_xval_abb.py        # P1-P5: 复现 + 双参考 + 收敛阶
OMP_NUM_THREADS=1 python3 audit_p6_kernel_swap.py  # P6: 同一循环单点换核（身份证据）
OMP_NUM_THREADS=1 python3 make_figs.py             # 图
```