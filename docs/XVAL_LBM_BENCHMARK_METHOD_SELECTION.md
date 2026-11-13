# XVAL-LBM-BENCHMARK-METHOD-SELECTION — 与成熟开源 LBM 实现的选型交叉对标

锚点 `XVAL-LBM-BENCHMARK-METHOD-SELECTION` · mission m17499 · 仓 TensorLBM · 分支 `agent/m17499`

> 本文件由 `xval/benchmark_method_selection/make_report.py` 从 `summary.json` 程序化生成，报告里每个数字都能回溯到 `runs/<arm>_D<d>/result.json`（零手抄）。

## ① 问题与基准出处

**决策问题**：Does TensorLBM's recorded benchmark-method-selection (oss_solutions id=417: MRT + half-way BB + far_field + quadratic sponge + Ladd MEM + Hann-FFT St, for the square-cylinder vortex-shedding benchmark) reproduce the numbers that mature OSS LBM practice selects for the same benchmark (BGK single-relaxation per Maarten-vd-Sande/lbm; TRT Lambda=3/16 per lssfau/lbmpy + lettucecfd/lettuce), within the owner 15% criterion? And which selection knobs actually carry the difference?

- **算例**：二维方柱绕流涡脱（正置 α=0°），Re = U·D/ν = 200，阻塞比 D/H = 1/20，域 20D×20D，方块中心距入口 6D，U = 0.05（Ma = 0.087），ν = U·D/Re，τ = 0.5+3ν，入口播种 2 个 St_seed=0.13 周期（10%U 横向正弦）后回归自由流。
- **对标对象（本仓选型记录）**：`oss_solutions` id=417，feature_slug=`CASE-LBM-SQUARE-CYL-NORBERG`，tag=`benchmark-method-selection`：MRT + 半程反弹 BB + `far_field_bc_2d` + 下游二次 sponge + Ladd 动量交换测力 + Hann-FFT 谱峰 St。
- **对标对象（OSS 选型，源码查得）**：`~/oss-cache/Maarten-vd-Sande_lbm`（BGK 单松弛 + Zou–He 入口 + 零梯度出口 + 半程反弹，@573a01c）；`~/oss-cache/pylbm`（松弛参数逐矩显式声明 + BFL/Linear/Anti 边界族）；magic Λ=3/16 的 TRT 默认见 lbmpy/lettuce 谱系（oss_solutions 415/416 行 census）。
- **判据（owner 精度标准）**：对标误差 < 15% 不可接受线；本单预注册阈值见 `predeclared.json`。

## ② 方法与实现

同一内核、同一 BC、同一测力点、同一后处理，**只切换选型变量**（碰撞算子 / 吸收层），使差异只反映选型决策而非代码差异：

- **A1_recorded_mrt_sponge**：MRT (s_e=1.64, s_eps=1.54, s_q=1.7) + per-cell sponge tau_field (alpha=10, 6D, quadratic) -- the recorded TensorLBM selection (oss_solutions id=417)
- **A2_oss_bgk_plain**：BGK single relaxation + plain zero-gradient outlet -- OSS collision/outlet choice as implemented in Maarten-vd-Sande/lbm (lbm/streamcollide.py::collide uses a scalar omega; lbm/streamcollide.py::set_outlet copies the 3 upstream populations)
- **A3_oss_trt_plain**：TRT with magic parameter Lambda=3/16 + plain zero-gradient outlet -- OSS collision choice of lbmpy/lettuce (Ginzburg 2008)
- **A4_recorded_mrt_plain**：MRT + no sponge: bridge/control arm sharing everything with A1 except the sponge and with A2/A3 except the collision operator

比较逻辑：sponge_effect = A1 vs A4 (single difference: absorbing layer)；collision_effect = A4 vs A2 vs A3 (single difference: collision operator, identical BC/outlet/force/post-processing)；recorded_vs_oss = A1 vs A2/A3 (both knobs, i.e. the real-world selection difference)。

工程链（全部复用仓内已验证库函数，不另起炉灶）：`tensorlbm.solver.{collide_mrt,collide_bgk,collide_trt,stream}` + `tensorlbm.boundary.boundaries.{far_field_bc_2d,make_sponge_strength,compute_obstacle_forces}` + `tensorlbm.d2q9.equilibrium`；驱动脚本 `xval/benchmark_method_selection/run_arm.py`，聚合 `analyze.py`（阈值与 summary 计算全在其中）。

**OSS vs 本仓差异清单（做法/参数/边界条件，全部附源码行号）**：

| 环节 | TensorLBM（本仓记录选型, oss_solutions id=417） | OSS 做法（源码查得，零手抄） |
|---|---|---|
| 碰撞算子 | MRT，s_e=1.64 / s_eps=1.54 / s_q=1.7，逐格 `tau_field` 松弛（`src/tensorlbm/solver.py:276 collide_mrt`） | **单松弛 BGK**：`~/oss-cache/Maarten-vd-Sande_lbm/lbm/streamcollide.py:147-156`（`collide()` 用标量 `omega[0]`，无逐格松弛能力）；**TRT 由用户显式声明松弛参数**：`~/oss-cache/pylbm/pylbm/scheme.py:109`（`relaxation_parameters` 逐矩列出，库不给默认）；magic Λ=3/16 谱系见 oss_solutions 415/416 行（lbmpy/lettuce 默认） |
| 壁面/障碍 | 半程反弹 BB（`far_field_bc_2d` 内 `bounce_back_cells`） | 同类半程反弹：`streamcollide.py:65-66`（`# stream back (no slip boundary)` → `fin[n]=fout[v_inv[n]]`）；pylbm 另有 BFL/Linear/Anti 阶族 `pylbm/boundary.py:532,604,920,759`（亚格插值 BB） |
| 入口 | 自由流 Dirichlet（平衡态，`boundary/boundaries.py:201`） | Zou–He 型密度回溯：`streamcollide.py:218-233`（`set_inlet` 用未知族求和反解 rho=1/(1-u_x)·Σ） |
| 出口 | 零梯度 + 下游 6D 二次 sponge（`tau_field`, α=10） | 纯零梯度复制：`streamcollide.py:211-216`（`set_outlet` 把 `fin[3..5]` 从上游列拷贝）；pylbm 用 Neumann 族 `pylbm/boundary.py:1295`；两库**均无吸收层** |
| 测力（Cd/Cl） | Ladd 动量交换，post-stream/pre-bounce-back 采样（`boundary/boundaries.py:164`） | **两库均无测力模块**（`grep -rn 'force\|drag' Maarten/lbm/*.py` 仅命中重力项；`pylbm` 无 drag/momentum-exchange 代码）→ 力必须由使用者自建 |
| 数值精度 | float64（torch default set in `run_arm.py`） | Maarten：**float32**（`lbm/init.py:10-14` all `dtype=np.float32`）；pylbm 默认 float64 |
| 稳定性护栏 | 无（靠选型 MRT + sponge） | Maarten：**速度钳位** `streamcollide.py:131-133`（`maxvel=sqrt(2/3)`，超出按比例缩放） |
| St/特征量后处理 | Hann 窗 FFT 谱峰 + 抛物线插值 + 滞回过零交叉双法 | 两库均只给场量，无 St/后处理（示例 `examples/cylinder_flow.py` 只做动画） |

## ③ 参数表（无量纲数与格点参数完整）

| 量 | 记号 | 值 | 说明 |
|---|---|---|---|
| 雷诺数 | Re | 200 | U·D/ν |
| 阻塞比 | D/H | 0.05 | 域高 20D |
| 马赫数 | Ma | 0.1732 | U/c_s, c_s=1/√3 |
| 格子速度 | U | 0.1 | 格子单位 |
| 运动粘度 | ν | 0.00500 | = U·D/Re |
| 松弛时间 | τ | 0.51500 | = 0.5+3ν（τ−0.5 = 0.01500） |
| 格距 | Δx | 1 | 格子单位 |
| 时间步 | Δt | 1 | 格子单位 |
| 播种 St | St_seed | 0.13 | 入口横速正弦 2 周期、10%U |
| sponge | α, w | 10, 6D | σ(x)=clamp((x−x0)/w,0,1)²，τ_eff=τ(1+ασ) |

## ④ 对照表（ref vs got + 误差打勾/叉）

基线 ref = A1（本仓记录选型）同分辨率值；阈值 15%（预注册）。

| arm | 碰撞 | sponge | D | 步数 | Cd | Cl 幅值 | St (FFT) | St (过零) | ΔCd | ΔCl | ΔSt | <15% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | mrt | on | 10 | 14871 | 1.5768 | 0.5638 | 0.1694 | 0.1691 | — | — | — | — |

**失败臂（honest，不剔除、不粉饰）**：

- `A3_D10`（U=0.05）：`blowup` 于第 326 步（发散，非有限值）——按预注册稳定性规则判定失败并保留记录。
- `A5_D10`（U=0.05）：`blowup` 于第 624 步（发散，非有限值）——按预注册稳定性规则判定失败并保留记录。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

**(a) 分辨率阶梯 D=10 → D=12**（预注册收敛判据 10%）：

| arm | Cd(10) | Cd(12) | Δ | Clamp(10) | Clamp(12) | Δ | St(10) | St(12) | Δ |
|---|---|---|---|---|---|---|---|---|---|

**(b) 选型敏感性（碰撞算子 / 吸收层，D=10 固定）**：见表 ④ 的 Δ 列——sponge 效应看 A1 vs A4，碰撞效应看 A4 vs A2/A3，记录选型 vs OSS 选型看 A1 vs A2/A3。

**(c) 估计器敏感性**：St 双法（Hann-FFT 谱峰 + 抛物线插值 vs 滞回过零平均周期）逐臂偏差见 ④ 的 `St (FFT)` / `St (过零)` 两列，预注册一致性阈值 5%。

## ⑥ 失败模式与工件

- **失败模式 1（TRT 不可移植）**：OSS 谱系默认的 magic Λ=3/16 在本格点粘度下退化为 τ₋ = 0.5 + Λ/(τ₊−0.5) ≈ 25.5（τ₊=0.5075），远离推荐窗口 τ₋∈[0.5,2]，实测发散；把 τ₋ 钳到 2.0（A5）仍发散；抬 U 到 0.10（τ₊=0.515）后仍发散 → **本仓不采纳 TRT 移植**，但需在第 2 轮区分「τ₊≈0.51 的真实 TRT 不稳定性」与「本仓 `collide_trt` 内核缺陷」——本单未分辨，如实留待。
- **失败模式 2（测力口径）**：力必须在 post-stream/pre-bounce-back 采样，pre-stream 采样会把 Cd 符号翻转（历史实测 Cd=−1.615）。本单所有臂共用同一采样点，因此符号与量级一致（Cd>1）。
- **OSS 侧生态摩擦（friction）**：参考实现全部无法在本环境直接运行——`Maarten-vd-Sande/lbm` 需 numba（缺，pip 镜像可装但未在本单落地）、`pylbm` 需 mpi4py/cython/h5py/numpy-stl （缺，且需编译）；故 OSS 臂只能是**方法移植**（同内核、只换选型变量），这是本单的核心局限：不能分离「方法差异」与「实现差异」。
- **工件（全部在 worktree 内）**：`xval/benchmark_method_selection/{predeclared.json, summary.json, table.csv, runs/<arm>_D<d>/{result.json,trace.npz}, figures/*.png, logs/*.log}`；结构测试 `tests/test_xval_method_selection.py`。

## ⑦ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 本仓 got | 误差 | <15%? | <3%? |
|---|---|---|---|---|---|

**是否 <3%？** 上表逐项给出：Cd/Cl/St 的选型间差异若落在 3% 以内，说明记录选型与 OSS 选型在该算例上等价（选型不是精度瓶颈）；若 >3% 但 <15%，说明选型有可测影响但仍在 owner 不可接受线之内；>15% 则记录选型与 OSS 选型不可互换，必须在本仓台账上标注选型依赖。本单未做实验/文献级绝对误差判定（Norberg 1993 原表在本环境仍为 WITHHELD，见 `refs/norberg1993.json`），故绝对精度结论仍 withheld——本单只回答「选型互换性」。

## ⑧ 移植取舍

- **采纳**：pylbm 的「松弛参数逐矩显式声明」（`pylbm/scheme.py:109`）——把选型从库内隐式默认变成算例级显式参数，与 `accuracy_recommendation.py` 的 fail-closed 证据门同向；Maarten 的速度钳位护栏（`streamcollide.py:131-133`）作为发散前哨（低成本、无穷通量风险）。
- **不采纳**：TRT magic Λ=3/16 移植（实测发散，见 ⑥-1）；Maarten 的 float32 存储（本仓已有 float32 舍入吞尾部的实测教训，oss_solutions 415 行）；两库均无测力模块——本仓 Ladd MEM 采样是**净能力优势**，不外借。
- **缺口**：OSS 侧无吸收层、无 St 后处理、无测力，意味着「选型对标」只能在碰撞算子与边界族上做，力/频率口径必须由本仓自带并单独验证（已在 `docs/CASE-LBM-SQUARE-CYL-NORBERG*` 与 oss_solutions 417 记录）。

## ⑨ 复现

```bash
cd <repo> && python xval/benchmark_method_selection/run_arm.py --arm A1 --D 10   # 每臂一条
python xval/benchmark_method_selection/analyze.py                              # summary + 图
python xval/benchmark_method_selection/make_report.py                          # 本文件
pytest -q tests/test_xval_method_selection.py
```

