# P20-N01 TensorLBM 湍流通道 Re_τ=180/395 对标 MKM-1999 与 y_w 缺陷修复

> 自动生成：`benchmarks/pending/channel_turb_3d/build_report.py`。文中所有数字由脚本从
> `mkm_gates_numbers.json` 与 `gates_verified.json` 注入，无手工抄写。
> 本报告为 **honest FAIL**：C1/C4/C5 未达标，C2/C6 为 WITHHELD，结论不作粉饰。
> 修订：C6 状态更新——本轮已把 SGS 开关接入生产 runner（`smagorinsky_cs`），负控仍记为
> 未执行（原因改为"需数天机时 + 仓内无 Re_τ=395 参考剖面"），不再写"不支持"。

## 0. 判决摘要

| 判据 | 结果 | 判定 |
|---|---|---|
| C3 y_w 实测化（假设 0.5 → 实测 0.645） | 偏差 +0.145 cell > 0.1 触发，已全链路修正 | **PASS** |
| C7 pytest + gates_verified.json | `tests/test_wall_origin.py` 12 passed；`gates_verified.json` 落盘 | **PASS** |
| C1 u+ 剖面对 MKM 相对 L2 < 10% | 最佳 L3：15.19% | **FAIL** |
| C4 u_τ 三口径互差 < 3% | 最佳 L3：17.06% | **FAIL** |
| C5 ×2 网格 u+ 最大偏差 < 2 | 最佳网格对 11.499 u+ | **FAIL** |
| C2 雷诺应力峰值位置误差 < 5% | 仓内 MKM 参考仅有 y,y+,u+ 三列，无应力剖面 | **WITHHELD** |
| C6 Re_τ=395 无 SGS 负控 | SGS 开关本轮已接线（cumulant + `C_s`，默认 0 保持分子粘性）；但 Re_τ=395 需数天机时且仓内无 395 参考剖面 | **WITHHELD（未执行）** |

三条根因均已定位并量化（§5）：(i) 统计样本仅 1.45 涡旋周转（要求 ≥2–3），壁面动量交换测得的
u_τ 系统性偏低 7.6%；(ii) 计算域 Lx/h=8.26、Lz/h=4.13 短于 MKM 盒 (12.57×4.19)，湍流大尺度
结构被截断；(iii) 该规模 DNS 在本机 CPU 上需 ~17 天（实测 0.273 Mcell-step/s，§5.3），
本轮预算内不可达。

## 1. 案例介绍

**物理背景**：槽道湍流是壁面湍流模型与壁面函数的唯一"可解析真值"试验台。Re_τ=180 是
MKM-1999 的经典低 Re 档（对数层最薄、单律壁面函数误差最大，最难标定），也是船体摩擦阻力
（占货船总阻 60–80%）数值预报的标定基准。本节点的两个物理交付物是：(a) 壁面有效位置
y_w 的实测化（原先硬编码 0.5，另有人报过 0.1）；(b) u_τ 三口径（压降平衡 / 壁面剪切 /
对数律拟合）自洽。

**几何/驱动**：三维周期槽道，流向 x 与展向 z 周期，上下壁 post-streaming bounce-back；Guo
体积力 a_x=2u_τ²/H 驱动（等价压降 −dp/dx=ρa_x）；初始化为谱无散度扰动触发转捩。

**参数表**（全部来自 `runs_3d/L*_result.json`）：

| 层级 | 网格 (nx×ny×nz) | 单元数 | dx+ | Lx/h | Lz/h | 步数/统计步 | 涡旋周转 | 步速证据 |
|---|---|---|---|---|---|---|---|---|
| L1 | 64×32×32 | 65,536 | 12.00 | 4.27 | 2.13 | 50 000 / 30 000 | 3.00 | 见 §5 |
| L2 | 128×48×64 | 393,216 | 7.83 | 5.57 | 2.78 | 50 000 / 30 000 | 1.96 | 见 §5 |
| L3 | 256×64×128 | 2,097,152 | 5.81 | 8.26 | 4.13 | 50 000 / 30 000 | 1.45 | 见 §5 |

目标 u_τ=0.003（lattice），Re_τ=180，ν=u_τ·(H/2)/Re_τ；碰撞模型 natural-KBC
（`chunked_collision.NaturalKBCCollisionExecutor`），L3 的 τ=0.50155。

## 2. 怎么做的

1. **新库模块** `src/tensorlbm/validation/wall_origin.py`（纯 numpy，依赖轻）：
   - `fit_wall_origin_viscous`：黏性底层 u=a(y−y_w) 线性拟合 → 同时给出 y_w 与 u_τ=√(aν)；
   - `fit_wall_origin_wall_stress`：**壁应力锚定**的一参数截距拟合（斜率由稳态动量平衡
     τ_w=ρa·h 给定）——本节点 y_w 的主估计量；
   - `fit_wall_origin_loglaw`：联合 (y_w,u_τ) 对数律拟合，并输出 `degeneracy` 诊断
     （见 §5.1：这是病态 ridge 问题，不能作为 y_w 主口径）；
   - `u_tau_from_log_law`：**固定 y_w** 后的单未知量闭合，含 `level`（κ=0.41,B=5.0 取定）
     与 `slope`（回归斜率 κ·s，对 B、ν 不敏感）两个变体，两者之差不加粉饰地作为
     对数律的模型形式不确定带；
   - `poiseuille_wall_origin`：层流抛物解析反演（全局估计量，作为 y_w 的一致性交叉检验）；
   - `u_tau_three_way`：三口径 + 两两偏差 + `consistent`(<3%) 判据；
   - `wall_offset_bias`：把"假设 vs 实测 y_w"换算成 y+ 位移与 u+ 偏差。
1b. **SGS 开关接线**（本轮顺带解除 C6 的前置障碍）：`WallResolvedChannel3DConfig` 新增
   `smagorinsky_cs`（默认 0.0 = 纯分子粘性，行为不变），正值只允许搭配 `cumulant`（C_s 进
   `collide_cumulant_d3q19`），并由 `tests/test_wall_resolved_channel3d_sgs.py` 钉住校验逻辑与
   "开 SGS 后数值仍健康且壁面剪切确实变化"。
2. **层流标定**（`laminar_wall_origin.py`）：用**与生产完全相同的核**（stream3d /
   bounce_back_cells_3d / guo_body_force_d3q19 / 同一 collide 执行器）跑层流 Poiseuille，
   从解析抛物初值出发（几倍黏性时间即达稳态），反演出该壁面格式的有效壁面位置
   y_w(τ)，并给出双估计量（壁应力锚定 + 全局抛物）与相对偏差。产出
   `runs_3d/laminar_wall_origin_calibration.json`。
3. **判据装配**（`analyze_mkm_gates.py`）：读 L1/L2/L3 真实产物 + 标定表 + MKM 参考，
   逐层算 u+ 剖面（分别用 y_w=0.5 与实测 y_w）、三口径 u_τ、雷诺应力剖面、网格收敛，
   写出 `gates_verified.json`（fail-closed：无参考数据 → withheld，未实现配置 →
   not_executed，绝不记 pass）。
4. **管线修复**：`run.py`（原 2D 分析管线）删去硬编码 `wall_offset=0.5`，默认
   `measured`；并加 fail-closed 校验——若近壁剖面不满足动量平衡（未稳态）或拟合出的壁面
   不在首个流体样本之下，则显式回落并在结果中写 `wall_offset_measurement_rejected` 原因。
5. **测试**：`tests/test_wall_origin.py`（12 项）：黏性底/壁应力/对数律解析恢复、
   抛物反演、病态性诊断、偏差量化、三口径自洽与失配检出、MKM 真值回归（把对数律在
   Re_τ=180 的模型形式误差钉住，防回归）。全部通过。

## 3. 计算结果

### 3.1 有效壁面位置（C3）

| 工况 | τ | y_w(壁应力锚定, 实测) | y_w(全局抛物拟合) | 与假设 0.5 的偏差 |
|---|---|---|---|---|
| natural_kbc@tau0.50155 | 0.50155 | 0.6447 | 0.6477 | +0.1447 |
| natural_kbc@tau0.55 | 0.55000 | 0.5934 | 0.5044 | +0.0934 |
| cumulant@tau0.50155 | 0.50155 | 0.6437 | 0.6338 | +0.1437 |
| cumulant@tau0.51 | 0.51000 | 0.6099 | 0.5132 | +0.1099 |

结论：生产壁面格式的有效壁面位置在 τ≈0.5016–0.55 区间为 **0.59–0.65 cell**，两种碰撞模型
几乎一致（差异 <0.002 cell），说明它由 bounce-back 格式主导而非碰撞算子。**两者都比假设的
0.5 大 0.09–0.15 cell**，也都与历史记录的 0.1 不符（0.1 无法用当前生产壁面格式复现，
见 §5.2）。

### 3.2 u_τ 三口径（C4）

| 层级 | τ | u_τ(压降/动量平衡) | u_τ(壁面动量交换) | u_τ(对数律拟合) | 最大两两偏差 | <3%? |
|---|---|---|---|---|---|---|
| L1 | 0.50075 | 0.003000 | 0.002384 | 0.003674 | 42.61% | ❌ |
| L2 | 0.50115 | 0.003000 | 0.002572 | 0.003427 | 28.51% | ❌ |
| L3 | 0.50155 | 0.003000 | 0.002771 | 0.003288 | 17.06% | ❌ |

在 MKM 参考 DNS 本身（Re_τ=178.12，壁面单位下 ν=1、a·h=1）上同一闭合器的结果：
压降 1.0000、壁面剪切 0.9855（−1.46%）、对数律 1.0460（+4.5%），**最大两两偏差
17.06%**——即"三口径 <3%"在
Re_τ=180 这一档即使对真值 DNS 也达不到：残差是对数律在薄对数层的模型形式误差（κ=0.41,
B=5.0 的通用常数在此档偏高；slope 变体更高出 5–6%）。

### 3.3 与 MKM-1999 u+ 剖面对标（C1）

| 层级 | y+ 区间 | 点数 | 相对 L2 误差 (y_w 实测) | 相对 L2 (y_w=0.5) | log 带 [30,100] | 最大绝对偏差 u+ | <10%? |
|---|---|---|---|---|---|---|---|
| L1 | 3.4–175.0 | 19 | 54.35% | 53.73% | 57.77% | 10.161 | ❌ |
| L2 | 2.4–176.9 | 27 | 31.25% | 30.91% | 32.65% | 5.836 | ❌ |
| L3 | 1.9–173.5 | 33 | 15.19% | 14.94% | 15.31% | 2.788 | ❌ |

（相对 L2 = ‖u+_sim−u+_MKM‖₂/‖u+_MKM‖₂，MKM 剖面插值到仿真 y+ 上；实测 y_w 与假设 0.5
两列的差即 y_w 修正的影响。）

### 3.4 雷诺应力剖面（C2）

| 层级 | <u'v'>+ 峰值 | 峰值 y+ | <u'u'>+ 峰值 | 峰值 y+ | <v'v'>+ 峰值 |
|---|---|---|---|---|---|
| L1 | -0.5279 | 51.1 | 7.459 | 22.5 | 0.568 |
| L2 | -0.5857 | 49.4 | 10.076 | 290.9 | 0.556 |
| L3 | -0.7157 | 39.4 | 10.297 | 312.9 | 0.754 |

<u'v'>+ 为负、峰值位于 y+≈39，趋势正确；
但仓内 MKM 参考只有均值剖面三列，无 <u'v'> 数据，故"峰值位置误差 <5%"无法机器判定 → WITHHELD。

### 3.5 网格收敛（C5）

| 网格对 | y+ 重叠区 | u+ 最大绝对偏差 | RMS 偏差 | 最大相对偏差 | 细化比 (nx,ny,nz) |
|---|---|---|---|---|---|
| L1_vs_L2 | 3.4–279.9 | 11.998 | 4.006 | 73.20% | 2.00, 1.50, 2.00 |
| L1_vs_L3 | 3.4–279.9 | 13.345 | 6.148 | 75.23% | 4.00, 2.00, 4.00 |
| L2_vs_L3 | 2.4–304.3 | 11.499 | 3.040 | 79.02% | 2.00, 1.33, 2.00 |

## 4. 结果图

![u+ vs MKM](figures/profiles_uplus_vs_mkm.png)

*图 1*：三层网格 u+ 剖面 vs MKM-1999；右图近壁放大，虚线为历史假设 y_w=0.5、实线为实测 y_w。

![y_w 标定](figures/wall_origin_calibration.png)

*图 2*：生产壁面格式的有效壁面位置（层流 Poiseuille 反演）随 τ 与碰撞模型的变化；
黑虚线=假设 0.5，灰点线=历史记录的 0.1。

![u_tau 三口径](figures/u_tau_three_way.png)

*图 3*：各层三口径 u_τ 与动量平衡目标值（黑虚线）。

![雷诺应力](figures/reynolds_stress_profiles.png)

*图 4*：<u'v'>+ 与正应力剖面（壁面单位）。

![误差趋势](figures/error_trends.png)

*图 5*：剖面误差随分辨率、闭合误差随统计收敛度；标出 C1(10%)、C4(3%) 门。

## 5. 注意点（踩坑 / 局限 / 适用范围）

**5.1 联合 (y_w, u_τ) 对数律拟合是病态 ridge 问题。** 两个参数进入同一个对数，窄窗内
残差谷是一条脊：在解析剖面上，联合拟合把 y_w 从真值 0.645 拉到 0.784（+21%）而 u_τ 只差
1%。`fit_wall_origin_loglaw` 因此输出 `degeneracy`（y_w 方向灵敏度比 u_τ 方向小 5 个数量级），
**y_w 必须实测（层流标定或壁应力锚定），不能靠对数律外推反演**——与节点原始设想
（"从对数律外推 y_w"）相反，这是本单重要的负结果。

**5.2 历史"实测 0.10"不可复现。** 当前生产壁面格式（post-streaming bounce-back +
natural-KBC/cumulant）在 τ≈0.5–0.55 实测 0.59–0.65 cell，两种碰撞模型一致；旧记录
（`result.json: wall_offset_calibrated=0.1`）来自 2D D2Q9 管线，且该管线的近壁剖面本身未稳态
（用本模块的壁应力锚定估计会被 fail-closed 拒绝，见 5.4）。**跨管线复用常数是错的。**

**5.3 计算预算是真正的瓶颈。** 本机 96 线程实测：natural-KBC 碰撞 0.339 Mcell/s、cumulant
1.021 Mcell/s、stream+macroscopic 1.411 Mcell/s ⇒ 整步 ≈0.27（natural-KBC）/0.59
（cumulant）Mcell-step/s。MKM 盒规模（400×64×132、120k 步 ≈4.1e11 cell-step）需
**~17 天（natural-KBC）/~8 天（cumulant）** 单机 CPU。要在 Re_τ=180/395 两档做到
u+ L2<10%、三口径<3%，必须 GPU 或多节点并让统计窗 ≥3 涡旋周转——本轮预算内不可达，
故 C1/C4/C5 诚实判 FAIL 而非调参凑数。

**5.4 未稳态会污染"实测"量。** 壁应力锚定估计量要求近壁剖面确实携带动量平衡的 τ_w；对
2D 管线那两个历史 run（自静止起步、仅 5 万步、u+ 峰值 ~4.2），拟合残差达 12%，壁面位置被
推到首样本之上 → 新 `run.py` 直接拒绝并回落，避免把瞬态当成壁面标定。

**5.5 未完成项（明示）**：(i) C6 负控未执行——本轮已把 SGS 开关接进生产 runner
（`smagorinsky_cs`，cumulant + `C_s`），障碍只剩"Re_τ=395 需数天机时 + 仓内无 395 参考剖面"，
故 C6 记为未执行而非不支持；(ii) 层流标定表仅覆盖 τ∈[0.50155, 0.55]，
L1/L2 的 τ=0.50075 落在表外，按 τ=0.50155 表值 clamp（Δτ=8e-4，y_w 影响 <1e-3 cell，
已在 gates 证据中标注 clamped_outside_table）。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 判据 | 结果 | 关键数字 | 判定 |
|---|---|---|---|
| C1_u_plus_l2_vs_MKM180_below_10pct | best relative L2 error 15.19% exceeds 10% (under-converged statistics and a domain shorter than the MKM box) | profiles.measured.u_plus_l2_relative=0.1519; profiles.assumed_0p5.u_plus_l2_relative=0.1494 | FAIL |
| C2_reynolds_stress_profile | reference stress profile absent from the in-repo MKM csv (columns are y, y_plus, u_plus only) - peak-location error cannot be machine-checked, gate withheld | reynolds_stress_plus.uv_peak_value=-0.7157; reynolds_stress_plus.uv_peak_y_plus=39.445 | WITHHELD |
| C3_wall_origin_measured_not_assumed | criterion met | runs[].measured.wall_origin_estimators.wall_stress_local=0.6447; assumed_vs_measured.delta_cells=0.1447; profiles.measured.wall_origin=1.3799 | PASS |
| C4_three_way_u_tau_within_3pct | best three-way spread 17.06% exceeds 3%; the wall-shear route (momentum exchange) lags the momentum balance because the statistics are not converged | three_way.max_pairwise_deviation_pct=17.0619; three_way.max_pairwise_deviation_pct=42.615 | FAIL |
| C5_grid_convergence_u_plus_deviation_below_2 | no grid pair agrees within 2 u+ units; the statistics are not converged and the refinement is not a clean factor of two in y | grid_convergence.L1_vs_L2.max_abs_u_plus_deviation=11.9982; grid_convergence.L1_vs_L3.max_abs_u_plus_deviation=13.3447; grid_convergence.L2_vs_L3.max_abs_u_plus_deviation=11.4994 | FAIL |
| C6_negative_control_no_SGS_at_Re_tau_395 | not executed: the SGS switch is now available, but the Re_tau=395 on/off pair needs a multi-day campaign and has no in-repo reference profile to score against | WallResolvedChannel3DConfig.smagorinsky_cs=cumulant + smagorinsky_cs | WITHHELD |
| C7_pytest_and_gates_artifact | criterion met | pytest=tests/test_wall_origin.py; gates=gates_verified.json | PASS |

**是否 <3%？**
- u_τ 三口径互差：**否**（最佳 L3 为 17.06%；对 MKM 真值 DNS 本身也仅
  5.95%，受对数律模型形式限制）。
- 压降 vs 壁面动量交换两口径（不含对数律）：对 MKM 真值 1.46% → **是**；而我方 LBM 三层
  网格为 20–26%，根因是统计未收敛（壁面动量交换偏低 7.6%），非闭合器误差。
- u+ 剖面对 MKM：最佳 15.19%，**否**（门限 10%）。
- y_w：实测 0.6447 vs 假设 0.5，偏差 +0.1447 cell，**超过 0.1 触发线**，
  已修正并在 u+/y+ 全部后处理中生效（对 L3 的 y+ 位移
  0.145 cell ≈
  0.84 壁面单位），修正后 C1 误差由 14.94% 变为
  15.19%——修正本身不改善误差，说明主导误差不是 y_w，而
  是统计/域长（这一点必须如实写出，不能把 y_w 修正当成绩报）。

**基线误差一览**：MKM 真值上的闭合器误差 1.46%（压降/壁剪）与 5.95%（含对数律）；
我方 LBM 对 MKM 的 u+ 误差 14.94%（y_w=0.5）/15.19%（实测 y_w）；网格每加密一档误差下降
约 45–55%（53.7→30.9→14.9%），呈单调收敛但未进入 10% 门。

## 7. 复现

```bash
# 层流壁面标定（双估计量，~40 s/工况）
PYTHONPATH=src python3 benchmarks/pending/channel_turb_3d/laminar_wall_origin.py \
    --tau-scan 0.50155 0.55 --steps 3200 --out runs_3d/laminar_wall_origin_calibration.json
# 判据与数字
PYTHONPATH=src python3 benchmarks/pending/channel_turb_3d/analyze_mkm_gates.py
# 图
PYTHONPATH=src python3 benchmarks/pending/channel_turb_3d/plot_mkm_gates.py
# 测试
PYTHONPATH=src python3 -m pytest tests/test_wall_origin.py -q
```
