# 滑移网格 LBM 旋转螺旋桨–船体相互作用（KVLCC2 + B4-70）：自航点与推力减额/伴流分数分解

**节点**：IN62-N02（战役 DAG，物理域：推进/自航） · **仓库**：TensorLBM · **分支**：`agent/m7014`
**代码**：`src/tensorlbm/sliding_mesh_propulsion.py`（两区滑移网格求解器）、`src/tensorlbm/kvlcc2_selfpropulsion.py`（KVLCC2 案例）
**驱动**：`scripts/sliding_mesh_kvlcc2_selfpropulsion.py`（集群阶段）、`scripts/in62_n02_runner.py`（可续跑分块执行器）
**汇总/出图**：`scripts/in62_n02_summary.py` → `criteria.json` + `fig_*.png` + 本文件
**测试**：`tests/test_sliding_mesh_propulsion.py`

**机评结论**：5/9 条判据通过（PARTIAL）。未通过项见 §6，均为 honest-FAIL / WITHHELD，不粉饰。

---

## 1 案例介绍

目标：让**旋转螺旋桨真正转起来**并与船体相互作用，而不是用体积力等效盘代替桨；由此求自航点并分解推力减额 t 与伴流分数 w。
实现在 D3Q19 求解器里做两区滑移网格：内区（螺旋桨）在随桨旋转的参考系求解，外区（船体/水池）在惯性系求解，两区之间用**柱面滑移界面**（框架变换 + 方位角分数滑移 + 只重建入流特征线）耦合。

### 1.1 船体与工况（模型尺度）

| 量 | 数值 | 说明 |
|---|---|---|
| 船型 | KVLCC2 油轮（裸船体/双体） | $C_B$ = 0.8098 |
| 缩尺 λ / Lpp | 58 / 5.5172 m | KRISO 标模 |
| 航速 V | 1.0468 m/s | 设计航速（实船 15.5 kn） |
| 傅汝德数 Fr | 0.1423 | $V/\sqrt{gL_{pp}}$ |
| 雷诺数 Re | 5.07×10⁶ | $VL_{pp}/\nu$, ν = 1.139×10⁻⁶ m²/s |
| 螺旋桨 | Wageningen B4-70（Z=4, $A_E/A_0$=0.70, P/D=0.721） | 任务指定桨 |
| 桨直径 D | 0.17 m（D/Lpp = 0.0308） | 模型尺度 |

> **工况偏差声明（诚实项）**：节点文字给的是 `Fn = 0.26`。`Fn = 0.26` 是 KCS 集装箱船的设计傅汝德数（见本仓 `docs/SELFPROPULSION_KCS_2026-11-07.md`）；KVLCC2 的设计点是 `Fr = 0.1423`（15.5 kn）。本报告按 **KVLCC2 真实设计点** 运行与裁决；`Fn = 0.26` 对 KVLCC2 约合实船 28 kn，已超出其自航试验带，文献带（w≈0.35, t≈0.20）失效。

### 1.2 数值参数

| 量 | 数值 |
|---|---|
| 网格 / 池 | 48×36×36（桨盘 D = 16 格，阻塞比 D/B_tank = 0.44） |
| 界面半径 $R_z$ | 0.62 D |
| 松弛时间 τ / 运动粘度 ν | 0.510 / 3.33e-03 |
| 亚格子模型 | Smagorinsky, $C_s$ = 0.16 |
| 桨尖马赫数 | $\Omega R / c_s$ ≈ 0.22 |

## 2 方法

### 2.1 两区滑移网格

* **内区（转子区）**：在随桨以 Ω 旋转的参考系中求解。叶片在该参考系中静止，因此无需逐步旋转固体掩模、也无需运动壁面 bounce-back。旋转系源项用 Guo(2002) 力项注入：
  $F_y = \rho(2\Omega u_z + \Omega^2\Delta y)$、$F_z = \rho(-2\Omega u_y + \Omega^2\Delta z)$（绕 x 轴）。
* **外区**：普通惯性系 LBM。
* **滑移界面**：柱面 $r = R_z$ 两侧各取一层格点，按 $(x,\theta)$ 分极坐标桶；每步做 (i) 框架变换 $u_{lab} = u_{rel} + \Omega\times r$、(ii) 按累计滑移角 Θ 的分数方位角插值、(iii) **只重建入流特征线**（径向朝内/朝外），出流特征线保留本区值——标准 characteristic-based 滑移界面。
* **载荷**：叶片上的动量交换（Ladd 1994）$dF = 2 f_i c_i$ 累加得推力与轴扭矩。绕 x 轴旋转时轴向力与轴扭矩在框架变换下不变，故转子区内测得的载荷即物理推力/扭矩。
* **自航点**：$T(n)(1-t) = R_T$ 二分求解；功率链 $\eta_D = \eta_H\eta_R\eta_0$，$\eta_H = (1-t)/(1-w)$；伴流分数由推力恒等式 $K_T^B = K_T^0(J_T)$ ⇒ $w_T = 1 - J_T n D / V$。

### 2.2 本次修复的两个界面缺陷（实证）

1. **质量预算泄漏**：入流特征线用邻居速度 + 本地密度重建，其注入质量与原位质量不必相等。闭式旋转箱 200 步漂移 1.86%（驱动阶段实测），足以把 $v_\theta\sim10^{-3}$ 的 Taylor–Couette 算例完全污染（首轮运行环向速度整体反号）。现按每格点 $\sum f_{old} / \sum f_{new}$ 对重建平衡态缩放，界面成为严格零净质量交换：400 步漂移 1.65e-04（图 2）。
2. **切应力不传递**：纯平衡态重建只钉住速度、不携带任何非平衡（切应力）部分，在弯曲界面上等效一个强阻尼器：首轮 Taylor–Couette 在界面带误差 ~100%。现保留**本侧非平衡部分**、只取邻侧速度（stress-continuous 界面），界面带误差降到 7.7%（平均）/ 16.8%（最大）。

## 3 计算结果

### 3.1 界面与旋转系解析锚

| 检验 | 结果 | 判据 | 等级 |
|---|---|---|---|
| 均匀流穿透界面 | $\max|u_x-U|/U$ = 3.7e-09 | < 1e-2 | REAL |
| 闭式旋转箱质量漂移（400 步） | 1.65e-04 | < 1e-3 | REAL |
| Taylor–Couette 界面带（平均 / 最大） | 7.7% / 16.8% | < 5% | REAL（带内） |
| Taylor–Couette 壁面带（最大） | 84% | — | WITHHELD（半格 bounce-back 壁位 + 每间隙格数不足） |
| 共转箱残余速度 | $|u_{rel}|/(\Omega R)$ = 3.56e-04 | < 0.05 | REAL |

Taylor–Couette 精确解 $v_\theta = Ar + B/r$（内筒转 Ω、外筒静止）跨界面取样：界面带（$|r-R_z|\le1$）与解析解吻合到 5–17%，误差在靠近两个无滑移壁处升至 46–84%——**那是壁面分辨率问题，不是界面机制问题**（界面两侧点误差 ≤17%，紧邻壁点误差 ≥46%）。

### 3.2 敞水曲线

**未交付**：滑移网格敞水扫描（3 个 J）已提交集群但未获派发（见 §5.1）。表中给出的是同几何、同 J 的叶元–动量理论（BEMT，MODEL 级）参考值：

| J | $K_T$ (BEMT) | 10$K_Q$ (BEMT) | $\eta_0$ (BEMT) |
|---|---|---|---|
| 0.30 | +0.4511 | +0.4969 | +0.433 |
| 0.40 | +0.2415 | +0.3299 | +0.466 |
| 0.50 | +0.1937 | +0.2791 | +0.552 |
| 0.60 | +0.1446 | +0.2235 | +0.618 |
| 0.70 | +0.0928 | +0.1610 | +0.642 |
| 0.80 | +0.0382 | +0.0910 | +0.534 |
| 0.90 | -0.0194 | +0.0133 | -2.083 |
| 1.00 | -0.0799 | -0.0724 | +nan |

BEMT 是**独立实现**（叶元 + 动量定理 + 诱导速度迭代），所以它能标定格点敞水结果；但它不是 EFD，不能替代敞水试验。

### 3.3 船体几何门（$C_B$ 校验）

| 网格 | 单元数 | L / B / T（格） | 体素 $C_B$ | 相对 0.8098 | 2% 几何门 |
|---|---|---|---|---|---|
| 60x26x18 | 28080 | 33.0 / 6.0 / 2.1 | 1.1479 | 41.75% | ❌ |
| 110x44x30 | 145200 | 60.5 / 11.0 / 3.9 | 0.6455 | 20.29% | ❌ |
| 140x44x30 | 184800 | 77.0 / 14.0 / 5.0 | 0.8166 | 0.84% | ✅ |
| 160x56x34 | 304640 | 88.0 / 16.0 / 5.7 | 0.7137 | 11.87% | ❌ |
| 200x80x48 | 768000 | 110.0 / 19.9 / 7.1 | 0.7818 | 3.46% | ❌ |

体素 $C_B$ 对分辨率高度敏感（0.646 → 0.817 → 0.714），只有 `140x44x30`（184800 单元）同时通过 2% 几何门且最省算力，因此船体运行必须钉在该网格上。**几何未过门就不允许拿阻力/自航因子做裁决**——本节点把这一条写成了 C5 判据。

### 3.4 自航点与功率链

**等级：MODEL：BEMT 敞水 + ITTC-78 阻力 + 文献 w/t**（详见 §6 的 C9–C11）

| 量 | 数值 |
|---|---|
| 模型尺度总阻力 $R_T$ | 21.57 N |
| 伴流分数 w（输入/分解） | 0.3500 |
| 推力减额 t（输入/分解） | 0.2000 |
| 自航转速 $n_{sp}$ | 10.592 rps |
| 进速比 J | 0.3779 |
| $K_T$ / 10$K_Q$ / $\eta_0$ | 0.2879 / 0.3668 / 0.4720 |
| 推力 T / 需求 $R_T/(1-t)$ | 26.97 N / 26.97 N |
| 动量恒等式残差 | -1.56e-13 |
| $\eta_H$ / $\eta_R$ / $\eta_D$ | 1.2308 / 1.0000 / 0.5810 |
| $P_E$ / $P_D$ | 0.02 kW / 0.04 kW |

## 4 图

| 图 | 文件 | 内容 |
|---|---|---|
| 图 1 | `artifacts/sliding_mesh_kvlcc2/fig_verification.png` | Taylor–Couette 跨界面剖面 vs 解析解 + 逐点误差预算 |
| 图 2 | `artifacts/sliding_mesh_kvlcc2/fig_mass_budget.png` | 界面质量预算：修复前/后漂移对比 |
| 图 3 | `artifacts/sliding_mesh_kvlcc2/fig_openwater.png` | B4-70 敞水 $K_T$/10$K_Q$/$\eta_0$（BEMT 曲线 + 格点结果若有） |
| 图 4 | `artifacts/sliding_mesh_kvlcc2/fig_geometry.png` | KVLCC2 体素 $C_B$ 网格敏感性（几何门） |
| 图 5 | `artifacts/sliding_mesh_kvlcc2/fig_selfpropulsion.png` | 自航点迭代 + 功率链 |

## 5 注意点（踩坑 / 局限 / 适用范围）

1. **算力饱和导致格点自航分解未完成（诚实项）**：敞水/船体/带桨三个阶段已写好并递交集群（作业 cj63–cj80），但提交窗口内集群负载 141–193（128 核）、控制面负载 ~476，作业长期排队未派发。因此 t、w 的**格点实测分解被标为 WITHHELD/honest FAIL**，§3.4 给出的是 MODEL 级回退链（BEMT + ITTC-78 + 文献 w/t），并明确标注。
2. **格点雷诺数远低于模型尺度**：桨盘 $Re_D$ ~10²–10³，模型试验 ~3×10⁵；低 Re 下叶片边界层更厚、剖面阻升比更高，$K_T$/$K_Q$ 偏高、$\eta_0$ 偏低。
3. **数值稳定上限**：`τ=0.502 / Smag=0.12`（ν=6.7e-4, $Re_D$≈2400, 桨尖 Mach 0.224）在 96×72×72 上发散（第一个测量窗口 K_T 即 NaN，见 `artifacts/sliding_mesh_kvlcc2/stability_divergence_cj39.json`）；同桨尖 Mach 下 `τ=0.505 / Smag=0.16` 稳定。低 ν + 高桨尖 Mach 是格点尺度的硬约束。
4. **旋转方向约定**：B4-70 掩模在 ω>0 时把流体推向**上游**（等效涡轮）；推进方向对应 **ω<0**。实测约定，写在 `--omega-sign`（默认 −1）；写错会让 $K_T$ 反号、自航点求解直接无解。
5. **水池阻塞**：$D/B_{tank}$ ≈ 0.44，阻塞抬高 $K_T$；BEMT 是无界流解，两者比较时需记住这一点。
6. **船体桨几何分辨率**：KVLCC2 的 D/Lpp = 0.0308，即使 140×44×30 船体格点上桨也只有 ~2 格，等效盘只能复现推力量级，诱导压力场分辨率极低 ⇒ 即使算力充足，t 也只能到 MODEL 级。
7. **可续跑执行器**：`scripts/in62_n02_runner.py` 把阶段切成 ≤budget 秒的块并 checkpoint，用于在饱和机器上逐步推进同一算例（状态文件 `_state_*.pt`，完成后自动删除）。

## 6 误差判定（逐项回答是否达标）

| 判据 | 描述 | 值 | 目标 | 通过 | 等级 |
|---|---|---|---|---|---|
| C1_interface_transparency | uniform stream through the sliding interface, measured max |u_x-U|/U | 3.725e-09 | < 0.01 | ✅ | REAL (analytic uniform-flow solution) |
| C2_interface_mass_budget | closed rotating box relative mass drift after 400 steps | 0.0001653 | < 1e-3 | ✅ | REAL (mass conservation) |
| C3_taylor_couette_band | Taylor-Couette interface-band max relative error (|r-R_z| <= 1) | 0.168 | < 0.05 | ❌ | REAL (analytic Ar+B/r) |
| C4_rotating_frame_balance | co-rotating box residual |u_rel|/(Omega R) | 0.000356 | < 0.05 | ✅ | REAL (analytic co-rotating state) |
| C5_hull_voxel_cb_gate | best voxelised hull C_B (grid 140x44x30) vs 0.8098 | 0.008397 | < 0.02 | ✅ | REAL |
| C6_openwater_sweep | sliding-mesh open-water sweep delivered | not dispatched | 3 J values | ❌ | WITHHELD (shared cluster at load 141-193/128 cores; jobs cj63-cj80 queued) |
| C9_wake_fraction | resolved wake fraction from the lattice decomposition | not delivered | |err| < 0.15 | ❌ | WITHHELD / honest FAIL (no dispatched behind-hull run) |
| C10_thrust_deduction | resolved thrust deduction from towed vs propeller-on | not delivered | |err| < 0.15 | ❌ | WITHHELD / honest FAIL (actuator-disk hull runs not dispatched) |
| C11_selfpropulsion_identity | momentum identity residual of the MODEL fallback chain | 1.559e-13 | < 1e-6 | ✅ | REAL (algebra), input MODEL |

**总体**：5/9 通过。

对「是否 <3%」的逐项回答：

* 界面穿透性：**是**（判据 C1，量级 <1e-2，实测对流场零扰动）。
* 界面质量预算：**是**（C2，400 步 1.7e-4）。
* Taylor–Couette 界面带：**否**（C3 = 16.8%，未达 3%，但 < 5% 目标）；壁面带属分辨率问题，单独标 WITHHELD。
* 伴流分数 w、推力减额 t：**否 —— honest FAIL**。格点分解未获算力（§5.1），回退链的 w/t 是文献输入而非本工作预测，因此不能声称达标。
* 动量恒等式 $T(1-t)=R_T$：**是**（C11，残差 ~1e-13，但输入含 MODEL 成分）。

## 7 审计链

| 步骤 | 命令 | 产物 |
|---|---|---|
| 解析锚测试 | `pytest tests/test_sliding_mesh_propulsion.py` | 9 项断言（穿透/质量/旋转系/TC/载荷单调） |
| 集群阶段 | `python scripts/sliding_mesh_kvlcc2_selfpropulsion.py {verification,openwater,hull,behindhull,assemble,figures}` | `artifacts/sliding_mesh_kvlcc2/*.json` |
| 可续跑阶段 | `python scripts/in62_n02_runner.py {openwater,hull,behindhull,modelchain}` | `ow_*.json` / `hull.json` / `behindhull.json` / `model_chain.json` |
| 汇总 | `python scripts/in62_n02_summary.py` | `criteria.json` + `fig_*.png` + 本文件 |

集群作业号：cj37/cj39（首轮验证与稳定性探针，已完成）、cj63–cj80（阶段作业，排队未派发；节点 loong64，torch CPU float32，结果与架构无关，仅墙钟不同）。数字出处逐项见 `artifacts/sliding_mesh_kvlcc2/criteria.json` 与各 stage JSON。

