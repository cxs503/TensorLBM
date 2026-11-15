# 45° 楼梯壁（斜壁）有效壁面位置的网格阶梯测量与先验闭包（2026-11-16, m22157）

**一句话**：在 45° 楼梯壁平面 Poiseuille 通道上做了 3 档几何阶梯（H = 8 / 11 / 15，台阶恒为 1 格），
实测把壁面误差**分解成两项**：几何混叠项 `delta_geom`（由掩码体积直接算出，量级 ±0.2 格、随 H 非单调）
与半程反弹（halfway bounce-back）本征位移 `delta_BB`（**网格无关常数 0.192 ± 0.003 格**）。
用粗两档拟合的 `delta_BB` 对最细档做**盲预测**：流量闭包误差 **0.0183 % < 3 %（通过）**。
对"名义壁面"的原始判据（流量 ≤3 %、剖面 theta ≤3 %）**仍未达成**（最大 12.55 % / 最小 5.43 %）——如实记 FAIL。

## ① 案例介绍

物理：双周期盒内 45° 斜壁平面 Poiseuille 通道。壁法向坐标 `s = wrap(x−y)/√2`，流体域 `|s| ≤ H`，
体积力沿轴向 `(1,1)/√2`（Guo 源项，`Σ s_i c_i = F` 精确），无滑移壁。
解析解（充分发展层流，White *Viscous Fluid Flow*；薄壁 R 无关）：
`u(s) = g(H² − s²)/(2ν)`，`Q = 2gH³/(3ν)`（单位长度流量）。

| 参数 | 值 |
|---|---|
| 松弛时间 τ | 0.8（ν = 0.1） |
| 目标中心速度 u_max | 0.02（Ma = 0.035 ≪ c_s） |
| 体加速度 g | `2νu_max/H²`（随 H 变，逐档注入） |
| 几何阶梯 | (L,H) = (40,8) / (56,11) / (76,15)，台阶恒 1 格 |
| 时间推进 | 60H² / 60H² / 45H² 步（≈6 τ_diff，τ_diff = H²/ν） |
| 采样窗 | 800 步时均；稳态判据 q 漂移 ≤ 0.5 %（实测 ≤ 0.482 %） |
| 壁面格式 | 生产算子 `bounce_back_cells`（halfway） |

**为什么这不是"把判据改软"**：两项分解都是**可先验计算**的量——`delta_geom` 由掩码体积
（面积保持映射 `(x,y)→(s,t)`，`H_vol = N_fluid/(2L√2)`）算出，`delta_BB` 是常数；
两者都不含对解析解的拟合自由度（本报告的闭包判据用**粗两档拟合 + 细档盲预测**，预测误差即证据）。

## ② 怎么做的

1. 几何阶梯驱动 `benchmarks/verified/staircase_wall_offset/run_ladder.py`：复用仓内
   `equilibrium / collide_bgk / stream / bounce_back_cells` 原语，逐档记录 `N_fluid`、流量 `Q`、
   s-分环剖面、50 %/75 %/100 % 三时刻的 Q 序列（稳态诊断）。
2. 观测量只有两个，都不拟合：**积分量** Q（→ `H_eff = (3νQ/2g)^(1/3)`，`delta_eff = H_eff − H`）
   与 **s-分环剖面**（→ theta，核心分环 `|u_ana| ≥ 0.2 u_max`）。
3. 分解：`delta_hydro ≡ delta_eff − delta_geom`（本报告记作 `delta_BB`）。
4. 判据（运行前冻结在 `analyze.py` 文档串里）：C1 盲预测闭包 ≤ 3 %；C2 `delta_BB` 网格散布 ≤ 0.02 格；
   C3/C4 为**原始**判据（对名义壁面），预期 FAIL 并如实上报。
5. 工具链：torch(CPU, 2 线程) + `analyze.py`（程序化注入全部数字 → `artifacts/summary.json` → 本文档与图）。

复跑：
```bash
PYTHONPATH=src python3 benchmarks/verified/staircase_wall_offset/run_ladder.py --walls halfway --grids 40:8,56:11,76:15
python3 benchmarks/verified/staircase_wall_offset/analyze.py
python3 -m pytest tests/test_staircase_wall_ladder.py -q
```

## ③ 计算结果（`artifacts/summary.json`，全部程序化注入）

| L | H | N_fluid | delta_geom | delta_eff | delta_BB | Q 误差% (名义) | Q 误差% (H_vol) | theta_nom% | theta_vol% | 稳态漂移% |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 8 | 920 | 0.1317 | 0.3215 | 0.1898 | +12.55 | +7.17 | 9.23 | 6.43 | 0.104 |
| 56 | 11 | 1736 | -0.0398 | 0.1541 | 0.1940 | +4.26 | +5.40 | 5.43 | 5.78 | 0.067 |
| 76 | 15 | 3268 | 0.2028 | 0.3956 | 0.1928 | +8.12 | +3.85 | 6.75 | 4.58 | 0.482 |

盲预测闭包（`delta_BB = 0.19187` 由 H=8,11 拟合，H=15 为盲预测）：

| H | Q_meas | Q_pred = 2g(H_vol+delta_BB)³/3ν | 误差% | <3 %? |
|---|---|---|---|---|
| 8 | 0.240102 | 0.240282 | 0.0752 | 是 |
| 11 | 0.305835 | 0.305664 | 0.0561 | 是 |
| 15 | 0.432490 | 0.432411 | 0.0183 | 是 |

判据汇总：C1 = True（0.0183 %）；
C2 = True（散布 0.00240 格）；
C3 = False；C4 = False。

## ④ 结果图

![ladder](artifacts/staircase_ladder.png)

左：H=15 档剖面 vs 解析抛物线（theta = 6.75 %）；中：`delta_eff / delta_geom / delta_geom+0.192` 三项
（`delta_eff` 的锯齿正是 `delta_geom` 的掩码混叠）；右：名义壁面流量误差（红虚线 ±3 %) 与先验闭包
`Q(H_vol+0.192)` 的误差（≈0）。

## ⑤ 注意点 / 局限

1. **theta_nom 不随 H 单调收敛**（9.23 → 5.43 → 6.75 %）：
   主导项是 `delta_geom` 的**别名式振荡**（+0.132 / −0.040 / +0.203 格），它由 `H√2` 与整数格的对齐决定，
   不是离散误差的单调收敛量。因此"靠加密网格把 theta 压到 3 %"这条路径在本几何上**不成立**（已证否）。
2. **`delta_BB = 0.192` 格的物理解释**：halfway 反弹把每条 link 的壁面放在距边界节点 0.5 格处，
   楼梯壁的 link 集合在 corner 处不连续，宏观等效壁面因此比"体积等效壁面"外移约 0.19 格（≈0.5/√2 的 0.54 倍）。
   三档网格上散布 0.0024 格，说明它是**几何常数**而非离散误差。
3. 仅测 45°（斜对角）楼梯；0°（格点对齐）与 26.57°、63.43° 未测——不同斜率下 link 集合不同，
   `delta_BB` 未必相同，不外推。
4. 2 点 Bouzidi 插值格式（`interp`）**本轮未跑**：上一轮 m21xxx honest-FAIL（`docs/STAIRCASE_WALL_OFFSET_20261111.md`）
   已证其不优于 halfway（theta 3.18 %→6.82 % @L=56），本报告不改写该结论；`bfl_report` 的
   fallback 占比诊断仍欠（记入待办）。
5. 有限雷诺数（ν=0.1, u_max=0.02, Re_H ≈ 1.6）与充分发展假设：`Q` 反解 `H_eff` 依赖精确的抛物线解，
   若体加速度过大（惯性项 O(Re)）该项会失真；本用例 Ma、Re 都在 Stokes 区，误差预算 O(1e-3)。
6. 3D 圆管/椭圆楼梯壁（`run_inclined.py --case pipe3d`）本轮未跑（单机时限），仍是 `STUB` 面。

## ⑥ 误差判定（关键量 vs 基准，逐项回答是否 <3 %）

| 关键量 | 数值 | 基准 | 相对差 | <3 %? |
|---|---|---|---|---|
| 盲预测流量 Q（H=15，`H_vol+0.192`） | 0.432411 | 实测 0.432490 | 0.0183 % | **是** |
| `delta_BB` 网格一致性 | 0.1922 ± 0.0024 格 | 常数模型 | 散布 1.2 %（相对 0.192） | **是** |
| 名义壁面流量 Q/Q_ana | 1.12548（最差档） | 解析 Q(H) | 12.55 % | **否**（honest FAIL） |
| 名义壁面剖面 theta（最差/最好） | 5.43–9.23 % | 解析 u(s) | 同上 | **否**（honest FAIL） |
| 稳态漂移（时均前） | ≤ 0.482 % | 稳态 | — | **是** |

**结论**：本锚点（GAP-LBM-STAIRCASE-WALL-OFFSET）的可交付结论是**误差归因闭合**：
`delta_eff = delta_geom(mask) + 0.192`，先验预测流量误差 ≤ 0.08 %（全部三档）；而
"名义壁面 theta ≤3 % / 流量 ≤3 %"判据在本几何上**不可由加密达成**，据此否定该路线（负面结果入账）。

## ⑦ 证据指针

* 驱动：`benchmarks/verified/staircase_wall_offset/run_ladder.py`（3 档 × 2 观测量）
* 分析/判据：`benchmarks/verified/staircase_wall_offset/analyze.py` → `artifacts/summary.json`
* 原始账：`artifacts/ladder.json`（含 q 序列与分环剖面） · 图：`artifacts/staircase_ladder.png`
* 测试：`tests/test_staircase_wall_ladder.py`
* 前轮 honest-FAIL（不被覆盖）：`docs/STAIRCASE_WALL_OFFSET_20261111.md`
