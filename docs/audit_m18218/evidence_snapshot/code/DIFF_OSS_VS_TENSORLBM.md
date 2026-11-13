# 差异清单：TensorLBM D2Q9 自由面 vs OSS oracle `Maarten-vd-Sande/lbm`

锚点 `XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS`｜第 1 轮·任务榜·开源对标（XVAL）
对标来源：`/root/oss-cache/Maarten-vd-Sande_lbm`（inventoried snapshot，master）
逐条读源码得出（`lbm/util.py`, `lbm/init.py`, `lbm/streamcollide.py`,
`lbm/update_types.py`, `lbm/time_evolve.py`, `examples/dambreak.py`），零手抄。

数值结论与图见 `REPORT_XVAL_ORACLE_MINIMAL_D2Q9_FS.md`；本文件的 port 对应
`oss_oracle_port.py`，其与**未改动**上游代码的一致性见
`artifacts/port_fidelity_*.json`。

---

## 1. 做法差异（模型层）

| 面 | OSS `Maarten-vd-Sande/lbm`（Thuerey 最小实现） | TensorLBM `free_surface_lbm_2d.py`（Körner/Thürey 参考实现） | 影响 |
|---|---|---|---|
| 碰撞 | 纯 BGK；`fout = fin + ω(feq − fin) − ρ t_q (c_q·g)`（重力作**分布增量**，无 `(1−ω/2)` 因子） | BGK + **fill 加权 Guo 力**（含 `(1−1/(2τ))` 源项修正）、`u*` 修正的平衡态 | 静态液压场二者一致；瞬态下 OSS 的力项不满足二阶矩一致性，误差 O(Ma²) |
| 自由面判据 | 二态 bitflag（FLUID/INTERFACE/GAS/OBSTACLE）+ 每格 `mass`；`ε = mass/ρ` | 同（LIQUID/INTERFACE/GAS/SOLID）+ 独立质量账 `m`，`ε = m/ρ_l` | 同族 |
| 界面（气侧）重建 | `f_in = feq_q(ρ=1,u) + feq_opp − f_out,opp`（ABB，用**本步** u） | ABB + Laplace/气泡压强项（默认关）+ 亚格/常压策略 | 常压、无表面张力时等价 |
| 界面-界面质量交换 | Thürey 4 分支选择器乘 `(ε_s+ε_n)/2`；但上游 `evolve` 传 `rho_prev=copy(mass)` 使 **ε≡1**（见 §3 Q1） | Körner 反对称配对交换（`paired_liquid_debit`，界面-界面精确 ±dm） | OSS 的加权因子在本仓语境里退化为常数 |
| 表面法向修正 | 有代码，但因 Q1 **恒不触发**（normal≡0） | 由 ε 场中心差分得到，参与转换/再分配 | OSS 的该能力事实上是死代码 |
| 超额质量再分配 | 沿法向 `η = max(0, c·n)` 加权；`η_total=0` 时退化为**界面邻居均分**；若无界面邻居则**静默丢弃** | Körner 式再分配 + 守恒夹紧（`sum(m)` 严格不变） | OSS 有质量守恒缺口（实测见报告 §5） |
| 转换阈值 | `fill_offset = 0.003`、`lonely_tresh = 0.1`，I→L: `m > 1.003ρ` 或 `m ≥ 0.9ρ` 且无气相邻居；I→G: `m < −0.003ρ` 或 `m ≤ 0.1ρ` 且无液相邻居，或 `NO_IFACE/NO_FLUID` | 同类阈值 + 转换驱动包络（保证 L/G 之间恒有一层 INTERFACE） | 同族，参数取值不同 |
| 速度限幅 | `update_rho_u` 硬限 `|u| ≤ sqrt(2/3)=0.8165` | 无（依赖 Ma 控制） | OSS 的限幅在自身 демо 参数下**长期饱和**（实测 u_max=0.8165），是抹平而非物理 |
| 气相密度 | `ρ_gas = 0`（场里不更新），ABB 用 `ρ=1` 平衡态 | `rho_gas` 显式参数（本 XVAL 取 1.0） | 一致 |

## 2. 参数/边界条件差异

| 面 | OSS | TensorLBM（本 XVAL 公共算例） |
|---|---|---|
| 出厂算例 | `nx=ny=170`，`ω=1.0`（τ=1），`g=(0,−0.1)`，750 步，块 `x∈[0,70) y∈[40,170)` | 由 `run_xval.py::DESIGN` 钉死：`a=32`, `h0=2a`, `nx=5a`, `ny=h0+a`, `τ=1.0`, `g=5e-5` |
| 初始列 | 顶**贴天花板**（`maxy=170=ny`，无顶隙）；界面环 `mass=0.5` 且 `rho=0.5`（Q4） | 顶留 1a 气隙（开放自由面，符合 M&M 实验）；界面环 `m=0.5`、`ρ=ρ_l=1` |
| 墙 | 四边 `OBSTACLE` 位掩码 + bounce-back（半格壁不在格心） | 四边 `solid_mask` + 半格反弹壁（流体侧半格） |
| 指数越界守卫 | `if 0 < ix_next >= nx: continue`（链式比较，只挡右越界；靠四边墙兜底） | 无越界读（`solid`/padding 兜底） |
| 时间推进 | `evolve` 每步 `print(time)`，返回 `velocities` 栈（名字叫 `u`，见 Q6） | 纯函数式 `free_surface_step_2d` 返回 `(f, mass, flags)` |
| 精度 | float32 全流程 | float32（torch） |
| 松弛参数语义 | `ω`（`ω=1/τ`） | `τ` |

## 3. 上游 quirk 清单（实测/读码；移植前必知）

| # | 位置 | 事实 | 处理 |
|---|---|---|---|
| Q1 | `time_evolve.py:evolve` | `mass_prev=copy(mass); rho_prev=copy(mass)` —— `rho_prev` 取了 `mass`。流式阶段 `ε = mass/rho_prev ≡ 1`（mass>0），法向中心差分 `≡0`，于是 (a) 界面交换权重恒 1，(b) 法向修正恒不触发 | port 默认**照抄**（`rho_prev_alias=True`），并提供 `False` 消融臂度量其影响 |
| Q2 | `streamcollide.py:get_epsilon` | `elif cell_type & CT.GAS == 2`：`==` 优先级高于 `&`，判据恒 False；GAS 落入 else 分支返回 **0.5** 而非 0 | port 照抄（`get_epsilon` 注释标注） |
| Q3 | `streamcollide.py:lbm_streaming` | 界面分支只写 `n=1..8`，界面格 `f0` 不再流式（滞后一步） | 照抄 |
| Q4 | `examples/dambreak.py` | 列顶贴天花板；界面环 `rho=0.5`（`rho += mass`） | 照抄为 `interface_rho="oss"`；开放顶隙算例用 `"liquid"` |
| Q5 | `time_evolve.py` | `u_prev` 在第 0 步未定义，只因第 0 步不发生 G→I 转换才不炸 | 照抄（driver 侧一次性给 `u_prev`） |
| Q6 | `time_evolve.py` / `examples/dambreak.py` | `evolve` 返回 `velocities` 而示例写作 `u, cell_types = evolve(...)`；把返回值再喂回去会在 `update_rho_u` 处 `IndexError: index 1 is out of bounds for axis 0 with size 1` | 驱动侧取 `velocities[-1]` 续跑（`oss_original.run_oss` 注释） |
| Q7 | `streamcollide.py:update_rho_u` | `u /= rho` 无零除保护；界面格 `Σf_q → 0` 时上游**在 30 步内变 NaN**（实测 `dam_break_a16`，见 `port_fidelity_a16.json` 的 `upstream_first_nonfinite_step: 30`） | port 加零除守卫（**唯一语义偏差**，声明于报告 §5）；一致性对照在上游变 NaN 前逐位成立 |

## 4. 移植取舍（本仓结论）

1. **不移植** OSS 的质量记账/限幅/Q1–Q7 这些"最小实现"的省事写法：它们是
   demo 级简化，且在物理工况（Ma≈0.1）下直接 NaN（Q7）或饱和限幅。
   TensorLBM 现有 Körner 实现（独立质量账 + 反对称交换 + 半格壁）保留。
2. **可借鉴但已在仓内**：气相侧 ABB 重建、四边位掩码壁、`fill_offset/lonely_tresh`
   式转换阈值、超额质量"无界面邻居"退化路径（仓内为守恒夹紧，优于 OSS 的静默丢弃）。
3. **移植面（本次新增，独立于求解器）**：把 OSS 作为**可复现 oracle** 收进
   `xval/oracle_minimal_d2q9_fs/`（`oss_original.py` 直接驱动上游源码 +
   `oss_oracle_port.py` 逐位等价向量化 port），用于后续 free-surface 改动的回归对照；
   结论是"对照口径要用 M&M 实验系，而不是这个 OSS 实现系"。
4. **不建议**把 OSS 数值结果当作精度基准（accuracy oracle）：它只在自身
   `g=0.1 / Ma≈6` 的演示参数域里"看起来像"溃坝；量化基准应继续用
   Martin & Moyce (1952) + Ritter (1892)（`src/artifacts/`）。

## 5. 复现

```bash
# 上游（未改动代码 + numba stand-in）
python xval/oracle_minimal_d2q9_fs/oss_original.py --nx 40 --ny 24 --a 8 --h0 16 --steps 5
# port 与上游逐位对照
python xval/oracle_minimal_d2q9_fs/port_fidelity.py --all-cases
# 全矩阵 XVAL
python xval/oracle_minimal_d2q9_fs/run_xval.py --matrix
# 图
python xval/oracle_minimal_d2q9_fs/make_figures.py
```