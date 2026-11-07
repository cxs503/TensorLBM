# 阻力链 CLOUD 补测：场量云图与角度分布（mission m4927）

> scenario_key = `paper-mat-resistance-chain-de-mock-cloud` ｜ mission_id = 4927 ｜ 仓 = TensorLBM ｜ git commit = `unknown`
> 工件目录：`artifacts/resistance_chain_cloud/`（fields/*.npz、table1/2/3 CSV、fig1/fig2 PNG、cloud_index.json）
> 生成脚本：`benchmarks/resistance_chain_cloud/run.py`（LBM 计算）+ `analyze.py`（表/图）+ `tests/test_resistance_chain_cloud.py`（判据）

## ① 案例介绍（物理背景 / 几何 / 参数表）

阻力链（课题 `resistance-chain-de-mock`）的既有 8 个 mission 报告只给出**标量**证据：
总阻力 R_t / 总阻力系数 C_t / 批量成功率。缺的是 **CLOUD＝空间/角度分布证据**：
阻力到底是「哪些表面压力、哪些涡、哪段尾流」贡献的。本工作用本仓 D2Q9 LBM 求解器对
船体二维截面做自由流绕流，产出**时均场量云图 + 表面压力/摩阻系数角度分布**，
并让角度分布**积分回到阻力**（场内 → 力），从而实现「场量→阻力」的闭环闭合。

| 项 | 设置 |
|---|---|
| 几何 | 圆形截面（直径 D，Braza 锚）；椭圆截面 L/B=4 船体水线截面（潜力流解析参考） |
| 参考长度 | D = 2a（椭圆流线向半轴 a，横向半轴 b = a/4） |
| 域 | 22D × 20D，体心距入口 6D，下游 8D sponge（τ_eff = τ(1+10σ)） |
| 阻塞比 | 1/20 = 5% |
| 边界 | 入口/上下自由流 Dirichlet + 出口零梯度 + 障碍 full-way bounce-back |
| 求解器 | `tensorlbm.solver.collide_mrt` + `stream` + `d2q9.equilibrium`（库路径，零手写内核） |
| 激励 | 入口 2 周期 St=0.165、10% 幅值扰动，之后回归自由流（涡街即刻起振） |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **场量求解**：D2Q9 MRT（`collide_mrt`）+ 统一 gather streaming；τ = 0.5+3ν，ν = U_in·D/Re。
   Cd/Cl 用 Ladd 动量交换（`compute_obstacle_forces`，post-stream/pre-bounce-back），
   与仓内已验证基准 `benchmarks/verified/cylinder/re100/run.py` 同一调用序列。
2. **时均统计**：起振后按 t* 窗口逐步累加 ρ、u、v、ω_z 及其二阶矩（每 4 步采样一次），
   得到时均场、脉动 RMS 场（涡街强度）；Cd 的统计不确定度用 10 块 block-average 的 SEM → U95。
3. **表面角度分布**：取障碍的阶梯边界结点（含至少一个流体四邻域），
   在**首层流体格心**上取 p=(ρ−1)c_s²、切向速度 u_t，得
   `Cp(φ)=(p−p∞)/(½ρU∞²)`，`Cf(φ)=τ_w/(½ρU∞²)`（半程 bounce-back，壁面位于 0.5Δx，
   τ_w=2μu_t）。按 10° 分箱（36 箱）并做 2 次循环 3 点平滑。
   φ 定义：`φ=atan2((y−yc)/b, (x−xc)/a)`，φ=0 指向下游，φ=±180° 为上游驻点。
4. **场→力闭环**：`Cd_p = Σ_boundary Cp·n_x /D`、`Cd_f = Σ_boundary Cf·t_x /D`，
   与 Ladd 总 Cd 对比给出分解闭合误差（下表 `Cd_pf_rel_err_pct`）。
5. **收敛判据**：三级网格 Roache GCI（Fs=1.25，r=D₂/D₁），非单调/非渐近时给出
   `fallback_reason` 并退化为保守带。
6. **参考值**：圆截面 Cd=1.35±0.05、St=0.164~0.165（Braza et al. 1986, JFM 165；
   与仓内 `benchmarks/verified/cylinder/re100/README.md` 一致）；
   椭圆截面**无实测 Cd 参考**，故 Cd 标 NO-DATA-TBD，只用**解析潜力流 Cp(φ)**
   （保角映射 z=ζ+c²/ζ，见 `potential_cp_ellipse`）作角度参考。

## ③ 计算结果（表 1：逐网格关键量 / 参考值 / 真误差 / U95 / 场量统计矩）

## ④ 结果图（场量云图 + 与参考剖面比较）


### 表 2（节选：逐角度级 Cp/Cf 与解析参考、真误差、时间标准差、网格带）

### 表 3（尾流/空间分布统计，节选）

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **逐格动量交换不是局部量**：库内 `compute_obstacle_forces` 对**全部**固体格求和
  （full-way bounce-back 下内部格的成对项在总和中相消，但单格值不为零）。
  因此把逐格动量交换当局部力会失配（实测 `checksum_rel_err` 见 run.json，量级 O(1)），
  本工作的表面分布改用**流体侧**首层格心的 ρ 与 u_t（半程 bounce-back 标准做法），
  并用 `Cd_p+Cd_f` 与 Ladd 总 Cd 对比作为分解闭合判据。
- Re=100 二维层流涡街，**不是**实船高雷诺数湍流：本工件的作用是给阻力链补「场量/角度分布」
  证据面与场→力闭环，其 Cd/St 精度对标的是二维圆柱基准，不能外推为实船阻力精度。
- 潜力流 Cp 参考仅在**附着流窗**（|φ|>90°，前半体）有意义；分离区（后半体）差异是物理的，
  不构成误差判据。
- 二维截面 = 无限展长假设，无自由面、无兴波：兴波阻力分量不在本工件覆盖范围内
- 椭圆截面无实测 Cd 参考 → 该列标 NO-DATA-TBD，禁止把潜力流值当测量值。

## ⑥ 误差判定（逐项显式回答「是否<3%」）

| 关键量 | 本工作 | 参考值 | 误差 | 是否 <3% |
|---|---|---|---|---|

**显式回答：无可用参考值 → NO-DATA-TBD（不得声称 <3%）**。

无基准项声明原因：
- 椭圆截面 Cd/St：本仓与课题内**无**实测/文献基准登记，故 NO-DATA-TBD；
  只给出解析潜力流 Cp(φ) 的角度对比与真误差列。

**NO-DATA-TBD**：椭圆船体水线截面（L/B=4）算例在本 mission 的计算窗内未跑完，
本轮只交付圆截面场量证据；椭圆截面的字段/角度表的生成脚本已就位
（`run.py --geo ellipse`、`surface.py`、`analyze.py --geos ellipse`），缺的是算力时间，不是代码。

## ⑦ 与既有报告的关系（不改写任何既有数字）

| report id | 标题 | 本工件补的是什么 |
|---|---|---|
| #5539 | IN10-C1 冰阻力全谱与速度-功率曲线：三态模型族 + L2 锚语料 25/25 + PC6 VP 矩阵 | 空间/角度分布证据面（CLOUD）：时间平均场云图 + 表面 Cp/Cf 角度分布 + 场→力闭环 |
| #5393 | m4842 多船型标模阻力基准：Holtrop vs 公开水池试验（逐工况误差 + honest-FAIL） | 空间/角度分布证据面（CLOUD）：时间平均场云图 + 表面 Cp/Cf 角度分布 + 场→力闭环 |
| #5384 | 千工况批量计算（P6-02）：5 船型参数扫描 1300 工况 / 成功率 100.0% | 空间/角度分布证据面（CLOUD）：时间平均场云图 + 表面 Cp/Cf 角度分布 + 场→力闭环 |

既有报告的标量数字（如 #5539 的 2876 kN、#5384 的 1300 工况/100% 成功率、#5393 的逐工况误差）
原样引用，**未在本工作中重算或改写**。

