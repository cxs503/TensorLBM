# P50-N06 — 粘弹性聚合物减阻（Toms）验证面：Oldroyd-B/FENE-P 精确 Poiseuille 锚 + 可实现性门

**仓库** TensorLBM · **模块** `src/tensorlbm/viscoelastic/` · **测试** `tests/test_viscoelastic_toms.py`
**产物** `artifacts/viscoelastic/study.json` + 6 图 · **生成器** `python -m tensorlbm.viscoelastic.study`

> 证据分级：层流锚 / 可实现性 / 能量账 / 网格收敛 = **REAL**；湍流减阻 ΔB(Wi) =
> **WITHHELD**（未做粘弹性湍流 DNS/LES，如实申报，见 §5）。所有数字由
> `study.json` 程序化产出（零手抄），报告中的值可在该文件检索到同名键。

---

## 1. 案例介绍（物理背景 / 参数）

聚合物稀溶液在湍流中显著降阻（Toms 现象）。其**可机器判裁的物理内核**有三件：

1. **本构**：Oldroyd-B 上随体导数 + FENE-P 有限拉伸；
2. **层流精确锚**：Oldroyd-B 平面 Poiseuille 的稳态解；
3. **可实现性**：构型张量 `C = <R R>` 必须半正定且 `tr C >= 3`。

关键的解析事实（本验证面的物理支点）：Oldroyd-B 在**稳态剪切**中
`tau_xy = eta_p * gdot` 与 `lambda` 无关，因此

* 速度剖面**精确等于**用**总粘度** `nu0 = nu_s + nu_p` 计算的牛顿抛物线：

  `u(y) = G/(2 nu0) * [ (H/2)^2 - (y - y_c)^2 ]`，`y_c = (H+1)/2`；

* 一阶法向应力差 `N1 = tau_xx - tau_yy = 2 eta_p lambda gdot(y)^2`（随 `lambda` 线性），
  而构型张量 `C_xx = 1 + 2 Wi^2`、`C_xy = Wi`、`C_yy = C_zz = 1`，`Wi = lambda gdot`。

### 参数表（`study.json:parameters`，格子单位，`rho = 1`）

| 参数 | 值 | 说明 |
|---|---|---|
| `nu0` | 0.05 | 总运动粘度 `nu_s + nu_p` |
| `beta = nu_s/nu0` | 0.5（锚）/ 0.25 / 1.0（负控） | 溶剂占比 |
| `u_max` 目标 | 0.01（Ma ≈ 0.017） | 压住 LBM 压缩性误差 |
| `G` | `8 nu0 u_max / H^2` | 体积力驱动 |
| `Wi`（壁面） | 1.0 / 5.0 | `Wi = lambda * gdot_wall` |
| `H`（流体行数） | 32（锚）/ 16, 64（网格收敛） | 壁面在半格 `y = 0.5, H+0.5` |
| 本构模式 | 稳态（隐式局部解，`constitutive_dt = 1e6`） | 见 §2 与 §5 局限 |

---

## 2. 怎么做的（方法 / 工具链）

**求解器**（`viscoelastic/lbm.py`）：D2Q9 BGK 溶剂 + Guo(2002) 离散力 + 聚合物应力散度反馈：

1. 由 `f` 取 `rho, u`；
2. 构型张量 `C` 推进（`conformation.advance_conformation`）：上随体
   `dC/dt + u·∇C = L C + C Lᵀ - (f C - a I)/λ`，**应变+松弛隐式**（逐点 3×3 Cramer 解，
   未知量取**偏量** `Y = C - I`，使 `λ→0` 时不丢精度），对流项显式；
3. 聚合物应力 `tau_p = (nu_p/λ)(f C - a I)`，其散度 `∂_y tau_xy` 作为附加体积力；
4. BGK 碰撞（溶剂 `nu_s`）+ Guo 力项 + 流式 + **半程反弹**（壁面正好落在 `y=0.5`，
   若把反射写进壁面格点则等效宽度变 `H+1`，`u_max` 虚高 `((H+1)/H)^2`，H=16 时 10% —— 这是本节点踩过的坑）；
5. 可实现性投影（特征值夹紧 + `tr C >= 3` 保底 + FENE-P 迹上限）。

**算子**：`ddy_2nd` 内部中心差分、首末流体行用三点单侧格式（对抛物线**精确**），
使 Oldroyd-B 的离散不动点与解析抛物线的差压到 LBM 截断量级（`study.json:c1`）。

**工具链**：`torch float64`、`pytest`、matplotlib（`Agg`）；全程 `PYTHONPATH=src`。

---

## 3. 计算结果（判据数字，来自 `study.json:criteria`）

| 判据 | 要求 | 实测 | 判定 |
|---|---|---|---|
| c1 速度剖面（Wi=1，H=32，32 点） | ≤ 2% | **0.104%** | PASS |
| c1 速度剖面（Wi=5，H=32，32 点） | ≤ 2% | **0.103%** | PASS |
| c1 `N1` 偏差（Wi=1 / Wi=5） | ≤ 3% | **0.027% / 0.023%** | PASS |
| c2 牛顿极限（λ→0 退化） | ≤ 1e-6 | **9.39e-3（H=32）** | **FAIL（如实）** |
| c3 可实现性（投影开，Wi=5 全域） | `tr C ≥ 3-1e-9`、`C ⪰ 0` | `min tr C = 3.000000`，`min eig = 0.505` | PASS |
| c3 负控（关投影，显式高 Wi） | 必须越界 | 第 1 步 `min eig = -1.0` | PASS |
| c4 能量账（解析锚场） | ≤ 1e-6 | **2.5e-16** | PASS |
| c4 能量账（LBM 场） | — | 1.18e-3（受截断限制，网格加密 8.7e-4 → 1.7e-4） | 报告值 |
| c5 减阻律 ΔB(Wi) | 单调且 ≤ 11.7 | **未测（WITHHELD）** | **FAIL（如实）** |
| c6 网格×2 误差比 | ≥ 3.2 | **4.01（观测阶 2.00）** | PASS |
| c7 负控 β=1 | 减阻 0 / 应力 0 | Δu = 0.0，`N1 = 0` | PASS |
| c7 负控 关上随体导数 | `N1` 偏差 > 20% | **100%** | PASS |
| c7 负控 Wi 超上限 | 明确报发散 | `ViscoelasticDivergenceError: tr C=34.69 ≥ L²=20` | PASS |
| c8 FENE-P 支 | 可实现 + 剪切变稀 | `tr C_max` 4.37/11.56 < L²=50；流量 +0.36%/+4.9% | PASS |

**能量账分解**（`c4`，Wi=5 锚）：`P_in = 8.326e-7`，`eps_s = 4.155e-7`，`eps_p = 4.163e-7`，
两者各占 50%（`beta = 0.5` 自洽），弹性能变化率 `dE_el/dt = -2.5e-10`（稳态≈0）。

**FENE-P 稳定性**：`Wi=3` 时 `tr C` 峰值 11.56 < `L² = 50`，投影后 `min tr C = 3.000`。

### 图（`artifacts/viscoelastic/`）

| 图 | 内容 |
|---|---|
| `fig1_oldroyd_b_profiles.png` | LBM 剖面 vs 解析抛物线（Wi=1, 5）+ 逐点偏差对数图 |
| `fig2_conformation_and_n1.png` | `C_xx`、`C_xy`、`N1` 场 vs 解析（`N1` 相对误差 2.3e-4） |
| `fig3_realizability_gate.png` | 投影/未投影的 `min eigenvalue` 与 `min tr C`（红虚线 `tr C = 3`） |
| `fig4_grid_convergence.png` | 剖面误差 vs H（log-log）+ 斜率 -2 参考线 |
| `fig5_virk_reference.png` | Virk 三段律 + MDR 渐近线（ΔB = 0/2/5/8），注明未测 |
| `fig6_energy_budget.png` | `P_in / eps_s / eps_p` 分解（解析锚 vs LBM 场） |

---

## 4. 能量账的离散一致性与逐项误差判定

预算用 **work form**：`P_in = Σ G u_j`，`eps_s = -nu_s Σ u_j L2(u)_j`，`eps_p = -Σ u_j D(tau_xy)_j`，
其中 `L2` 是**遵守半格壁面**的二阶拉普拉斯（首行 `-4u_1 + (4/3)u_2`）。于是

`P_in - eps_s - eps_p = Σ_j u_j r_j`（`r_j` = 离散动量残差），**无求和分部假设、无边界项**。
解析锚场（二次速度 + 线性剪应力）被整套算子精确表示 ⇒ 残差 2.5e-16；
LBM 场残差 1.18e-3 即截断误差，网格加密时 8.7e-4 → 1.7e-4（降 5.1 倍）⇒ 账本实现正确，
残差不是"声称的 1e-6"，而是**如实测量的 1.18e-3**。

**误差判定表（关键量 vs 基准）**

| 关键量 | 基准 | 误差 | < 3%？ |
|---|---|---|---|
| `u(y)` 逐点（Wi=1, H=32） | Oldroyd-B 解析抛物线 | 0.104% | 是 |
| `u(y)` 逐点（Wi=5, H=32） | 同上 | 0.103% | 是 |
| `N1(y)` 峰值（Wi=1/5） | `2 eta_p lambda gdot^2` | 0.027% / 0.023% | 是 |
| `C_xy(y)`（Wi=5） | `Wi(y)` | < 2%（实测 0.02%） | 是 |
| `P_in` 闭合 | 解析锚 | 2.5e-16 | 是 |
| λ→0 剖面 vs 牛顿 | 纯牛顿解 | 9.4e-3（H=32） | 是（但未达 1e-6 门，见 §5） |
| ΔB(Wi) | Virk MDR | — | **无基准（未测）** |

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **`C ⪰ I` 不是可实现性条件**（本节点最大坑）。Oldroyd-B 稳态剪切下
   `C = [[1+2Wi², Wi],[Wi, 1]]` 的最小特征值是 `1+Wi²-Wi√(1+Wi²)`（Wi=1 时 **0.586 < 1**）。
   早期版本把投影下限设成 1，结果 `C_xy` 被削掉 14%、剖面误差 17%。物理条件只有
   **`C ⪰ 0` 且 `tr C ≥ 3`**。`tr C ≥ 3` 单独也不充分（`[[1,2],[2,1]]` 迹为 3、特征值 -1），
   门必须同时测两者（`test_trace_gate_alone_is_insufficient`）。
2. **半程反弹写进壁面格点 = 壁面外移一格**：等效宽度 `H+1`，`u_max` 虚高 `((H+1)/H)²`
   （H=16 时 +10%，正是第一次 c1 不过的原因）。反射必须写回**流体**首行。
3. **首末流体行的算子必须对抛物线精确**（三点单侧）；用「壁面格点值 + 中心差分」会在
   壁邻点引入 O(1) 误差，直接毁掉 2% 门。
4. **耦合收敛**：聚合物力滞后反馈在 `nu_p/nu_s ≈ 1` 时振荡；本节点用
   「本构稳态局部解（`constitutive_dt=1e6`）+ 力松弛」把迭代变成单调收缩。
   **代价**：这是**稳态**求解路径（验收门全部针对稳态锚），时间精确的瞬态解只在
   均匀剪切本构层验证过稳定性与可实现性。
5. **c2 牛顿极限未达 1e-6（honest FAIL）**：聚合物应力以**中心差分力**进入动量方程，
   溶剂粘度则是 LBM 松弛算子，两个**离散**不动点只相差截断量级：
   3.88e-2 (H=8) → 1.89e-2 (H=16) → 9.39e-3 (H=32)，每次加密约减半（观测阶 1.04）。
   因此正确表述是"λ→0 时以离散精度收敛到牛顿解"，而非固定网格上 1e-6 相等；
   要拿到 1e-6 需让两个算子同一化（例如把聚合物应力并入 LBM 有效松弛时间）。
   `N1`（∝λ）在 λ≲0.5 时被 `C_xx-1` 的相消吃掉有效位，故小 λ 段只验证"可忽略"。
6. **c5 减阻律未测（honest FAIL / WITHHELD）**：2D 体积力槽道在 `Re_tau < 5772` 会
   再层流化；3D FENE-P 槽道 `Re_tau ≥ 100` 需 ~2.6e6 格子点 × 1e5 步，超出本节点预算。
   本节点只交付 **Virk 三段律 + MDR 上界 + 验收机具**（单调、`ΔB ≤ 11.7` 上限、
   log-law 截距反演 `delta_b_from_log_law`），**不声称任何 ΔB(Wi) 测量值**。
7. **未实现 Fokker-Planck LBM**：构型张量是"LBM 格点上的有限差输运 + 隐式局部解"，
   不是在构型空间布分布函数的 LBM。任务书里"(a) 用构型张量 C 的 LBM 演化"按此实现，
   差异如实登记。
8. 适用范围：层流平面 Poiseuille、`Wi ≤ 5`、`beta ∈ [0.25, 1]`、`H ∈ [12, 64]`、`Ma ≤ 0.02`。
   高 Wi 稳态 Oldroyd-B 在真实槽道中物理上会失稳（本节点的稳态路径不覆盖该瞬态失稳）。

---

## 6. 复现

```bash
PYTHONPATH=src python -m tensorlbm.viscoelastic.study --out artifacts/viscoelastic   # study.json + 6 图
PYTHONPATH=src python -m pytest tests/test_viscoelastic_toms.py -q                    # 37 项（含 3 组负控 + 1 个 strict-xfail 记录 honest FAIL）
```

`tests/test_viscoelastic_toms.py` 中的 `test_newtonian_limit_strict_1e6_gate` 以
`pytest.mark.xfail(strict=True)` 固化 c2 的 honest FAIL：它**必须失败**，一旦哪天真通过，
说明算子被统一，届时删除该标记并更新本文件。