# m23014 · DFG Re=100 圆柱升力幅值 35% 偏差：四假设逐一检验

> 锚点 **LBM-P0-DFG** · 仓 TensorLBM · 分支 `agent/m23014` · 案例 `CASE-CYLINDER-DRAG-RE100`
> 预注册：`docs/m23014/PREREG.json`（新算例之前冻结）· 判定引擎：`docs/m23014/analyse_m23014.py`
> 本文件由 `docs/m23014/make_doc.py` 从 `analysis.json` **程序化生成**，零手抄；重跑即重渲染。

## ① 案例介绍

物理背景：槽道内圆柱绕流，Re=100 的层流卡门涡街。几何按 Schäfer & Turek (1996)
DFG 基准 **case 2D-2**：槽道 2.2×0.41，柱心 (0.2, 0.2)，D=0.1，堵塞比 D/H=0.2439；
入口抛物线 (U_max/U_mean=1.5)，出口 p=0，壁面/柱面半程反弹，力用 Ladd 动量交换。
被质疑的头条数字是升力幅值误差约 **+35%**。

| 参数 | 值 |
|---|---|
| Re = U·D/ν | 100 |
| U_mean（格子） | 0.0833（τ 随 D 变，Re 精确保持） |
| 松弛/碰撞 | MRT（`tensorlbm.solver.collide_mrt`），D2Q9 |
| 边界 | Zou/He 入口(抛物线)+压力出口；壁面与柱面半程反弹 |
| 统计窗 | transient_periods=5，avg_periods=12（瞬态显式剔除） |
| 升力幅值口径 | 脱落频率上的最小二乘基频幅值（非整条记录峰峰值） |

## ② 怎么做的

1. **先冻结判据**：`PREREG.json` 写死四假设、外锚（S&T Table 3）、量化验收
   （Cl<10% 且 Δp<5% 且 Cd<5% **同时**达成；收敛跨度 <3%）、以及决策规则与停止规则。
   阈值在看到任何新结果之前固定，事后不得改动。
2. **不重复实现**：开工先查 `docs/AUDIT_m16827_CYLINDER_DFG_CL_AMP.md` —— 该单已把
   +36.04% 分解为「~13.4pp 估计器伪差（整条记录峰峰 + 瞬态入窗）+ 残余 +17.8% 真误差」，
   并明确留下 **未做 D≥56 档**、**未重跑** 两项欠账。本单接的是这两项欠账。
3. **估计器口径统一**（本单第一个坑）：冻结的 `level_D*.json` 由 m16827 修复**之前**的代码写出，
   其 `cl_amp` 是旧口径（整条记录峰峰）。若直接拿新算例的新口径数字与之比较，就是本单要查的
   那类错误本身。因此冻结档的四个观测量一律取自 `rejudge_cl_amp.json` 的 `values_new`
   （同一无瞬态窗口重判），新档用 `run_case` 自带新口径，两侧一致。
4. **四假设各自的判别性实验**（全部预注册）：
   - ① 网格：补 D48/D56 两级（`run.py --levels 48/56`），看悬崖是否收敛、跨度是否 <3%。
   - ② 黏性/τ：同 D40 几何、Re=100 不变，把 U_mean 由 0.0833 降到 0.06（τ 0.600→0.572）。
   - ③ 边界/堵塞：用求解器**同一个** `cylinder_mask` 重算离散面积、堵塞比、半程反弹有效直径、
     探针位置；不重跑物理。
   - ④ 采样：冻结时序上做「瞬态占比」分解 + 窗长 4→12 周期扫描。
5. **独立验证**：本单的独立估计器（ACF 定周期 + 最小二乘基频）必须复现 m16827 的冻结值
   到 1% 以内才允许用于结论（`tests/test_m23014_dfg_hypotheses.py` 锁定），
   避免结论是我的第二次实现伪差。
6. **计算**：重算全部递交调度系统（`cj4520`=D56、`cj4521`=D48、`cj4522`=D40@u0.06），
   本机只跑 <8 核、<10 min 的几何审计与冻结时序分析。

## ③ 计算结果

### 收敛阶梯（3 档；冻结档取自 m16827 重判，新档为本单新增：—）

| level | D [cells] | steps | τ | C_D | err | C_L amp | err | St | err | Δp* | err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D20 | 20 | 14000 | 0.550 | 3.4558 | +6.991% | 1.1786 | +17.857% | 0.3097 | +3.224% | 5.3453 | +7.768% |
| D28 | 28 | 20000 | 0.570 | 3.3872 | +4.868% | 1.1775 | +17.749% | 0.3035 | +1.168% | 5.3682 | +8.231% |
| D40 ✅ | 40 | 27200 | 0.600 | 3.2608 | +0.953% | 1.0283 | +2.835% | 0.3011 | +0.379% | 5.1653 | +4.139% |

✅ = 该档 Cd/Cl/Δp 三项同时落入 5% 容差。

### 相邻档收敛跨度

| pair | C_D | C_L amp | St | Δp* |
|---|---|---|---|---|
| D20->D28 | 2.02% | 0.09% | 2.03% | 0.43% |
| D28->D40 | 3.88% | 14.50% | 0.79% | 3.93% |

判据：跨度 <3% 才算收敛。

### 四假设判决表

| hypothesis | discriminating test | number | verdict |
|---|---|---|---|
| ① grid convergence (≥3 levels) | cliff at D28→D40; span test | span C_L D28->D40 = +14.503% | CONFIRMED as the residual mechanism |
| ② turbulence / viscosity model | same D40 geometry, U_mean 0.0833→0.06 (τ shift, Re=100 fixed) | frozen D20 pair, Δτ=0.014: C_L amp -4.112%, Cd -1.285%, Δp -1.566%, St -0.309% → tau_model_marginal | marginal / not the driver |
| ③ boundary distance / blockage | mask area vs πr², blockage vs 0.1/0.41, half-way BB effective diameter | area within 1%; nominal blockage = spec; D_eff = D+1 (O(1/D), monotone) | falsified as the explanation (monotone bias cannot produce a non-monotone error) |
| ④ sampling window / duration | transient share + window-length drift 4→12 periods | transient D20 4.1%, D28 11.9%, D40 22.2%; window drift D20 1.50%, D28 0.74%, D40 2.18% | transient CONTRIBUTES; sampling duration falsified (<2%) |


## ④ 结果图

- `docs/figures/m23014_dfg_hypotheses.png`：四联图。(a) 四个观测量按参考带中点归一化随分辨率变化，
  灰色带=±5% 容差；(b) 带符号误差 vs 分辨率；(c) **相邻档跨度**——收敛判据的直接可视化，
  红线=3% 判据；(d) H4 窗长稳定性与 H3 半程反弹几何偏差（D_eff=D+1，O(1/D)）的对照。
- `docs/figures/m23014_cl_series.png`：最粗与最细档的 C_L(t)，红色区=被旧口径吞进统计窗的
  启动瞬态——这就是「整条记录峰峰值」为何系统性高估幅值的直接证据。
- 图为 matplotlib 程序化生成，图中数字由 `analysis.json`/`h3_h4_audit.json` 注入。
  本单为数值验证/根因判定，二维算例的场量云图不作为判定依据（判定量是力的时间历程与标量统计）。

## ⑤ 注意点（坑、局限、适用范围）

1. **口径陷阱（本单最大风险）**：新旧两代 `cl_amp` 语义不同。若不区分，D48/D56 会看起来
   「突然变好」而实际只是换了估计器。分析器现在按档记录 `cl_amp_estimator` 与
   `window_provenance`，并保留旧峰峰值只作审计。
2. **收敛不是单调的**：Cl 幅值在 D20/D28 平台约 +17.8%，D40 骤降到 +2.8%。
   这种「悬崖」形态**不能**由任何平滑的 O(1/D) 几何偏差解释（H3 的 5.00%→1.79% 是单调的），
   因此 H3 即便存在也不能当根因；真正的问题是**达到悬崖的临界分辨率在哪**。
3. **τ 与网格绑定**：固定 Re 与固定 U_mean 加密网格必然使 ν_lb∝D、τ 上升
   （0.550/0.570/0.600/0.620/0.640）。这是正确的定马赫加密，但意味着
   「加密」同时改变了 τ，故 H2 用**同几何变 U_mean** 单独隔离 τ，避免与 H1 混淆。
4. **参考带的证据级别**：C_L 幅值带 0.99–1.01 是仓内转抄（S&T 原表主列是 Cd/Δp/St），
   属 **WITHHELD** 级溯源，本单无网络无法核对；因此「Cl 落入带内」的证据强度低于
   Cd/Δp/St 落入带内。Cd/Δp/St 三带是文献通用值。
5. **未做**（如实登记为 not_run，不算通过）：与独立 CFD/文献逐值对表；D=64 以上档
   （仅在 D48↔D56 跨度为 3–6% 的边界情形下按预注册的应急条款才追加）。

## ⑥ 误差判定（关键量与基准逐项；显式回答是否 <3%）

| quantity | reference | value | error | within tol? | tol |
|---|---|---|---|---|---|
| cl_amp (D40) | 0.99–1.01 (S&T Table 3 transcription) | 1.0283 | +2.835% | YES | 10% |
| cl_amp (D40) | same, matrix tolerance | 1.0283 | +2.835% | YES | 5% |
| cd (D40) | 3.22–3.24 | 3.2608 | +0.953% | YES | 5% |
| dp (D40) | 4.92–5.00 | 5.1653 | +4.139% | YES | 5% |
| st (D40) | 0.295–0.305 | 0.3011 | +0.379% | YES | 5% |


### 偏差与 WITHHELD 登记（预注册计划 vs 实际执行）

**fail_stage: `convergence-ladder`** — The two new refinement levels (D48, D56) could not be executed: the compute scheduler kept them queued for the entire mission window (76 min, in-flight queue growing 87 -> 108, nodes healthy). The convergence branch of the pre-registered decision tree is therefore WITHHELD. Everything that could be decided from frozen data plus zero-compute geometry analysis was decided and is reported.

| id | 预注册 | 实际 | 对判决的影响 |
|---|---|---|---|
| D1 | Execute two new grid levels D48 and D56 (hybrid: `run.py --levels 48 56`) to decide the H1 convergence branch. | Submitted cj4520 (D56, 40000 steps) and cj4521 (D48, 34000 steps) to the scheduler. Neither started: cj4520 sat 'queued' for 76.2 min while the scheduler's in-flight count rose from 87 to 108; cj4521 was withdrawn at 50.7 min. | The H1 convergence branch is WITHHELD, not decided. The verdict is therefore applied at the finest EXECUTED level (D40) instead of the pre-registered finest (D56). |
| D2 | Run D40 at U_mean=0.06 (tau 0.600 -> 0.572) as the H2 discriminator. | cj4522 submitted, never started (same queue), withdrawn. | No weakening for the branch decision (H2 came out 'marginal' on both the frozen D20 pair and the pre-registered margin), but the smaller tau span (0.014 vs the planned 0.028) means H2's sensitivity is bounded, not tightly measured. |
| D3 | Interrupt the analysis with an extra D64 level only if the D48<->D56 span falls between 3% and 6%. | Not reached (D48/D56 never ran). | none |

预注册阈值**未改动**；改变的是「哪些实验被执行」。后续任务可直接用
`docs/m23014/DEVIATIONS.json` 里的 `resubmit_verbatim` 逐字重投。


**是否 <3%？** 逐项回答：
- `cd` = +0.953% → **是**
- `cl_amp` = +2.835% → **是**
- `dp` = +4.139% → **否**
- `st` = +0.379% → **是**

**判定（预注册规则，未事后改动）：`HONEST_FAIL_CONVERGENCE`**
— PREREG rule 2: primary gate met but span gate failed -> no converged level

- 主门控（Cl<10% 且 Δp<5% 且 Cd<5%，最细档 D40）：
  Cl +2.835%、Δp +4.139%、Cd +0.953%
  → **通过**
- 次门控（相邻档跨度 <3%）：C_L +14.503%、
  Δp +3.929%、Cd +3.878%
  → **未通过**

矩阵状态**不由本单改动**；若出现 honest-FAIL，FAIL 行保留、不抹除、不静默删除。
