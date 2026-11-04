# 2D 圆柱绕流阻力验证欠账：把 Step E 从「无证据」推进到「有证据的不收敛」 (TensorLBM, m3650)

## 摘要

`docs/LBM_VERIFICATION_TEACHING.md` 的 **Step E** 声称「圆柱 Re=200, Cd=1.30 (Schäfer–Turek)，
D=24→48→96→200 收敛到 0.8%」，但同时存在两个问题：(1) **基准归属错**——Schäfer–Turek（DFG 2D-2）
是 **Re=100、带约束 D/H≈0.244** 的算例，Cd≈3.22–3.24、St≈0.295–0.305，不是 Re=200/Cd=1.30；
(2) **零可执行证据**——仓内 `tests/test_cylinder_grid_convergence.py` 喂给分析器的是**合成记录**
（`cd = 1.33 + 10/D**2`），它验证的是分析器的算术，不是求解器；`tests/test_cylinder_cd.py`
的判据是 `1.0 < Cd < 4.0`，3 倍误差照样绿。

本单把这条欠账做成**可执行证据**：写了 `tests/validation/cylinder_drag_benchmark.py`（真跑
D2Q9 BGK + 半路反弹 + Ladd 动量交换力），按 DFG 2D-2 几何（D/H=0.2439、2D 上游、20D 下游）
与两个边界集实跑，并逐次记录**定常态判据**（双窗均值漂移 <1%）。结论是**诚实的 FAIL/WITHHELD**：

* `apply_simple_channel_boundaries`（plug 入口 + 零梯度出口，无压力锚点）**阻力不收敛**：
  无质量修正时 Cd 从 2.86 单调爬到 3.99（+39%，2.8×10⁴ 步）；加 `correct_mass` 固定总质量后
  仍在爬（B=0.125 算例 1.84→2.11，+14.6%）——说明病根是**动量收支**而非总质量。
* `apply_zou_he_channel_boundaries`（Zou/He 速度入口 + 压力出口）在 Ma=0.17 与 Ma=0.10 下
  **都发散成 NaN**，因此本仓目前**没有**能支撑定量圆柱阻力的通道边界集。
* 唯一达标停机的算例（DFG 几何 + simple BC + correct_mass，漂移 0.60%）给出
  **Cd=3.0438（对 3.23 差 -5.76%）**、**St=0.2623（差 -12.56%）**，
  但它停在一个**尚未真正定常**的平台上（后续再跑即恢复上漂，见下），所以这些数字**不作为验证值引用**。

因为「不收敛窗口里的数字」不能当验证，圆柱 Cd 仍标 **WITHHELD**——但与之前不同：现在是
**有脚本、有日志、有判据、有归因**的 WITHHELD，任何一次修边界集之后可以一条命令复算。

## ① 案例介绍

**物理背景**：圆柱绕流是钝体阻力的标准验证题，阻力由壁面动量交换（Ladd 1994）给出，
`compute_obstacle_forces` 在后迁移、反弹之前对固体格点做 `F=2Σc_i f_i`。可验证量有两类：
**Cd**（阻力系数，对边界条件/阻塞比极敏感）与 **St**（涡脱频率，对分辨率中等敏感）。

**几何与参数**（DFG 2D-2 相似配置，格点化到 D=10）：

| 量 | 取值 |
|---|---|
| 圆柱直径 D | 10 格点 |
| 通道高 H | 41 格点（D/H = 0.2439，与 DFG 相同） |
| 计算域 | 上游 2D + 下游 20D（Lx=220 格点） |
| 雷诺数 | Re = U·D/ν = 100 |
| 格子速度 U | 0.10（simple）／0.06（zou_he 复测） |
| ν / τ | 0.01 / 0.53（U=0.10），0.006 / 0.518（U=0.06） |
| 弛豫 | BGK（`collide_bgk_matmul`，与 `collide_bgk` 逐位一致到 1.3e-5） |
| 边界 | 入口 plug、出口零梯度（simple）或 Zou/He 速度入口+压力出口；上下壁+圆柱半路反弹 |
| 统计 | 最小预热 8000 步（3.6 个流通时间）+ 自适应窗口，漂移 <1% 才停机 |

**基准（文献，按适用范围严格分开）**：DFG 2D-2 约束算例 Cd=3.22–3.24、St=0.295–0.305
（Schäfer & Turek）；无约束 2D Re=100 则 Cd=1.32–1.35（Dennis & Chang 1970）、St=0.164–0.166。
**二者不可混用**——原文档正是把无约束量级（1.30）挂到了约束基准名下。

## ② 怎么做的

1. **单一真源**：`tests/validation/cylinder_drag_benchmark.py`，`Config`（几何/Re/步数/边界集/质量修正）
   → `run_cfg` → `Measurement`（Cd 均值/脉动、St、定常性漂移、停机原因、墙钟时间）；
   `--pilot/--only/--full` 三种粒度，`--emit-dir` 落 JSON+CSV。
2. **先把性能坑钉住**：在 load15≈420/128 核的机器上，默认 torch 线程池让 9k 格点的小张量跑成
   **~500 ms/步**；`torch.set_num_threads(1)` 后降到 **~7 ms/步**（快 70 倍），`collide_bgk_matmul`
   再省 1.8 倍。所有结果都标了 `wall_seconds`。
3. **定常态闸门**：预热后每 2000 步比较前后半窗 Cd 均值，漂移 <1% 才判 `stationary`，否则跑满
   `max_steps` 并标 `max_steps`；NaN 记为 `diverged`。**只有 stationary 窗口的数才可引用**。
4. **诊断实验**：① simple BC 原样跑 2.8×10⁴ 步（看是否收敛）；② simple BC + `correct_mass`
   （区分「总量漂移」与「动量失衡」）；③ zou_he BC 在 Ma=0.17/0.10 两个马赫数下复测（看是否稳）。
5. **文档纠正**：`docs/LBM_VERIFICATION_TEACHING.md` 的 Step E 段落改写为正确的基准归属 +
   指向本报告的 WITHHELD 判据；`tests/test_cylinder_drag_benchmark.py` 里加一条**文档漂移闸门**，
   防止 `Cd=1.30 (Schäfer-Turek)` 这类错误归属再写回文档。

## ③ 计算结果（程序化注入，零手抄）

| run | D | H(格点) | B=D/H | 网格 | Ma | ν | St | 状态 | 墙钟 s |
|---|---|---|---|---|---|---|---|---|---|
| dfg2d2_simple | 10 | 41 | 0.2439 | 220x41 | 0.173 | 0.0100 | 0.2623 | stationary | 78 |
| dfg2d2_zouhe | 10 | 41 | 0.2439 | 220x41 | 0.104 | 0.0060 | NaN | diverged | 161 |

| 量 | 实测 | 基准 | 文献区间 | 相对误差 | 是否 <3% |
|---|---|---|---|---|---|
| Cd (dfg2d2_simple, drift 0.60%) | 3.0438 | 3.2300 | 3.22-3.24 | -5.76% | 否 |
| St (dfg2d2_simple) | 0.2623 | 0.3000 | 0.29-0.30 | -12.56% | 否 |

**漂移证据（从运行日志直接解析，非手抄）**

| run | 观测 | 结论 |
|---|---|---|
| dfg2d2_simple (no mass correction) | Cd 2.86 (t=2500) -> 3.37 (t=10000) -> 3.99 (t=28000), monotone +39% | unbounded density pumping: plug inlet + zero-gradient outlet, no pressure anchor
| free_B0.125 (with correct_mass) | Cd 1.84 (t=1.0e4) -> 2.11 (t=2.4e4), monotone +14.6% | mass pinning does not remove the momentum imbalance of the BC set

## ④ 结果图

左图：三条实跑阻力历史（simple 无修正 / simple+B=0.125 带质量修正 / zou_he），叠加 DFG 锚点 3.23
与无约束区间 1.32–1.35，可见无一停到平台（zou_he 在 NaN 前冲出画幅）；右图：唯一达标停机窗口的
Cd/St 对 DFG 锚点的相对误差，虚线为 ±3% 判据线。数据取自 `docs/cylinder_drag_registry.json`
与运行日志，图由 `tests/validation/make_cylinder_drag_figure.py` 生成。

![cylinder drag](assets/cylinder_drag_benchmark.png)

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

- **基准归属是验证的第一道门**：同一句「Schäfer–Turek 圆柱基准」可以指约束（Cd≈3.23, Re=100）
  或无约束（Cd≈1.32, Re=100）两个完全不同的算例，混用会让「误差 0.8%」这类结论纯属巧合。
- **plug 入口不是 DFG 的抛物入口**：DFG 中心线速度为均值的 1.5 倍，plug 入口下圆柱见到的
  来流只有 DFG 的三分之一到一半，所以即便几何相同，Cd/St 也不该直接对标 DFG；这也是本单
  把 dfg2d2 数字标为「不可引用」的原因之一。
- **质量守恒 ≠ 动量守恒**：`correct_mass` 固定总质量后阻力仍在爬，说明 `simple` 边界的病灶在
  入口/出口的动量通量不平衡（出口零梯度复制会反射），不是密度漂移。
- **Zou/He 通道边界在本仓当前实现下发散**（Ma=0.17 与 0.10 皆 NaN），Ma 降档不是解药；
  若要继续，应走兼容性/特征边界（`nscbc_outlet_2d`、`far_field_bc_2d`、
  `stabilize_outlet_backflow`）或周期-自由流构造。
- **定常态闸门必须显式**：早期版本预热只给 2000 步（0.9 个流通时间），Cd 还在 2.86 就统计，
  得出的 3.08 与后来的 3.04、3.37 全都「看起来像结果」。宁可标 FAIL，也不要发布非定常窗口均值。
- 适用范围：D2Q9/BGK、半路反弹、定常或周期涡脱、单相不可压；不含 MRT/cumulant、
  插值反弹（弯壁精度）、湍流模型。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---|---|
| dfg2d2 Cd（漂移 0.60% 的窗口） | DFG 2D-2 = 3.23 | -5.76% | **否**（且窗口未定常，不作验证值） |
| dfg2d2 St | DFG 2D-2 = 0.300 | -12.56% | **否**（同上） |
| zou_he 两个马赫数 | — | NaN（发散） | **无法判定** |
| free_B0.125 Cd（B 扫描点） | 无约束 1.32 需 B→0 外推，单点不可比 | 漂移 +14.6%，不可用 | **无法判定** |

结论：**本单没有任何一项圆柱阻力数字达到 <3% 的可引用标准**。如实结论是：仓库当前的通道
边界集不足以支撑圆柱阻力定量验证，这是**边界条件欠账**（不是 LBM 内核欠账——内核已由
Step A–D 的解析链证明到 ≤0.34%）。圆柱 Cd 维持 **WITHHELD**，但已从「零证据」升级为
「有可执行反证据 + 明确归因 + 复算命令」。

## 证据指针

- 代码：`tests/validation/cylinder_drag_benchmark.py`、`tests/test_cylinder_drag_benchmark.py`、
  `tests/validation/make_cylinder_drag_figure.py`
- 注册表：`docs/cylinder_drag_registry.json`（含 anchors / runs / comparisons / drift_observations / verdict）
- 原始日志：`/tmp/cyl_full2.log`（无修正漂移）、`/tmp/cyl_v3.log`（correct_mass）、`/tmp/cyl_zh2.log`（zou_he NaN）
- 复算：`OMP_NUM_THREADS=1 python -m tests.validation.cylinder_drag_benchmark --full --emit-dir artifacts/cylinder_drag`
- 测试：`python -m pytest tests/test_cylinder_drag_benchmark.py -q`
