# MCMP 分层 Poiseuille 验收判据的操作特性审计（gate operating characteristics）

**锚点**：`FEAT-LBM-VERIFY-BENCH`（TensorLBM MCMP 校准与**证伪**基准工装，feature_ledger status=in_progress）
**任务**：m16895（paper 席自主工时）
**日期**：2026-11-12
**代码**：`src/tensorlbm/audit/gate_oc.py`（通用判据机器，零领域语义）、
`cases/mcmp_stratified_poiseuille/gate_oc_run.py`（本算例判据集，领域侧）
**测试**：`tests/test_gate_oc_mcmp.py`（16 passed）
**产物**：`benchmarks/verified/mcmp_stratified_poiseuille/gate_oc.json`、`figures/gate_oc/fig{1,2,3}_*.png`
**不重跑任何 LBM**：本单是纯后处理审计（本机 load15≥125，按容量纪律只做轻量活）。

---

## 1. 案例介绍（问题背景）

`GAP-LBM-MCMP-POISEUILLE` 的验收记录由四条判据构成：界面位置 <5 %、两相流量 <10 %、
静态控制伪电流 / 特征速度 <0.5 %、误差随分辨率单调下降，另有两条前置闸门（分层 φ>0.5、
定常漂移 <1e-3）。上游对抗复核（m16034）已发现 R3/R4 两条记录用**不同** `u_max_analytic`
参考归一化同一判据（差 1.3 %）。

本单问的是更根本的一层：**这些判据本身能不能失败？** 一个从未在任何缺陷测量上返回 FAIL
的判据不是证据，是装饰。本仓历史上反复出现的失效模式不是"数字算错"，而是
（a）**恒真判据**（m16308 对 RCP 锚点的判定）、（b）**记录阈值与代码比较值静默不一致**。
本单为此建立机器：判据 = 显式声明（metric 键 + 运算符 + 阈值 + 依据一句话），然后被
三角化、被开关点探测、被变体（mutant）证伪。

## 2. 怎么做的（方法 / 关键设置 / 工具链）

三层机器（`src/tensorlbm/audit/gate_oc.py`，全部纯函数、无 I/O、无领域常量）：

1. **判决语义**：`PASS` / `FAIL` / `WITHHELD` / `ERROR`。缺失或非有限的度量一律
   `WITHHELD`，**永不 PASS**（判据不能被"没有数据"满足）；聚合时 `WITHHELD` 会拖低总判。
2. **阈值三角化**：同一阈值必须在三方独立载体上一致 —— 审计声明值 / 验收记录
   `thresholds` 块 / **生产源码中实际比较的常量**。源码常量用 `ast` 解析
   （`parse_source_constant`：不 import、不执行，因此不受 torch 等重依赖与运行时改写影响）。
   任一方不一致即记 defect；`None` 记 **missing evidence**（不是"一致"）。
3. **开关点 + 灵敏度 + 变体证伪**：`switch_probe` 在 `t(1±1e-3)` 两侧取值，要求
   下侧 PASS、上侧 FAIL —— 抓"隐藏宽松"（`x < 2t`）与"比较取反"；`detection_curve`
   给出被判据看见的最小缺陷；`discriminate` 用合规/缺陷混合探针集检验若干**退化变体**
   （阈值×2、阈值×0.5、运算符取反）是否全部被杀 —— 变体存活 = 探针集（即"已验证"声明）
   证据不足。

领域侧脚本 `gate_oc_run.py` 只做数据装配：声明 6 条判据（值与依据对应 SPEC §6 与
`result.json` 的 `thresholds` 块），从 `result.json` / `result_r3.json` 抽取每条记录度量，
从 `r3_benchmark.py` / `r4_benchmark.py` **解析** `TARGET` 常量，然后：三角化 → 机器复算
每条记录的 PASS/FAIL 声明 → 计算利用率（`|metric|/阈值`，1.0=判据触发）→ 生成报告与图。

工具链：Python 3.11 + matplotlib(Agg)；无 torch、无 LBM 运行；全程 <2 s。

## 3. 计算结果（零手抄：本表数字全部取自 `gate_oc.json`）

### 3.1 判据集与记录上的判决分布

| 判据 | 运算符 | 阈值 | PASS | FAIL | WITHHELD | 利用率 min/max | 开关点探测 | 开关点取值 |
|---|---|---|---|---|---|---|---|---|
| iface_5pct | abs_le | 0.05 | 19 | 4 | 10 | 0.2759 / 18.29 | PASS->FAIL | 0.04995 / 0.05005 |
| flow_mix_10pct | abs_le | 0.1 | 11 | 12 | 10 | 0.01414 / 1.609 | PASS->FAIL | 0.0999 / 0.1001 |
| flow_component_10pct | abs_le | 0.1 | 17 | 6 | 10 | 0.1175 / 1.309 | PASS->FAIL | 0.0999 / 0.1001 |
| spurious_static_0p5pct | abs_le | 0.005 | 0 | 3 | 30 | 13.24 / 69.97 | PASS->FAIL | 0.004995 / 0.005005 |
| stratified_phi_0p5 | ge | 0.5 | 16 | 4 | 13 | 0.5 / 3.484 | PASS->FAIL | 0.5005 / 0.4995 |
| steady_1e-3 | abs_le | 0.001 | 0 | 0 | 16 | —（全 WITHHELD） | PASS->FAIL | 0.0009995 / 0.0010005 |

**五条判据在真实记录上都真的返回过 FAIL**（非恒真）；**第六条（定常闸门）在记录上是
100 % WITHHELD** —— 记录只存了布尔 `steady`，没有存 `steady_rel_change` 数值，因此这条
判据**无法被独立复算**，只能被"信任"。这是本单发现的第一条实质缺陷（见 §5-①）。

### 3.2 声明值 vs 记录值 vs 源码比较值（三角化）

| 载体对 | 一致？ | 缺证据项 |
|---|---|---|
| 声明 vs `result.json thresholds` | **一致**（0 分歧） | — |
| 声明 vs `r3_benchmark.py TARGET`（ast 解析） | 无分歧 | `steady_1e-3`（run_sweep 里是裸字面量 `1e-3`） |
| 声明 vs `r4_benchmark.py TARGET`（ast 解析） | 无分歧 | `stratified_phi_0p5`（r4 未声明分层闸门）、`steady_1e-3` |

即：**没有发现静默放宽**（这是好消息，并首次被机器证明，而非人工阅读），但两条判据存在
"载体缺口"，r4 判据集缺分层闸门（若 r4 单独使用，混相记录不会被拒）。

### 3.3 判决可复算性与变体证伪

- **机器复算**：92 条（记录 × 判据）声明判决全部由记录中的度量值重算一致，
  **mismatch = 0 / 92（0.00 %）**；16 条 WITHHELD（均为 §3.1 的缺失度量）。
- **变体全杀**：6 条判据 × 3 个退化变体（阈值×2 / 阈值×0.5 / 运算符取反）= 18 个变体，
  `mutants_surviving = []`，说明探针集对"阈值写错/比较取反"两类错误有分辨力。
- **开关点全通过**：`switch_probe_ok` 全 True，判据切换点与声明阈值一致（无隐藏宽容）。

### 3.4 对锚点结论的量化补充

`spurious_static_0p5pct` 的利用率最小值 **13.24**、最大 **69.97**：静态控制伪电流是
判据阈值的 13–70 倍，与上游报告的 "~15–70×" 一致，且**在本次全部 3 个静态控制档上无
PASS 记录**（0 PASS / 3 FAIL / 30 WITHHELD）——判据③的 honest-FAIL 不是口径产物。

## 4. 结果图

| 图 | 文件 | 内容 |
|---|---|---|
| 图 1 | `figures/gate_oc/fig1_gate_utilisation.png` | 每条记录在每条判据上的利用率散点（红线 = 判据触发）；可见 spurious 判据整簇在 10× 之外，flow 判据跨 1.0 两侧 |
| 图 2 | `figures/gate_oc/fig2_switch_point.png` | 判据操作特性：判决随"缺陷 = 阈值×(1+f)"的阶梯曲线，全部在 f=0 处由 PASS 翻 FAIL |
| 图 3 | `figures/gate_oc/fig3_mutant_kill.png` | 变体杀伤矩阵（行=判据，列=退化变体，格=能被杀死的探针数；0=变体存活） |

（纯后处理审计，无场量云图；云图与剖面属 `GAP-LBM-MCMP-POISEUILLE` 既有报告，
本单不重跑因而不产生新的场数据。）

## 5. 注意点 / 局限 / 适用范围

① **发现（记录面缺口）**：`steady_1e-3` 判据不可复算 —— 验收记录只存布尔值，判据
6/6 记录全 WITHHELD。修法是把 `steady_rel_change` 数值写进记录（建议列入该锚点后续小任务）；
在补齐之前，"定常"这一前置条件的证据等级应记 **WITHHELD**，而不是 PASS。

② **发现（判据集覆盖差异）**：r4 的 `TARGET` 不含分层闸门 `phi`，因此 r4 记录若单独使用，
混相 run 不会被拒 —— 同一锚点的两套判据集覆盖不同，比较时须显式说明用的是哪一套。

③ **局限（口径未裁决）**：本单**没有**裁决 m16034 提出的 `u_max_analytic` 参考口径分歧
（nominal 0.7/0.3 混合规则 vs 共存态规则，差 ~1.3 %）。本单给的是"该分歧对判据③结论
无影响"的机器证据（利用率 13–70 倍，远超 1.3 % 的量级），而不是"哪一口径物理上正确"。

④ **局限（审计的自我边界）**：`gate_oc.py` 只能证明"判据在给定记录上会失败/切换点正确"，
不能证明判据**阈值本身选得合理**（10 % 是否够严，属工程判断，SPEC 预注册在先）。
变体证伪的强度也受探针集设计约束：本单探针集为单度量合规/缺陷谱，未覆盖"多度量联合
伪造"（例如同时把 Q 和 u_max 按同一比例缩放使比值不变）—— 这是下一步的自然延伸。

⑤ **适用范围**：`gate_oc.py` 与领域无关，可直接用于任何"记录 + 阈值"型验收面
（本仓 `benchmarks/verified/*` 大多是这个形态）。

## 6. 误差判定（逐项 + 显式回答是否 <3 %）

本单不产生新的物理量测量，因此没有"数值解 vs 解析解"的误差可列；可列的是**审计自身
的判决误差**（能否忠实复算记录声明）：

| 项目 | 基准 | 实测 | 误差 | <3 % ? |
|---|---|---|---|---|
| 判决复算一致性 | 记录声明的 PASS/FAIL（92 条） | 机器复算 92/92 一致 | **0 / 92 = 0.00 %** | **是** |
| 开关点位置 | 声明阈值（6 条） | 探测上侧 `t(1+1e-3)` 全部 FAIL | 相对偏差 ≤1e-3 = 0.1 % | **是** |
| 变体可分辨性 | 每个变体至少被杀 1 次 | 18/18 被杀 | 存活 0 | **是** |
| 阈值三方一致性 | 声明值 | 记录值 / 源码 parsed 值 | 分歧 0，缺证据 3 项 | **是**（分歧 0 %） |
| 定常判据可复算性 | 记录中应含度量数值 | 0/16 记录含数值 | 100 % 不可复算 | **否 → honest-FAIL/WITHHELD**（如实记录，未粉饰） |

**结论**：审计机器本身的三项精度指标全部 <3 %（实测 0 %、0.1 %、0）；唯一超出的是
"定常判据可复算性"，这是**记录面缺陷**，如实记为 WITHHELD 并给出修法（§5-①），不掩盖。

## 7. 复现

```bash
cd <repo>
PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/gate_oc_run.py     # 生成 gate_oc.json + 3 图
PYTHONPATH=src   python -m pytest tests/test_gate_oc_mcmp.py -q             # 16 passed
```

## 参考

- Kruger et al. (2017) *The Lattice Boltzmann Method: Principles and Practice*, ch. 9（伪电流量级）。
- `docs/MCMP_STRATIFIED_POISEUILLE_SPEC.md` §6（判据预注册）。
- `docs/REPORT_qr_pilot_a.md`（QR 独立盲裁 A 臂 FAIL）。
- 本仓历史同类缺陷：m16308（恒真判据）、m16034（参考口径不一致）。