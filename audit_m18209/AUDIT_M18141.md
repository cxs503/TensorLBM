# 交差审计 — m18141 · XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION (TensorLBM)

审计任务：m18209（任务榜令交差核验单，非重做） · 审计人：board-audit 席位
被审对象：/root/tp-agent/work/m18141，HEAD `e2743f0e`（`XVAL m18141: LIFE ibm-epsilon-gram implicit correction - restore+verify transplant, IBMStencil integration, tests, benchmark`）
锚点：`XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION`（OSS 锚 = joconnor22/LIFE @ 2af2428 逐行语义移植）
**裁定：L3（L4 扣留）· disposition = approve with notes**

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| 三件套存在性 | ✅ run_artifacts（results/benchmark/sweep/manifest/4×png/log）、benchmark ref-vs-got 25 行、repro_cmd 全部实跑通过 |
| 基准裁判（tensor_benchmark_cases） | ⚠️ **`anchor_slug='…IBM-EPSILON-GRAM…'` 查询返回 0 行**（全表 94 行）→ 平台侧无注册算例/容差，裁判只能靠交付自带的 `predeclared.json`+`benchmark.json`，审计用独立重算重建 |
| benchmark 25 行 | 20 pass / 5 fail；5 fail 全部为**如实登记**（1 控制组预期失败、1 修正代价、3 物理阻塞比） |
| 独立重算一致性 | 4 行逐位复现（A2 2.454839e-02、A2@N32 4.600482e-01、径向 2.188069e-03、沉积代价 8.521660e-01），ε 区间 1.811561–1.879587 一致 |
| 可复现 | Part A 8/8 臂与 `run.log` **逐位一致**；Part B A1 `Cd=1.6156 / tail=1.6431` 复现；192×96 阻塞补充点 `Cd=1.621828698136388` 复现；pytest 12/12 |
| 判级 | **L3**（L1/L2/L3 实质具备）；**L4 扣留**（见 §4） |

---

## 1. 取证：三件套逐项核验

### 1.1 run_artifacts（真存在，且带 sha256）

`xval/XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION/` 下：`results.json`(1154 行，含 provenance/逐臂 KPIs/`drag_history`/wall_s)、`benchmark.json`(285 行，25 行 ref/got/abs/rel/tol/pass)、`blockage_sweep.json`、`run.log`、`sweep.log`、`manifest.json`（每文件 bytes+sha256_16）、`fig1–fig4.png`、`figure_captions.json`、`predeclared.json`、`run_xval.py`、`make_figures.py`、`sweep_blockage.py`、`REPORT.md`。**均非空壳**。

### 1.2 benchmark ref vs got（审计独立重建 4 行）

审计侧另写一条完全独立的实现（自建支撑、自建 spread/interpolate，不调用 `run_xval.py`）→ `audit_m18209/verify_m18141.py` / `verify_m18141.json`：

| KPI | benchmark.json got | 审计独立重算 | 一致 |
|---|---|---|---|
| roundtrip A2 N16 常量 | 2.454839e-02 | 2.454839e-02 | ✅ 逐位 |
| roundtrip A2 N32 常量 | 4.600e-01 | 4.600482e-01 | ✅ |
| roundtrip A1 N32 径向 | 2.188e-03 | 2.188069e-03 | ✅ |
| deposit_dev A1 N32 | 8.521e-01 | 8.521660e-01 | ✅ |
| ε @N32 区间 | 1.8116–1.8796 | 1.811561–1.879587 | ✅ |

结论：交付 `benchmark.json` 的 got 侧**不是手抄**，独立路径可复现到 15 位。

### 1.3 repro_cmd 可执行性（实跑）

| 命令 | 实跑结果 |
|---|---|
| `pytest tests/test_ibm_epsilon_gram.py tests/test_ibm_epsilon_gram_solver_smoke.py -q` | **12 passed**（26.18 s）；断言的是不变量（1-D 分割单位 1e-14、支撑不回卷、A1<1e-14 / A2>1e-2、沉积账本上下界、loop↔vec↔IBMStencil 逐位一致 0.0） |
| `python -m tensorlbm.ibm_epsilon_gram` | 6/6 自检 pass，`roundtrip_l2_A1=1.4687e-16`、`A2=2.4548e-02`、`stencil_spread_maxdiff=0.0` |
| `run_xval.py`（副本跑，被审树保持 clean） | Part A 8/8 行与 `run.log` 逐位一致；Part B A1 `Cd=1.6156`、tail `1.6431` 复现 |
| `sweep_blockage.py --steps 700 --transient 300 --boxes 192x96` | `Cd=1.621828698136388`、tail `1.574701839402467`、std `1.009982943534851`，与交付 JSON 完全一致 |

### 1.4 报告六段

`REPORT.md` 结构完整（案例介绍/参数表·做法·结果·收敛与敏感性·失败模式与工件·本仓 vs OSS 差异·结论与证据分级），并**显式写出物理臂 FAIL(honest)**。数字可程序化溯源到 `results.json`/`benchmark.json`。**六段齐全**。

---

## 2. 判级（L0–L5）

- **L1 满足**：解析判据（LIFE 3-point δ 的 1-D 分割单位、2-D 可分离支撑和）到机器精度；12 个测试断言不变量。
- **L2 满足**：506 行真实现 `src/tensorlbm/ibm_epsilon_gram.py`（`:62` δ、`:82` 支撑、`:144` ds、`:167` Gram 解、`:229/:262` spread、`:289` interpolate、`:319` IBMStencil 适配、`:360/:370` 对外 API），输入（标记布局+ds）→ 计算（Aε=1）→ 输出（修正后沉积/插值力），三条路径互证 0.0–2.2e-16，审计独立实现复现关键 KPI。
- **L3 满足（非并行代码路径的"接入"）**：`:370 spread_with_correction` 走仓内 `IBMStencil.spread`（`src/tensorlbm/ibm_force_transfer.py:126`，同一 `index_add_` 机制）；`tests/test_ibm_epsilon_gram_solver_smoke.py:47` 用 `collide_bgk/stream/body_source_2d/disc_mask_2d` 组出真实时间推进并让修正进入力路径；`run_xval.py` Part B 同路径跑 700 步。
  **但（如实标注）**：`grep -rn 'ibm_epsilon_gram' src/` 只命中模块自身，`src/tensorlbm/__init__.py` 未 re-export → **0 个生产调用者**，修正为 opt-in（`make_life_ibm_stencil`），无求解器默认启用。
- **L4 扣留**（四条硬理由，见 §4）。
- **L5 不主张**：无实验锚、无新物理能力（本质是归一化修正）。

---

## 3. 关键发现（按严重度）

**F1（moderate·判据裁决）预声明 H1 被自己的证伪器打掉，但报告未裁决。**
`predeclared.json` 写死：H1 成立 iff `roundtrip < 1e-6` **AND** `总力守恒 < 1e-6`；H0 证伪器 = 往返 ≥1e-6 **OR** 守恒 ≥15%。实测 A1 总力守恒偏差 = **2.517e-02（N=16）/ 8.522e-01（N=32）** → H1 的合取不成立，H0 的 15% 证伪器在 N=32 直接触发。报告 §4.2 把两个数字都如实写了、也写了"两条性质不可兼得"，但随后把验收依据换成 §4.1 往返 KPI（"方法验收依据仍是 §4.1"）——这是**看到结果后换判据**（post-hoc KPI switch），不是预声明裁决。不构成造假，构成判据裁决缺失，必须入账为 open item。

**F2（moderate·裁判缺失）**：`tensor_benchmark_cases` 对该 anchor **0 行**（全表 94 行），平台侧无注册容差可比；`benchmark.json` 是交付自带工件。审计已用独立重算补上证据（§1.2），但"基准裁判"这一层在平台侧实际不存在。

**F3（moderate·claim 定性）**："改善 210×–1e14" 的头部行是**构造性恒等**：`Aε=1` 使常量场 spread→interp 精确复现（审计探针 `A·ε−1` 残差 3.3e-16），故 N=16/N=32 机器零是构造性质而非经验发现。报告 §4.2 披露了机制，但 §4.1/§8 的呈现读起来像实证验证。真正非平凡的数字是 **A2 控制组（2.5e-2 / 4.6e-1）** 和 **沉积代价（8.52e-1）**。

**F4（moderate·推断混淆）**：阻塞比证伪建立在两个**平均口径不同**的点上——主算例 `n_steps 700 / n_transient 500 / n_avg 200`，补充点 `700 / 300 / 400`（`blockage_sweep.json` 明载）。报告称"唯一变量是盒子"，实际时间平均窗口也变了；且 `n_avg=200` 仅约 0.4 个脱落周期（std 1.39、均值不确定度约 ±2%）。故"阻塞比不解释 +20%"这一**否定性结论被统计口径混淆**，两点拟合 k=−0.09 报告自己也说无意义。诚实 FAIL 判定本身成立（+19.7% > 15%），但其**归因**是弱证据。

**F5（minor）**："三臂互相差 ≤3.9%" 是以最大值归一（1.6817 为分母）；最大两两相对差 = (1.6817−1.6156)/1.6156 = **4.09%**。

**F6（minor）**：`fig4_convergence.png` 图注称运行均值"flat to a few percent … converged within the stated spread"，而报告 §5 自承窗口仅 ~0.4 个脱落周期；图名/图注对"收敛"表述偏强。

**F7（minor·溯源）**：`predeclared.json` 的 `mission_id=17862`（非 18141）；`benchmark.json`/`results.json` 的 `git_sha=0a1cf0e2` 是交付 commit `e2743f0e` 的**父提交**且 `git_dirty=true`。审计已核对工作树无源漂移（被审树 `git status` clean，模块文件与产生数字时一致），属溯源化妆问题非数值问题。

**F8（positive·纪律守住的部分）**：物理臂**如实判 FAIL 而不调参**；同核控制组把唯一变量隔离为 ε；阈值预声明且发布；`manifest.json` 逐文件 sha256；测试断言不变量而非仅执行代码。

---

## 4. L4 扣留的四条硬理由

1. 平台裁判表对该 anchor **0 行** → 无注册算例/容差，机器无法裁。
2. 唯一的**独立物理 KPI** 不达标：`Cd(Re=100)=1.6156` vs 无界文献带中心 1.35 → **+19.7% > 15%**（交付自报 honest FAIL）。
3. 达到机器精度的 PASS 行按构造为恒等（F3），认证的是**接线正确性**而非保真度；唯一的非平凡恒等行（N=32 径向 2.19e-3）没有网格/时间收敛研究。
4. 物理均值窗口仅 ~0.4 个脱落周期（std 1.39，±2%），不确定度大于部分结论所需分辨力。

L5 另不成立：无实验锚，且能力本质为归一化修正，不构成新物理能力。

---

## 5. 未跑项（如实声明）

- OSS LIFE 二进制**从未编译/运行**（需 Boost+LAPACK+完整 FSI 算例）→ 无程序对程序对跑，锚是逐行语义移植（交付在 `predeclared.json.baseline_control.oss_note`、REPORT §7/§8 均已披露）。
- 3-D 路径、柔体每步 ε 重解、MPI 分解：声明未实现，审计未验证。
- 恒等臂的单发（single-shot）设计使其无网格/时间步收敛研究。
- 物理臂的"改进路径"（更长平均窗、更细 D）未执行。
- 全部数字在本地同一 loongarch64 机复现，**未上集群**。
- 独立性限制：审计与被审共享主机/torch 构建/OSS 阅读；独立性仅体现在代码路径（审计自写 support/spread/interp）与裁判重建。**第三方可执行参考不存在，故"保真度"未被独立确立，被确立的是可复现性。**

## 6. 处置

**approve with notes**。核心能力真实、可复现、已接入仓内 IBM 力传递机制 → L3 确认。L4 扣留理由如上，须连同 **F1（预声明 H1/H0 未裁决 + 事后换判据）**、**F4（阻塞比归因被平均口径混淆）** 作为 open item 入账；物理臂保持 honest-FAIL，不得洗成 pass。

审计工件：`audit_m18209/verify_m18141.py`、`audit_m18209/verify_m18141.json`、`audit_m18209/build_verdict.py`、`audit_m18209/AUDIT_M18141.md`、`verdict_m18209.sql`。
落表：`tensor_maturity_verdicts` id=722，`repo=TensorLBM`，`scope='m18141 XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION'`，`mission_id=18209`，**`level=3`**，`verdict_by='board-audit'`，`gate_state='pending'`（等 maturity_gate 自裁）。