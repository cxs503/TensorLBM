# 交差核验单 · m17966 → m17843
**XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER**（TensorLBM ↔ loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow）
审计人：`board-audit`（任务 m17966）｜审计日期：2026-11-13｜性质：**非重做**，独立取证 + 判级 + 落表

---

## 0. 结论先行

| 项 | 结论 |
|---|---|
| 数字真实性 | **真**：交付的 20 个关键数字，审计方用**自写 NumPy 实现**独立复算，最大相对偏差 **≤2.4e-13**（面板 B） |
| 可复现性 | **真**：交付驾驶员 3 个 stage 重跑与 `results.json` **逐位一致**；4 张 PNG 重生成**字节数完全相同** |
| 测试 | **真**：`pytest tests/test_les_graph_unroll_bptt.py` 审计方实跑 **6 passed in 12.98 s** |
| 入库 | **真**：commit `ac4445e9`（14 files, +3240），工作树 clean（对比 m17846 的 0-commit 交付） |
| 基准裁判 | **交付时缺**：`tensor_benchmark_cases` 该锚点 **0 行**（无预声明 metric/tolerance）→ 本单补登记 2 行 |
| 物理臂 | **诚实 FAIL**：broadband 算例终端场误差 **72.15%**（>15% 门）、预声明「较无模型改善 ≥30%」实测 **0.17%** |
| 外部锚 | **仅源码级**：OSS 未被执行（无外网），`oss_source_notes.json` 自述 "No line of upstream code was copied"，XVAL 实为**规范→实现**，非 code-to-code parity |
| **判级** | **L2**（独立功能实现）｜L3 拒：零集成调用方；L4 拒：物理臂越门 + 外部锚未执行 + 交付时无基准登记 |

---

## 1. 被审对象与身份

| 项 | 值 |
|---|---|
| mission | 17843（`tensor_dev_missions` status=`succeeded`, 09:58:50Z finished） |
| 交付 commit | `ac4445e9` on base `be78a87b`（branch `agent/m17843`，worktree `/root/tp-agent/work/m17843`，`git status` clean） |
| 交付件 | `src/tensorlbm/les_graph_unroll/{solver2d,calibrate,reference2d}.py`（222/264/150 行）、`scripts/xval_graph_unroll_bptt_les.py`（428）、`scripts/..._figures.py`（133）、`tests/test_les_graph_unroll_bptt.py`（120）、`docs/XVAL-LBM-...md`（275）、`artifacts/xval_graph_unroll_bptt_les/{results.json(1552 行), oss_source_notes.json, fig1..fig4.png}` |
| 平台报告 | `tensor_reports` id **16797**（mission 17843） |
| 交付时基准行 | `tensor_benchmark_cases` anchor_slug 该值：**0 行** |

审计执行环境：Python 3.11.6 / torch 2.7.1 / numpy 2.4.6 / loongarch64，`PYTHONPATH=/root/tp-agent/work/m17843/src`，
所有复算脚本在 `/root/tp-agent/work/m17966/docs/board_audit/m17966/`，工件在 `/root/tp-agent/work/m17966/artifacts/m17966/`。

---

## 2. 做法（独立性从何而来）

审计方**另写一套 NumPy D2Q9 + Smagorinsky/OSS-proxy 实现**（`audit_m17843.py` §「Independent D2Q9」），
不 import 交付的 `calibrate/solver2d` 做任何主判据；只在两处触及交付代码：
① `unroll()` 等价性对照（A0，判据本身是"两套实现同解"）；② 读取 `results.json` 的已存数字做比对。

| 探针 | 内容 |
|---|---|
| **A0 等价性** | 交付 torch 图链 `cs=0` 五步 vs 审计方 NumPy BGK 五步，32²，τ=0.515563 |
| **A1 闭包** | 自写 Smagorinsky τ 场 `0.5(τ₀+√(τ₀²+18√2 C_s²Q/ρ))` 与 OSS proxy `√(2Σ(f-f^eq)²EEk)`；剪切模衰减拟合 ν_t，对照解析 `C_s²|S|=C_s²·A·k`；并复算 OSS/文献 pre-factor 比 |
| **A2 孪生** | 自建 32² 同解参考（植入 C_s=0.17/0.30，快照 50/100/150/200）→ 自写中心差分 `dJ/dC_s`（h=1e-3）× 4 个 C_s 点，与交付 autograd 比；自写 golden-section 定全局最优 |
| **A3 物理臂** | 自建 **128²** BGK 参考 + 4×4 盒滤波，重算 4 条臂（no_model/upstream_0.17/bptt/golden）的**终端相对 L2 场误差** |
| **A4 文档算术** | 用 `results.json` 原始 got/ref 重算 docs §6 判定表的每个 err% |
| **A5 收敛** | 直接读 128→256 KE 相对差并与 docs 声明核对 |
| **R 复现** | 在 `/tmp/m17966rt` 副本重跑 `closure`、`twin tgv 0.17`、`phys tgv`、figures，与已提交工件比对 |
| **P 反证** | `probe_bptt_budget.py`：同一孪生问题把 BPTT 迭代数 20→40→80，检验「8% 回收误差」是方法极限还是优化预算极限 |

---

## 3. 核验结果（逐项，全部可复核）

### 3.1 等价性与闭包（A0/A1）

* A0：`cs=0` 五步，交付 torch 链 vs 审计 NumPy 链 **max|Δf| = 0.0**（逐位相同）→ 交付的「graph_unroll 与显式 BGK 同一求解器」成立。
* A1（ν_t 实测 vs 解析 `C_s²|S|`）：

| C_s | 交付 err% | 审计 err% | ν_t 交付 / 审计 | 相对差 |
|---|---|---|---|---|
| 0.10 | 4.7456 | 4.7456 | 1.028338e-04 / 1.028338e-04 | 0.0 |
| 0.17 | 4.5599 | 4.5599 | 2.966628e-04 / 2.966628e-04 | 0.0 |
| 0.30 | 3.9838 | 3.9838 | 9.187729e-04 / 9.187729e-04 | 0.0 |
| 0.50 | 2.6342 | 2.6342 | 2.519023e-03 / 2.519023e-03 | 0.0 |

* OSS proxy（C_s=0.17）实测 ν_t=7.039777e-05，err vs 文献解析 **75.19%**（交付同值）；
  pre-factor 比（文献/OSS）实测 **4.214094** vs 期望 `18√2/6 = 4.242641`，**偏差 0.673%** ⇒ 交付「<10%」成立。

### 3.2 孪生标定与梯度（A2/P）

| 问题 | 交付 BPTT C_s | 植入真值 | 审计回收误差 | 交付自称 |
|---|---|---|---|---|
| TGV, C_s=0.17 | 0.156391 | 0.17 | **8.01%** | 8.0% ✅ |
| TGV, C_s=0.30 | 0.272560 | 0.30 | **9.15%** | 9.2% ✅ |
| Fourier, C_s=0.17 | 0.160723 | 0.17 | **5.46%** | 5.5% ✅ |

* 梯度：交付 autograd vs **审计自写 NumPy 中心差分**，4 点相对差 **0.00396 / 0.00471 / 0.00935 / 0.00437 %**（≤0.0094%）；
  且**审计 FD 与交付 FD 互差 ≤1.5e-13**（两套独立实现算出同一条 FD）⇒ X1「0.001–0.009%」成立，不是自证同一段代码。
* 审计方用交付的 golden 最优点复算损失：`J(0.169999) = 8.96e-15`、`J(0.300000)=4.70e-15`、`J(0.169999)=1.77e-13`
  ⇒ 交付的**无导数最优就是真全局最优**（与植入常数到 5e-6），故 8% 不是可辨识性上限——见 §4-F4 与反证 P。

### 3.3 物理臂（A3）——审计方自建 128² 参考重算

| 算例 | 臂 | 交付终端场误差% | 审计独立重算% | 差 |
|---|---|---|---|---|
| TGV (Re=500, 32²) | no_model | 14.6213423 | 14.6213423 | 4e-15 |
| | upstream 0.17 | 14.5926761 | 14.5926761 | 1e-14 |
| | **bptt** | 14.5962382 | 14.5962382 | 0 |
| | golden | 14.5926710 | 14.5926710 | 4e-15 |
| Fourier 6-mode | no_model | 87.5126869 | 87.5126869 | 4e-15 |
| | upstream 0.17 | 85.4764449 | 85.4764449 | 4e-16 |
| | **bptt** | **72.1462251** | 72.1462251 | 0 |
| | golden | 75.9470139 | 75.9470139 | 4e-15 |

⇒ 交付的"误差地板"（14.62% / 87.51%）是**离散化地板**而非模型效应，BPTT 在 TGV 上仅把 14.6213% 拉到 14.5962%（**相对改善 0.1717%**，预声明门 ≥30%），
在 Fourier 上把点式误差从 87.5% 拉到 72.1%（仍 4.8 倍于门）且 KE 衰减误差 155.1%。交付文档**逐条自认**这些 FAIL。

### 3.4 文档算术与复现（A4/A5/R）

* docs §5.1：`J(0)=0.0148933865`、`J_min=0.0148714295 @ C_s=0.125` ⇒ 增益 `0.1475%`≈"0.15%" ✅（与 `results.json` 完全一致）。
* docs §5.2：「t=50 时 53%、t=200 时 87%」实测 `53.62% / 87.51%` ✅。
* docs §5.5：起点敏感性 0.02→0.1564、0.45→0.1472，spread `5.88%`≈"5.9%" ✅。
* docs §5.4 与 A5：128→256 KE 差 `0.2692%`（tgv）/ `0.0137%`（fourier）✅。
* R 复现：`closure`、`twin tgv 0.17`、`phys tgv` 三个 stage 重跑 → `stage1_twin.tgv.cs_truth_0.17` 块 **JSON 序列化后逐字符相等**；`stage2_physical.tgv.terminal_field_err_pct` / `ke_decay_err_pct` 全位相同；4 张 PNG 重生成字节数 **102604/174801/94396/39571 完全一致**。
* 测试：`PYTHONPATH=src python3 -m pytest tests/test_les_graph_unroll_bptt.py -q` → **6 passed, 1 warning, 12.98 s**（交付自称 10.5 s，仅机器差异）。

---

## 4. 发现（按严重度排序）

### F1【阻断 L4｜治理】交付时无基准裁判行
`SELECT ... WHERE anchor_slug='XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER'` → **0 行**；`LIKE '%unroll%'/'%bptt%'` 亦 0 行。
owner 规则「基准判据是唯一裁判」在此单**无从机器复核**，只能由本审计补登记。**本单已补 2 行**（§6）：
方法臂 `passed`、物理臂 `failed`。→ 任何"gate 自动 confirmed"的说法在交付时**无依据**。

### F2【阻断 L3｜集成】零调用方
`grep -rn "les_graph_unroll" src/ tests/ scripts/`：命中仅 `src/tensorlbm/les_graph_unroll/*` 自身 + `tests/` + `scripts/`。
`src/tensorlbm/__init__.py` 的 `__all__` **未导出**该模块；主求解器现成的可微路径 `src/tensorlbm/autograd_path.py` +
`collide_smagorinsky_bgk3d` 未被接入/复用。docs §5.6 亦自述"scoped to periodic boxes，墙体 counterpart 未接"。→ L3 不成立。

### F3【阻断 L4｜外部锚】OSS 未被执行，XVAL 实质是"规范→实现"
无外网（docs §7.1，`raw.githubusercontent.com` → 000）；`artifacts/.../oss_source_notes.json` 自述
`"No line of upstream code was copied; the mechanism was re-derived with PyTorch autograd"`，provenance 仅为平台记录
（`oss_inventory.id=14`、`oss_solutions ids=[37,39,35,36,38,40]`、`report 9555`、`cache_nodes=n315,n330`、`cache_commit=af07744`）。
⇒ 本锚点的 `external_anchor` 级别 = **源码引用（source-read）/ 数值 parity WITHHELD**，与 m17880（importlib 执行 OSS 文件 + sha256）、
m17890（逐字节 vendor + sha256 复算）两单的 code-to-code 对标**不同级**。交付未夸大为 parity，但下游若把它当"OSS 已对齐"引用即失效。

### F4【重要｜解释】"孪生回收 8.0% <15%"是优化预算，不是可辨识性
反证 P（`artifacts/m17966/probe_bptt_budget.json`）实测：迭代 20 → C_s=0.156391（err 8.005%, J=2.72e-8）；
迭代 40 → **同一平台** 0.156391（err 8.005%, J=2.72e-8，Adam 在对数参数上停滞）；迭代 80 → C_s=0.167869（err **1.254%**, J=6.61e-11）；
无导数最优 → 0.169999（err **5.4e-5%**, J=8.96e-15）⇒ 8% 的残差**不是方法/可辨识性极限**，而是 20 迭代的学习率-预算停点（4× 迭代即降到 1.25%）。
交付把它列为 X3「✅」在**数值上**成立，但**不能**被引作"闭包可辨识性达 8%"的结论。非阻断（交付未据此下强结论），但描述需修正。

### F5【重要｜语义】X5 TGV「14.60% ✅ (marginal)」不能读成"模型达标"
四条臂 14.5927–14.6213% 互相差 <0.3%，BPTT 相对改善 0.17%；14.6% 是 32² 离散化地板（128→256 自身 KE 收敛 0.27%，点式场不收敛）。
docs §5.1 已明说"one-constant Smagorinsky 在此配置不可辨识"，但 §6 判定表把它记成 ✅ 容易被下游误读为"LES 验证通过 15% 门"。
→ 建议改标注为 `n/a (floor)`。

### F6【重要｜文档】"13 段全部跑完"不可机检，且计数与实现不符
`scripts/xval_graph_unroll_bptt_les.py:411-420`：`stages = ... else [tuple(argv)]` —— 一次调用只跑**一个** stage；
`results["stages_done"] = sorted({...})` **每次覆写**，故已提交 `results.json` 里 `stages_done == ["oss_diff"]`（只有最后一次）。
实际可证的 stage 有 10 个（closure / twin×3 / phys×2 / convergence×2 / twin_start / oss_diff），section 键齐全但**没有 13**；
`stage_horizon`（脚本 328 行）存在却从未运行——docs §5.5 已诚实自述"Not executed"。→ 属记账夸大，不改任何数字。

### F7【次要｜系统性偏差】闭包 ν_t 一律偏高 +2.6…+4.7%
四个 C_s 全同号偏高（非随机），交付未分析来源（审计方判断最可能为 6 步对数拟合 + 振幅估计 `mean·2` 的一阶偏差，及 Π^neq 的 O(Ma³) 项）。
在 15% 门内不阻断；但若有人把此检当"<3% 精度"引用即错。

### F8【次要｜测试强度】`test_unroll_matches_reference_kernel_at_zero_cs` 是等价性检查而非物理验证
A0 给出 max|Δf| 恰为 0.0，说明两侧是同一组方程同一离散 → 该测试证明"实现一致性"，不证明正确性。
其余 5 个测试（闭包解析 / 梯度 vs FD / 孪生回收 / pre-factor）有独立参考面，强度够 L1-L2。

### F9【正向】无粉饰、无手抄、可复现三件套齐备
交付把 3 条硬 FAIL（fourier 72.15%、改善门 0.17%、fourier C_s 66.7%）写进 docs §5/§6 与 `result.honest-FAIL` 口径；
每个数字都能在 `results.json` 找到出处（零手抄）；工件入库（commit `ac4445e9`）。这在同类单里属高水位，但**不改变级别**。

---

## 5. 未执行的检查（如实标注）

* `all` 单命令全量重跑（交付自称 `runtime_seconds=516.7`）未做——只复现了 `closure` / `twin tgv 0.17` / `phys tgv` 三个 stage。
* `convergence tgv|fourier`（256²）与 `twin_start` 未重跑，仅做**读数核对**（与 docs 一致）。
* `horizon`（T=100 vs 200）**不存在可核数据**（交付未跑）。
* OSS 上游源码**无法执行**（无外网、仓库未 vendor）⇒ 上游数值 parity 本单**判 unknown/not_run**，不是 passed。
* 3D/带壁面路径未涉及（交付自己限定 periodic box）。

---

## 6. 落表

### 6.1 `tensor_benchmark_cases`（本单补登记 2 行，此前 0 行）
| case_name | metric | tolerance | status |
|---|---|---|---|
| `GRAPH-UNROLL-BPTT-TWIN-AND-GRADIENT` | 植入 C_s 回收误差 ×3 + autograd vs 独立中心差分 | 回收 <15%、梯度 <5%（owner 标准） | `passed`（8.01/9.15/5.46%；梯度 ≤0.0094%） |
| `GRAPH-UNROLL-BPTT-LES-PHYSICAL-TERMINAL-FIELD` | 粗网格 LES 终端相对 L2 场误差 vs 128² 盒滤波参考 | <15% | `failed`（tgv 14.60% 与无模型 14.62% 无差别；fourier 72.15% 越门） |

### 6.2 `tensor_maturity_verdicts`
`repo=TensorLBM, scope='m17843 XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER', mission_id=17966, level=2, verdict_by='board-audit'`
evidence 键：impl_files / entry / run_artifacts / tests / integration_callers / benchmark / repro_cmd / external_anchor（值见 DB 行）。

### 6.3 判级理由（L2）
* **L0/L1 越过**：不止文档或单元算式——222 行可微 D2Q9 + 264 行标定器 + 150 行 NumPy 参考，输入→计算→输出完整（`solver2d.unroll`/`calibrate.bptt_calibrate`）。
* **L2 成立**：真实代码 + 入口脚本 + 可复现工件（3 stage 逐位复现、6 测试实跑、4 图字节一致），且审计独立实现复算全部关键数字 ≤2.4e-13。
* **L3 拒**：F2 零调用方、未导出、未接 `autograd_path`。
* **L4 拒**：F1 交付时无基准行 + F3 外部锚未执行 + F6 物理臂 72.15% 越 15% 门、预声明改善门 0.17% ≪ 30%。
* **L5 拒**：无科研级新结论；核心发现（单常数不可辨识）属**负面结果**，是诚实的边界刻画，但不构成 L5 工程成果。

---

## 7. 处置（disposition）：**request changes → 返修单**

建议返修项（按性价比）：
1. **接主求解器公开 API**（`tensorlbm.__init__` 导出 + `autograd_path` 复用），才够 L3；顺带把 `stages_done` 改累积。
2. **把点式 L2 目标换成统计量目标**（能量/拟涡能衰减、谱）——交付已判"objective 用错"，这是唯一能把物理臂拉回 15% 门的路径；`stage_horizon` 与加速/收敛留一验证。
3. **基准行已由本单补登记**；返修时应把 `GRAPH-UNROLL-BPTT-LES-PHYSICAL-...` 由 `failed` 转 `passed` 才算消账。
4. docs §6 判定表把 X5 TGV 改标 `n/a (floor)`，X3 补注"20 迭代未收敛"，"13 段"改为实际 10 段的清单。
5. 闭包 +4% 单号偏差给出成因（拟合阶数/振幅估计），否则任何人引用"<3%"都站不住。

---

## 8. 工件清单

| 路径 | 内容 |
|---|---|
| `docs/board_audit/m17966/audit_m17843.py` | 独立 NumPy 重算器（A0–A5） |
| `docs/board_audit/m17966/probe_bptt_budget.py` | 反证 P：BPTT 迭代预算 vs 回收误差 |
| `docs/board_audit/m17966/make_figures.py` | 审计三联图 |
| `artifacts/m17966/audit_m17843.json` | 全部复算数字（含 `delivered_results_sha256`） |
| `artifacts/m17966/probe_bptt_budget.json` | 迭代预算探针输出 |
| `artifacts/m17966/fig_m17966_audit.png` | 面板 A 门槛图 / B 独立 parity / C 优化预算 |
| `/tmp/m17966rt/` | 交付副本重跑现场（closure / twin tgv 0.17 / phys tgv / figures 日志） |