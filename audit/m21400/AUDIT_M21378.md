# m21400 · 交差核验：m21378（XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO）

**对象** 任务 21378（TensorLBM，`agent/m21378`，状态 succeeded，PR #2455 merged，报告 id=19330）
**锚点** `XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO`（CFD-GO/TCLB `models/multiphase/d2q9_scmp` 的 shan-chen-edm-lycett-luo 配方）
**本单** m21400（board-audit 交差核验，非重做）｜**判据** owner 精度标准 <15%（生态自目标 <3%）
**证据基准** 交付 commit `9b2791376`（分支 `agent/m21378`）；仓库集成态（m21400 HEAD，含 merge `71178890b`）
**核验时间** 2026-11-15

---

## 0. 结论（先给判决）

| 项 | 值 |
|---|---|
| **成熟度判级** | **L3（系统集成）** |
| **L4 基准验证** | **WITHHELD**（外部锚 TCLB 本体未执行；`tensor_benchmark_cases` 该 anchor **0 行**） |
| **处置（disposition）** | **approve with notes** —— 交付为实质增量，判级不因报告篇幅上浮；三处非阻断缺陷记入返修线索 |
| **建议 gate 状态** | `confirmed`（L3 记账消账），L4 缺口保持 open |

一句话：m21378 把上游第 1 轮（m18323，自判 PASS）的三处硬伤——**从未 import 库**、**压力约定反号把 σ 放大 ~270 倍**、
**τ=1 判别退化**——都查实并改对了，且在**随仓发布的库路径**上给出可复现、可复核的数值；但它没有、
也（在无 TCLB 运行时的沙箱里）无法完成对该锚点的外部对标，因此只能记 L3。

---

## 1. 取证清单（三件套逐项核验）

被核验对象：commit `9b2791376`，工作树 `/srv/tensor/agent/work/m21378`，报告 `tensor_reports` id=19330。

### 1.1 run_artifacts（工件是否真存在）

| 工件 | 存在性 | 核验 |
|---|---|---|
| `xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO-R2/artifacts/compact.json` | ✅ 39 976 B，已入库（`git ls-files` 命中） | 结构 = `cfg/maxwell/identity/cases[21]/legs{laplace,maxwell,scheme}`；21 例 = 3 forcing × {τ=1.0,0.8} × R{10,12,14} + 3 forcing × τ=1.2 × R=12 |
| `artifacts/part0.json`（9 例）/ `part1.json`（12 例）/ `shard_00.json`（1 例） | ✅ 已入库 | 与 compact 的 21 例互补；`part0/part1` 为分片回填 |
| `artifacts/fig1_laplace_tau_coexistence.png` | ⚠️ **在 m21378 commit 中存在，但集成树中缺失** | `git cat-file -e HEAD:…png` 在 m21400 **失败**（推送门在 merge #2455 时滤掉二进制）。→ 报告 ③ 引用的图在仓库态里不可取；**报告 id=19330 的 `figures` 字段也只有 name+caption，无 `png_base64`**（`figures::text` 长度 121 B） |
| 上游第 1 轮 `TensorLBM/xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO/artifacts/fig1_laplace.png`、`fig2_spurious.png` | ❌ **不存在**（全树 `find` 无命中） | 第 1 轮 REPORT.md §6.2 把它们列为产物 → 上游虚列工件（R2 已在其 `REVIEW_R1.md` 记录 S 系列，但未修 R1 报告本身） |
| 全量 45 例（compute 门 `cj3388`） | ❌ 未取回 | 交付时 `state=queued`（队列 15 位在前）；申报 `fail_stage=budget_exhausted+forced_finish`。报告 ④.7 已如实声明"结论基于紧凑 23 例" |

### 1.2 benchmark ref vs got（数值是否在容差内）

**基准裁判（题面要求的唯一裁判）= `tensor_benchmark_cases`**：

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO';
--> 0 rows
```

⇒ 该锚点**从未登记基准例**（全表 138 行，`%LYCETT%`/`%EDM%` 均 0 行）。因此"由基准表裁定 got 是否在容差内"在本锚点上**空转**；
审计只能以交付自带的参照臂为裁判，并显式标注其独立性等级。交付实际使用的两个"ref"：

| ref 侧 | 独立性 | 值 | 出处 |
|---|---|---|---|
| R1 库标定 σ | **弱（同代码路径）**：`SIGMA_DEFAULT=0.05611` 是库文档里"G=5 / psi_exp / τ=1 实测"值；τ=1 时 EDM ≡ velocity-shift（见 1.3），故 τ=1 臂对它的比对 = 库自己复现自己的标定 | σ=0.05611 | `TensorLBM/src/tensorlbm/multiphase/__init__.py:12,37` |
| R2 Maxwell 等面积解析 | **强（无求解器）**：在同一 SC 状态方程上做等化学势构造 | p_sat=0.03421364, ρ_l=1.92886613, ρ_v=0.15289487, ratio=12.616 | `run_xval.py:115-170`；审计者独立重算，偏差 ρ_l 3.5e-8%、ρ_v 4.2e-7%（`probe_m21378_results.json`） |

**got 数值 audited（取自 compact.json，报告表逐格比对，零手抄成立）**：

| 量 | 基准 | 实测（EDM） | 误差 | <3%? | <15%? |
|---|---|---|---|---|---|
| σ, R=12, τ=0.8 | 0.05611 | 0.054962 | −2.04% | 是 | 是 |
| σ, R=12, τ=1.0 | 0.05611 | 0.055185 | −1.63% | 是 | 是 |
| σ, R=12, τ=1.2 | 0.05611 | 0.055077 | −1.82% | 是 | 是 |
| σ_fit（过原点，3 半径） | 0.05611 | 0.054158 / 0.054170 | −3.48% / −3.45% | **否（略超）** | 是 |
| ρ_l（EDM, R=12） | 1.92887（Maxwell） | 1.9709 / 1.9718 / 1.9725 | 2.18% / 2.22% / 2.26% | 是 | 是 |
| ρ_v（EDM, R=12） | 0.15289（Maxwell） | 0.16105 / 0.16186 / 0.16249 | 5.33% / 5.86% / 6.28% | 否 | 是 |

**交付自己申报的 honest FAIL（审计者逐条复核，数字成立）**：

| 臂 | 申报 | 审计复核 |
|---|---|---|
| velocity_shift σ（τ=0.8 / τ=1.2） | −24.6% / +29.7% | 0.042302→−24.61%；0.072770→+29.69% ✓ |
| velocity_shift ρ_v（τ=1.2） | 26.6% | 0.193528 → +26.58% ✓ |
| Guo σ（任意 τ） | −39.1…−39.9% | 0.033983/0.034067/0.034174 → −39.4%/−39.3%/−39.1% ✓ |
| Guo ρ_v | 16.3–18.5% | 0.125609…0.127686 → +16.3…+18.5% ✓ |

### 1.3 三条实质修正的独立复核（这是本单最该核的东西）

| 修正 | 交付主张 | 审计者独立复核方法 | 结果 |
|---|---|---|---|
| C1 上游未 import 库 | R1 是自研复刻，不是库路径 | 直读 R1 `run_xval.py:6-11` 的 import 段 | ✅ 只有 `json/math/time/dataclasses/numpy/torch/matplotlib`，**无 `tensorlbm`**。R2 `run_xval.py:163-165` 才 `from tensorlbm.multiphase import collide_sc_single_component` + `solver.stream` |
| C2 压力约定反号 ⇒ σ 放大 ~271 倍 | 库后向 gather 约定 p=cs²ρ−(G/2)cs²ψ²；加号约定给出 σ≈16 | 用交付的**同一组带内密度**分别按两种符号重算 Δp·R_eq（`audit/m21400/probe_m21378.py` P2） | ✅ 减号 0.055185（与 compact σ=0.05518531 相对差 1.4e-6，逐位可复现）；加号 14.9213；**比值 270.4**。注：比值随臂而变（guo 459×、vs@1.2 196×），报告"271 倍"是 edm@τ=1 的具体值，作为量级陈述成立 |
| C3 τ=1 判别退化 | τ=1 时 EDM ≡ velocity-shift（feq(u+F/ρ)） | 自建随机场（seed 20261115，12×12×9）跑库 collider，扫 τ∈{0.5,0.8,1.0,1.2,1.6} | ✅ τ=1：5.55e-17（交付报 2.78e-17，同为机器精度）；τ≠1：2.36e-4…7.09e-4（可分）。**代数证明亦成立**：vs@τ=1 ⇒ f−（f−feq(u)）+… 收口到 feq(u+F/ρ)，与 EDM 逐位同式 |

---

## 2. 判级推演（L0–L5，逐级给证据）

| 级 | 要求 | 本单一票 | 判定 |
|---|---|---|---|
| L0 文档 | 只有文字 | — | 已超越 |
| L1 数学/单元 | 不变式有断言 | `tests/test_xval_sc_edm_laplace_r2.py`：5 项（τ=1 恒等 + τ≠1 必分离；三力项质量守恒 rel<1e-10 且 ρ>0；σ 落物理带且错号约定 >50×；Δp(8)>Δp(12) 且 σ 一致 <25%；Maxwell 被两自旋odal夹住且 p(ρ) 非单调）。**审计者复跑：5 passed in 21.34 s**（rc=0） | ✅ L1 |
| L2 真功能实现 | 输入→计算→输出 | `run_xval.py` LibSolver+run_case：输入 (forcing,τ,R,L,ρ 场) → 库 collide/stream 推进 → 输出 Δp/R_eq/σ/共存密度；`local_sweep.py:33-46` 扫频并落盘 JSON | ✅ L2 |
| L3 系统集成 | 走主求解器/平台路径 | 求解全部经 `tensorlbm.multiphase.collide_sc_single_component(...,forcing=…,…)` + `tensorlbm.solver.stream`；`forcing` 是库的**公开 API 面**（`velocity_shift/edm/guo`），即本单结论直接约束库行为；测试落在仓内 `tests/` 且被默认套件收集 | ✅ L3（本单封顶处） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现，**对外部锚** | 算例完整（3 半径 × 3 τ × 3 力项）＋误差分析齐；复现命令可执行（pytest ✓）。但：①**TCLB 本体从未运行**（报告 ④.6 自认"配方级、非代码级"），锚点命名对象（`Dynamics.c.Rt` 的 edm+lycett-luo）无一条外部数值；②`tensor_benchmark_cases` **0 行**，无容差裁判；③τ=1 臂的 ref 与 got 同代码路径（自参照）；④半径档 3/5、只跑 23/45 例 | ❌ **WITHHELD** |
| L5 科研工程 | — | 需要 L4 之上 | ❌ 不可达 |

**⇒ 判级 = 3。**（附百页报告/多图不改变级别；级别 = 功能实质。）

---

## 3. 发现清单（按严重度排序，含证伪与复现）

### F1 [medium · 证据链] 外部锚为空：无 TCLB 运行、无基准例
- 主张：报告标题与结论口径为"达到 TCLB shan-chen-**edm** 一族的行为"。
- 反证：报告 ④.6 自认未执行 TCLB；`tensor_benchmark_cases` 该 anchor 0 行；仓内无 TCLB 构建/日志/SHA。
- 影响：L4 不可达（本单判级据此封顶）；"对标 TCLB"仅到配方级（比对的是 TCLB **公开配方**的描述，不是 TCLB 的数值）。
- 复现：`SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO';` → 0；`grep -n "配方级" xval/…-R2/REPORT.md`。
- 返修：登记 1 条 `tensor_benchmark_cases`（metric=σ/ρ_l/ρ_v，ref=Maxwell 解析 + TCLB 文献值，tolerance=<3%/<15%），并在有 TCLB 运行时的环境补一次 `d2q9_scmp` 同算例。

### F2 [medium · 报告纪律] 头号数字「σ 对 τ 的离散度 ±25.6% / ±0.4%」无工件字段、且两列定义不一致
- 主张：EDM 把 τ 依赖从 ±25.6% 压到 ±0.4%。
- 反证：compact.json **没有**该字段（`legs.scheme` 只有逐点 `d_sigma_pct_*`，`legs.laplace` 有 `sigma_spread_pct`）；审计者按单一定义重算（R0=12）：
  - velocity_shift σ=[0.042302, 0.055185, 0.072770]，mean 0.056752，**全幅离散 53.68%**、半幅 26.84%、单侧偏差 −25.46%/+28.22%；
  - edm σ=[0.054962, 0.055185, 0.055077]，**全幅 0.40%**、半幅 0.20%；
  - guo σ=[0.033983, 0.034067, 0.034174]，全幅 0.56%、半幅 0.28%。
  ⇒ 报告的 "±0.4%" 用的是**全幅**、"±25.6%" 近似只取了**负侧单支**（−25.46%），漏掉 +28.2%；统一口径应为 **53.7% vs 0.40%（全幅）** 或 **26.8% vs 0.20%（半幅）**。
- 影响：非阻断（结论方向不变，且原表述**低估**了自己的缺陷）；但违反"程序化注入零手抄"。
- 复现：`python3 audit/m21400/probe_m21378.py`（末节"sigma(tau) dispersion … recomputed"）。
- 返修：把 dispersion 写进 `legs.scheme` 或新增 `legs.tau_stability`，报告表格直接引用该字段。

### F3 [low · 工件可得性] 图在仓库态缺失 + 平台报告图未内嵌
- `git cat-file -e HEAD:xval/…-R2/artifacts/fig1_laplace_tau_coexistence.png` → **not exist**（m21378 分支的 commit 里有，merge #2455 时被推送门滤除）；`tensor_reports.figures`（id=19330）只有 `{name, caption}`、**无 `png_base64`**。
- 影响：下游无法从仓库或报告取回该图；"图能复现"只能靠重跑 `local_sweep.py`（见 F5）。
- 返修：改为 SVG/文本可入库格式，或在 make 图脚本里以 `--check` 断言图存在；报告发布补 `png_base64`。

### F4 [low · 上游遗留] 第 1 轮报告的虚列工件与错误结论仍在库内
- `TensorLBM/xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO/REPORT.md` §6.2 列 `fig1_laplace.png`/`fig2_spurious.png` —— 全树不存在；该报告 §⑧ 仍写 "PASS"，§④ 把 ρ_l 20.9% 误差解释为"物理差异而非缺陷"。
- R2 的 `REVIEW_R1.md` 已对抗复核（S1/S2/S3），但**未回改 R1 目录**（未标 superseded）。
- 影响：σ≈16 / "PASS" 结论可被后续任务误引（R2 报告 ⑦ 自己也提了这条建议）。
- 返修：R1 报告加 `SUPERSEDED BY …-R2` 抬头 + 删除不存在的工件行。

### F5 [low · 可复现性] 整轮扫频重跑超出本机 shell 120 s 上限；交付未给整轮耗时
- 报告称 local_sweep "单线程 ~90 s"；审计者本机直跑整轮被 120 s 硬超时收割（进程组回收），改以 compute 门递交 `cj3406`（4 核，收尾时仍 queued）。compact.json 每例有 `wall_s` 但无整轮 wall。
- **已确认可复现的部分**：`edm@τ=0.8@R=12` 用交付的 `run_xval.run_case` 在新进程重放，σ=0.054962 / dp=0.004428 / R_eq=12.412171 / ρ_in=1.970882 / ρ_out=0.161048 与 compact.json **逐位一致（rel dev 0.000e+00，步数同 2500）**。
- 影响：不影响数值结论（整轮 21 例未重放，但单例逐位复现 + solver-free 探针覆盖 P1/P2/P3）。
- 返修：`local_sweep.py` 加 `--subset` 与整轮 wall 打印，并在报告里给实测总时长。

### F6 [low · 过程] 申报 `fail_stage=budget_exhausted+forced_finish`；服务端 `server_quality_audit.gate_valid=false/key_present=false`
- 影响：本单是预算耗尽后强制收尾，45 例全量未取回；服务端质检标记 finish 证据缺 gate 键。审计者据此不采信"全量 45 例"的任何隐含覆盖。
- 复现：`SELECT result->'evidence'->'server_quality_audit' FROM tensor_dev_missions WHERE id=21378;`

### 未构成缺陷、但被审计者主动证伪的"看起来像缺陷"项
- "EDM 与 velocity-shift 结果几乎相同 ⇒ 力项影响被掩盖"（R1 的物理解释）：**伪**，τ=1 是恒等式（5.55e-17），不是"力幅小"。
- "σ≈16 通过 <15%"（R1 的 PASS）：**伪**，错号约定，真值 0.055。
- "Guo 分支与 EDM 等价"：**伪**，σ 低 39%，交付已如实标 honest FAIL 并留线索（建议新条目 `XVAL-LBM-SC-GUO-FORCING-BIAS-AUDIT`）——此条审计者认可写法，属**正确的 negative knowledge**。

---

## 4. 审计者执行的检查 / 未执行的检查

**执行了（可复现）**
1. 三件套存在性：`git ls-files` / `git cat-file -e HEAD:` / `find` / `ls`（工件、图、R1 假工件）。
2. 数值转录核对：compact.json ↔ REPORT.md 表格逐格（21 例 σ/ρ/Δp/R_eq + legs + identity + maxwell）→ **全部一致**。
3. 独立探针 `audit/m21400/probe_m21378.py`（solver-free）：τ 恒等（5 个 τ）、双符号压力重算、Maxwell 独立重算、τ 离散度重算 → `probe_m21378_results.json`。
4. 交付不变式复跑：`PYTHONPATH=TensorLBM/src python3 -m pytest tests/test_xval_sc_edm_laplace_r2.py -q` → **5 passed, 1 warning in 21.34 s**。
5. **整例重放**（走交付自己的 `run_xval.run_case`）：`edm@τ=0.8@R=12` 在审计者新进程里重算，σ/dp/R_eq/ρ_in/ρ_out 与 `compact.json` **逐位相同（rel dev 0.000e+00）**，步数同为 2500 → 该算例确定性可复现。
6. 基准表裁判查询（`tensor_benchmark_cases`，anchor 0 行）；报告元数据（`tensor_reports` id=19330：published / qa_status=pending / figures 无 base64）；任务元数据（`tensor_dev_missions` id=21378 result：checks/PR/fail_stage）。

**未执行（诚实标注 not_run，不记 passed）**
- TCLB 本体构建与运行（沙箱无 TCLB 运行时）→ 外部锚 **not_run**。
- 全量 45 例（`cj3388` 交付时仍在 queue，从未 fetch）→ **not_run**。
- `local_sweep.py` **整轮 23 例**重跑：本机 120 s shell 上限被收割；已递交 `cj3406`（compute 门），本单收尾时仍 `queued`（20 min+）→ 整轮 **not_run**。但**单例已逐位复现**（见 §4 第 5 条），且全部关键量已由 solver-free 探针独立覆盖。
- 未复核 floating-point dtype 对齐（交付求解用 float32、恒等腿用 float64）：float32 下 σ 的舍入误差量级 ~1e-7，远小于 1.6–6% 的结论误差，**不构成精度风险**，但也未单独量化。

## 5. 独立性限制
- 审计者与交付者共用同一份源码树（m21400 HEAD 含 m21378 merge），无法排除"共享实现假设"——因此对"库约定"类主张（C2）只做到"与库文档/自洽性一致"，未做独立实现回归。
- Maxwell 参照是**解析自洽**参照（同一 SC 状态方程），不是实验/文献锚：它能证明共存密度与 EOS 一致，**不能**证明该 EOS/力项配方与 TCLB 或实验一致。
- 判级依据以工件与可执行检查为准，未采信报告文字中的任何未附工件数字（唯一例外已列 F2）。

## 6. 处置（disposition）与返修清单
**approve with notes → L3 confirmed**。交付是实质增量（把锚点问法从自研复刻搬到库路径，并修实三处硬伤），不是文档堆砌；但 L4 因外部锚空缺不可达。

返修（按优先级，均非阻断本单消账）：
1. 登记 `tensor_benchmark_cases` 一行（ref=Maxwell 解析 + 文献 σ，tolerance <3%/<15%），让锚点下次可被基准表裁定。
2. 补 TCLB `d2q9_scmp` 同算例（或改用一篇明确给出 SC/EDM σ 的文献值），填上 `external_anchor`。
3. F2 的 τ 离散度落字段 + 统一口径重述。
4. F3 图入库格式 / 报告图内嵌。
5. F4 R1 目录标 superseded、删假工件行。
6. 把 45 例全量跑完并取回工件（`cj3388` 从未 fetch）。