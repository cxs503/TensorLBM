# m15969 交差核验：m15952 / CASE-MCMP-LAPLACE 第 2 轮（TensorLBM）

审计对象：mission **m15952**（succeeded），交付提交 **cdb7be19**（PR #1881，合并于 `5b959fa1`）
被审锚点：`CASE-MCMP-LAPLACE` —— 2D 两组分 Shan–Chen 静止液滴 · Young–Laplace 定律 `Δp = σ/R`
审计单：mission **m15969**（board-audit，非重做）；分支 `agent/m15969`，基线 HEAD `2e89b2c3`。
**结论：L3**（独立功能实现 + 接入仓内两相求解器与算例池），**不确认 L4**：非循环裁判下 `Δp = σ/R` 的绝对值判据
**0/10 通过**，且无 `tensor_benchmark_cases` 登记行、第 2 轮证据未入库、文档所述复现命令在交付时以非零码退出。
第 1 轮由 m15891 授予的 L4 依据（σ 由同一组 Δp 拟合）经本轮证实为**循环判据**，应予撤销。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

两组分 Shan–Chen（MCMP），线性状态方程 `ψ(ρ)=ρ`，BGK，速度平移力格式，D2Q9 周期方盒。
静止液滴平衡时界面两侧压差服从 2D（圆柱界面）Young–Laplace 律

```
Δp = σ / R        （3D 球为 2σ/R）
```

| 项 | 取值（被审工件，`inputs/eos_sigma.json` / `inputs/independent_sigma.json`） |
|---|---|
| 耦合常数 G₁₂ | −2.5 / −3.0（两点） |
| 网格 | n = 96 主序列；n = 128 分辨率点（固定 R≈16） |
| 初始半径 r₀ | 12 / 16 / 20 / 24（n=96），16（n=128） |
| 液滴点数 | 10（2 耦合 × 4 半径 + 2 分辨率点） |
| 容差（算例池） | `<5%`（`tensor_case_pool` id=16 `MCMP-LAPLACE`，tol_policy `<5%`） |
| 参考量 | 解析律 `σ/R`；σ 取本轮**独立锚** `σ_EOS`（平界面 EOS–剖面/梯度理论积分） |
| 机器裁判 | `tensor_benchmark_cases WHERE anchor_slug='CASE-MCMP-LAPLACE'` → **0 行**（全表 25 行） |

被审工件改动面（cdb7be19）：`benchmarks/case_pool/mcmp_laplace/eos_sigma.py`（+277）、
`independent_sigma.py`（+338）、`docs/case_mcmp_laplace_round2.md`（+109）；运行工件全部落在
`runs/m15952_laplace/`（**被 `.gitignore:108` 忽略，未入库**）。

## ② 怎么做的（独立复算，不采信报告数字）

审计不重跑液滴链（4.5 min/耦合点），而是**从原始场量独立重算每个头条数字**，并对可执行面做授权复现：

1. **复算脚本**：`artifacts/m15969_audit/audit_m15969.py`（唯一入口，读 `artifacts/m15969_audit/inputs/`
   里的被审原始 JSON，输出 `audit_matrix.json` + `figs/fig1_ratio_and_residual.png`）。
   逐点重算 `Δp·R/σ_EOS`、误差 `100(Δp−σ_EOS/R)/(σ_EOS/R)`、逐耦合点 `Δp·R` 散度、分辨率敏感性。
2. **报告表格回代**：正则解析被审 `docs/case_mcmp_laplace_round2.md` 的 ②④⑤ 三张表，与重算值逐格比对
   （容差 1e-6…0.01），检验「程序化生成/数字零手抄」是否名副其实。
3. **崩溃复现**：以被审 `independent_sigma.json` 为输入直接调用 `_summary()`（文档复现命令的最后一步），
   捕获异常栈，确认交付时的退出行为。
4. **生成链复现**：在审计工作树内以同一份 `independent_sigma.json` 重跑
   `eos_sigma.py --out runs/m15952_laplace/eos_sigma_rerun.json`（rc=0，67.1 s），逐值与交付 JSON 对比，
   并把重新生成的文档与交付文档 diff（`artifacts/m15969_audit/regenerated_vs_committed.diff`）。
5. **登记面核对**：SQL 直查 `tensor_benchmark_cases`（0 行）与 `tensor_case_pool`（id=16，`<5%`）；
   `git ls-files runs/...` 与 `git check-ignore` 核对工件入库状态。
6. **器物检查**：`render()` 是否产出 ⑦ 段与 `verdict.json`；全仓是否有 `verdict.json` 的生成者。

## ③ 计算结果（数字表，全部由 `audit_matrix.json` 程序化注入）

### 表 1｜重算 vs 被审使命（头条数字一致性）

| 量 | 使命（`verdict.json`） | 审计重算 | 判定 |
|---|---|---|---|
| `Δp·R/σ_EOS` 均值 | 0.318143 | 0.318143 | ✅ 一致 |
| 区间 min / max | 0.309476 / 0.323224 | 0.309476 / 0.323224 | ✅ 一致 |
| 跨 10 点散度 | 4.32% | 4.3216% | ✅ 一致 |
| `Δp·R` 半径无关性 G−3.0 / G−2.5 | 2.677% / 4.207% | 2.682% / 4.215% | ✅ 一致 |
| 原始锚误差（逐点） | −67.68…−69.05% | −67.68…−69.05% | ✅ 一致 |
| 原始锚 `<5%` 通过数 | 0/10 | **0/10** | ✅ 一致 |

⇒ 被审报告的表 ④（10 行）**逐格与重算一致，0 处不符**；`verdict.json` 头条数字可复算。
诚实 FAIL 成立：**非循环裁判下绝对值判据为 0/10**。

### 表 2｜常数偏差的来源候选（审计补算，供下一轮）

| 归一化 | 逐耦合点残差 | 落在 ±5% 内 | 备注 |
|---|---|---|---|
| 原始 `σ_EOS` | −68.13% / −68.24% | 0/10 | 预注册判据，FAIL |
| `cs²·σ_EOS`（=σ/3） | **−4.389% / −4.726%** | **7/10** | 使命 ⑦ 记为「−0.04% / −0.05%，仍在 ±5% 之外」→ 见 F4 |
| `σ_EOS/π`（=`σ·0.31831`） | 均值偏差 **−0.052%** | 4/10（逐点散度 4.32% 吞掉） | 新候选：缺因子 π |
| `Δp·R` 与 `σ_EOS/π` 的一致度 | 峰值 −0.05%，区间 ±2.2% | — | 与 cs² 假设不可区分（π 与 3 相差 4.7%） |

结论：偏差是**常数归一化**问题（10 点上固定倍数），不是曲率/半径效应；`1/π` 与 `1/cs²` 两个候选在逐点散度内
不可区分，需要从离散相互作用直接推导切向应力归一化才能定案——本轮不据任何候选改判。

### 表 3｜可执行面核验

| 检查 | 结果 |
|---|---|
| `eos_sigma.py` 重跑（锚 + 文档 + 图） | rc=0，67.1 s（使命 76.7 s），`σ_EOS` **逐比特相同**，逐点误差差 0 |
| 文档复现命令 #1（`independent_sigma.py`）末步 `_summary()` | 交付时 **ZeroDivisionError**（`independent_sigma.py:306`），以被审工件 0.1 s 内复现 |
| 重新生成的文档 vs 交付文档 | diff 73 行：①行序不同（交付件按 `--G -3.0 -2.5` 生成）②**⑦ 段整体缺失** |
| `runs/m15952_laplace/*` 入库 | `git ls-files` = 空（`runs/` 被 `.gitignore:108` 忽略）→ JSON/verdict/图均只在 mission 工作树 |
| `verdict.json` 生成者 | 仓内 0 处（`grep -rl verdict.json benchmarks/case_pool/mcmp_laplace` = 空） |
| `tensor_benchmark_cases` 登记 | 0 行 → 无机器裁判；容差来源为 `tensor_case_pool` id=16 |

图：`artifacts/m15969_audit/figs/fig1_ratio_and_residual.png`
（左：`Δp·R/σ_EOS` 随 R 的平台 + 1/π、cs² 参考线；右：原始锚与 cs² 归一化假设的逐点残差与 ±5% 带）。

## ④ 图：与解析/基线的比较

* 左图：10 个点的 `Δp·R/σ_EOS` 落在 0.3095–0.3232 的水平带上（无 R 依赖）⇒ `1/R` 标度被数据支持，
  偏差是公共倍数；红虚线 `1/π=0.31831` 与灰点线 `cs²=1/3` 分别为两个候选归一化。
* 右图：原始锚逐点残差 −67.7…−69.1%（全部在 ±5% 带外）；乘 `cs²` 后 7/10 点落回 ±5% 带内，
  且残差随 R 单调漂移（R 越大越接近 0）——说明 `cs²` 修正后剩余的是**离散/有限半径**效应，
  与「常数归一化 + 曲率修正」图象自洽，但不足以定案。

## ⑤ 注意点（坑与局限）

* **循环判据的陷阱**：第 1 轮 `run_case.py` 用同一组 `Δp_i` 过原点最小二乘得 `σ_fit`，再拿 `σ_fit/R_i` 判
  `Δp_i`——任何公共倍数误差原样通过。这类判据在「PASS」时最危险：本轮实测绝对值偏差 68%，第 1 轮判据仍 PASS。
* **锚共用的 EOS**：`σ_EOS` 与液滴 `Δp` 共用 `p_iso = cs²(ρ₁+ρ₂)+G₁₂cs²ρ₁ρ₂`，公共尺度误差在比值中相消，
  故该锚只能界定**切向应力归一化**，不能独立检验 EOS 的绝对尺度（独立性声明在「与 Δp/R 无关」意义上成立）。
* **路由 B 自门否决**：离散 Kirkwood–Buff（两点 SCMC 压力张量）在平界面上 `P_NN` flatness≈44%、σ≤0，
  连负对照 `isotropic` 同样 43%——这是模型应力张量本身的问题，不是门盲；本轮判死合理。
* **文档化 ⑦ 段脱离生成链**：`render()` 只写到「与第 1 轮判据的差异」，⑦ 是手工追加，其数字未经生成器校验，
  恰好也是唯一出现单位错的地方（见 F4）。凡是「生成式报告」，判决段必须同样生成。
* **未入库的 `runs/`**：本轮全套证据（JSON、verdict、图）都不在仓库；`eos_sigma.py` 在缺 `independent_sigma.json`
  时把 `drops` 置空（`checks=[]`），仍会**静默**写出一份「0/0 点」文档——降级路径无护栏。

## ⑥ 误差判定（关键量 vs 基准，逐项，显式回答 <3%）

| 关键量 | 基准 | got | 误差 | <3%？ |
|---|---|---|---|---|
| `Δp` @ G−3.0, R=11.65 | `σ_EOS/R` | 0.018011 | **−68.53%** | ❌ 否 |
| `Δp` @ G−3.0, R=15.70 | `σ_EOS/R` | 0.013531 | **−68.16%** | ❌ 否 |
| `Δp` @ G−3.0, R=19.67 | `σ_EOS/R` | 0.010885 | **−67.91%** | ❌ 否 |
| `Δp` @ G−3.0, R=23.60 | `σ_EOS/R` | 0.009135 | **−67.68%** | ❌ 否 |
| `Δp` @ G−3.0, n=128, R=15.64 | `σ_EOS/R` | 0.013486 | **−68.37%** | ❌ 否 |
| `Δp` @ G−2.5, R=11.49 | `σ_EOS/R` | 0.016059 | **−69.05%** | ❌ 否 |
| `Δp` @ G−2.5, R=15.65 | `σ_EOS/R` | 0.012105 | **−68.23%** | ❌ 否 |
| `Δp` @ G−2.5, R=19.66 | `σ_EOS/R` | 0.009722 | **−67.95%** | ❌ 否 |
| `Δp` @ G−2.5, R=23.61 | `σ_EOS/R` | 0.008156 | **−67.72%** | ❌ 否 |
| `Δp` @ G−2.5, n=128, R=15.52 | `σ_EOS/R` | 0.012085 | **−68.26%** | ❌ 否 |
| `Δp·R` 半径无关性（G−3.0 / G−2.5） | 常数 | — | 2.68% / 4.22% | ✅ 是（标度律，非绝对值） |
| `σ_EOS` 跨分辨率（n=96→128） | 同值 | — | +0.00% / −0.92% | ✅ 是（自洽性） |
| `Δp` vs `cs²·σ_EOS/R`（审计补算，非预注册） | `cs²σ_EOS/R` | — | −4.39% / −4.73% | ❌ 否（7/10 点在 ±5% 内） |

**是否 <3%：关键量（液滴压差 vs 独立表面张力锚）无一通过，最小偏差 67.68%。** 被审使命对此的诚实申报
（`0/10 → FAIL`）与重算一致，属合格的 honest-FAIL；不得以「第 1 轮 PASS」或「报告完整」粉饰。

---

## 缺陷与判级

### 判级（L0–L5）

**L3**。判据与实质：

| 级别 | 判 | 依据 |
|---|---|---|
| L0/L1 | 否 | 不是文档/单元级产物 |
| L2 | 是 | 真代码：`eos_sigma.py` + `independent_sigma.py` 输入→LBM 松弛→积分→误差输出 |
| **L3（本级）** | **是** | 驱动仓内**既有**两相求解器公共 API（`tensorlbm.mcmp_verification.{estimate_coexistence,simulate_planar_slab,simulate_static_droplet,measure_laplace}`；`run_case.py:228-230` 直接调 `tensorlbm.multiphase.collide_sc_two_component` / `tensorlbm.solver.stream`），算例位于仓内算例池 `benchmarks/case_pool/mcmp_laplace/`，有入口脚本与已入库的第 1 轮 `artifacts/summary.json` |
| L4 | **否（撤销）** | ①非循环裁判下判据 **0/10 FAIL** → 不是「基准验证通过」的算例；②`tensor_benchmark_cases` 该 anchor **0 行**，无机器裁判登记；③第 2 轮全套工件未入库、文档复现命令交付时非零退出、⑦/verdict.json 无生成者 → 可复现性不闭合 |
| L5 | 否 | 无科研级结论：常数归一化的来源未定（π/cs² 不可区分），文献离散压力张量未实现 |

### 缺陷（按严重度）

**F1（HIGH · 判级/科学状态）** 第 1 轮 L4 建立在循环判据上，本轮以独立锚证伪绝对值判据（0/10，−67.7…−69.1%）。
证据：表 1/表 3 + `audit_matrix.json:recompute.pass_at_5pct`。影响：`CASE-MCMP-LAPLACE` 不能以「L4 已验证」结账。

**F2（MEDIUM · 可复现缺陷，本单已修）** 文档复现命令 #1 末步 `ZeroDivisionError`（`independent_sigma.py:306`）：
配对选择只看 flatness、不执行报告自己的 `σ>0` 门，选中 `bwd`（σ≡0）。以被审工件 0.1 s 复现。
修复：按「flatness<5% 且 σ>0（两分辨率）」筛选，无合格配对时置 `best=None` + `gate_failed` 且不出行；
新增 `tests/test_mcmp_laplace_anchor_gate.py`（3 项不变量断言：全族被门否决时不再除零并显式标旗、
合格配对选择、最平但 σ=0 的配对不得被选中）→ **3 passed**。修复后同一工件复演：`rows=0, gate_failed={G-2.5:True, G-3.0:True}`，不抛异常。

**F3（MEDIUM · 报告可复现性）** `docs/case_mcmp_laplace_round2.md` 自称「程序化生成、数字零手抄」，
但 ⑦ 判定段与 `runs/.../verdict.json` **无任何生成者**（`render()` 不含 ⑦ / 不写 verdict.json；仓内 0 处写 verdict.json）。
重跑 `eos_sigma.py`（rc=0，σ_EOS 逐比特相同）重新生成的文档**没有 ⑦**：`regenerated_vs_committed.diff` 73 行，其中 24 行是 ⑦ 整段。

**F4（MEDIUM · 数字单位错）** ⑦ 称 cs² 归一化残差「G−3.0 −0.04%，G−2.5 −0.05%，仍在 ±5% 边界之外」；
`verdict.json` 实为 −0.042055 / −0.047110（分数）⇒ **−4.39% / −4.73%**，报告小 100 倍；且按此值 7/10 点
**落在 ±5% 之内**，与「边界之外」相反。头条 0/10（原始锚，预注册判据）不受影响，故整体 honest-FAIL 成立，
但该段作为「候选解释」低估了一个明显候选，须按真值重写。

**F5（LOW · 无依据断言）** 「倍数在 10 个点上一致到 0.3%」——无任何工件含 0.3%：实测 `ratio` 散度 4.32%
（0.3095–0.3232）；最接近的真实数字是 σ_EOS 跨分辨率 0.92% 与逐 G 的 Δp·R 散度 2.68%/4.21%。该数字应删或改为 4.3%。

**F6（LOW · 工件保管）** `runs/` 被 `.gitignore:108` 忽略，第 2 轮证据（2 个 JSON、verdict.json、图）**全部未入库**；
且 `eos_sigma.py` 在缺 `independent_sigma.json` 时 `drops=[]` → 会**静默**产出「0/0 点」文档（无护栏）。
建议：证据迁至 `artifacts/m15969.../` 或 `benchmarks/case_pool/mcmp_laplace/runs/` 并入库，缺失输入直接硬失败。

**F7（LOW · 溯源）** 交付 `eos_sigma.json` 的 levels 顺序为 [−3.0, −2.5]，即由 `--G -3.0 -2.5` 生成，
而文档写明 `--G -2.5 -3.0` ⇒ 交付件与文档命令不是同一条；重新生成后 ②④⑤ 行序改变。命令与工件须一致。

**F8（INFO · 独立性成立但有限）** 锚 2 只用平界面剖面 + 体相 EOS，不含液滴 Δp/R ⇒ 独立性声明成立；
局限：与液滴共用 `p_iso`，公共尺度误差相消，只能界定归一化（非 EOS 绝对尺度）。

### 检查了 / 没检查（不虚报）

* 已跑：重算矩阵（10 点）；报告 ②④⑤ 表格回代（10/10 行 0 不符，Route-B 8 行、σ_EOS 3 行）；崩溃复现；`eos_sigma.py`
  重跑（rc=0，逐比特一致）；文档重生成 diff；`git` 入库状态；SQL 查两表；新测试 3 passed。
* **未跑**：完整 `independent_sigma.py` 液滴链重跑（4.5 min，已递交集群 **cj1214**，报告时排队中；崩溃修复由
  「失败步直接复演 + 3 项断言」证明，不用一次全链重跑当证明）；`tests/test_laplace_law.py`（>110 s，超本机单次时限，未判读）；
  未复算 LBM 碰撞/迁移内核本身（属仓级单测范围）；未取得 Shan (2008) 离散压力张量做锚 1 重建（仓库内无实现，
  使命亦如实列为后续项）。

### 处置（disposition）

**request changes / 重开（reopened）**。收账条件（下一轮）：
1. 把非循环判据登记进 `tensor_benchmark_cases`（anchor `CASE-MCMP-LAPLACE`，metric=`Δp` vs 独立 σ/R，tolerance `<5%`），
   并如实登记当前 **FAIL**（或先解决归一化再登记 PASS）——不允许只用 `tensor_case_pool` 的 tol 说事；
2. 解决常数归一化（从离散相互作用推导切向应力权重；π 与 cs² 两个候选须有一个有推导支撑）；
3. ⑦ 判定段与 `verdict.json` 纳入生成链；F4/F5 按真值重写；
4. 工件入库（不再落 `runs/`），`eos_sigma.py` 缺输入硬失败；
5. 文档命令与工件生成的 G 顺序一致。

L4 在 1–4 完成且判据通过后才可恢复；本轮功能实质维持 **L3**。
---

## 附：交付指针

* verdict：`verdict_m15969.sql` → `tensor_maturity_verdicts` id=422（repo=TensorLBM, scope=`m15952 CASE-MCMP-LAPLACE`, level=3, verdict_by=board-audit）
* 平台报告：`tensor_reports` id=15105（scenario_key=CASE-MCMP-LAPLACE, mission_id=15969，含两图）
* 重算产物：`artifacts/m15969_audit/`（`audit_m15969.py` 唯一入口；`audit_matrix.json`；`repro_evidence.json`；
  `regenerated_vs_committed.diff`；`inputs/` 被审原始 JSON 快照；`figs/fig1_ratio_and_residual.png`、`figs/fig2_route_b_gate.png`）
* 修复：`benchmarks/case_pool/mcmp_laplace/independent_sigma.py`（`_summary` 按 σ>0 门筛配对 + `gate_failed` 标旗）
  ＋`tests/test_mcmp_laplace_anchor_gate.py`（3 passed）
