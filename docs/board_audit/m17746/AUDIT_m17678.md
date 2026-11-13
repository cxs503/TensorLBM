# Board audit — m17678 · XVAL-LBM-CONSERVATIVE-ALLEN-CAHN-H-POPULATION

**Auditor mission** m17746 (board-audit) · **Subject** m17678 (TensorLBM, status `succeeded`, branch `agent/m17678`)
**Subject worktree** `/root/tp-agent/work/m17678` · **Audited artifact** `artifacts/xval_m17678/xval_tclb_cac_h_population.json`
(+ `_quick.json`, `curves.json`, `fig1..fig4`)
**Verdict** **L2** — independent functional implementation. L3/L4 **not** met (reasons in §⑤, §⑥).
**Disposition** *request changes / reopen for L3–L4* — the physics numbers are good; the promotion blockers are
**registration, integration and commit**, not accuracy.

> This is a **交差核验单**, not a redo. The auditor re-ran the tests and the repro, re-derived nothing by hand,
> and regraded no metric: the numbers below are read out of the subject's own artifact by
> `scripts/board_audit_m17746.py` (→ `artifacts/board_audit/m17746/audit_m17678.json`, `fig_audit_m17678.png`).

---

## ① 案例介绍（物理背景 / 算例 / 参数表）

**物理背景.** 保守 Allen–Cahn（CAC）相场方程用 **h-群（h-population）** LBM 求解，靠**反扩散/逆梯度源项**
(`∝ 4φ(1−φ)/W · c·n`) 让界面维持有限宽度而不被扩散抹平。本仓已有 `cac_lbm` / `cac_lbm2d`（Liang et al. PRE 97,
033309, 2018 路线，称为 **Arm A**）；CFD-GO/**TCLB** `models/multiphase/d3q27_pf_velocity`（Fakhari & Mitchell
一系）解**同一个方程**但离散配方不同。本单目的：**字段级交叉验证（XVAL）**两套实现，并产出一个 TCLB 平权核。

**算例与参数**（取自工件 `/static2d/params`、`/adv2d/params`、`/static3d/params`；格子单位 dx=dt=1, cs²=1/3）：

| 算例 | 网格 | R | W(=IntWidth) | M | ω=1/(3M+½) | W_eff=3MW | R/W | U(CFL) | 步数 |
|---|---|---|---|---|---|---|---|---|---|
| static2d | 128² (D2Q9) | 25 | 4 | 1/3 | 0.6667 | 4 | 6.25 | 0 | 2000 |
| adv2d | 128² (D2Q9) | 20 | 4 | 1/3 | 0.6667 | 4 | 5 | 0.05（精确 32 格） | 640 |
| static3d | 48³ (D3Q27) | 10 | 3 | 1/3 | 0.6667 | 3 | 3.33 | 0 | 400 |
| M 扫描 | 64² | 12 | 4 | 1/12…1/2 | 0.667–1.5 | 1…6 | 3 | 0 | 600 |
| W 扫描 | 96² | 18 | 2,3,4,6 | 1/3 | 0.6667 | =W | 9…3 | 0 | 800 |
| U 扫描 | 128² | 20 | 4 | 1/3 | 0.6667 | 4 | 5 | 0.01–0.05 | 16/U |

无粘性/无 g-LBE 耦合 ⇒ 本单**无 Re**，无量纲群只有 M、W、U(CFL)、R/W、L/R。运行环境：
torch 2.7.1 · python 3.11.6 · float64 · CPU 128 核 · threads 8（工件 `/env`）。

## ② 怎么做的（方法 / 工具链）

1. **Job 1 结构层（零手抄）**：`scripts/xval_m17678_tclb_cac_h_population.py`（921 行）在运行期从 TCLB **源码文本**解析出
   q15/q27 权重、`tmp1` 反扩散系数、`calcF_phi` 逆梯度式、collision 行（L928）、`model.R` 的 `IsotropicGrad`/`myLaplace`
   公式、`src/lib/lattice.R` 格子矩阵、`Dynamics.R` 的 `AddSetting` 默认值；逐项与 TensorLBM 实现比对（P1/P2/P2b/P9a/P9b）。
2. **Job 2 物理层（解析参考）**：**Arm B** = 本单新增 `src/tensorlbm/phasefield/cac_h_population.py`（473 行，TCLB 平权核，
   D2Q9/D3Q19/D3Q27）；**Arm A** = 既有 `tensorlbm.cac_lbm2d`（给定速度场驱动）。参考值＝解析 tanh 剖面、**精确平移**、
   **通量形式的质量严格守恒**；再加跨臂场 L2（XVAL 主判据）。
3. **工具链**：干净室重写（TCLB 是 GPL-3.0，只读参考，公式属公开文献）；探针 = `/tmp/TCLB` 源码 sha256 进工件
   （审计复核：`model.R` = `c8a220a9…f6f6`，与本单工件记录**逐字符一致**，源码树仍在，可追溯）。

## ③ 计算结果（数字表，来自工件，非手抄）

`scripts/board_audit_m17746.py` 从工件 JSON 读出，审计不重算：

| 代号 | 度量 | 参考 | got | <15% | <3% |
|---|---|---|---|---|---|
| P1 | 反扩散源项 F_phi vs TCLB 抽取式 | TCLB 文本 | **0.0**（max abs 0.0） | ✅ | ✅ |
| P2 | 拉普拉斯 vs `myLaplace` | TCLB 文本 | **2.30e-16** | ✅ | ✅ |
| P2 | 梯度 vs `IsotropicGrad` | TCLB 文本 | **1.428**（x 轴；y 轴 3.3e-16） | ❌ | ❌ |
| P9a | 单步 vs `h−ω(h−h_eq+½F)+F` | TCLB 文本 (3D) | **0.2577** | ❌ | ❌ |
| P9b | 跨臂单步 (2D, u=0) | Arm A | **6.25e-10** | ✅ | ✅ |
| P3 | 静态质量漂移 2000 步 | 0 | **7.44e-14** | ✅ | ✅ |
| P4 | 界面半径漂移 | 0 | **0.204%** | ✅ | ✅ |
| P5 | 径向剖面 vs 解析 tanh（B / A） | 解析 | **0.312% / 0.312%** | ✅ | ✅ |
| P6 | 平移质心误差 B / A | 精确平移 | **1.310% / 1.476%** | ✅ | ✅ |
| P6 | 平移 φ 场 L2 B / A | 精确平移 | **1.690% / 1.098%** | ✅ | ✅ |
| P7 | **跨臂场相对 L2（静态 / 平移）** | — | **1.169e-6 / 0.6578%** | ✅ | ✅ |
| P8 | 宽度律 W_meas/W_pred（M=1/12,1/6,1/3,1/2） | 3MW | 4.167 / 2.084 / **1.031** / 0.687 | ❌(3/4 点) | ❌ |
| P3d | 静态 3D 球：质量 / 半径 / 剖面 | 0/0/解析 | 1.50e-14 / 0.048% / 0.97% | ✅ | ✅ |

**机器计数（审计脚本输出）**：13 条被评级度量中 **11/13 < 15% 且 < 3%**；两条不过的是 P2 梯度与 P9a。

**图**：`artifacts/board_audit/m17746/fig_audit_m17678.png` — 13 条度量的 |误差| 对 15% 判据线与 3% 线的对数条形图
（绿 <3% / 黄 3–15% / 红 >15%，逐条标注数值）。原单 4 张图：`fig1_static2d_profile_mass.png`、
`fig2_advection_cut.png`、`fig3_sensitivity.png`、`fig4_static3d_sphere.png`（审计未复算其像素内容）。

## ④ 三件套核验（交差硬要求）与逐项裁判

| 三件套 | 结论 | 证据 / 审计动作 |
|---|---|---|
| **run_artifacts 文件** | ✅ 存在，但**未入库** | `artifacts/xval_m17678/`：`xval_tclb_cac_h_population.json`(17 069 B)、`_quick.json`、`curves.json`(24 714 B)、4×PNG。`git ls-files` 对 6 条交付路径**全空** ⇒ 只活在 worktree |
| **benchmark ref vs got 数值** | ⚠️ 数值齐、**平台无登记判据** | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CONSERVATIVE-ALLEN-CAHN-H-POPULATION'` → **0 rows**（ILIKE `%ALLEN%`/`%conservative%` 亦 0）。故"唯一裁判"缺席，只能用工件+解析参考+TCLB 文本裁判 |
| **repro_cmd 可执行** | ✅ quick 可执行并复现（full 未复核） | 审计实跑 `python3 scripts/xval_m17678_tclb_cac_h_population.py --quick` → 100.0 s 完成，工件内 `quick` JSON 重写；**未**复跑 full（353 s 档），故 full 数值未经审计独立重算 |
| **报告六段齐全** | ✅（report id 16667；16665 为同题重复稿） | ①案例②方法③参数表④ref vs got 表⑤敏感性/注意点⑥本仓 vs OSS 差异清单，全部在位；且**主动披露** 3 处 honest FAIL 未并入结论 |
| **测试** | ✅ 3 passed | `python3 -m pytest tests/test_xval_m17678_cac_h_population.py -q` → `3 passed in 9.34s`（审计实跑） |

## ⑤ 判级：L2（三级门槛的逐条裁决）

| 级 | 判据 | 裁决 | 依据 |
|---|---|---|---|
| L0 文档 | — | 越过 | 有实现，不适用 |
| L1 数学/单元验证 | 算子级恒等 + 单元测试 | **满足** | P1 源项 0.0、P2 拉普拉斯 2.3e-16、P9b 6.25e-10；3 个单测绿 |
| **L2 独立功能实现** | 真代码 输入→计算→输出 | **满足（定为 L2）** | `src/tensorlbm/phasefield/cac_h_population.py` 473 行：`Stencil/get_stencil`、`isotropic_gradient/laplacian`、`interface_normal`、`counter_term`、`h_equilibrium`、`CACHPopulation`(step/mass)、`update_from_tclb`；D2Q9/D3Q19/D3Q27 三档；由 921 行驱动脚本端到端跑通 2D/3D |
| L3 系统集成 | 接入主求解器/平台 | **不满足** | `grep -rn "cac_h_population\|CACHPopulation" src/ benchmarks/ tests/ examples/` 排除自身/脚本/test 后 **0 命中**；未进 `phasefield/__init__.py` 的 `__all__`；无 registry / capability contract 登记；唯一调用者是自己脚本第 42 行与 test 第 9 行 |
| L4 基准验证 | 完整算例+误差分析+可复现 | **不满足**（数值层面接近） | ①`tensor_benchmark_cases` **0 行**，平台无机器判据；②3 处 gates 自报 FAIL（P2 梯度 1.428、P9a 0.258、P8 4/1 点），其中两项是"结构恒等"主张的直接反证，未收敛；③6 条交付路径**全未入库**（`git ls-files` 空）⇒ 检出即丢失；④审计只复现了 quick 档，full 档数值无独立复算（见⑥） |
| L5 科研工程成果 | — | 不适用 | 本单是 XVAL 一致性研究，非工程产品；且无实验数据 |

**为什么不是 L4（尽管主判据 0.6578% 远优于 15% 线）**：本仓板审已有先例（m17664 同签名：0 call site ⇒ L3/L4 不met，
"promotion blocker is coverage + registration + commit, not accuracy"）。本单同签名，且多一条
**未入库**——一次 `git push` 后这 6 个文件不在任何分支里。数值再好也不改变级别；级别=功能实质。

## ⑥ 注意点 / 审计局限 / 缺陷清单（按严重度）

**F1（blocking，L3/L4）交付物未入库**：6 条路径 `git ls-files` 全空（模块/脚本/test/2 篇文档/1 个 JSON）。
mission 状态 `succeeded` 但产物只在 `/root/tp-agent/work/m17678`。worktree 一旦回收，m17678 的全部证据不可恢复。

**F2（blocking，L4）无登记基准**：`tensor_benchmark_cases` 该锚 0 行 ⇒ 平台的机器判据缺席，验收只能靠工件自证；
板审只能核"数值是否在合理容差内"，不能核"是否命中平台判据"。

**F3（major）两个"结构恒等"只成立一半**：P2 梯度按轴为 1.428 / 3.3e-16（同一 instrument 下 y 轴完美、x 轴差 143%），
报告归因于**自家 R→Python 重建缺陷**（单模场上重建式返回全零）——归因合理但**未证明**；因此
"梯度系数与 TCLB 相同、只是实现路径不同"这一条只有拉普拉斯一个实证支柱。P9a（3D 单步 0.258）同理归因于参考侧。
两处均被如实标为 honest-FAIL 且未并入结论（**符合证据纪律，不计降级**），但它们直接削弱 L4 的"误差分析闭环"。

**F4（major）宽度律 P8 在 3/4 个 M 点被证伪**：W_meas/W_pred = 4.167 / 2.084 / **1.031** / 0.687。仅 M=1/3（3MW=W 的配对点）
成立。报告的处置（把 `W` 重释为"反扩散系数"、要求 M=1/3 配对）是诚实的语义修正，但**意味着"界面宽度 = 3MW"作为一般律不成立**；
下游若有人照搬 `width=IntWidth` 会拿到错宽度。这条应写进模块 docstring 的显式断言（现已有，建议再加一行"仅 M=1/3 时物理宽度=W"）。

**F5（minor）quick 档不能当复现证据**：审计实跑 quick 得到 adv 质心 5.27%、场 2.47%，而 graded full 档是 1.31% / 1.69%；
quick 还额外让 `P6_pass` 变 False。文档只写 `[--quick]`，未声明"quick 数值不等于 graded 数值"。建议脚本在 quick 工件里写
`"not_graded": true`，避免后人拿 quick 数字对表。

**F6（minor）拟合器符号缺陷残留在工件**：`width_from_profile` 的 `atanh` 参数符号反了（返回负宽度），报告已披露、
模块已修，但 `xval_tclb_cac_h_population.json` 的 P8 数值带缺陷痕迹。

**审计局限（audit limits，如实申报）**
1. **未复算 full 档**（353 s 档）——审计只跑 quick（100 s）与单测；③④表中的全部数值都来自**被审方工件**，
   不是审计独立重算的。因此"got 数值是否真如其声称"在本单是 **withheld/verified-by-provenance**，不是 auditor-reproduced。
2. **未复核 4 张 PNG 的像素内容**，只确认文件存在与大小。
3. **TCLB 侧未执行**：只核源码 sha256 与文本解析，未编译运行 TCLB（架构/许可限制）；故"ref"是**源码文本**派生的参考，
   不是 TCLB 运行输出。
4. **无物理精度结论**：两臂一致 ⇒ 证明"移植/配方对齐"，**不**证明物理精度；本单无 g-LBE 耦合、无壁面润湿，报告自己也把
   耦合精度单列。
5. 未核 `/tmp/TCLB` 之外的上游来源（是否与 GitHub 上游同 commit 未验证，仅核 sha256 与工件自述一致）。

## ⑦ 误差判定（显式回答"是否 <3%"）

- **主判据（跨臂场 L2）**：静态 **1.169e-6**、平移 **0.6578%** ⇒ **<3%** ✅（<15% ✅）。
- **对解析参考**：P5 剖面 0.312%、P6 质心 1.31% / 场 1.69%、P3d 3D 剖面 0.97% ⇒ 全部 **<3%** ✅。
- **守恒**：质量漂移 7.44e-14（2D）/ 1.50e-14（3D）＝机器精度 ✅。
- **不满足 <3% 的项（3 条，全部为报告自报 honest FAIL，审计不代为粉饰）**：P2 梯度 1.428 ❌、P9a 3D 单步 0.2577 ❌、
  P8 宽度律 3/4 点 ❌。
- **结论**：**物理/主判据层 <3% 成立；结构层 2/3 项（源项、拉普拉斯）+ 跨臂 1 步 + 跨臂多步全部 <3%；**
  梯度 instrument 与 3D 单步 instrument 未收敛，宽度律仅在 M=1/3 成立 —— 这三条**不能**计入"通过"。

---

### 处置（disposition）
**request changes**：级别 **L2** 落表；`maturity_gate` 预期 `reopened`。返修清单（按性价比排序）：
1. **commit 6 条交付路径**（F1）—— 一行 `git add` 即可解锁可检出性；
2. **登记 `tensor_benchmark_cases`** 该锚 3 行（P7 static L2 ≤1e-4、P7 adv L2 ≤1e-2、P5 剖面 ≤1e-2）（F2）；
3. **接一个主求解器调用点** 或进 `phasefield/__init__.__all__`（L3）——例如让 `multiphase3d_d3q27` 可切到 Arm B 核；
4. 修 P2 梯度 instrument（单模场重建取项）与 P9a 参考侧（F3）；quick 工件标 `not_graded`（F5）。
推荐**不做**：为了让 P8"过"而改参考值——honest FAIL 不可覆盖。