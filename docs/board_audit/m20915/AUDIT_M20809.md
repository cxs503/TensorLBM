# 交差审计：m20809 · XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D（TensorLBM）

* 审计任务：m20915（任务榜 owner 2026-11-11 交差核验令）
* 被审对象：mission 20809，commit `57afcd8b57c8b3f9d027283d439d33f4a04f03fe (branch agent/m20809, NOT merged into main 2bfb5ee1a)`
* 被审工作树：`/root/tp-agent/work/m20809`；审计工作树：`/root/tp-agent/work/m20915`
* 判决：**L3**（系统集成）；maturity_gate 复核状态见 `tensor_maturity_verdicts.gate_state`

> 本文全部数字由 `artifacts/board_audit_m20915/recompute_checks.py`（独立重算）与
> `render_audit_doc.py`（程序化注入）生成，无手抄。审计方不信任交付的 `ref` 块，
> 而是从 `(A,B,kappa,R)` 重算 `sigma=2A^2/(3B)sqrt(2 kappa/A)` 与 `dp=2 sigma/R`。

## 1. 取证范围

| 项 | 结果 |
|---|---|
| 交付 commit | `57afcd8b5`「m20809: matured mission deliverables」，24 文件 / +4159 行 |
| 分支归属 | `git branch -a --contains 57afcd8b5` → 仅 `agent/m20809`；main（`2bfb5ee1a`）未含 |
| run_artifacts | `results.json`（8 例）+ 8 个 `cases/*.json` + 2 个 withheld 冒烟 + 六段文档 + 2 张图 |
| 平台报告 | `tensor_reports WHERE mission_id=20809` → 1 行（title/summary/2 figures，qa_status=pending） |
| 基准裁判 | `tensor_benchmark_cases WHERE anchor_slug='…YOUNG-LAPLACE-D'` → **0 行**；`tensorbench_results WHERE mission_id=20809` → **0 行** |
| 前序同锚判决 | `tensor_maturity_verdicts id=868`（m20769 → **L1**，当时 NaN 发散、零工件、未提交） |

## 2. 三件套核验（run_artifacts / ref-vs-got / repro_cmd）

### 2.1 逐例 got vs 分析基准（审计方独立重算，绝对值口径）

| 用例 | n | R | xi | operator | force_form | 2σ/R（重算） | dp_got | 相对误差 | <3% | <15% |
|---|---|---|---|---|---|---|---|---|---|---|
| `A_iso_phi_grad_mu` | 48 | 12 | 2.00 | isotropic_d3q19 | `minus_phi_grad_mu` | 0.022222 | 0.019142 | -13.86% | **否** | 是 |
| `L_c7_phi_grad_mu` | 32 | 7 | 2.00 | central7 | `minus_phi_grad_mu` | 0.038095 | 0.030825 | -19.08% | **否** | **否** |
| `L_iso_phi_grad_mu` | 32 | 7 | 2.00 | isotropic_d3q19 | `minus_phi_grad_mu` | 0.038095 | 0.029295 | -23.10% | **否** | **否** |
| `C_n48` | 48 | 12 | 2.00 | isotropic_d3q19 | `mu_grad_phi` | 0.022222 | 0.021008 | -5.46% | **否** | 是 |
| `L_c7_mu_grad_phi` | 32 | 7 | 2.00 | central7 | `mu_grad_phi` | 0.038095 | 0.037486 | -1.60% | 是 | 是 |
| `L_iso_mu_grad_phi` | 32 | 7 | 2.00 | isotropic_d3q19 | `mu_grad_phi` | 0.038095 | 0.036594 | -3.94% | **否** | 是 |
| `L_xi1_R5` | 32 | 5 | 1.00 | isotropic_d3q19 | `mu_grad_phi` | 0.026667 | 0.021481 | -19.45% | **否** | **否** |
| `L_xi1_R6` | 32 | 6 | 1.00 | isotropic_d3q19 | `mu_grad_phi` | 0.022222 | 0.017788 | -19.95% | **否** | **否** |

* 交付 `ref` 与独立重算一致（最大偏差 <1e-12）：**True**；交付 `error.dp_rel` 与重算一致（绝对值口径）：**True**。
* 带宽记账：**1/8** 例 <3%；**4** 例超出 15% 不可接受带；最差 **+23.10%**（`L_iso_phi_grad_mu`）。
* 按力形分组：`minus_phi_grad_mu` 3 例，|err| ≤ 23.10%（全部 >15%）；`mu_grad_phi` 5 例，|err| ≤ 19.95%（xi=2 的 3 例在 15% 带内，xi=1 的 2 例带外）。
* 半径阶梯拟合（独立过原点最小二乘）：`sigma_fit=0.053563660` vs 解析 `0.066666667` → **-19.65%**；仅 2 点（R=[5.0, 6.0]，xi=1.00），1 自由参数且无截距 → 交付报告中的「最大单点残差 +0.38%」不是独立校验。
* 网格阶梯：交付 1 个 n（[48]），交付「跨度 +0.00%」在单点上为空断言（`span_is_vacuous=true`）。

### 2.2 预声明协议覆盖（runner 自身 `case_table()` 是协议）

预声明 15 例，落入 `results.json` 的声明用例 **2/15 = 13%**；另有 6 例为**表外替代用例**。

| 轴 | 预声明 | 交付（声明内） | 缺失 |
|---|---|---|---|
| A 力约定×算子 | 4（A_iso_mu_grad_phi, A_iso_phi_grad_mu, A_c7_mu_grad_phi, A_c7_phi_grad_mu） | 1（A_iso_phi_grad_mu） | A_c7_mu_grad_phi, A_c7_phi_grad_mu, A_iso_mu_grad_phi |
| B 半径阶梯 | 4（B_R8, B_R10, B_R12, B_R14） | 0（—） | B_R10, B_R12, B_R14, B_R8 |
| C 网格阶梯 | 3（C_n40, C_n48, C_n56） | 1（C_n48） | C_n40, C_n56 |
| D 界面阶梯 | 3（D_kappa0.05, D_kappa0.1, D_kappa0.2） | 0（—） | D_kappa0.05, D_kappa0.1, D_kappa0.2 |
| E 时间收敛 | 1（E_steps4000） | 0（—） | E_steps4000 |

表外替代用例：`L_c7_mu_grad_phi`, `L_c7_phi_grad_mu`, `L_iso_mu_grad_phi`, `L_iso_phi_grad_mu`, `L_xi1_R5`, `L_xi1_R6`（n=32；xi=2 组 R=7，xi=1 组 R=5/6 —— 均非声明轴的几何）。

### 2.3 repro_cmd 可执行性（审计方实跑）

| 检查 | 命令 | 结果 |
|---|---|---|
| 从零跑一个真算例 | `run_laplace_calibration.py --quick --out $Q` | 2 例 5.2 s / 5.8 s 跑通；`dp` 与交付 `artifacts/_smoke_fe_cal` **abs_diff = 0.000e+00**（位级一致） |
| 聚合位级重复 | `--aggregate` + `diff` vs 交付 `results.json` | 仅 `generated` 时间戳 1 行差异（`m20915_aggregate_diff.log`） |
| 交付方测试 | `pytest tests/test_free_energy_laplace_calibration.py -q` | **7 passed**（72.08 s，审计方重跑） |
| 旧行为回归 | `pytest tests/test_multiphase3d_d3q27_free_energy.py tests/test_phasefield_common_operators.py -q` | **28 passed**（13.64 s）→ `free_energy_step_3d` 新参数默认值未破坏旧路径 |
| 平台端点 | `import backend.routers.benchmarks` | 路由 `/free_energy_laplace` 存在，`run_free_energy_laplace=True` |

结论：**三件套中的 run_artifacts 与 repro_cmd 是实的**（位级可复现）；
**ref-vs-got 的裁判席位是空的**——本锚点在 `tensor_benchmark_cases` 中 0 行，无容差可比，只能退化为「与解析闭式对比」的内部分析基准，而该内部分析基准的结论是**精度未达标**。

## 3. 报告六段核验（`docs/FREE_ENERGY_YOUNG_LAPLACE_CALIBRATION_20261114.md`）

| 段 | 是否齐全 | 实据 / 缺陷 |
|---|---|---|
| ① 案例介绍（物理/几何/参数表） | 齐 | 参数表列 A,B,Gamma,tau_g、判据 `dp=2σ/R`、σ 闭式与来源 |
| ② 方法/流程/工具链 | 齐 | 冻结 φ 淬火协议、两轴（force_form / operator）、几何体相掩膜、`quench_gate`、集群作业编号 |
| ③ 结果数字表 | 齐但有缺陷 | 表 1 完整；**表 2 渲染为空表**（渲染器 group 过滤名不匹配）；表 3 标题承诺「界面阶梯」但 0 数据 |
| ④ 图（场量/对照） | 齐 | `figures/fe_laplace_calibration_law.png`、`…_convergence.png`；平台报告内嵌 2 图 |
| ⑤ 注意点/局限 | 有但一处归因不实 | §5.4 把表 2 误差归因于「n=48 镜像液滴」，而表 2 数据实为 n=32/xi=1 |
| ⑥ 误差判定（逐项 + 是否<3%） | 齐且诚实 | 显式列出 4 项并回答「**否**」，未粉饰 |

报告纪律（owner 2026-10-29 升级的六要素）**达标**：数字由 `render_calibration_report.py` 程序化渲染，honest-FAIL 如实披露。缺陷集中在渲染器与局限说明，而非伪造数据。

## 4. 判级（L0–L5 逐级）

| 级 | 判据 | 本单 | 依据 |
|---|---|---|---|
| L0 | 文档成果 | 达到 | 99 行六段文档 + 平台报告 |
| L1 | 数学/单元验证 | 达到 | σ 闭式与 dp=2σ/R 独立重算 1e-12 内吻合；7 个 pytest 门 |
| L2 | 独立功能实现（输入→计算→输出） | 达到 | 798 行生产模块 + 407 行 runner；审计方从零执行 `--quick` 与交付位级一致 |
| L3 | 系统集成（接入主求解器/平台） | **达到（本判决级别）** | `multiphase3d.free_energy_step_3d` 新增 operator/force_form（默认旧行为，28 回归测试全过）+ 包级导出 + REST `POST /benchmarks/free_energy_laplace` |
| L4 | 基准验证（完整算例+误差分析+可复现） | **未达** | ①本锚点 `tensor_benchmark_cases` **0 行**、`tensorbench_results` 0 行 → 无裁判；②预声明扫描 **2/15 例**；③8 例中 **4 例 >15% 不可接受带**、最差 +23.10%、σ 拟合 +19.65% |
| L5 | 科研工程成果 | 未达 | 无外部数值对标；关键判据自评「<3%：否」 |

**判决：L3**。附百页报告不改变级别——级别取功能实质：生产模块 + 主步进器选项 + 平台端点是实的（L3），但「完整算例 + 容差内」的基准验证不成立（非 L4）。

## 5. 缺陷清单（按严重度）

| # | 严重度 | 缺陷 | 实据 / 复现 |
|---|---|---|---|
| F1 | high | L4 裁判缺位：anchor 在 tensor_benchmark_cases 无任何注册用例 | 0 行；仓内亦无 register_benchmark_cases.sql；tensorbench_results 0 行 |
| F2 | high | 预声明扫描只交 2/15 例（13%），三条轴实质缺失 | B 半径阶梯(R=8..14)、C 网格阶梯(n=40,56)、D 界面阶梯(kappa=0.05,0.1)、E 时间收敛(4000 步) 全缺；6 个 lite 用例不在预声明表内（n=32、xi=1，界面更薄） |
| F3 | high | 精度未达标且被矛盾表述稀释 | 1/8 <3%，4/8 >15% 带外，最差 +23.10%；sigma 拟合 -19.65%；但报告标题突出 +1.6%~+3.9%（n=32 lite 子集） |
| F4 | medium | 交付工件未合入 main | 57afcd8b5 仅 agent/m20809；main 2bfb5ee1a 的 laplace_face.py 无 LAPLACE_FACTOR、multiphase3d 无 operator/force_form、benchmarks 路由无 free_energy_laplace 端点 |
| F5 | medium | 报告渲染缺陷：表 2 半径阶梯渲染为空表 | render_calibration_report.py:245 过滤 group=='radius_ladder'，交付数据 group 实为 'lite_radius_ladder' → 表头下 0 行，而下方拟合段却引用 2 个点；aggregate 侧 run_laplace_calibration.py:195-196 有 fallback，渲染侧没有 |
| F6 | medium | 表 3 标题承诺「界面阶梯（R=12, n=48）」但该轴 0 数据；「网格阶梯误差跨度 +0.00%」在单点上是空断言 | grid_ladder 仅 C_n48 一个 n；span=max-min=0.0 被写成收敛跨度；xi_ladder 无任何用例 |
| F7 | low | 拟合的「最大单点残差 +0.37%」不是独立校验 | 2 点 / 1 参数 / 过原点 → 残差只反映两点共线性；唯一有信息量的是 sigma_fit 相对误差 -19.65% |
| F8 | low | 误差符号口径为绝对值 | 交付 error.dp_rel 全为正；实测均小于基准（got/ref<1），真实符号为负；报告表未声明口径 |
| F9 | medium | docs §5 注意点 4 对半径阶梯误差的归因（镜像液滴、n=48）与交付数据不符 | 表 2 数据实为 n=32、xi=1、R=5/6；n=48 说法错误，镜像效应与薄界面效应未分离 |

### 5.1 说明（为何 F1/F2/F3 是 blocking 而非建议）

* **F1**：交差核验的唯一裁判是 `tensor_benchmark_cases`。本锚点无注册行，等于「无容差可比」，此时任何「通过」声明都不可机器复核，L4 结构性不可达。
* **F2**：预声明扫描是协议。13% 的交付率意味着半径/网格/界面/时间四条收敛轴没有数据，「结论可外推到生产参数」不成立；且 6 个替代用例改变了 (n, xi)，不能算「本地复跑声明子集」。
* **F3**：owner 标准「误差 >15% 不可接受」。生产默认力形 3 例全部 >15%（13.86% 亦属边界外），保守力形在 xi=1 亦 >15%。标题突出 +1.6%~+3.9% 而不标注其几何范围，属选择性呈现。

## 6. 审阅声明（独立性限制 / 未跑检查 / 夸大主张 / disposition）

**独立性限制**：审计方在独立工作树（m20915, main `2bfb5ee1a`）完成审阅，未参与 m20809 的实现；被审工件来自 m20809 分支检出，审计脚本只读不写被审工作树。审计方无法访问集群作业 `cj2978/cj2999` 的调度回执（仓内无工件），因此「两节点负载高、只回收部分网格」这一对缺例的解释**未获独立验证**，仅按交付方陈述记录。

**未跑的检查（不记为通过）**：
* 未重跑 15 例 ×1600 步全量扫描（~245 s/例单核，属重计算；且 artifacts 已逐例落盘、--aggregate 位级可复现）
* 未对集群作业 cj2978/cj2999 的原始回执做核对（仓库内无回执工件；交付方声明两节点负载高只回收部分网格）
* 未实测 REST 端点端到端（需以本分支重启 :8001 服务；仅做到 import + 路由存在 + 内部函数调用链审读）
* 未做 n=48 保守力形对照点（A_iso_mu_grad_phi）——交付方缺失，审计方不在本单代跑
* 未做外部数值对标（无外部基准，文献仅形式引用）

**被审方的夸大/不精确主张**：
* 报告标题/摘要「保守 Korteweg 力形达 +1.6%~+3.9%」：数字真实但取自 n=32/R=7 lite 子集；声明网格 n=48/R=12 上保守力形唯一交付点 C_n48 为 -5.46%，未在 n=48 做同网格对照即宣称「能通过 Young-Laplace」偏宽。
* 「网格阶梯误差跨度（max-min）= +0.00%」：单点网格上无意义，实际未做网格收敛性验证。
* 「其余子集在本机以同一代码路径复跑（L_* 用例）」：L_* 并非声明子集的本地复跑，而是 n=32/xi=1 的表外替代用例，措辞掩盖了协议偏离。
* docs 表 2 标题「半径阶梯：Young-Laplace 线性（保守力形，iso 算子）」：实际数据是 n=32、xi=1、R=5/6 的 lite 用例（且渲染为空表），与声明轴（n=48、xi=2、R=8..14）不同。
* 「已把这条要求写进参数文档与 pytest 门」的前提是分支工件；main 无此端点/参数，对外不可见。

**Disposition：approve with notes（L3）**。返修门（reopened 条件）：
* G1 注册 benchmark case 并回写 `tensorbench_results`（含容差与 passed 判定），使 L4 可判；
* G2 补齐声明轴：`A_iso_mu_grad_phi`（n=48, xi=2 保守力形对照）、`B_R8/10/12/14`、`C_n40/56`、`D_kappa*`、`E_steps4000`；
* G3 修复 `render_calibration_report.py:245` 的 group 名（`lite_radius_ladder`）与表 3 标题，去掉单点 span 断言；
* G4 修正 docs §5.4 的误差归因；合入 main。

## 7. 图

![case errors](artifacts/board_audit_m20915/fig_m20915_case_errors.png)

![parity & coverage](artifacts/board_audit_m20915/fig_m20915_parity_coverage.png)

## 8. 交付回执（审计单自身）

| 项 | 值 |
|---|---|
| verdict 落表 | `tensor_maturity_verdicts` id=869，repo=`TensorLBM`，scope=`m20809 XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D`，level=3，verdict_by=`board-audit`，mission_id=20915 |
| maturity_gate | 交单时 `gate_state=pending`（gate 为平台侧自动裁定；confirmed=消账 / reopened=返修，以 gate 复核为准） |
| 平台报告 | `tensor_reports` id=18958（mission_id=20915，2 图内嵌，status=published） |
| verdict SQL | `verdict_m20915.sql`（幂等：ON CONFLICT DO UPDATE，可重放） |
| 证据目录 | `artifacts/board_audit_m20915/`（重算脚本+JSON+日志+2 图+投稿脚本） |

复核入口（任何人可重跑）：
```
cd artifacts/board_audit_m20915
PYTHONPATH=/root/tp-agent/work/m20809/src python3 recompute_checks.py /root/tp-agent/work/m20809 recompute_checks.json
python3 make_figures.py && python3 render_audit_doc.py
```
