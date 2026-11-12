# m15894 交差核验审计 — m15470 CASE-MCMP-CONTACT-ANGLE（壁面接触角 Young 定律 / TensorLBM）

**审计对象**：mission 15470（TensorLBM，succeeded，PR #1825）→ commit `4256e939`（分支 `agent/m15470`）
**审计单**：m15894（任务榜交差核验单，非重做）
**裁定**：**L2（独立功能实现）**，`honest FAIL` 成立 —— 算例本身是可信的否证成果，但被检验的能力（Young 关系当接触角设定器）**未通过**。
**落表**：`tensor_maturity_verdicts` id=383，`repo='TensorLBM'`, `scope='m15470 CASE-MCMP-CONTACT-ANGLE'`, `level=2`, `verdict_by='board-audit'`
**审计证据包**：`audit/audit_m15894/evidence.json`（由 `audit/verify_m15470.py` 程序化生成）+ 复算工件 `audit/audit_m15894/audit_spot_a_coarse_70.json` + 图 `audit/audit_m15894/fig_m15894_contact_angle_audit.png`

---

## ① 案例介绍（被审计对象 + 物理背景）

**物理**：静止座滴在固体壁面上的平衡接触角由 Young 方程给出

$$\gamma_{lv}\cos\theta=\gamma_{sv}-\gamma_{sl}\qquad(\text{Young, 1805})$$

Shan–Chen 单组分伪势（SCMP）模型用壁面伪势 $\psi_w$ 注入润湿，库内把"给一个目标角"实现成线性映射
`cos θ = (ψ_w − ψ_v)/(ψ_l − ψ_v)`（`TensorLBM/src/tensorlbm/wetting/__init__.py:164-184`）。
m15470 检验的假设 = **这条映射能不能当"接触角设定器"用**（预注册判据 $|\theta_{\rm meas}-\theta_{\rm Young}|<10°$）。

**几何/参数（取自算例 artifact `contact_angle_sweep.json` 的 meta，非手抄）**

| 项 | 值 |
|---|---|
| 网格 coarse / fine | 96×64，壁 2 行，R=13，8000 步 / 144×96，R=20，12000 步 |
| 碰撞 / 耦合 | BGK $\tau=1.0$；SC $G=5.0$；$\psi(\rho)=1-e^{-\rho}$ |
| 共存密度 | $\rho_l=1.957$, $\rho_v=0.1596$；$\psi_l=0.858718$, $\psi_v=0.147516$ |
| 润湿旋钮 | $f_w=\psi_w/\psi_l\in[0.30,0.80]$ |
| 无量纲 | $Bo=0$（$g=0$，Young 定律应严格成立）、$Ca=0$、$Ma=0$ |

## ② 怎么做的（审计方法：核验而非重做）

| 步骤 | 做法 | 判据 |
|---|---|---|
| S1 取证 | `git ls-tree` + `find` 清点 `artifacts/CASE-MCMP-CONTACT-ANGLE`（41 文件），解析 sweep，记录 sha256 | 工件**真存在**且含现场量（trace/mass drift/interface points） |
| S2 基准裁判 | `psql -U postgres -c "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-MCMP-CONTACT-ANGLE'"` | 该表 25 行中**本锚 0 行** |
| S3 数值复算 | 从 sweep JSON **重算** $\theta_{\rm meas}-\theta_{\rm Young}$、pass 计数（5°/10° 双阈值）、B2 hold-out | 与报告表逐位对齐 |
| S4 可复现 | 直接执行 `run_case_cli.py --theta 70 --grid coarse --law young`（审计侧独立复跑，48 s） | 返回码 0 且数字逐位一致 |
| S5 测试核验 | 复跑 mission 声称的 pytest 子集 | 8 passed（与声称一致） |
| S6 持久性 | `git ls-tree origin/main -- artifacts/...`、`git show --stat` PR merge 提交 | 数字能否从 trunk 复核 |
| S7 平台裁定 | `GET /tensor/bench/l4-verdict/15470` | 机器裁定状态 |

工具：`psql`（只读）、`git`、`pytest`、平台 API；审计脚本与全部原始输出留在 `audit/`。

## ③ 计算结果（程序化复算，零手抄）

**stage A（ref = Young 设定角 vs got = 实测角）**

| 网格 | $f_w$ | ref $\theta_{\rm Young}$ | got $\theta_{\rm meas}$ | Δ | <5°? | <10°? |
|---|---|---|---|---|---|---|
| coarse | 0.4551 | 70.0 | 111.39459 | **+41.39** | ✗ | ✗ |
| coarse | 0.1718 | 90.0 | 179.99994 | **+90.00** | ✗ | ✗ |
| coarse | — | 110.0 | OUT_OF_RANGE（需 $\psi_w<0$，上限 101.97°） | n/a | ✗ | ✗ |
| fine | 0.4551 | 70.0 | 117.70241 | **+47.70** | ✗ | ✗ |
| fine | 0.1718 | 90.0 | 179.99994 | **+90.00** | ✗ | ✗ |
| fine | — | 110.0 | OUT_OF_RANGE | n/a | ✗ | ✗ |

pass 计数：**0/6（5° 代码阈值）**，**0/6（10° 声明阈值）**；最大 |Δ|=89.9999°。

**stage B（粗网格标定 → 细网格 hold-out）**：70°→89.12（Δ+19.12）、90°→106.01（Δ+16.01）、110°→121.44（Δ+11.44）→ **0/3**。

**误差分析（算例内已有，审计复算一致）**：网格漂移 6.31°（$f_w=0.4551$）/25.0°（强润湿端）；同 $f_w$ 亚稳裂 11.11°（已超 10° 容差）；解析圆帽估计器控制偏差 ≤1.106°（2 网格×3 角）→ 40–90° 漂移是**模型误差**而非测量误差。

**审计侧独立复跑**：`theta_measured = 111.39459003071705`，与 mission 记录 **abs diff = 0.0°**（逐位一致）。
**pytest**：`8 passed, 6 deselected in 6.22s`（与 mission 声称一致）。

## ④ 结果图

![audit figure](audit/audit_m15894/fig_m15894_contact_angle_audit.png)

左：ref（Young 设定角，灰）vs got（实测，蓝/红），标注 Δ；110° 两例标 OUT_OF_RANGE。
右：实测壁面律 $\theta(f_w)$（B1 粗网格扫描）与库内线性 cos 假设对比 —— 拟合斜率 5.03 vs 库 1.21（**4.2 倍**），中性壁 $f_w^*=0.512$ vs 库 0.172。图内数字全部由 `contact_angle_sweep.json` 现算。

## ⑤ 注意点（审计发现，按严重度）

| # | 发现 | 严重度 | 证据 |
|---|---|---|---|
| F4 | **持久性缺口**：PR #1825 合并提交 `ee578b8f` 只带 2 个文件（`wetting/__init__.py` +11 行、`docs/CASE-MCMP-CONTACT-ANGLE.md` +153 行）；41 个 run artifact **不在 main**（`git ls-tree origin/main -- artifacts/CASE-MCMP-CONTACT-ANGLE` → 0 文件；main 树内 `contact_angle_sweep.json` 0 命中）。数字只在 `agent/m15470` 分支/工作树可复核 | **阻断 L4** | `git show --stat ee578b8f`；`git ls-tree -r origin/main` |
| F5 | **无机器裁判**：`tensor_benchmark_cases` 本锚 0 行（最近行 `GAP-LBM-MCMP-POISEUILLE` 度量是分层 Poiseuille 界面/流量/spurious current，不能裁接触角）；`tensorbench_results` 本单 0 行 → 平台 `l4-verdict = CLAIMED_NOT_PROVEN` | **阻断 L4** | S2/S7 查询输出 |
| F3 | **判据口径不一致（非阻断）**：文档/commit 写 10°，代码 `meta.accept_deg=5.0`、`tol_deg=5.0` → 实跑**更严 2 倍**；两阈值下结论相同（0/6、0/3） | 中 | sweep meta、逐例 `tol_deg` |
| — | **ref 自指**：ref $\theta_{\rm Young}$ 由被检验的同一反演函数生成；外部的只有 Young 方程**形式**，文献页级核验 mission 自报 WITHHELD（离线） | 中（证据等级） | 文档 §① 表 + 审计复核 |
| F6 | **无集成调用方**：harness 仅存于 `tests/benchmarks/wetting_contact_angle/`，无主求解器/平台调用；库侧改动仅 11 行 docstring 警告 | 阻断 L3 | grep；`git show 4256e939` |
| F7 | 测量链本身可信（控制实验 ≤1.106°）→ 这是**可信的否证**，不是坏工装 | 正面 | `estimator_control` 6 点 |

**适用范围**：结论对"把 `wall_psi_from_contact_angle()` 当角度设定器"成立；对同分辨率同初始帽的**标定式**处方（`invert_calibration`）不适用（B2 已否证跨网格可迁移）。$f_w\ge0.70$（粗）/$\ge0.80$（细）出现 `NO_INTERFACE`，圆帽反演失效。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 量 | 基准值 ref | 实测 got | 相对误差 | <3%？ | <10° 判据？ |
|---|---|---|---|---|---|
| θ, coarse, 70° | 70.0° | 111.39° | **+59.1%** | ✗ | ✗（+41.39°） |
| θ, coarse, 90° | 90.0° | 180.00° | **+100%** | ✗ | ✗（+90.00°） |
| θ, coarse, 110° | 110.0° | 不可表达 | — | ✗ | ✗ |
| θ, fine, 70° | 70.0° | 117.70° | **+68.1%** | ✗ | ✗（+47.70°） |
| θ, fine, 90° | 90.0° | 180.00° | **+100%** | ✗ | ✗（+90.00°） |
| θ, fine, 110° | 110.0° | 不可表达 | — | ✗ | ✗ |
| θ, B2 hold-out 70/90/110° | 70/90/110° | 89.12/106.01/121.44° | +27.3/+17.8/+10.4% | ✗ | ✗（+19.12/+16.01/+11.44°） |

**是否 <3%：否。** 全部关键量误差 10–100%，远超 3% 线，也超预注册 10° 容差；**honest FAIL 成立，不粉饰**。
不可比的量：$\sigma$、共存密度等物性参数本单未与外部数值对比（无基准 → 不作误差声明）。

---

## 裁定与机器裁定（判级附件）

| 级 | 判定 | 依据 |
|---|---|---|
| L0 文档 | 不足 | 有真实多相 LBM 计算，不是文档 |
| L1 数学/单元 | 达到但不止 | 解析圆帽控制 ≤1.106° 是单元级验证 |
| **L2 独立功能** | **✅ 达到（裁定级）** | 34 个算例记录、输入→计算→输出真实、逐位可复现、估计器自证 |
| L3 系统集成 | ❌ 否证 | 无主求解器/平台调用方；库改动仅 docstring |
| L4 基准验证 | ❌ 否证 | 0/6 与 0/3 失败 → 无能力被验证；ref 自指；0 裁判行；工件不在 main |
| L5 科研工程 | ❌ 否证 | 无独立外部锚（WITHHELD） |

**审计独立性声明/局限**：在同一台 loongarch64 机器、同一工作树上复跑（无跨架构复核）；`artifacts` 的簇作业来源未核（`tensorbench_results` 无行，无法绑定 cluster_job）；外网不可用 → 文献页级核验未做；`/tensor/bench/cases` 类 REST 端点不存在（404），裁判表经 `psql` 只读查询。
**未跑项**：全量 34 例重跑（只复跑 1 例，其余依赖逐位一致的确定性与 sweep 内 trace 自洽）、`repro.sh` 全链（约 6–8 min，未跑；只跑单例入口）。

## 处置（disposition）

**approve with notes**：m15470 的 honest FAIL 结论**采信**（数值独立复算 + 逐位复现一致），判级 **L2 confirmed**；
note-1（阻断 L4，非阻断消账）：若要把本算例升级为 L4 基准，须 ① 把 41 个 run artifact 提交进 main；
② 在 `tensor_benchmark_cases` 注册 `CASE-MCMP-CONTACT-ANGLE`（metric=接触角°，tolerance=10°）；
③ 写入 `tensorbench_results`（让平台可复核）；④ 引入**外部锚**（文献/实验 $\theta(f_w)$ 点）替代自指 ref。
note-2：统一判据口径（文档 10° vs 代码 5°）。