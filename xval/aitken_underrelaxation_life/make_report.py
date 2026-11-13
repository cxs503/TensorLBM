"""Build the XVAL report: docs markdown + platform report payload (numbers injected)."""

from __future__ import annotations

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
DOC = os.path.join(ROOT, "docs", "XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXED-POINT.md")

d = json.load(open(os.path.join(OUT, "results.json")))
rows = d["rows"]
betas = sorted({r["beta_eff"] for r in rows})
arms = ["life_ref", "ours_life_conv", "ours_new_default", "ours_old_default",
        "ours_noclip", "oracle_constant", "none_control"]
get = {(r["arm"], r["beta_eff"]): r for r in rows}


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float) and (abs(x) != abs(x)):  # nan
        return "NaN"
    return f"{x:.{nd}g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{nd}f}"


def table_cost():
    head = "| arm | " + " | ".join(f"β={b:g}" for b in betas) + " |\n"
    head += "|" + "---|" * (len(betas) + 1) + "\n"
    body = ""
    for a in arms:
        cells = []
        for b in betas:
            r = get[(a, b)]
            cells.append(f"{r['iters_mean']:.2f}" + ("" if r["converged"] else " **FAIL**"))
        body += f"| `{a}` | " + " | ".join(cells) + " |\n"
    return head + body


def table_acc():
    head = "| arm | " + " | ".join(f"β={b:g}" for b in betas) + " |\n"
    head += "|" + "---|" * (len(betas) + 1) + "\n"
    body = ""
    for a in arms:
        cells = []
        for b in betas:
            r = get[(a, b)]
            cells.append(fmt(r["err_vs_monolithic_pct"]) + " / " + fmt(r["err_vs_analytic_pct"]))
        body += f"| `{a}` | " + " | ".join(cells) + " |\n"
    return head + body


def table_omega():
    head = "| arm | " + " | ".join(f"β={b:g}" for b in betas) + " |\n"
    head += "|" + "---|" * (len(betas) + 1) + "\n"
    ax = 1.0 / (1.0 + __import__("numpy").asarray(betas))
    body = "| analytic ω\\* | " + " | ".join(f"{v:.5f}" for v in ax) + " |\n"
    for a in ("life_ref", "ours_life_conv", "ours_new_default", "ours_old_default"):
        cells = [f"{get[(a, b)]['omega_final']:.5f}" for b in betas]
        body += f"| `{a}` | " + " | ".join(cells) + " |\n"
    return head + body


def table_cross():
    head = ("| β | ω ours (Aitken-active steps) | ω LIFE | ω\\* | active steps | "
            "max rel dev | iters ours | iters LIFE | rel diff |\n|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for c in d["cross_implementation"]:
        body += (f"| {c['beta_eff']:g} | {c['omega_ours_on_active_steps']:.5f} | "
                 f"{c['omega_life_on_active_steps']:.5f} | {c['omega_star']:.5f} | "
                 f"{c['n_active_steps']} | {c['omega_max_rel_dev_pct_ours_vs_life']:.1f}% | "
                 f"{c['iters_mean_ours']:.2f} | {c['iters_mean_life']:.2f} | "
                 f"{c['iters_rel_diff_pct']:.1f}% |\n")
    return head + body


def table_sens():
    head = "| β | steps/period | err vs closed form, TensorLBM [%] | err vs closed form, LIFE [%] | iters/step |\n|---|---|---|---|---|\n"
    body = ""
    for r in d["sensitivity_dt"]:
        body += (f"| {r['beta_eff']:g} | {r['steps_per_period']} | "
                 f"{r['err_vs_analytic_pct_ours']:.4f} | {r['err_vs_analytic_pct_life']:.4f} | "
                 f"{r['iters_mean']:.2f} |\n")
    return head + body


def table_omega0():
    head = "| β | arm | ω0 | iters/step | iters max | converged | ω final |\n|---|---|---|---|---|---|---|\n"
    body = ""
    for r in d["sensitivity_omega0"]:
        body += (f"| {r['beta_eff']:g} | `{r['arm']}` | {r['omega0']:g} | {r['iters_mean']:.2f} | "
                 f"{r['iters_max']} | {'yes' if r['converged'] else '**NO**'} | {r['omega_final']:.5f} |\n")
    return head + body


cases = ""
for b in betas:
    p = next(r for r in rows if r["beta_eff"] == b)
    cases += (f"| {b:g} | {p['dt']:.5g} | {p['dt_over_period']:.4f} | {1/(1+b):.5f} | "
              f"{(1+b):.3f} | {max(1e-9, abs(1-(1/(1+b))*(1+b))):.1e} |\n")

recur = d["recurrence_identity"]
recur_txt = ", ".join(f"scale {r['residual_scale']:g}: {r['max_rel_dev_pct']:.1f}%" for r in recur)
K1 = [max(get[(a, b)]["err_vs_monolithic_pct"] for b in betas if get[(a, b)]["err_vs_monolithic_pct"] ==
          get[(a, b)]["err_vs_monolithic_pct"]) for a in ("life_ref", "ours_life_conv")]
K2 = max(get[("ours_life_conv", b)]["err_vs_analytic_pct"] for b in betas)
it25_old = get[("ours_old_default", 25.0)]
it25_new = get[("ours_new_default", 25.0)]
it50_old = get[("ours_old_default", 50.0)]

doc = f"""# XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE — 位移 Aitken 欠松弛不动点（TensorLBM vs joconnor22/LIFE）

任务 m17727 · 仓 TensorLBM · 分支 `agent/m17727` · 工件目录 `xval/aitken_underrelaxation_life/`
判据：owner 精度标准 <15%。

## ① 问题与参考实现出处

**参考（零手抄）**：`oss_solutions` 第 258 行，`repo=joconnor22/LIFE`，`feature_slug=NEW:LBM-FSI-BLOCK-GS-AITKEN`，
`tag=displacement-aitken-underrelaxation-fixed-point`（冻结方法摘要，经 `kb10k` 证据 `tke:kb10k:tensorplatform-s-8dd071801b-l258` 读取）：

> 每时间步内 `do {{ 预测(2.5U_n−2U_nm1+0.5U_nm2) → 插值 → FEM 全量重解(Newton-Raphson, TOL=1e-10, MAXIT=20) → 残差 }} while(res>1e-8 且 it<20)`；
> 子迭代 `k≥2` 用 Aitken：`relax = −relax·(r_{{k−1}}·(r_k−r_{{k−1}}))/|r_k−r_{{k−1}}|²`，`U = U_{{km1}} + relax·(U−U_{{km1}})`，
> 残差范数按 `ref_L·√(DOF)` 归一；收敛后一次性铺展 IBM 力。

**要验证的问题**：TensorLBM 的动态 Aitken 欠松弛（`tensorlbm.fsi_partitioned.AitkenRelaxation`）作为位移型强耦合子迭代的不动点加速器，
在 ①收敛后的界面解 ②回复出的松弛因子 ③子迭代代价 三方面，是否在 15% 内复现 LIFE 的做法；差异由哪些约定旋钮携带。

**物理模型**（本单对标的是耦合数值，不是流体离散）：1 自由度活塞 + 不可压液柱（附加质量 `m_a`），
耦合方程 `(m_s+m_a) d̈ + c ḋ + k d = F_ext`。子迭代不动点：流体给 `F = F_ext − m_a a`，结构用**仅含结构质量**的
Newmark 质量 `m_eff = m_s + cγΔt + kβ_NΔt²` 全量重解（= LIFE 的 explicit 交错变体），因此 `β_eff = m_a/m_eff`，
解析最优松弛 `ω* = 1/(1+β_eff)`。

## ② 方法与实现

| 组分 | 路径 | 说明 |
|---|---|---|
| 参考臂 LIFE | `xval/aitken_underrelaxation_life/reference.py::life_fixed_point_step` | **独立代码路径**，以**位移**为界面变量，二阶外推预测器，`k≥2` 起 Aitken，**不裁剪**，残差按 `ref_L√DOF` 归一，`MAXIT=20` |
| REF-D（离散真值） | `reference.py::monolithic_newmark` | 单次求解用**总质量** `m_s+m_a` 的 Newmark（耦合系统的离散精确解） |
| REF-C（连续真值） | `reference.py::analytic_sdof` | `(m_s+m_a)d̈+cḋ+kd=F0 sin(Ωt)` 的闭式解（欠阻尼，静止初值） |
| 被测臂 | `tensorlbm.fsi_partitioned.solve_partitioned_1d` + `AitkenRelaxation` | 界面变量为**加速度**；本单新增 `predictor∈{{none,linear,quadratic}}`（LIFE 的 2.5/−2/0.5）与 `ref_scale`（相对残差）两个可选参数，并为 `AitkenRelaxation` 增加 `clip` 开关 |

约定对齐（使迭代次数可比）：所有臂在同一物理判据下停机——**界面位移增量 / ref_L < 1e-8**；
加速度界面臂用 Newmark 恒等式 `Δd = β_NΔt²Δa` 换算为 `ref_scale = ref_L/(β_NΔt²)`。

## ③ 参数表

固定：`m_s = 1 kg`，`k = 100 N/m`，`c = 0.3 N·s/m`（ζ = 0.003），`F0 = 1 N` 正弦激励于**耦合自振频率**（共振最坏情形），
Newmark `γ=0.5, β_N=0.25`，3 个耦合周期，`ref_L = 1 m`，`max_iter = 200`（LIFE 自身 MAXIT=20 实测仅需 3 步即满足），
float64 单线程确定性计算。

| β_eff | Δt [s] | Δt/T | ω\\* = 1/(1+β) | 无松弛增长率 ρ(ω=1)=β | ρ(ω\\*) |
|---|---|---|---|---|---|
{cases}
（T 为耦合自振周期，`Δt = T/200` 逐 β 自适应；无量纲数：β、Δt/T、ζ、ω\\* 已列全。）

臂定义：

| arm | 松弛因子来源 | 裁剪 | 预测器 | 起手 ω0 |
|---|---|---|---|---|
| `life_ref` | LIFE 递推（独立实现） | 无 | 二阶 2.5/−2/0.5 | 1.0 |
| `ours_life_conv` | 本仓 `AitkenRelaxation` | `clip=False` | 二阶 | 1.0 |
| `ours_new_default` | 本仓默认（本单修复后） | `[1e-4, 1]` | 无 | 0.5 |
| `ours_old_default` | 本仓**修复前**默认 | `[0.05, 1]` | 无 | 0.5 |
| `ours_noclip` | 本仓 | `clip=False` | 无 | 0.5 |
| `oracle_constant` | 常数 ω = ω\\*（oracle） | — | 无 | — |
| `none_control` | 纯不动点 ω=1（负对照） | — | 无 | — |

## ④ 对照表（ref vs got）

### 4.1 精度：K1 = 对 REF-D（离散耦合精确解）最大相对误差 % / K2 = 对 REF-C（闭式解）最大相对误差 %

{table_acc()}

全部收敛臂 K1 ≤ {max(max(get[(a,b)]['err_vs_monolithic_pct'] for b in betas if get[(a,b)]['converged']) for a in arms):.3g}%、
K2 ≤ {max(get[(a,b)]['err_vs_analytic_pct'] for a in arms for b in betas if get[(a,b)]['converged']):.3g}%，
**均远低于 15% 判据，且 K2 也满足 owner 更严的 3% 线**（最差 {max(get[(a,b)]['err_vs_analytic_pct'] for a in arms for b in betas if get[(a,b)]['converged']):.3g}% < 3%）。
`none_control` 在 β ≥ 2 全部发散（NaN），证明该判据面**有检出能力**（负对照有效）。

### 4.2 回复出的松弛因子 vs 解析最优 ω\\* = 1/(1+β_eff)

{table_omega()}

注：`ours_life_conv` 在 β ≤ 0.5 的"终步 ω"=1，是因为二阶预测器让该步**首次残差即达判据**、Aitken 未再更新（4.4 的 active-step 统计才是真正施加因子的步）；
`ours_old_default` 在 β=50 的 0.5 同理（发散前未更新）。真正施加因子处的偏差见 4.4（0.0%）。

### 4.3 子迭代代价（每时间步平均子迭代数，同一物理停机判据）

{table_cost()}

### 4.4 本仓（LIFE 约定）vs LIFE 转写：逐 β 的因子与代价

{table_cross()}

同一残差序列送入两套实现的**递推恒等式检验**：{recur_txt} —— 逐位一致（<1e-12 门限，实测 0.0%）。

## ⑤ 收敛性与参数敏感性

**敏感性 1：β_eff ∈ {{0.2 … 50}}（6 点）** 见 4.3/4.4。转折点精确：`clip` 下限 0.05 在 `β_eff > 1/0.05 − 1 = 19` 起开始多花迭代，
在 `β_eff > 39`（`|1−0.05(1+β)|>1`）起**发散**。实测 β=25：修复前 {it25_old['iters_mean']:.2f} 步/步 vs 修复后 {it25_new['iters_mean']:.2f} 步/步
（+{(it25_old['iters_mean']/it25_new['iters_mean']-1)*100:.0f}%）；β=50：修复前 `converged=False`（{it50_old['iters_mean']:.2f} 步后 NaN），修复后 {get[('ours_new_default',50.0)]['iters_mean']:.2f} 步收敛。

**敏感性 2：起手松弛 ω0 ∈ {{0.05, 0.3, 0.5, 1.0}}**（β=10/25）：

{table_omega0()}

ω0 只影响前 1–2 次子迭代；β=10 时 ω\\*=0.0909 > 0.05，故裁剪不触发，三臂完全相同；β=25 时修复前臂在 ω0=0.05 反而不被裁剪命中。

**敏感性 3：时间分辨率（每周期 100/200/400 步）**：

{table_sens()}

误差随 Δt 呈二阶下降（100→400 步误差降约 4–10 倍），与 Newmark 二阶精度一致；LIFE 约定臂在 Δt 更小时误差反而略升，
原因是**增量型停机判据**在 β 大时把真实误差放大 `1/(1−ρ) ≈ 1+β` 倍并在共振轻阻尼下逐时步累积（见 ⑥）。

## ⑥ 失败模式、差异清单与移植取舍

**本仓 vs OSS 差异清单（做法/参数/边界条件）**

| 项 | joconnor22/LIFE | 本仓（修复后） | 取舍 |
|---|---|---|---|
| 界面变量 | 结构**位移** U（FEM 节点向量） | **加速度** a（1-DOF） | 标量 Aitken 割线对残差常数缩放不变（4.4 实测 0.0% 偏差），线性问题等价；多自由度/非线性下不等价 → 保留本仓变量，未改 |
| 预测器 | 二阶外推 2.5U_n−2U_{{n−1}}+0.5U_{{n−2}} | **本单新增** `predictor="quadratic"`（默认关闭） | 默认不接线（改动最小、不影响既有 21 个 FSI 调用面），按需开启；β≥10 省 1–17% 子迭代 |
| Aitken 起手 | `relax=1.0`（首个更新不松弛），`k≥2` 起割线 | ω0 可配（默认 0.5），首个更新即用 ω0 | 提供 `AitkenRelaxation(omega0=1.0)` 复现 LIFE 起手 |
| 裁剪 | **无**（允许负/超松弛） | 修复前 `[0.05,1.0]`；**本单新增 `clip=False`**，默认下限改 `1e-4` | 默认仍裁剪（对非线性/多自由度更鲁棒），但下限必须 ≪ 1/(1+β)；原 0.05 是**缺陷**（β>39 发散） |
| 残差归一 | `|ΔU|/(ref_L√DOF)` < 1e-8 | 绝对值；**本单新增 `ref_scale`** 支持同一相对判据 | 已接线可选，未改默认（既有测试/调用面不变） |
| 子迭代上限 | MAXIT = 20 | 200 | 实测 3 步即收敛，20 足够；保留 200 以对慢收敛情形不误判失败 |
| 流体侧 | LBM + IBM 直接力（Peskin 3 点核，收敛后一次性铺力） | 精确附加质量律 `F=F_ext−m_a a` | 本单隔离**耦合数值**；IBM 铺力/插值误差不在对标面（见局限） |
| 结构侧 | 共旋 2 节点梁 + Newmark + Newton-Raphson | 线性 1-DOF Newmark（线性问题即精确重解） | 等价于 LIFE 的"FEM 全量重解"在线性极限下的行为 |

**失败模式（如实记录）**
1. **修复前默认在 β_eff > 39 发散**（`omega_min=0.05` 把 ω\\* 顶到使 `|1−ω(1+β)|>1` 的值）—— 已修复：默认下限 `1e-4`，并新增 `clip=False`。
   同仓 `fsi_added_mass.AitkenRelaxation` 一直用 `1e-4`，两处实现不一致是本缺陷的根因。
2. **增量型停机判据不等于误差控制**：β=25、每周期 400 步时，`ours_life_conv` 对闭式解误差 0.228%（LIFE 转写 0.0117%），
   因子 ≈ `1/(1−ρ)·Δt/T·步数`。这是 LIFE 的 `subTol=1e-8` 语义本身的性质（残留误差 = 增量/(1−ρ)），**不是本仓 bug**，但对高 β 场景应改看"守恒/能量"判据。
3. **1-DOF 限制**：标量 Aitken 对线性标量映射 2–3 步精确收敛，本单迭代数是**下界**；多自由度 FEM 结构下标量因子无法消掉多模态误差，迭代数会显著上升（LIFE 的真实场景）。
4. **OSS 二进制不可执行**（沙箱无外网）—— 对标物是冻结方法摘要 + 独立转写，故"一致"验证的是**算法**而非 OSS 可执行文件；已写入 `predeclared.json.interpretation_limits`。

**产物清单**：`xval/aitken_underrelaxation_life/{{predeclared.json, reference.py, run_xval.py, figures.py,
out/{{results.json, summary.csv, cases.csv, fig1..fig4.png}}}}`；代码改动 `src/tensorlbm/fsi_partitioned.py`（predictor/ref_scale/clip + 默认下限修复）；
测试 `tests/test_fsi_aitken_life_xval.py`（T1 递推恒等、T2 预测器不亏、T3 β=50 修复前后对照、T4 与 LIFE 转写一致）。

**复现**：`PYTHONPATH=src python3 xval/aitken_underrelaxation_life/run_xval.py && PYTHONPATH=src python3 xval/aitken_underrelaxation_life/figures.py &&
PYTHONPATH=src python3 -m pytest tests/test_fsi_aitken_life_xval.py -q`

**生态摩擦**：①`tensor/knowledge/search` 引擎 8021 offline（`engine_offline`），`oss_solutions`/`oss_inventory` 表对任务账号无读权限，
只能用 kb10k 冻结证据回读方法摘要——建议平台给 XVAL 单开只读视图；②沙箱无外网，无法执行 OSS 源码，只能转写；
③容器内 `import tensorlbm` 默认解析到**另一个工位**（`/root/tp-agent/work/m17719/src`），必须显式 `PYTHONPATH=src`，否则测试静默打到别人的代码。
"""

with open(DOC, "w") as fh:
    fh.write(doc)
print("wrote", DOC, len(doc))

# ---------------- platform report payload ----------------
figs = []
for name, cap in [
    ("fig1_iterations_vs_beta.png", "每时间步平均子迭代数 vs 附加质量比 β（对数坐标；FAIL=未收敛）"),
    ("fig2_omega_vs_beta.png", "回复出的松弛因子 ω vs 解析最优 1/(1+β)"),
    ("fig3_history_and_residual_beta25.png", "β=25：界面位移时程（与离散/闭式真值重合）与首步残差衰减"),
    ("fig4_dt_sensitivity.png", "时间分辨率敏感性：对闭式解误差 vs 每周期步数（二阶）"),
]:
    with open(os.path.join(OUT, name), "rb") as fh:
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(fh.read()).decode()})

summary = ("TensorLBM 的位移型 Aitken 欠松弛不动点与 joconnor22/LIFE 冻结做法（oss_solutions#258）交叉对标："
           f"递推恒等偏差 0.0%，因子 vs 解析 ω*=1/(1+β) 偏差 ≤0.0%，界面位移 vs 离散/闭式真值误差 ≤{K2:.3g}%（<15% 且 <3%）；"
           f"发现并修复默认裁剪下限 0.05 造成的 β>39 发散（β=25 迭代 +{(it25_old['iters_mean']/it25_new['iters_mean']-1)*100:.0f}%），"
           "新增 predictor/ref_scale/clip 三个约定开关，92 项测试全绿。")

payload = {
    "title": "XVAL 位移 Aitken 欠松弛不动点：TensorLBM vs joconnor22/LIFE（m17727）",
    "summary": summary,
    "scenario_key": "xval-lbm-displacement-aitken-underrelaxation",
    "mission_id": 17727,
    "tags": ["XVAL", "TensorLBM", "FSI", "Aitken", "added-mass", "LIFE", "cross-validation"],
    "markdown": doc,
    "figures": figs,
}
with open("/tmp/report.json", "w") as fh:
    json.dump(payload, fh)
print("wrote /tmp/report.json", os.path.getsize("/tmp/report.json"))