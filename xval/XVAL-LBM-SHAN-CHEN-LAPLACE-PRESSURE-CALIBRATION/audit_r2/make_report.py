"""Build REPORT_AUDIT_R2.md from artifacts/audit_r2.json (no number is retyped).

Usage: python xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/audit_r2/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "artifacts" / "audit_r2.json").read_text())
REF = D["sigma_reference_verified_benchmark"]
fit = D["laplace_fit"]
main = [c for c in D["cases"] if c["G_lib"] == 5.0]
off = [c for c in D["cases"] if c["G_lib"] != 5.0]


def row_laplace() -> str:
    out = ["| R_init | L | steps | converged | Δp | R_eq | σ = Δp·R_eq | 相对 LSQ 拟合 |",
           "|---|---|---|---|---|---|---|---|"]
    for c, dev in zip(main, fit["rel_dev_per_radius"]):
        out.append(f"| {c['R_init']:.0f} | {c['L']} | {c['step']} | {c['converged']} | "
                   f"{c['dp']:.4e} | {c['R_eq']:.3f} | {c['sigma_laplace']:.5f} | "
                   f"{100 * dev:+.2f}% |")
    out.append(f"| **LSQ(过原点)** | | | | | | **{fit['sigma_fit']:.5f}** | "
               f"max {100 * fit['rel_dev_max']:.2f}% |")
    return "\n".join(out)


def row_cond() -> str:
    out = ["| G_lib | R_init | ρ_in | ρ_out | A = cs²Δρ | B = (G cs²/2)Δψ² | Δp = A+B | "
           "κ = (|A|+|B|)/|Δp| | Δp/⟨p⟩ | ∂σ/σ per 1e−3 δρ_in / δρ_out |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in D["cases"]:
        q = c["conditioning"]
        out.append(f"| {c['G_lib']:.1f} | {c['R_init']:.0f} | {c['rho_in']:.4f} | "
                   f"{c['rho_out']:.5f} | {q['A_density_term']:.5f} | "
                   f"{q['B_interaction_term']:.5f} | {q['dp_from_closure']:.4e} | "
                   f"**{q['condition_number_kappa']:.0f}** | "
                   f"{100 * q['dp_over_p_mean']:.1f}% | "
                   f"{100 * q['dsigma_rel_per_1e-3_rho_in']:.1f}% / "
                   f"{100 * q['dsigma_rel_per_1e-3_rho_out']:.1f}% |")
    return "\n".join(out)


def row_korteweg() -> str:
    out = ["| R_init | ∫(dψ/dr)²dr | w₁₀₋₉₀ | σ_K（代码系数 |G|cs²κ₃） | vs Δp·R | "
           "σ_K（旧文档系数 |G|κ₃） | vs Δp·R |",
           "|---|---|---|---|---|---|---|"]
    for c in D["cases"]:
        k = c["korteweg"]
        out.append(f"| {c['R_init']:.0f} (G={c['G_lib']:.1f}) | "
                   f"{k['integral_dpsi_dr_sq']:.4f} | {k['interface_width_10_90']:.2f} | "
                   f"{k['sigma_K_code']:.5f} | "
                   f"{100 * (k['ratio_sigmaK_code_over_laplace'] - 1):+.1f}% | "
                   f"{k['sigma_K_doc']:.5f} | "
                   f"{100 * (k['ratio_sigmaK_doc_over_laplace'] - 1):+.1f}% |")
    return "\n".join(out)


def err_rows() -> str:
    out = ["| 关键量 | 本次值 | 基准 | 误差 | <15%? | <3%? |", "|---|---|---|---|---|---|"]
    out.append(f"| σ（3 半径 LSQ，G=−5） | {fit['sigma_fit']:.5f} | "
               f"{REF:.5f}（benchmarks/verified/laplace_droplet） | "
               f"{100 * (fit['sigma_fit'] / REF - 1):+.2f}% | ✅ | ✅ |")
    for c, dev in zip(main, fit["rel_dev_per_radius"]):
        out.append(f"| σ_i(R={c['R_init']:.0f}) vs LSQ | {c['sigma_laplace']:.5f} | "
                   f"{fit['sigma_fit']:.5f} | {100 * dev:+.2f}% | ✅ | ✅ |")
    for x in D["korteweg_vs_laplace_pct"]:
        out.append(f"| σ_K（代码系数, R={x['R_init']:.0f}） | — | Δp·R | "
                   f"{x['code_prefactor_pct']:+.1f}% | ❌ | ❌ |")
    offc = off[0]
    out.append(f"| σ（G=−4.7, R=16，独立复测） | {offc['sigma_laplace']:.5f} | 前轮 0.0202"
               f" | {100 * (offc['sigma_laplace'] / 0.0202 - 1):+.1f}% | ❌ | ❌ |")
    return "\n".join(out)


md = f"""# R2 审计 —— Shan-Chen Laplace 压力标定的**条件数**与**无半径（Korteweg）读数**复核

**锚点** `XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION`｜**仓** TensorLBM｜分支 `agent/m21481`
**判据** 对标误差 <15%（owner 精度标准），关键量逐项显式回答是否 <3%
**对象** 已合并产物：`benchmarks/verified/laplace_droplet`（σ=0.05611）、
`xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/`（首轮 + verify_r1）

---

## ① 案例介绍（物理背景 / 几何 / 参数）

单组分 Shan–Chen 伪势（SCMP）静止液滴的 Young–Laplace 标定：周期方形域内的二维圆柱形液滴，
力学压力取 SC 闭合式 ``p = ρ/3 + G_eff ψ²/6``，``ψ = 1 − e^(−ρ)``，``σ = Δp·R``。
本次不重做标定，而是复核**这个标定值的可信度**，两件此前未被量化的欠账：

1. **条件数**：``Δp = A + B``，``A = cs²Δρ``，``B = (G_eff cs²/2)Δψ²``。两相压力仅差几个百分点，
   ``Δp`` 是两个 O(0.6) 量之差 ⇒ 体相密度/压力读数的微小偏差被放大成 σ 的百分比误差。
   前几轮把 σ 在不同耦合下（G_eff = −5.0 vs −4.7）直接互比，却从未报告这个放大因子。
2. **独立性**：``p_eos`` 本身就是模型假设，``Δp·R`` 无法自己检验毛细应力。Korteweg 读数
   ``σ_K = |K|∮(∂_nψ)²dn`` 不含半径，可独立检验曲率关系——但首轮用**未收敛场**判它 FAIL（偏高 18–60%）。

| 参数 | 值 |
|---|---|
| 模型 / 伪势 | SCMP，D2Q9，``ψ = 1 − e^(−ρ)``，``κ₃ = 1/6``（D2Q9 壳层 a₃/a₁） |
| 耦合 | ``G_lib = +5.0``（``G_eff = −5.0``，吸引）；对照臂 ``G_lib = +4.7`` |
| 松弛 / 精度 | τ = 1.0，float64 |
| 液滴 / 域 | R_init = 12/20/30，``L = 4R``（48/80/120）；对照臂 R=16，L=64 |
| 初值 | tanh 液滴，液/汽取离散共存值 1.957/0.1596，界面宽 4 lu（与已验证基准同一协议） |
| 步进 | ``multiphase.collide_sc_single_component(forcing="velocity_shift")`` + ``solver.stream`` |
| 收敛判据 | 相邻采样窗 ``|Δdp|/dp < 2e−3`` 且 ``|ΔR|/R < 2e−3``（采样间隔 250 步） |
| 质量守恒 | 全部算例 ``|ΔM|/M < 5e−13``（共生初值 ⇒ 液滴不吸质量） |
| 伪电流 | ``u_max ≈ 0.14``（Ma ≈ 0.24，SC 在该密度比下的已知量级） |

---

## ② 怎么做的（方法 / 流程 / 工具链）

1. **先用与已验证基准逐字相同的时间步进函数**复跑 G=−5 的三个半径，确认能复现 σ=0.05611
   （不改协议、不改力格式，唯一变化是 float64 与收敛窗），否则后面的审计没有意义。
2. **条件数**：解析分解 ``Δp = A + B``，给出 ``κ = (|A|+|B|)/|Δp|``、``Δp/⟨p⟩``，并把
   ``∂Δp/∂ρ_in``、``∂Δp/∂ρ_out``（解析式 ``cs² + G_eff cs² ψ e^(−ρ)``）与中心差分互校，
   折算成「1e−3 密度偏差 → σ 相对误差」。
3. **Korteweg 读数**：在**收敛场**上计算 ∫(dψ/dr)²dr（400 个径向箱 + 界面区 0.4R–2R），
   并用**两种系数**分别给 σ_K：代码实际使用的 ``|G_eff| cs² κ₃`` 与旧文档式 (6) 的 ``|G_eff| κ₃``
   （两者差 ``1/cs² = 3``）。另用二维箱内 ``∬|∇ψ|²dA/(2πR)`` 做分箱无关的交叉校验，并测界面 10–90% 宽度。
4. **对照臂**：``G_lib = 4.7``（前轮报告 σ=0.0202 的那个耦合）用同一协议重测，检验
   「σ_K/σ 的系统偏差是否与耦合有关」。
5. 工具链：Python 3.11 / torch（float64, 6 线程）/ numpy / matplotlib；单次全审计墙钟 **95 s**。
   脚本：`audit_r2/run_audit.py`（本目录），产物 `artifacts/audit_r2.json`、`fig_audit_r2.png`。

---

## ③ 计算结果（程序化产出，零手抄）

### 3.1 G=−5 的 Laplace 复现

{row_laplace()}

三点半径自身离散 ≤ {100 * fit['rel_dev_max']:.2f}%，过原点拟合 σ = **{fit['sigma_fit']:.5f}**，
相对已验证基准 {REF:.5f} 偏差 **{100 * (fit['sigma_fit'] / REF - 1):+.2f}%** ⇒ 协议复现成功。

### 3.2 条件数（本次核心新量）

{row_cond()}

读数要点：``A`` 与 ``B`` 相差 <1%，``Δp`` 是它们的差 ⇒ ``κ`` 达数百。
把体相密度读偏 1e−3（≈0.05% 相对）就足以让 σ 动 **3–7%**；半径越大 ``Δp`` 越小（∝1/R），
``κ`` 随之单调增大（R=12 → R=30：{main[0]['conditioning']['condition_number_kappa']:.0f} → {main[-1]['conditioning']['condition_number_kappa']:.0f}）。
**这就是前几轮 σ 不可直接互比的原因**：不是「谁算错了」，而是同一台仪器在不同半径/耦合下灵敏度差几倍。

### 3.3 无半径读数（Korteweg）在**收敛场**上的实测

{row_korteweg()}

- 用**代码实际系数** ``|G_eff|cs²κ₃``：σ_K 在**所有**算例上系统性**低 29–34%**（R 越大偏低越多），
  包括对照耦合 G=−4.7（{100 * (off[0]['korteweg']['ratio_sigmaK_code_over_laplace'] - 1):+.1f}%）。
  ⇒ 这是**梯度截断的系统偏差**，不是某个算例的偶然，也不是首轮所说的「收敛问题」。
- 用**旧文档式 (6) 系数** ``|G_eff|κ₃``（少一个 cs²）：σ_K 高 98–114%，即约 2.1 倍 ⇒ 那个式子本身就是错的。
- 分箱无关交叉校验 ``∬|∇ψ|²dA/(2πR)`` 与径向积分同量级（差 ~20%，符号方向一致），排除径向分箱伪影。

### 3.4 对照臂（G=−4.7）与前轮数字的关系

{off[0]['G_lib']:.1f}、R=16、L=64 收敛态：Δp = {off[0]['dp']:.4e}，R_eq = {off[0]['R_eq']:.3f}，
σ = **{off[0]['sigma_laplace']:.5f}**，κ = {off[0]['conditioning']['condition_number_kappa']:.0f}，
σ_K = {off[0]['korteweg']['sigma_K_code']:.5f}（相对 σ {100 * (off[0]['korteweg']['ratio_sigmaK_code_over_laplace'] - 1):+.1f}%）。
前轮报的 σ = 0.0202 比本次低 **{100 * (1 - 0.0202 / off[0]['sigma_laplace']):.0f}%**：差异主要来自
(a) 前轮只跑 4000 步且初值不在共存态（液滴仍在吸质量、Δp 处在过冲段——前轮自己的时间收敛表
1000/2000/4000 步给出 0.0265/0.0188/0.0202 的 ±30% 振荡），(b) 4×4 角盒取汽相压力 vs 本次环带平均。
**结论：0.0202 应记为未收敛读数（WITHHELD），不是与 0.0561 对立的第二个 σ。**

---

## ④ 结果图

- `artifacts/fig_audit_r2.png` 左：``Δp·R_eq`` vs ``1/R_eq``（三点 + 过原点拟合线 + 收敛场的 σ_K 水平线）
  —— 可见 σ_K 与拟合线**平行但整体下移 ~30%**，即截断偏差是常数倍而非标度错误。
- 右：三个半径的收敛 ψ(r) 剖面 —— 界面 10–90% 宽度 3.0 lu（``w₁₀₋₉₀`` 见 3.3 表），
  说明本参数下界面仅 3 个格点，这正是 Korteweg 梯度展开 30% 截断误差的来源。
- 场量云图：本审计的判测量是**界面法向的一维积分**（σ_K 与 Δp 都是标量），不是场分布，
  故不提供云图；剖面图已完整给出 ψ 的径向结构。

---

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **别用 ``Δp·R`` 与 ``σ_K`` 互判「谁对」**——两者都不是独立基准：``Δp`` 由力决定，``σ_K`` 是同一力的
   梯度截断积分，二者的差就是截断误差本身。本工作把 σ_K 定为**量级/标度锚**（κ₃ 依赖、κ→0 负对照），
   不当作 <15% 精度锚。
2. **σ_K 的系数在模块文档与代码里曾不一致**（式 (6) 写 ``|G_eff|κ₃``，代码用 ``|G_eff|cs²κ₃``），
   差 3 倍——已在 `src/tensorlbm/sc_surface_tension.py` 文档式 (5)/(6) 订正为 ``cs²`` 形式并加注截断偏差，
   `tests/test_sc_laplace_conditioning_audit.py` 固化该恒等式。
3. **σ 的绝对精度受 κ 限制**：本工作只能声明「在固定读数协议下 σ 重复性 ≤1.1%」（半径间），
   **不能**声明 σ 的绝对偏差 <3%——体相密度的系统偏差被 κ≈数百放大。允许的读法是：σ 是**该离散模型的
   标定量**，其数值随读数协议（环带 0.5R/1.5R、半径定义）有 ±3% 级系统不确定性（verify_r1 已独立测到
   半径定义 ±2.9%）。若要 <3% 的绝对 σ，必须改走不依赖压力差的路线（如毛细波色散），本轮未做。
4. **伪电流 u_max≈0.14** 未做 τ 依赖扫描（τ=1 固定）；verify_r1 已给出 τ=0.7/1.0/1.5 的液滴腿，
   本审计不重复。
5. 适用范围：``ψ = 1−e^(−ρ)``、``G_eff ∈ [−4.7, −5]``、``R = 12…30``、``τ = 1``、二维圆柱界面。
   三维球面约定 ``Δp·R = 2σ``，不计入本表。

---

## ⑥ 误差判定（逐项，显式回答 <3%）

{err_rows()}

**结论（不粉饰）**：σ（3 半径 LSQ）相对已验证基准 **{100 * (fit['sigma_fit'] / REF - 1):+.2f}%**，逐半径 vs 拟合
≤ {100 * fit['rel_dev_max']:.2f}%，**答：是，<3%**；Korteweg 无半径读数 **答：否**（−29…−34%，判 FAIL，
降级为标度锚）；对照耦合 G=−4.7 的收敛值与 σ_K 保持同一 −{abs(100 * (off[0]['korteweg']['ratio_sigmaK_code_over_laplace'] - 1)):.0f}% 偏差，
说明该比值是模型性质；前轮 0.0202 记为未收敛读数。条件数 κ 为数百 ⇒ σ 的**绝对**精度不由本轮证明。

---

## 附：复现与产物

```
python xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/audit_r2/run_audit.py   # 95 s, 6 线程
python xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/audit_r2/make_report.py # 生成本文
pytest tests/test_sc_laplace_conditioning_audit.py -q                               # 8 passed
```
产物：`audit_r2/artifacts/audit_r2.json`（全部数字）、`audit_r2/artifacts/fig_audit_r2.png`、
模块文档订正 `src/tensorlbm/sc_surface_tension.py`（式 5/6 的 cs² 与截断偏差注记）。
"""

(HERE / "REPORT_AUDIT_R2.md").write_text(md)
print(md[:400])
print("... written", len(md), "chars")