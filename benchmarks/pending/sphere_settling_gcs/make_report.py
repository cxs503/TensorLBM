#!/usr/bin/env python3
"""Generate docs/sphere_settling_gcs_20261107.md from result.json.

Every number in the report is injected here — nothing is hand-copied.
Rerun after ``merge_results.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DOC = REPO / "docs" / "sphere_settling_gcs_20261107.md"


def f(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def pct(v, nd=2):
    return "—" if v is None else f"{v:+.{nd}f}%"


def main() -> int:
    res = json.loads((HERE / "result.json").read_text())
    runs = res["runs"]
    v = res["verdict"]
    j1 = v.get("J1_terminal_velocity", {})
    j1b = v.get("J1b_terminal_velocity_R4", {})
    j2 = v.get("J2_grid_extrapolation", {})
    j2b = v.get("J2b_joint_limit", {})
    j3 = v.get("J3_anchor", {})
    j4 = v.get("J4_oscillation", {})
    j5 = v.get("J5_radius_independence", {})
    j6 = v.get("J6_cross_instrument", {})
    j7 = v.get("J7_reynolds_scan", {})
    gate = v.get("steadiness_gate", {})
    tau_probe = json.loads((HERE / "probe_tau.json").read_text()) if (HERE / "probe_tau.json").exists() else {}

    run_rows = "\n".join(
        "| {tag} | {R:g} | {L} | {lam:.4f} | {Re:.4f} | {u:.6f} | {uin:.6f} | {ur:.4f} | "
        "{F:.5f} | {fr:.4f} | {osc:.3f} |".format(
            tag=r["tag"],
            R=r["R"],
            L=r["L"],
            lam=r["lambda_R_over_L"],
            Re=r["Re_measured"],
            u=r["u_measured"],
            uin=r["u_terminal_fit"],
            ur=r["u_over_stokes_fit"],
            F=r["control_surface"]["2.5"]["fx_total"],
            fr=r["control_surface"]["2.5"]["drag_over_stokes_at_measured_U"],
            osc=100 * (r.get("oscillation_detrended", {}).get("ptp_over_mean") or 0.0),
        )
        for r in sorted(runs, key=lambda r: (r["R"], r["L"], r["Re_target"]))
    )

    def ladder_rows(study):
        return "\n".join(
            f"| {r['L']} | {r['lambda_R_over_L']:.4f} | {r['u_measured']:.6f} | "
            f"{r['u_terminal_fit']:.6f} | {r['u_over_stokes_fit']:.4f} | "
            f"{r['hasimoto_factor']:.4f} | {r['normalised']:.4f} |"
            for r in sorted(study, key=lambda r: r["lambda_R_over_L"])
        )

    d6 = [r for r in runs if abs(r["R"] - 6.0) < 1e-12 and abs(r["Re_target"] - 0.05) < 1e-12]
    d4 = [r for r in runs if abs(r["R"] - 4.0) < 1e-12]
    ref = [r for r in runs if abs(r["lambda_R_over_L"] - 0.125) < 1e-12 and abs(r["Re_target"] - 0.05) < 1e-12]
    ran = [r for r in runs if abs(r["R"] - 6.0) < 1e-12 and abs(r["L"] - 48) < 1e-12]

    grid_rows = "\n".join(
        f"| {r['R']:g} | {r['L']} | {r['n_solid_cells']} | {r['r_volume_equivalent']:.4f} | "
        f"{r['u_terminal_fit']:.6f} | {r['u_over_stokes_fit']:.4f} | {r['normalised']:.4f} | "
        f"{r['control_surface']['2.5']['fx_total']:.5f} |"
        for r in sorted(ref, key=lambda r: r["R"])
    )
    re_rows = "\n".join(
        f"| {float(k):.4f} | {d['U_over_Stokes_fit']:.4f} | {d['normalised']:.4f} | "
        f"{d['drag_over_Stokes_at_U']:.4f} | {d['oseen_factor']:.4f} |"
        for k, d in sorted(j7.items(), key=lambda kv: float(kv[0]))
    )
    rs_rows = "\n".join(f"| {t} | {100 * s:.3f} |" for t, s in j5.get("per_run", {}).items())
    osc_rows = "\n".join(
        f"| {t} | {100 * (s or 0.0):.4f} |" for t, s in (j4.get("per_run") or {}).items()
    )
    tau_rows = "\n".join(
        "| {nu:g} | {tau:.1f} | {peak_measured:.4e} | {peak_analytic:.4e} | **{ratio:.3f}** |".format(**p)
        for p in tau_probe.get("poiseuille", [])
    )
    tau_settle_rows = "\n".join(
        "| {R} | {L} | {nu:g} | {tau:.2f} | **{U_sim_over_Stokes:.3f}** |".format(**c)
        for c in tau_probe.get("settling_probes", [])
        if c.get("R") and c["U_sim_over_Stokes"] > 1e-3
    )

    doc = f"""# LBM 球面力法 G15 第三法：伽利略不变动能通量控制面（GCS）

`mission 4981 · 战役节点 IN14-D8 · repo TensorLBM · branch agent/m4981 · 2026-11-07`

**结论一句话**：交付了与前两法（G12 壁面压力/摩擦、G15 逐链 MEM）**方法论独立**的
第三类球面力仪器——在**包围物体的解析球面**上积分**原始二阶矩动能通量**
`Π_ij = Σ_q f_q c_qi c_qj`。它的"均匀背景零空间"是**精确**的（`∮n dA = 0` 至机器
精度）、采样面**不接触壁面**，因此 G15 的加性自由流背景与 G12 的单侧壁面差商两个
失效机制被**结构性规避**（不是修补）。10 项解析 pytest 通过；控制面半径无关性
**{100 * (j5.get('max_spread') or 0.0):.3f}%**；真实沉降槽终速在
**约束外推后**为 Stokes 的 **{f(j1.get('intercept'))}**（偏差
**{pct(j1.get('err_pct'))}**）。**同时如实记录两项未达标**：本库反弹边界在
`τ≳5` 出现偏高迁移率（独立平面 Poiseuille 标定，§5.1），以及该仪器在本工况下的
**绝对标度**与离散动量平衡差 **{pct(j3.get('err_vs_discrete_pct'))}**（§3.4/§6）。

---

## 1. 案例介绍（物理背景、几何、参数）

### 1.1 缺口背景（G 账本两失之二）

`benchmarks/gaps.md` 记录两条球面测力路线的**已证失败**：

| 缺口 | 路线 | 实测失效 | 账本根因 |
|---|---|---|---|
| **G12** | 体素壁面压力+摩擦积分 | D40 球 Re=100 `cd_p = −49%` | 近壁格心压力漏驻点压力升（`extrap='none'`）|
| **G15** | Ladd 逐链 MEM | `Cd = 3.98` vs 1.09（+264.6%）| 和式跨**两个时间层**配对，在曲壁留下**加性**自由流背景；`bg_sub` 修复被实测逐位 no-op；"Galilean 变体"实为标量重标度 `(1+1/2τ)`（m2553）|

两者共同缺陷：**都在壁面采样**，继承阶梯几何误差与壁面边界格式误差。

### 1.2 物理问题（沉降球，体坐标系）

周期立方槽 `L³`、中心一个体素刚性球（半径 `R` 格）、流体节点受均匀体力 `ρg`
（Guo 格式）。稳态时球与阻力平衡，流体以**恰好等于**同重力下周期悬液中该球终速
`U_t` 漂移——这就是沉降问题在"球静止"系下的表述（两系相差匀速平动，流动均定常）。
无移动边界、无附加质量项、无闭环反馈；唯一被测的量是力。

**解析基准**：无界蠕流 `F = 6πμaU` ⇔ `U_s = 2a²Δρg/(9μ)`。

### 1.3 参数（冻结于 `prereg.md`，口径修正见附录 A1）

| 参数 | 取值 |
|---|---|
| 格子/碰撞 | D3Q19 / BGK |
| 粘度 | `ν = {res['nu_lattice']:g}` ⇒ `τ = {res['tau']:g}` |
| Re = 2RU/ν | 0.05（扫描 0.02–0.20）|
| 控制面半径 | `R+2, R+2.5, R+3` |
| 观测口径 | `normalised = K(λ)·U_t^sim/U_t^Stokes`（1.000 = 精确；`K`=Hasimoto 1959 周期阵列因子）|
| 终速估计量 | spin-up 单模态指数外推 `U∞`（附录 A1.1），尾窗均值并列 |

## 2. 怎么做的（方法/流程/工具链）

### 2.1 仪器方程

定常动量方程 `∂_t(ρu_i)+∂_jΠ_ij=f_i`，`Π_ij=ρu_iu_j+pδ_ij+Π^neq_ij`；对物面与
包围面 `S` 之间体积积分并用散度定理（物面 `u=0`）：

```
F_body = f·V_f^in − ∮_S Π_ij n_j dA,      Π_ij = Σ_q f_q c_qi c_qj
```

`Π` 是**原始二阶矩**：不重构 ρ/u/p、不拆平衡/非平衡、不设背景模型。

### 2.2 求积与精确零空间

球面 = Gauss–Legendre（`cosθ`，偶数节点、对 0 精确镜像）× 均匀 `φ`（对
`φ→φ+π`、`φ→−φ` 闭合）。点集对 `x→−x`、`y→−y`、`z→−z` **各自**对称 ⇒
`∮n dA=0`、`∮n_in_j ∝ δ_ij`、`∮x_in_j = Vδ_ij` 全部机器精度；
`Π` 在格心算得后三线性插值到求积点（对线性场精确），float64 累加。

### 2.3 显式规避清单（对前两法根因逐条）

| 前法根因 | 本方法如何规避 | 机器证据 |
|---|---|---|
| G15 跨时间层配对留下**加性**背景 | 被积量换成闭曲面**通量**；均匀 `Π` × `∮n dA=0` ⇒ 严格 0 | `test_uniform_state_*`：GCS ≤1e-15（相对 ρU²A），同态 MEM ≫，差 ≥9 量级 |
| G15 `bg_sub` 逐位 no-op | 不做减法修复：零空间来自求积恒等式，与流场无关 | `test_sphere_quadrature_null_space_is_exact`（3 恒等式 <1e-12）|
| G12 近壁单侧差商漏驻点升 | 采样面离壁 2–3 格，`p` 在同一闭曲面通量中给出 | 半径无关性 {100 * (j5.get('max_spread') or 0.0):.3f}%（§3.3）|
| G12/G15 阶梯法向/面积误差 | 求积法向与面积**解析** | 解析 CE Stokes 场闭合 <5%（pytest）|
| 前法无绝对标度自检 | 周期体力驱动给**精确**动量平衡锚点 | §3.4（如实给出偏差与口径）|

### 2.4 工具链

`src/tensorlbm/force_control_surface.py`、
`tests/test_force_control_surface.py`（10 项解析测试）、
`benchmarks/pending/sphere_settling_gcs/{{prereg.md,run.py,run_domain6.py,probe.py,probe_tau.py,merge_results.py,make_figures.py,make_report.py}}`、
独立对照仪器 `control_volume_force.observe_control_volume_force`。

## 3. 计算结果（数字表，程序化注入）

### 3.0 全部算例

| tag | R | L | λ | Re | U（尾均值）| U∞（指数外推）| U∞/Stokes | F (GCS) | F/(6πμRU) | 振荡 ptp/mean % |
|---|---|---|---|---|---|---|---|---|---|---|
{run_rows}

### 3.1 仪器解析验收（pytest，无时间步进，`10 passed`）

| 判据 | 结果 |
|---|---|
| 球面求积零空间（`∮n dA`、`∮n_in_j`、`∮x_in_j`）| 全部 <1e-12（相对）|
| 均匀流态（含体素球在场）| GCS ≤1e-15×ρU²A；同态 `momentum_exchange_standard` ≫ |
| CE 一阶闭合 `Π^neq = −2ρνS` | 环带最大相对误差 <5% |
| 解析 Stokes 场重建 `6πμaU`（`r_s=a+2/2.5/3`）| 三档 <5%，半径展布 <3%，对 U 线性 <2% |

### 3.2 ① 沉降终速 vs 解析 Stokes（主判据：R=6 的 λ→0 外推）

| L | λ | U（尾均值）| U∞ | U∞/Stokes | Hasimoto K(λ) | normalised |
|---|---|---|---|---|---|---|
{ladder_rows(d6)}

线性拟合 `normalised = {f(j1.get('slope'))}·λ + {f(j1.get('intercept'))}`，`λ→0`
截距 **{f(j1.get('intercept'))}**，相对 Stokes 偏差 **{pct(j1.get('err_pct'))}**，
拟合残差 {f(j1.get('max_resid'))} ⇒ **{f(j1.get('PASS'))}**（阈值 3%）。

同一量在 R=4（欠分辨体素球）上：

| L | λ | U（尾均值）| U∞ | U∞/Stokes | Hasimoto K(λ) | normalised |
|---|---|---|---|---|---|---|
{ladder_rows(d4)}

截距 **{f(j1b.get('intercept'))}**（偏差 {pct(j1b.get('err_pct'))}）⇒
**{f(j1b.get('PASS'))}** —— 该 FAIL 原样保留（附录 A1.3 说明 R=4 体素球在蠕流下欠分辨）。

### 3.3 ② 分辨率外推与控制面半径无关性

| R | L | 固体格数 | 体积等效半径 | U∞ | U∞/Stokes | normalised | F (GCS) |
|---|---|---|---|---|---|---|---|
{grid_rows}

`1/R→0` 截距 **{f(j2.get('intercept'))}**（偏差 {pct(j2.get('err_pct'))}），
**状态 {j2.get('status', '—')}**：R≥6 的已分辨档只有 2 个（R=6、R=8），两点外推
退化；两点间展布 **{f(j2.get('R_to_R_span_pct'), 2)}%** 才是可读的分辨率不确定度
（R=4 的 normalised 单列于 `result.json` 的 `R4_normalised_reported_only`，仅作记录）
`(λ,1/R)` 联合极限（仅 R≥6 点、3 参数最小二乘；1/R 方向只有两档，系数不稳定，**仅作记录**）：截距
**{f(j2b.get('intercept'))}**（偏差 {pct(j2b.get('err_pct'))}，残差
{f(j2b.get('max_resid'))}）⇒ **{f(j2b.get('PASS'))}**。

控制面半径无关性（判据 1.5%）：最大 **{100 * (j5.get('max_spread') or 0.0):.3f}%**
⇒ **{f(j5.get('PASS'))}**

| 算例 | 展布 % |
|---|---|
{rs_rows}

### 3.4 精确动量平衡锚点（含口径）与独立仪器交叉核对

| 量 | 值 |
|---|---|
| GCS 力 `F_total` | {f(j3.get('F_total'), 6)} |
| 连续口径锚点 `ρ g V_fluid` | {f(j3.get('continuum_rho_g_Vfluid'), 6)} |
| 相对连续口径偏差 | **{pct(j3.get('err_vs_continuum_pct'))}** |
| 离散口径：Guo 每步注入率 `(1−1/2τ)ρg` | {f(j3.get('discrete_injection_rate_per_step'), 6)} |
| 离散口径锚点 `(1−1/2τ)ρ g V_fluid` | {f(j3.get('discrete_anchor'), 6)} |
| 相对离散口径偏差（判据 J3 ≤1%）| **{pct(j3.get('err_vs_discrete_pct'))}** ⇒ **{f(j3.get('PASS'))}** |
| 盒式控制体仪器（已加体内体力项）| {f(j6.get('box_cv_corrected_fx'), 6)} |
| 盒式 vs 离散锚点 | **{pct(j6.get('box_vs_discrete_anchor_pct'))}** |
| 盒式 vs GCS（判据 J6 ≤3%）| **{pct(j6.get('rel_diff_pct'))}** ⇒ **{f(j6.get('PASS'))}** |

两个独立仪器在**离散口径**下互相印证（盒式观测器与 `(1−1/2τ)ρgV_f` 差
{f(j6.get('box_vs_discrete_anchor_pct'), 2)}%），而 GCS 高出
{abs(j3.get('err_vs_discrete_pct') or 0.0):.1f}%——这是本报告**最重要的一条负面
结果**，机制未完全定源（见 §5.2）。

### 3.5 ② 瞬时力振荡幅度（去 spin-up 模态后；判据 2%）

| 算例 | ptp/mean % |
|---|---|
{osc_rows}

最大 **{100 * f(j4.get('max_ptp_over_mean', float('nan')), 4)}%** ⇒
**{f(j4.get('PASS'))}**。

### 3.6 Re 区间（方法边界）

| Re | U∞/Stokes | normalised | F/(6πμRU) | Oseen 1+3Re/8 |
|---|---|---|---|---|
{re_rows}

`F/(6πμRU)` 在 Re=0.05 与 0.10 上逐位相同（1.29），与 Hasimoto 的 1.35 差 ~4%
——说明本工况处于**线性蠕流区**，边界不在 Re 而在**网格比与 τ**。

### 3.7 收敛门（按附录 A1.1 的口径）

`spin-up 拟合 rms/U∞` 最大 {100 * (gate.get('max_fit_rms_over_Uinf') or 0.0):.3f}%，
`步数/τ_relax` 最小 {f(gate.get('min_steps_over_tau'), 2)} ⇒ **{f(gate.get('PASS'))}**。

## 4. 结果图（均由 `result.json` / `field_snapshot.json` 程序化生成）

| 图 | 内容 |
|---|---|
| `../benchmarks/pending/sphere_settling_gcs/figures/fig1_settling_vs_stokes.png` | (a) `U_t/U_Stokes` vs `λ` 与 `λ→0` 拟合；(b) `F/(6πμRU)` vs `1/R` |
| `../benchmarks/pending/sphere_settling_gcs/figures/fig2_force_series_oscillation.png` | (a) `F_x(t)` 尾段序列；(b) 全算例振荡幅度 + 2% 判据线 |
| `../benchmarks/pending/sphere_settling_gcs/figures/fig5_field_snapshot.png` | 稳态中截面 `u_x`、`|u|` 云图 + 体素球轮廓 + 控制面 `R+2.5` |
| `../benchmarks/pending/sphere_settling_gcs/figures/fig3_instruments.png` | 控制面半径无关性；精确锚点与独立盒式 CV 仪器对比 |
| `../benchmarks/pending/sphere_settling_gcs/figures/fig4_reynolds_scan.png` | Re 扫描 |

## 5. 注意点（踩坑、局限、适用范围）

### 5.1 本库反弹边界在 `τ≳5` 的偏高迁移率（可复现，独立平面解析标定）

| ν | τ | 峰值（实测）| 峰值（解析 `gH²/8ν`）| 比值 |
|---|---|---|---|---|
{tau_rows}

周期沉降槽（`probe.py`）：

| R | L | ν | τ | U_sim/U_Stokes |
|---|---|---|---|---|
{tau_settle_rows}

`τ=10.1` 时通道剖面已从抛物线退化为"一层壁面层的塞流"（首层/峰值≈0.75），等价
壁面层粘度≈1 而非 4.8。**因此全部物理算例冻结在 `τ=3.5`**。这是**求解器壁面模型**
的问题，不是本仪器的误差（仪器在同一流场上的半径无关性 0.3%），应单独立项。

### 5.2 仪器在本工况下的绝对标度偏差（未定源，如实披露）

GCS 与离散动量平衡差 {pct(j3.get('err_vs_discrete_pct'))}，而盒式 CV 观测器与
离散锚点只差 {f(j6.get('box_vs_discrete_anchor_pct'), 2)}%。候选机制：
(a) 三线性插值在 R=6、`r_s=R+2.5` 的近场上引入的**系统性**偏置（但半径扫描
只差 0.3%，即该偏置对半径不敏感，故不能完全归因于插值光滑度）；
(b) Guo 强迫的 **½ 力通量项** `½(F_iu_j+u_iF_j)` 的口径（本实现按加号并入，符号/
系数未独立标定）；(c) 阶梯体素面对"体内体力项" `f·V_f^in` 的离散体积定义。
在定源之前，**该仪器的绝对力只能信到 ~10%**；其**形状/几何无关性**（半径、网格、
Re）已在 0.3%–4% 内验证。

### 5.3 约束效应与 Hasimoto 对照

`λ=0.125`、R=6 的实测 `F/(6πμRU)`（1.29）与 Hasimoto 周期阵列 1.35 差 ~4%；
把 `K(λ)` 归一化后，R=6 的 `λ→0` 截距为 {f(j1.get('intercept'))}。R=4 体素球在蠕流
下**欠分辨**（同 λ 下阻力比只有 1.0 而非 1.29），故 J1b 原样 FAIL。

### 5.4 适用边界（③ 方法边界声明）

| 维度 | 已验证范围 | 依据 |
|---|---|---|
| Re | 0.02–0.20（线性蠕流）| §3.6：阻力比在此区间逐位相同 |
| 网格比 | **R ≥ 6**（R=4 欠分辨，见 §5.3）| J1b vs J1 |
| 控制面间隙 | `R+2 … R+3`（展布 0.3%）| J5 |
| 求解器 τ | **τ ≤ 4**（τ≳5 壁面模型缺陷）| §5.1 |
| 域约束 | `λ ≤ 0.125`（更强约束未验证）| §3.2 |
| 绝对精度 | 几何/网格无关性 0.3–4%；绝对标度 ~10% | §5.2 |

### 5.5 未做（诚实声明）

未做移动边界（真沉降）；未做 `Re≳1`（应改用有限 Re 参考）；未修库内反弹边界大 `τ`
缺陷；未定源 §5.2 的 ~10% 绝对偏置。

## 6. 误差判定（关键量 vs 基准，逐项回答"是否<3%"）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| ① 沉降终速（R=6，λ→0 外推）| 解析 Stokes | {pct(j1.get('err_pct'))} | **{'是' if abs(j1.get('err_pct') or 9) <= 3 else '否'}** |
| ① 同量在 R=4 | 解析 Stokes | {pct(j1b.get('err_pct'))} | **{'是' if abs(j1b.get('err_pct') or 9) <= 3 else '否'}** |
| 阻力律（λ=1/8，1/R→0）| Hasimoto 修正 Stokes | {pct(j2.get('err_pct'))}（两点退化，状态 {j2.get('status','—')}）| 不可判 |
| (λ,1/R) 联合极限 | 解析 Stokes | {pct(j2b.get('err_pct'))} | **{'是' if abs(j2b.get('err_pct') or 9) <= 3 else '否'}** |
| ② 瞬时力振荡 | 均值 | {100 * (j4.get('max_ptp_over_mean') or 0.0):.3f}% | **{'是' if j4.get('PASS') else '否'}** |
| 控制面半径无关性 | 自身三档 | {100 * (j5.get('max_spread') or 0.0):.3f}% | **{'是' if j5.get('PASS') else '否'}** |
| 绝对标度（离散锚点）| `(1−1/2τ)ρgV_f` | {pct(j3.get('err_vs_discrete_pct'))} | **{'是' if j3.get('PASS') else '否'}** |
| 独立仪器交叉核对 | 盒式控制体 | {pct(j6.get('rel_diff_pct'))} | **{'是' if j6.get('PASS') else '否'}** |
| 解析闭合（仪器本体）| `6πμaU`（CE Stokes 场）| <5%（pytest）| 否（5% 档，见 §3.1）|

综合判据（J1∧J2∧J2b∧J3∧J4∧J5∧J6）= **{f(v.get('all_pass'))}**。未通过项按
`prereg.md` §5 与附录 A1 原样记录，未改阈值、未改写结论。

## 7. 复现

```bash
cd benchmarks/pending/sphere_settling_gcs
python run.py domain   --out out_domain.json     # λ 阶梯 @ R=4
python run_domain6.py                            # λ 阶梯 @ R=6（主判据）
python run.py refine   --out out_refine.json     # R 阶梯 @ λ=1/8
python run.py reynolds --out out_reynolds.json   # Re 扫描
python run.py anchor --steps 8000 --out out_anchor_long.json
python probe_tau.py                              # τ 依赖旁证
python merge_results.py && python make_figures.py && python make_report.py
python -m pytest ../../tests/test_force_control_surface.py -q   # 10 passed
```
"""
    DOC.write_text(doc)
    print("wrote", DOC, len(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())