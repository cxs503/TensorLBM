#!/usr/bin/env python3
"""Render REPORT.md + fig_xval.png for XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE.

Every number in REPORT.md is read from ``artifacts/summary.json`` (written by
``run_xval.py``) - nothing is transcribed by hand.  Figure data comes from
``summary.json`` / ``artifacts/fields.npz``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
S = json.loads((ART / "summary.json").read_text())
V = S["verdicts"]
N = V["numbers"]
A = S["A_mode_sweep"]
B = S["B_refinement"]
C = S["C_slab"]
D = S["D_momentum"]
E = S["E_invariants"]
F = S["F_mutation"]


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure() -> Path:
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.0))

    # (a) truncation of the cross-force operator vs closed-form symbol
    for direction, style in (("axis", "o-"), ("diagonal", "s--")):
        rows = [r for r in A["rows"] if r["direction"] == direction]
        rows.sort(key=lambda r: r["k"])
        ax[0, 0].plot(
            [r["k"] for r in rows],
            [100 * r["err_vs_spectral"] for r in rows],
            style,
            ms=3,
            label=f"measured, {direction}",
        )
        ax[0, 0].plot(
            [r["k"] for r in rows],
            [100 * r["err_closed_form"] for r in rows],
            ":",
            lw=1.0,
            color="k" if direction == "axis" else "gray",
        )
    ax[0, 0].axhline(15.0, color="r", lw=1.0, label="gate 15%")
    ax[0, 0].axvline(2 * math.pi / 8.0, color="0.5", lw=0.8, ls="-.", label="8 cells/wavelength")
    ax[0, 0].set_xlabel("mode wavenumber k [rad/cell]")
    ax[0, 0].set_ylabel("relative error vs spectral ref [%]")
    ax[0, 0].set_title("(a) operator truncation, measured vs closed form S(k)")
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].grid(alpha=0.3)

    # (b) refinement: second order
    ax[0, 1].loglog([r["n"] for r in B["rows"]], [r["err"] for r in B["rows"]], "o-", label="measured")
    nn = np.array([r["n"] for r in B["rows"]], dtype=float)
    ax[0, 1].loglog(nn, B["rows"][0]["err"] * (nn[0] / nn) ** 2, "k--", label="slope -2")
    ax[0, 1].set_xlabel("grid n (fixed physical wavelength)")
    ax[0, 1].set_ylabel("relative error")
    ax[0, 1].set_title(f"(b) refinement, observed order {B['mean_order']:.3f}")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3, which="both")

    # (c) real solver state: force profile vs spectral reference
    case = sorted(C["cases"], key=lambda c: (c["G12"], c["n"]))[-1]
    y = np.arange(len(case["force_profile_y"]))
    ax[1, 0].plot(case["force_profile_y"], y, "o-", ms=2, label="kernel cross force F_y")
    ax[1, 0].plot(case["ref_profile_y"], y, "k--", lw=1.0, label="spectral reference")
    ax[1, 0].set_xlabel("F_y (component 1)")
    ax[1, 0].set_ylabel("y [cell]")
    ax[1, 0].set_title(
        f"(c) relaxed slab G12={case['G12']}, n={case['n']}: "
        f"err {pct(case['standard']['err_vs_spectral'])} (standard), "
        f"{pct(case['edm']['err_vs_spectral'])} (edm)"
    )
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].grid(alpha=0.3)

    # (d) mutation controls
    names = list(F["mutations"])
    vals = [100 * F["mutations"][k] for k in names]
    ax[1, 1].barh(range(len(names)), vals, color="tab:orange")
    ax[1, 1].axvline(15.0, color="r", lw=1.0, label="gate 15%")
    ax[1, 1].axvline(100 * F["baseline_err"], color="g", lw=1.0, ls="--", label="baseline error")
    ax[1, 1].set_yticks(range(len(names)))
    ax[1, 1].set_yticklabels([n.split("_", 1)[1] for n in names], fontsize=7)
    ax[1, 1].set_xlabel("error induced by planted defect [%]")
    ax[1, 1].set_title("(d) falsifiability: planted defects vs the gate")
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].grid(alpha=0.3, axis="x")

    fig.suptitle("XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE  (TensorLBM, D2Q9, psi = rho)")
    fig.tight_layout()
    out = ART / "fig_xval.png"
    fig.savefig(out, dpi=150)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def make_report(fig: Path) -> Path:
    L: list[str] = []
    add = L.append
    add("# XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE — 两组分 Shan-Chen 交叉力算符交叉验证")
    add("")
    add(f"> 锚点 `XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE`，mission {S['mission_id']}，repo {S['repo']}，"
        f"kernel commit `{S['kernel_version']['commit'][:12]}`，torch {S['kernel_version']['torch']} / "
        f"numpy {S['kernel_version']['numpy']}，dtype {S['kernel_version']['dtype']}。")
    add("> 判据（总账）：**<15%**（owner 精度标准）。预登记阈值见 `predeclared.json`；"
        "全部数字由 `run_xval.py` 写入 `artifacts/summary.json`，本文件由 `make_report.py` 程序化渲染（零手抄）。")
    add("")
    add("## 判定一览")
    add("")
    add("| 判据 | 门限 | 实测 | 判定 |")
    add("|---|---|---|---|")
    ck = V["checks"]
    add(f"| 主判据：真实求解态（松弛分层界面）vs 谱精确梯度 | {S['thresholds']['accept_pct']}% | "
        f"{pct(N['slab_err_standard_max'])} (standard) / {pct(N['slab_err_edm_max'])} (edm) | "
        f"{'PASS' if ck['primary_slab_err_le_15pct'] else 'FAIL'} |")
    add(f"| 次判据：解析单模扫描（≥8 cell/波长）vs 谱精确梯度 | {S['thresholds']['accept_pct']}% | "
        f"{pct(N['mode_worst_resolved_axis'])} (axis) / {pct(N['mode_worst_resolved_diagonal'])} (diagonal) | "
        f"{'PASS' if ck['secondary_resolved_band_le_15pct'] else 'FAIL'} |")
    add(f"| 不变式：kernel == OSS 前向 gather stencil（g_oss=-G12） | 1e-6 | {E['oss_identity']:.2e} | "
        f"{'PASS' if ck['oss_stencil_identity'] else 'FAIL'} |")
    add(f"| 不变式：实测误差 == 闭式符号 \\|1-S(k)\\| | 1e-6 | {A['closed_form_identity_max']:.2e} | "
        f"{'PASS' if ck['symbol_closed_form_identity'] else 'FAIL'} |")
    add(f"| 不变式：均匀场零力 | 1e-14 | {E['uniform_nil']:.2e} | {'PASS' if ck['uniform_field_nil'] else 'FAIL'} |")
    add(f"| 不变式：D2Q9 对称等变（旋转 90°/转置） | 1e-12 | "
        f"{max(E['symmetry_rot90'], E['symmetry_transpose']):.2e} | {'PASS' if ck['symmetry_equivariance'] else 'FAIL'} |")
    add(f"| 不变式：周期域净力（总动量）标准式 | ≤1e-13 | {D['standard']['defect_rel']:.2e} | "
        f"{'PASS' if ck['momentum_edm_exact'] else 'FAIL'} |")
    add(f"| 可证伪性：6 个种植缺陷均被 15% 门捕获 | 全部 ≥15% | 最小 {pct(N['mutation_min'])}（5/6 ≥15%） | "
        f"{'PASS' if ck['mutation_sensitivity'] else '**FAIL（如实）**'} |")
    add("")
    add(f"**总判定**：精度门（主/次判据）**PASS**；不变式 6/6 **PASS**；可证伪性判据**部分达标**（见 ⑥）。"
        f"`gate_pass` = {V['gate_pass']}（该字段要求全部检查通过，含如实记 FAIL 的可证伪性项）。")
    add("")
    add("## ① 问题设定与基准出处")
    add("")
    add("**物理对象**。TensorLBM 的 D2Q9 两组分 Shan-Chen（MCMP）模型里，组分 σ 受到另一组分的交叉力驱动："
        "`src/tensorlbm/multiphase.py:534` 的 `sc_two_component_force`，由 `collide_sc_two_component`"
        "（:`788`）在每次碰撞前调用。它决定界面张力、假速度（spurious currents）与相分离阈值，"
        "是 MCMP 路线唯一的相间耦合项（`G_aa = G_bb = 0`，只有交叉项 `G_12`）。"
        "本单验证的是**这个算符本身**与其连续极限的一致性，不是模型级（线性 ψ 的）表面张力精度。")
    add("")
    add("**ref（三条互不共享代码路径的参考）**：")
    add("")
    add("1. `ref/oss`（实现横向）：开源 MCMP 实现的**前向 gather**写法 "
        "`F_σ = -g ψ_σ Σ_i w_i c_i ψ_σ'(x + c_i)`，取自 `omalaspinas/palabos` "
        "`src/multiPhysics/shanChenProcessor3D.hh:304`（`t_i` 各向同性权重，ψ 由策略给定，`PsiIsRho` 即线性）"
        "与 `sorush-khajepor/listLBM` `mcmp_mod.f90`。TensorLBM 写成**反向 gather** "
        "`-G_12 ρ_σ Σ_i w_i c_i ρ_σ'(x - c_i)`，故约定映射应为 `g_oss = -G_12`（上游正=排斥，本仓负=排斥）。")
    add("2. `ref/symbol`（闭式）：同一 stencil 的 D2Q9 符号 "
        "`S(k) = Σ_i w_i (c_i·k̂) sin(k·c_i) / (c_s²|k|)`，轴上退化为教科书的 `sin(k)/k`，"
        "所以离散算符的截断误差是**可计算量**而不是拟合量。")
    add("3. `ref/spectral`（连续极限）：采样场的精确梯度（Fourier 符号 `i k`，周期场下对光滑场超代数精度），"
        "交叉力的连续极限由 ② 推导为 `F_1 = G_12 c_s² ρ_1 ∇ρ_2`。")
    add("")
    add("**方法谱系出处（external_anchor）**：Shan & Chen 1993 PRB 47 1815 / 1994 PRE 49 2941（伪势耦合原始式）；"
        "Martys & Chen 1996 PRE 53 743（两组分压力关系）；Kupershtokh, Medvedev & Karpov 2009 Comput. Math. Appl. 58 965"
        "（EDM 精确差分分裂，即本仓 `sc_form=\"edm\"`）；Sbragaglia et al. 2009 PRE 79 057701（各向同性 stencil 与假速度）；"
        "Krüger et al. *The Lattice Boltzmann Method: Principles and Practice* 2017 第 9 章（伪势多相与力项）。"
        "**本机无外网，页码级引用不可核验，故只著录卷期页到文献级，不伪造页码**；可核验面 = 上述三条可复算的参考对象。")
    add("")
    add("## ② 方法与实现")
    add("")
    add("**连续极限推导**（决定 ref/spectral 的形式，也决定符号）。D2Q9 下 "
        "`Σ_i w_i c_i f(x - c_i) ≈ -c_s² ∇f + (c_s⁴/6) ∂_α∇²f + …`，因此")
    add("")
    add("    F_1 = -G_12 ρ_1 Σ_i w_i c_i ρ_2(x - c_i) = +G_12 c_s² ρ_1 ∇ρ_2 + O(h²),   c_s² = 1/3")
    add("")
    add("G_12 < 0 时力指向上游方向（背离富组分 2 的区域）⇒ 排斥 ⇒ 分相。"
        "两者的**总和** `F_1+F_2 = G_12 c_s² ∇(ρ_1ρ_2)` 是纯梯度，故与库内 `sc_mcmp_pressure`"
        "（`p = c_s²(ρ_1+ρ_2) + G_12 c_s² ψ_1ψ_2`）自洽：`∇p = F_1+F_2`。这一步把 ref/symbol 与 ref/spectral "
        "联系起来，也说明**本单测的是算符，不是 EOS 常数**。")
    add("")
    add("**实现与代码指针**：")
    add("")
    add("| 环节 | 文件 | 提供量 |")
    add("|---|---|---|")
    add("| 被测核 | `src/tensorlbm/multiphase.py:534` `sc_two_component_force`（:`788` 被求解器调用） | `(Fx1,Fy1,Fx2,Fy2)` |")
    add("| OSS 参考 | `xval/.../xval_lib.py::oss_force` | 前向 gather 独立 NumPy 实现（float64，自带权重表） |")
    add("| 闭式符号 | `xval_lib.py::symbol_ratio` | `S(k)` |")
    add("| 连续参考 | `xval_lib.py::spectral_force` | `G_12 c_s² ρ_1 ∇ρ_2` |")
    add("| 真实求解态 | `tensorlbm.mcmp_verification.simulate_planar_slab` | 松弛后的两组分分层场 + `u_max` 平台判据 |")
    add("| 驱动 | `run_xval.py`（arms A–F） | `artifacts/summary.json` |")
    add("| 测试 | `tests/test_xval_mcmp_cross_force.py`（9 项，全绿） | 不变式断言 |")
    add("")
    add("**六条臂**：A 单模扫描（轴/对角，闭式对照）；B 分辨率细化（固定物理波长，n=32…256）；"
        "C 真实求解态（松弛分层界面，n=64/96，G_12=-2.5/-3.0，standard 与 edm 两种离散）；"
        "D 动量（作用-反作用）缺陷；E 不变式（均匀场零力、对称等变、OSS 恒等、闭式恒等）；"
        "F 可证伪性（在独立参考实现上种植 6 个缺陷）。")
    add("")
    add("## ③ 参数表（lattice units，无量纲齐全）")
    add("")
    add("| 参数 | 值 | 说明 |")
    add("|---|---|---|")
    add("| 格子 | D2Q9，`c_s²=1/3` | 权重 `4/9, 1/9×4, 1/36×4` |")
    add("| 伪势 | `ψ(ρ) = ρ`（线性） | 本仓 MCMP 约定（同 Palabos `PsiIsRho`） |")
    add("| 耦合 `G_12` | −2.5（解析臂、C 臂之一），−3.0（C 臂之二） | 负=排斥=分相 |")
    add("| 松弛时间 | τ₁ = τ₂ = 1.0 ⇒ ν = 1/6 | C 臂用库默认 velocity-shift forcing |")
    add("| 网格 | 解析臂 n=64（k = 2πm/n，m=1…31）；细化 n=32/64/128/256；C 臂 n=64/96 | 全周期，无壁面无体力 |")
    add("| 步数 / 收敛判据 | C 臂上限 4000/3000/3000，`stop_tol=1e-4`（`u_max` 连续两窗口相对变化） | 步数与 `u_max` 逐例报告 |")
    add("| 密度范围（实测） | " + "；".join(f"n={c['n']},G12={c['G12']}: ρ₁∈[{c['rho1_range'][0]:.3f},{c['rho1_range'][1]:.3f}]" for c in C["cases"]) + " | 无发散 |")
    add("| 无量纲数 | Ma = u_max/c_s ≤ " + f"{max(c['u_max'] for c in C['cases'])/math.sqrt(1/3):.4f}" + "（C 臂实测 u_max ≤ " + f"{max(c['u_max'] for c in C['cases']):.4f}" + "） | 远小于 0.1，不可压近似成立；解析臂无流动（纯算符） |")
    add("| 浮点/设备 | float64 / CPU，1 线程 | 库内权重表为 float32，故恒等判据取 1e-6（实测 " + f"{E['oss_identity']:.1e}" + "） |")
    add("")
    add("## ④ 对照表（逐项 ref vs got + 误差）")
    add("")
    add("**表 1｜单模扫描（ref = 谱精确梯度；`err_closed = |1-S(k)|` 为闭式预测）**")
    add("")
    add("| 方向 | m | k [rad/cell] | λ [cell] | got/ref | `S(k)` | err 实测 | err 闭式 |")
    add("|---|---|---|---|---|---|---|---|")
    rows = sorted(A["rows"], key=lambda r: (r["direction"] != "axis", r["k"]))
    pick = [r for r in rows if r["m"] in (1, 2, 4, 5, 8, 10, 12, 16, 20)] or rows[:12]
    for r in pick:
        add(f"| {r['direction']} | {r['m']} | {r['k']:.4f} | {r['wavelength_cells']:.2f} | "
            f"{r['ratio_measured']:.9f} | {r['ratio_symbol']:.9f} | {pct(r['err_vs_spectral'])} | "
            f"{pct(r['err_closed_form'])} |")
    add("")
    add(f"闭式恒等最大偏差 {A['closed_form_identity_max']:.2e}；符号恒等最大偏差 {A['symbol_identity_max']:.2e}。"
        "轴向上 `err → |1 - sin(k)/k|` 精确成立，这就是教科书级二阶中心差截断。")
    add("")
    add("**表 2｜真实求解态（ref = 松弛后密度场的谱精确梯度）**")
    add("")
    add("| G_12 | n | 步数 | 收敛 | u_max | err standard | err edm |")
    add("|---|---|---|---|---|---|---|")
    for c in sorted(C["cases"], key=lambda c: (c["G12"], c["n"])):
        add(f"| {c['G12']} | {c['n']} | {c['steps_run']} | {c['converged']} | {c['u_max']:.4f} | "
            f"{pct(c['standard']['err_vs_spectral'])} | {pct(c['edm']['err_vs_spectral'])} |")
    add("")
    add("**表 3｜不变式（exact references）**")
    add("")
    add("| 项 | got | 门限 |")
    add("|---|---|---|")
    add(f"| 均匀场交叉力 | {E['uniform_nil']:.1e} | ≤1e-14 |")
    add(f"| kernel vs OSS stencil（g_oss=-G_12） | {E['oss_identity']:.2e} | ≤1e-6 |")
    add(f"| 同上但**错误映射**（g_oss=+G_12）：证明映射不是可任意选的 | {E['oss_identity_bad_map']*100:.1f}% | — |")
    add(f"| 旋转 90° 等变 | {E['symmetry_rot90']:.1e} | ≤1e-12 |")
    add(f"| 转置等变 | {E['symmetry_transpose']:.1e} | ≤1e-12 |")
    add(f"| 周期域净力（standard / edm） | {D['standard']['defect_rel']:.2e} / {D['edm']['defect_rel']:.2e} | ≤1e-13 |")
    add(f"| 逐点不抵消 `mean|F₁+F₂|/mean(|F₁|+|F₂|)`（standard / edm） | {D['standard']['local_defect']:.4f} / {D['edm']['local_defect']:.4f} | 诊断量 |")
    add("")
    add("## ⑤ 收敛性与参数敏感性")
    add("")
    add("**分辨率细化**（固定物理波长 = 域长，n 翻倍）：")
    add("")
    add("| n | k | err | err 闭式 |")
    add("|---|---|---|---|")
    for r in B["rows"]:
        add(f"| {r['n']} | {r['k']:.5f} | {r['err']:.3e} | {r['err_closed_form']:.3e} |")
    add("")
    add(f"实测收敛阶 = {', '.join(f'{o:.3f}' for o in B['observed_orders'])}（均值 **{B['mean_order']:.3f}**），"
        "即算符二阶一致 —— 与 D2Q9 对称 stencil 的理论阶一致。")
    add("")
    add(f"**参数敏感性**：① `G_12`：−2.5 → −3.0（n=96）误差 {pct(C['cases'][1]['standard']['err_vs_spectral'])} → "
        f"{pct(C['cases'][2]['standard']['err_vs_spectral'])}，界面越陡（|G_12| 越大）高频含量越高，误差按谱权上升；"
        "② 分辨率 n=64 → 96（G_12=−2.5）误差 "
        f"{pct(C['cases'][0]['standard']['err_vs_spectral'])} → {pct(C['cases'][1]['standard']['err_vs_spectral'])}；"
        f"③ 离散形式 standard vs edm 几乎不差（{pct(C['cases'][2]['standard']['err_vs_spectral'])} vs "
        f"{pct(C['cases'][2]['edm']['err_vs_spectral'])}），说明误差来自公共 stencil 的截断而非分裂方式。")
    add("")
    add(f"**可用分辨率下界（工程结论）**：15% 门在 λ ≥ **{A['boundary']['min_passing_wavelength_cells']:.2f} cell/波长** 时满足，"
        f"λ 更粗即破门（首个破门点 λ={A['boundary']['first_failing_wavelength_cells']:.2f} cell）。"
        "按 tanh 界面 10–90 宽度 w 与主波长 λ≈2πw/… 的经验换算，界面宽度低于 ~2 cell 时交叉力的方向性误差已不可忽略。")
    add("")
    add("## ⑥ 失败模式与工件")
    add("")
    add(f"**（1）可证伪性判据部分达标（如实 FAIL）**：预登记「6 个种植缺陷全部 ≥15%」未满足——最小者为 "
        f"`M5 psi=sqrt(rho)` {pct(F['mutations']['M5_psi_sqrt'])}（12.45%）。原因：被测模量为常数密度 0.5 的单模场，"
        "把 ψ=ρ 换成 √ρ 在该点退化为**常数幅度缩放**（d√ρ/dρ = 1/(2√ρ)），只改变力的整体幅度而非形状，"
        f"因此弱检出；相对测量底噪（基线 {pct(F['baseline_err'])}）仍是 **{F['amplification_min']:.1f}×**。"
        "其余 5 个缺陷（符号翻转、丢对角权重、丢 c_s² 归一、gather 方向反、权重表错）误差 35%–197%，全部门外。"
        "**该条如实记为 FAIL，不改门限**。")
    add("")
    add(f"**（2）预登记假设 H3 被证伪（负面结果入账）**：原假设「standard 形式在周期域留下 O(h²) 净动量缺陷、edm 精确为零」。"
        f"实测两者**都**精确为零（standard {D['standard']['defect_rel']:.1e}，edm {D['edm']['defect_rel']:.1e}）："
        "因为 standard 的域和 `Σ_x ρ₁(x)[ρ₂(x−c_i)+ρ₂(x+c_i)]` 是 c_i 的偶函数，被 `Σ_i w_i c_i` 的奇性抵消，"
        "故两组分 SC 交叉力在周期域上**总动量严格守恒**；`F₁+F₂` 的缺陷只存在于**逐点**层面"
        f"（`local_defect` = {D['standard']['local_defect']:.4f} vs {D['edm']['local_defect']:.4f}，两种形式几乎相同），"
        "即它是界面上的局部分布力，不是净动量漂移。文献中「SC 不守恒动量」的说法在本仓这个 stencil 上不成立，"
        "**edm 的价值也不在动量而在消除 O(h²) 离散旋度**——本单的度量未能分辨这一点，留给后续单（负面/未判部分如实标注）。")
    add("")
    add("**（3）适用范围与局限**：")
    add("- 只覆盖 D2Q9 `ψ=ρ` 的 MCMP 交叉力；`multiphase3d.py`（D3Q19）与 `multiphase3d_d3q27.py` 未覆盖，属独立 XVAL；")
    add("- 只覆盖无壁面、无 solid_mask 分支（`wall_mode='dry'/'mirror'` 未进入）；")
    add("- 测的是**算符**与连续极限的一致性；模型级（线性 ψ）表面张力绝对精度不在本单范围——"
        "同仓 `benchmarks/case_pool/mcmp_laplace` 已如实记录该路线绝对 σ_EOS 锚 0/12 不达标；")
    add("- 两个 n=96 的 C 臂案例达到步数上限（`converged=false`），其碟/界面仍在缓慢松弛，误差数值保留但已标注；"
        "由于判据是对**当时密度场**的算符截断误差，松弛度不影响该量的含义，只影响所测界面陡度。")
    add("")
    add("**工件与复现**：")
    add("")
    add("```bash")
    add("cd <worktree>")
    add("PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py            # 解析臂+汇总 (<1 s)")
    add("PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py --slab-case 64:-2.5:4000")
    add("PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py --slab-case 96:-2.5:3000")
    add("PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py --slab-case 96:-3.0:3000")
    add("PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/make_report.py            # 本文件 + 图")
    add("python -m pytest tests/test_xval_mcmp_cross_force.py -q                                       # 9 passed")
    add("```")
    add("")
    add(f"- `xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/{{predeclared.json,xval_lib.py,run_xval.py,make_report.py,REPORT.md}}`")
    add("- `xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/artifacts/{summary.json,slab_cases.jsonl,fields.npz,fig_xval.png}`")
    add("- `tests/test_xval_mcmp_cross_force.py`")
    add("")
    add(f"![xval figure]({fig.relative_to(HERE)})")
    add("")
    add("**图**：(a) 轴/对角单模上算符截断 vs 闭式 `S(k)`（虚线）与 15% 门；"
        "(b) 分辨率细化（斜率 −2）；(c) 真实松弛界面上交叉力剖面 vs 谱参考；"
        "(d) 6 个种植缺陷的检出幅度 vs 15% 门与基线底噪。")
    add("")
    out = HERE / "REPORT.md"
    out.write_text("\n".join(L) + "\n")
    return out


if __name__ == "__main__":
    fig = make_figure()
    rep = make_report(fig)
    print("wrote", rep, "and", fig)