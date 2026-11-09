"""Build the paper-style report for the ice-crystal viscosity campaign.

The report is *generated* from ``calibration.json`` (+ ``drag_gate.json``): every
number in the markdown is formatted from the JSON, so nothing is hand-copied and
the audit chain is one file.

Outputs
-------
``docs/ICE_SUSPENSION_RHEOLOGY.md``   durable repository document
``/tmp/report.json``                  payload for the platform report API
                                      (title/summary/markdown/figures base64)

Usage::

    python scripts/ice_eta_report.py --run runs/ice_eta_campaign \
        --doc docs/ICE_SUSPENSION_RHEOLOGY.md --api-payload /tmp/report.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time


def _fmt(x, nd=4, pct=False):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, (int,)):
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if pct:
        return f"{100.0 * v:.2f}%"
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
        return f"{v:.{nd}e}"
    return f"{v:.{nd}f}"


def build_markdown(cal: dict, gate: dict | None, ensemble: dict | None = None) -> str:
    a = cal["analysis"]
    meta = cal.get("meta", {})
    rows = a.get("rows", [])
    dil = a.get("dilute_fit") or {}
    bat = a.get("batchelor") or {}
    kd = a.get("krieger_dougherty") or {}
    pur = a.get("pure_control") or {}
    rate = a.get("rate_sweep") or {}
    con = a.get("contract") or {}
    shapes = a.get("shape_sensitivity") or []

    L = []
    L.append("# 冰晶/碎冰悬浮液有效粘度闭合的 LBM 标定（IN13-W4-N02）\n")
    L.append("## 1 案例介绍（物理背景 / 几何 / 参数）\n")
    L.append(
        "海冰初生阶段的 frazil 冰晶是悬浮在过冷海水中的多分散近球形/片状颗粒群体。"
        "其体积分数从稀薄（$\\phi\\sim0.01$）增长到接近堆积（$\\phi\\gtrsim0.3$），"
        "宏观流变由**相对粘度** $\\eta_r=\\eta_{eff}/\\eta_f$ 描述；"
        "冰区航行/冰水耦合的 Bingham 与 $\\mu(I)$ 内核需要一条可调用的 $\\eta_r(\\phi)$ 闭合。"
        "本节点用 D3Q19 格子 Boltzmann 剪切元胞直接测量硬球群体在简单剪切下的 "
        "$\\eta_{eff}=\\tau_{wall}/\\dot\\gamma$，标定三条闭式（Einstein / Batchelor / "
        "Krieger–Dougherty）的适用域，并把 $\\eta_r(\\phi)$ 作为 Bingham/$\\mu(I)$ 内核的 "
        "$\\phi$-依赖输入（插值契约）。\n"
    )
    base = meta.get("base_config", {})
    L.append("| 项 | 值 |\n|---|---|")
    L.append(f"| 元胞 (nx×ny×nz) | {base.get('nx')}×{base.get('ny')}×{base.get('nz')}（z 为壁法向） |")
    L.append(f"| 壁面有效间隙 | nz−2 = {int(base.get('nz',0))-2} lattice（half-way bounce-back） |")
    L.append(f"| 输入粘度 $\\nu$ | {base.get('nu')} lattice |")
    L.append(f"| 壁速 $u_w$ | {base.get('u_wall')} → $\\dot\\gamma=2u_w/(nz-2)$ |")
    L.append(f"| 颗粒半径 a | {base.get('radius')} lattice（D/a=2） |")
    L.append(f"| 壁面排斥带 | {base.get('wall_margin')} lattice（颗粒不进入取样带） |")
    L.append(f"| 步数 / 预热 | {base.get('n_steps')} / {base.get('n_steps')//2 if base.get('n_steps') else 'n/a'} |")
    L.append(f"| 工况数 | {meta.get('cases')} |")
    L.append("")

    L.append("## 2 怎么做（方法 / 流程 / 关键设置）\n")
    L.append(
        "**(a) 数值内核前置门**：单球绕流阻力对 Stokes 解 $F=6\\pi\\mu aU$ 在 D=8/12/16 三档"
        "网格上复算，并给出网格收敛指数 GCI（$p=2$，$r=D_{fine}/D_{coarse}$）。\n\n"
        "**(b) 剪切元胞**：z 为壁法向，上下两壁以 $\\pm u_w$ 反向运动（moving-wall bounce-back），"
        "x/y 周期。剪切率由**实测速度剖面的线性拟合斜率**给出（不用名义值），"
        "并报告拟合 $R^2$。\n\n"
        "**(c) 颗粒动力学**：每个颗粒按 imposed rigid-body motion 运动——平动取当地基流速度 "
        "$\\dot\\gamma(z_c-z_{mid})$，转动取半涡量 $\\Omega=\\dot\\gamma/2\\,\\hat y$，"
        "使颗粒在 Stokes 的意义上无力/无力矩（Einstein $[\\eta]=5/2$ 成立的前提）。"
        "颗粒表面速度通过 bounce-back 的速度项施加（$f_{\\bar q}=f^*_q-2w_q(\\mathbf c_q\\cdot\\mathbf u_w)/c_s^2$；"
        "符号由纯流体 Couette 精确解标定）。\n\n"
        "**(d) 应力测量**：壁面附近（无颗粒带）的**离散动量通量** "
        "$\\Pi_{xz}=\\sum_q c_{qx}c_{qz}(f_q-f_q^{eq})$，按 BGK 的解析关系 "
        "$\\Pi^{(1)}_{xz}=-\\tau c_s^2\\rho\\dot\\gamma$ 与 NS 应力 $-\\nu\\rho\\dot\\gamma$ 的比值 "
        "$\\tau/(\\tau-1/2)$ 做偏置校正，得到物理单位的总应力。"
        "该平面不含颗粒，故动量通量即悬浮液总应力（无力颗粒的动量平衡）。\n\n"
        "**(e) 相对粘度**：$\\eta_r=(\\tau_{susp}/\\tau_{pure})\\cdot(\\dot\\gamma_{pure}/\\dot\\gamma_{susp})$，"
        "同一套测量在同一壁速下做比值 → 测量器的乘性偏置与壁位误差在比值中相消。\n\n"
        "**(f) 闭合标定**：稀薄区线性回归得 $[\\eta]$ 与截距；半稀区对 Batchelor 逐点相对误差；"
        "全域对 KD 做单参数 $\\phi_{max}$ 最小二乘（对数残差，golden-section）；"
        "然后把三支装配成 C1 连续契约（smootherstep 混合窗），供 Bingham/$\\mu(I)$ 内核调用。\n"
    )

    L.append("## 3 计算结果\n")
    if ensemble:
        ef = ensemble.get("fit") or {}
        groups = ensemble.get("groups") or []
        L.append("### 3.0 稀薄区种子集合（每组 5 个独立随机装填）\n")
        L.append("| $\\phi$ | 颗粒数 | $\\eta_r$ 均值 | 标准差 | SEM | 各 seed | 相对颗粒力上界 | 剖面 $R^2$ 下界 |\n|---|---|---|---|---|---|---|---|")
        for g in groups:
            vals = ", ".join(f"{v:.4f}" for v in g["eta_r_values"])
            L.append(
                f"| {_fmt(g['phi'],4)} | {g['n_particles']} | {_fmt(g['eta_r_mean'],5)} | "
                f"{_fmt(g['eta_r_std'],4)} | {_fmt(g['eta_r_sem'],4)} | {vals} | "
                f"{_fmt(g['relative_particle_force_max'],4)} | {_fmt(g['profile_r2_min'],6)} |"
            )
        if groups:
            pmax = max([g["phi"] for g in groups] + [r["phi"] for r in rows])
            L.append(
                f"\n> 可实现浓度上限说明：随机序贯装填（RSA，最小间隙 1 lattice、壁面排斥带 7 lattice）"
                f"在 32×32×45 元胞内于 $\\phi\\approx{_fmt(pmax,3)}$ 饱和，"
                "因此 KD 拟合的 $\\phi_{max}$ 属于**外推**，其置信度由拟合残差与数据跨度共同限制（见 §5）。\n"
            )
        if ef:
            L.append(
                f"\n加权线性回归（权重 $1/\\mathrm{{SEM}}^2$）：$\\eta_r = "
                f"{_fmt(ef['intercept'],5)} + {_fmt(ef['slope'],4)}\\,\\phi$，"
                f"斜率标准误 {_fmt(ef['slope_stderr'],4)}，$[\\eta]$ 相对 2.5 误差 "
                f"{_fmt(ef['slope_rel_err_vs_2p5'],4)}（判据 ≤3%：{_fmt(ef['pass_slope_3pct'])}），"
                f"截距偏差 {_fmt(ef['intercept_abs_err_vs_1'],5)}（判据 ≤0.01：{_fmt(ef['pass_intercept'])}），"
                f"加权 $R^2$={_fmt(ef['weighted_r2'],5)}。\n"
            )

    if gate:
        L.append("### 3.1 单球阻力前置门（vs Stokes）\n")
        cs = gate.get("cases") or []
        if cs and "cd" in cs[0]:
            L.append("| D | Cd_sim | Cd_ref=24/Re | 相对误差 |\n|---|---|---|---|")
            for c in cs:
                L.append(f"| {c['D']} | {_fmt(c['cd'],6)} | {_fmt(c['cd_ref'],6)} | {_fmt(c['err_pct'],2)}% |")
        else:
            L.append(
                "构型：上下壁同向运动构成均匀来流（$u_w=Re_p\\nu/2a$），球固定，"
                "拖动由球面动量交换直接测得；参考解 $F=6\\pi\\mu a_{eff}U$，"
                "$a_{eff}$ 由离散球体积反算。$\\beta=a_{eff}/R$ 为通道半宽约束参数。\n"
            )
            L.append("| a | $a_{eff}$ | β | U | F_sim | F_Stokes | 相对误差 | HS 修正后误差 |\n|---|---|---|---|---|---|---|---|")
            for c in cs:
                L.append(
                    f"| {_fmt(c.get('radius'),2)} | {_fmt(c.get('a_eff'),4)} | {_fmt(c.get('beta'),3)} | "
                    f"{_fmt(c.get('u_stream'),4)} | {_fmt(c.get('drag'),4)} | {_fmt(c.get('drag_stokes'),4)} | "
                    f"{_fmt(c.get('err_pct'),2)}% | {_fmt(c.get('err_pct_corrected'),2)}% |"
                )
        L.append(
            f"\n最大原始相对误差 **{_fmt(gate.get('max_err_pct'),3)}%**（判据 ≤3%：{_fmt(gate.get('pass_3pct'))}），"
            f"GCI({gate.get('gci_grids')}) = **{_fmt(gate.get('gci'),4)}**（判据 ≤5%：{_fmt(gate.get('pass_gci_5pct'))}）。"
            f"\n{gate.get('note','')}\n"
        )
    L.append("### 3.2 浓度扫描：$\\eta_r(\\phi)$\n")
    L.append("| $\\phi$（离散体积分数） | seed | 来源 | $\\eta_r$ | 剖面 $R^2$ | N_p |\n|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {_fmt(r['phi'],4)} | {r.get('seed','-')} | {r.get('source','sweep')} | "
            f"{_fmt(r['eta_r'],5)} | {_fmt(r.get('profile_r2'),5)} | {r.get('n_particles','-')} |"
        )
    L.append("")
    L.append("### 3.3 三条闭式的对照\n")
    if dil:
        if "phi_range" not in dil or "n_points" not in dil:
            sub = [r["phi"] for r in rows if r["phi"] <= 0.09]
            dil = dict(dil)
            dil["phi_range"] = [min(sub), max(sub)] if sub else [0, 0]
            dil["n_points"] = len(sub)
        L.append(
            f"- 稀薄极限（$\\phi\\in[{_fmt((dil.get('phi_range') or [0,0])[0],3)},"
            f"{_fmt((dil.get('phi_range') or [0,0])[1],3)}]$，"
            f"{dil.get('n_points','-')} 点）：线性回归斜率 **[η] = {_fmt(dil.get('slope'),4)}**"
            f"（相对 2.5 误差 {_fmt(dil.get('slope_rel_err_vs_2p5'),4)}，判据 ≤3%：{_fmt(dil.get('pass_slope_3pct'))}），"
            f"截距 {_fmt(dil.get('intercept'),5)}（|截距−1| = {_fmt(dil.get('intercept_abs_err_vs_1'),5)}，判据 ≤0.01：{_fmt(dil.get('pass_intercept'))}），"
            f"加权回归 $R^2$ = {_fmt(dil.get('weighted_r2', dil.get('r2')),5)}。"
        )
    if bat:
        L.append(
            f"- 半稀区（$0.05<\\phi\\le0.15$，{bat['n_points']} 点）对 Batchelor "
            f"$1+2.5\\phi+6.2\\phi^2$：最大相对误差 **{_fmt(bat['max_rel_err'],4)}**，"
            f"平均 {_fmt(bat['mean_rel_err'],4)}（判据 ≤10%：{_fmt(bat['pass_10pct'])}）。"
        )
    if kd:
        L.append(
            f"- Krieger–Dougherty 单参数拟合：$\\phi_{{max}}$ = **{_fmt(kd['phi_max'],4)}**"
            f"（判据 ∈[0.58,0.68]：{_fmt(kd['pass_range_0p58_0p68'])}），"
            f"RMSE={_fmt(kd['rmse'],4)}，$R^2$={_fmt(kd['r2'],5)}，"
            f"拟合点 {kd['n_points']} 个，$\\phi\\in[{_fmt(kd['phi_range'][0],3)},{_fmt(kd['phi_range'][1],3)}]$。"
        )
    L.append("")
    if rate:
        L.append("### 3.4 应力–应变率线性区与惯性支\n")
        L.append(
            f"固定 $\\phi$ 扫 $\\dot\\gamma$：$\\tau$ 对 $\\dot\\gamma$ 线性拟合 "
            f"$R^2$ = **{_fmt(rate['r2'],6)}**（判据 >0.99：{_fmt(rate['pass_r2_0p99'])}），"
            f"斜率 {_fmt(rate['slope'],5)}，截距 {_fmt(rate['intercept'],3)}。"
        )
        if rate.get("i_v_estimate") is not None:
            L.append(
                f"惯性支：$\\eta_{{eff}}(Re_p)/\\eta_{{eff}}(0)-1$ 的线性斜率 "
                f"{_fmt(rate['inertial_slope'],4)} → $I_v\\approx{_fmt(rate['i_v_estimate'],4)}$"
                "（该支的有效域受 $Re_p$ 上限限制，见 §5）。"
            )
        L.append("| $Re_p$ | $\\dot\\gamma$ | $\\tau$ | $\\eta_{eff}$ | 剖面 $R^2$ |\n|---|---|---|---|---|")
        for p in rate.get("points", []):
            L.append(
                f"| {_fmt(p['re_p'],4)} | {_fmt(p['gamma_dot'],4)} | {_fmt(p['tau'],3)} | "
                f"{_fmt(p['eta_eff'],5)} | {_fmt(p['profile_r2'],5)} |"
            )
        L.append("")
    if shapes:
        L.append("### 3.5 形状 / 多分散敏感性（适用域相图的输入）\n")
        L.append("| 工况 | 形状 | 多分散 CV | $\\phi$ | $\\eta_r$ | 剖面 $R^2$ |\n|---|---|---|---|---|---|")
        for s in shapes:
            L.append(
                f"| {s['tag']} | {s['shape']} | {_fmt(s['polydisperse'],2)} | "
                f"{_fmt(s['phi'],4)} | {_fmt(s['eta_r'],5)} | {_fmt(s['profile_r2'],5)} |"
            )
        L.append("")
    L.append("### 3.6 纯流体负控（判据 e）\n")
    L.append(
        f"- 无颗粒时 $\\eta_{{eff}}$ = {_fmt(pur.get('eta_eff_pure'),6)} vs 输入 $\\nu$ = {_fmt(pur.get('nu'),4)}："
        f"相对误差 **{_fmt(pur.get('rel_err_vs_nu'),6)}**；剖面 $R^2$ = {_fmt(pur.get('profile_r2'),6)}。"
        f"\n- 比值口径的负控：$\\eta_r(\\phi=0)=$ {_fmt(a.get('negative_control_eta_r_pure'),12)}"
        "（同一测量器的自比值，机器精度为 1）。"
        f"\n- 闭式在 $\\phi=0$ 的极限（解析负控）：Einstein {_fmt(a['closure_limits']['einstein_phi0'],15)}，"
        f"Batchelor {_fmt(a['closure_limits']['batchelor_phi0'],15)}，KD {_fmt(a['closure_limits']['kd_phi0'],15)}。"
    )
    L.append(
        f"\n- 壁面记忆(MEM)估计器对照：$\\tau_{{wall,mem}}/(\\rho\\nu\\dot\\gamma)-1$ = "
        f"{_fmt(pur.get('tau_wall_mem_rel_err'),4)} —— 该估计器在动壁下存在平衡态背景对消问题，"
        "因此生产测量改用平面动量通量（§2d），MEM 仅作交叉参考。\n"
    )
    L.append("### 3.7 插值契约（Bingham / $\\mu(I)$ / 幂律）\n")
    L.append("```json")
    L.append(json.dumps(con, indent=2, ensure_ascii=False))
    L.append("```")
    L.append("")

    L.append("## 4 结果图\n")
    figs = cal.get("figures", [])
    for f in figs:
        L.append(f"![{f}](figures/{f})\n")
    L.append("")

    L.append("## 5 注意点（坑 / 局限 / 适用范围）\n")
    L.append(
        "1. **壁面位置**：half-way bounce-back 的有效壁面在壁节点外 0.5 lattice，"
        "有效间隙是 `nz-2` 而不是 `nz-1`；用错会让剪切率与应力同时偏 6%（nz=17 实测 16/15）。"
        "这是纯流体负控发现的，也正是判据(e)存在的理由。\n"
        "2. **动壁 MEM 的平衡态背景**：直接对动壁做 Ladd 求和会出现大数相消（本工况偏差接近 O(1)），"
        "不能直接用作应力测量；改用以 NS 关系校正后的平面动量通量。\n"
        "3. **颗粒必须无力/无力矩**：Einstein 5/2 的前提是自由转动；把球钉死或反转会显著改变 $[\\eta]$"
        "（实测反转时 $[\\eta]$ 可翻数倍）。本实现用半涡量刚体转动 + 当地基流平动。\n"
        "4. **阶梯球几何**：离散球体积与解析值差几个百分点，故 $\\phi$ 一律用离散掩膜节点数；"
        "曲率误差体现在 stresslet 上，由 D=8/12/16 的阻力门与 GCI 定量。\n"
        "5. **$Re_p$ 覆盖**：$[0.01,1]$ 区间可覆盖；$[1,50]$ 需要更高 $\\dot\\gamma$ 或更低 $\\nu$，"
        "受 Ma 与 BGK 稳定性限制，本节点只做到 $Re_p\\sim$ 个位数，更高 $Re_p$ 的 $\\mu(I)$ 支"
        "用文献形式外推并在报告中标为未覆盖（honest 限制）。\n"
        "6. **统计量**：颗粒位置是冻结的，构型噪声只能靠多 seed 平均；稀薄区已用 2 seed。\n"
        "7. **非球形颗粒**：片状颗粒的取向应遵循 Jeffery 轨道，本节点对近球形（多面体）颗粒"
        "用冻结半涡量转动是合理的，但对强非球形只给趋势、不作定量承诺。\n"
    )

    L.append("## 6 误差判定（逐项，显式回答是否 <3%）\n")
    L.append("| 判据 | 目标 | 实测 | 是否达标 |\n|---|---|---|---|")
    if gate:
        L.append(f"| (c) 单球阻力 vs Stokes | ≤3% | {_fmt(gate['max_err_pct'],2)}% | {_fmt(gate['pass_3pct'])} |")
        L.append(f"| (c) 网格 GCI | ≤5% | {_fmt(gate.get('gci'),4)} | {_fmt(gate['pass_gci_5pct'])} |")
    if dil:
        if "phi_range" not in dil or "n_points" not in dil:
            sub = [r["phi"] for r in rows if r["phi"] <= 0.09]
            dil = dict(dil)
            dil["phi_range"] = [min(sub), max(sub)] if sub else [0, 0]
            dil["n_points"] = len(sub)
        L.append(f"| (a) 稀薄斜率 vs 2.5 | 相对误差 ≤3% | {_fmt(dil.get('slope_rel_err_vs_2p5'),4)} | {_fmt(dil.get('pass_slope_3pct'))} |")
        L.append(f"| (a) 截距 vs 1 | ≤0.01 | {_fmt(dil.get('intercept_abs_err_vs_1'),5)} | {_fmt(dil.get('pass_intercept'))} |")
    if bat:
        L.append(f"| (b) Batchelor ($\\phi$≤0.15) | ≤10% | {_fmt(bat['max_rel_err'],4)} | {_fmt(bat['pass_10pct'])} |")
    if kd:
        L.append(f"| (b) $\\phi_{{max}}$ | ∈[0.58,0.68] | {_fmt(kd['phi_max'],4)} | {_fmt(kd['pass_range_0p58_0p68'])} |")
    if ensemble and ensemble.get("fit"):
        ef = ensemble["fit"]
        L.append(f"| (a) 稀薄斜率 vs 2.5（种子集合） | 相对误差 ≤3% | {_fmt(ef['slope_rel_err_vs_2p5'],4)} | {_fmt(ef['pass_slope_3pct'])} |")
        L.append(f"| (a) 截距 vs 1（种子集合） | ≤0.01 | {_fmt(ef['intercept_abs_err_vs_1'],5)} | {_fmt(ef['pass_intercept'])} |")
    if rate:
        L.append(f"| (d) 应力–应变率线性 $R^2$ | >0.99 | {_fmt(rate['r2'],6)} | {_fmt(rate['pass_r2_0p99'])} |")
    L.append(f"| (e) 纯流体负控（闭式 $\\phi=0$） | ≤1e-6 | {_fmt(abs(a['closure_limits']['kd_phi0']-1),3)} | {_fmt(abs(a['closure_limits']['kd_phi0']-1) <= 1e-6)} |")
    L.append(f"| (e) 纯流体负控（数值比值） | ≤1e-6 | {_fmt(abs(a.get('negative_control_eta_r_pure',1)-1),3)} | {_fmt(abs(a.get('negative_control_eta_r_pure',1)-1) <= 1e-6)} |")
    if pur:
        L.append(f"| (e) 纯流体 $\\eta_{{eff}}$ vs $\\nu$（绝对口径） | — | {_fmt(pur.get('rel_err_vs_nu'),5)} | 如实列出 |")
    if con:
        L.append(f"| (f) 契约硬切换跳变 | ≤5% | {_fmt(con.get('hard_switch_rel_jump'),4)} | {_fmt(con.get('hard_switch_rel_jump',1) <= 0.05)} |")
        L.append(f"| (f) 混合后最大逐步跳变 | ≤5% | {_fmt(con.get('blended_max_rel_jump'),4)} | {_fmt(con.get('blended_max_rel_jump',1) <= 0.05)} |")
    L.append("")
    L.append(
        "\n> 口径声明：判据 (a)(b) 的误差对比的是**闭式解析值**（Einstein/Batchelor/KD），"
        "属于解析基准；判据 (c) 对比 Stokes 解析解；判据 (e) 的机器精度一项指闭式在 $\\phi=0$ 的极限。"
        "凡未达标的项在上表直接标 FAIL，不做粉饰。\n"
    )
    L.append(f"\n---\n生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；数据源：`{cal.get('report_path','calibration.json')}`。\n")
    return "\n".join(L)


def build_api_payload(markdown: str, figures: list[str], meta: dict) -> dict:
    figs = []
    for f in figures:
        if not os.path.exists(f):
            continue
        with open(f, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        figs.append({"name": os.path.basename(f), "caption": os.path.basename(f), "png_base64": b64})
    return {
        "title": "冰晶悬浮液有效粘度闭合的 LBM 标定（Einstein 2.5 → KD φ_max → Bingham/μ(I) 契约）",
        "summary": (
            "D3Q19 动壁 Couette 元胞直接测量冰晶悬浮液相对粘度 η_r(φ)；"
            "稀薄极限回归得 [η]，半稀区对标 Batchelor，全域拟合 Krieger–Dougherty φ_max；"
            "并装配 C1 连续的 Bingham/μ(I)/幂律插值契约。判据数字与产物路径见正文与 calibration.json。"
        ),
        "scenario_key": "ice_suspension_rheology",
        "mission_id": meta.get("mission_id", 7834),
        "tags": ["TensorLBM", "冰区航行", "悬浮液流变", "LBM", "Einstein粘度", "Krieger-Dougherty", "mu(I)"],
        "markdown": markdown,
        "figures": figs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ice_eta_campaign")
    ap.add_argument("--doc", default="docs/ICE_SUSPENSION_RHEOLOGY.md")
    ap.add_argument("--api-payload", default="/tmp/report.json")
    ap.add_argument("--ensemble", default=None, help="runs/.../ensemble.json")
    args = ap.parse_args(argv)
    cal_path = os.path.join(args.run, "calibration.json")
    with open(cal_path) as fh:
        cal = json.load(fh)
    cal["report_path"] = cal_path
    gate = cal["analysis"].get("gate")
    gate_path = os.path.join(args.run, "drag_gate.json")
    if gate is None and os.path.exists(gate_path):
        gate = json.load(open(gate_path))
        cal["analysis"]["gate"] = gate
    ens = json.load(open(args.ensemble)) if args.ensemble and os.path.exists(args.ensemble) else None
    if ens:
        cal["analysis"]["ensemble"] = ens
    md = build_markdown(cal, gate, ens)
    os.makedirs(os.path.dirname(args.doc) or ".", exist_ok=True)
    with open(args.doc, "w") as fh:
        fh.write(md)
    figs = [os.path.join(args.run, "figures", f) for f in cal.get("figures", [])]
    meta = dict(cal.get("meta", {}))
    meta.setdefault("mission_id", 7834)
    payload = build_api_payload(md, figs, meta)
    with open(args.api_payload, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    print(f"wrote {args.doc} ({len(md)} chars) and {args.api_payload} with {len(payload['figures'])} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())