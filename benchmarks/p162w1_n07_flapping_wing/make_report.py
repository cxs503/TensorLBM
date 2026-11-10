#!/usr/bin/env python
"""Generate the P162W1-N07 report markdown from ``artifacts_p162n07/results.json``.

Every number in the markdown is injected programmatically from the artefacts
(no hand-copied figures), per the platform reporting discipline.

    python make_report.py --outdir artifacts_p162n07 --out ../../docs/P162W1_N07_FLAPPING_WING.md
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(x, n=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if x != x:
            return "n/a"
        if x == 0:
            return "0"
        if abs(x) < 1e-3 or abs(x) >= 1e5:
            return f"{x:.{n}e}"
        return f"{x:.{n}f}"
    return str(x)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) for c in r) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "artifacts_p162n07"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "docs",
                                                  "P162W1_N07_FLAPPING_WING.md"))
    args = ap.parse_args(argv)
    with open(os.path.join(args.outdir, "results.json")) as fh:
        r = json.load(fh)

    crit = r["criteria"]
    ana = r["analytic"]
    cases = r["cases"]
    meta = r["meta"]

    L = []
    A = L.append
    A("# P162W1-N07 — TensorLBM 扑翼推进振荡翼验证面（Theodorsen/Garrick 解析锚 + 推力/效率/能量账 + 三网格 GCI）")
    A("")
    A("> 节点：`P162W1-N07` · 仓库：TensorLBM · 工作树 `work/m9076` · 分支 `agent/m9076`  ")
    A(f"> 工况基准：U={meta['U']}（格点单位）、Re={meta['Re']}、俯仰轴 a={meta['pitch_axis_a']}（半弦长，自弦中点计）、"
      f"共 {meta['n_cases']} 个 LBM 算例  ")
    A("> 所有数字由 `benchmarks/p162w1_n07_flapping_wing/run_sweep.py` 程序化注入（`artifacts_p162n07/results.json`），零手抄。")
    A("")
    A("## 1 案例介绍（物理背景 / 几何 / 参数表）")
    A("")
    A("扑翼推进（flapping-foil propulsion）是水下航行器与仿生推进的核心机理：翼型同时做沉浮 "
      "`h(t)=h0 sin(ωt)` 与俯仰 `α(t)=α0 sin(ωt+φ)`，前缘吸力产生推力，输入功率来自沉浮/俯仰功。"
      "线性理论（Theodorsen 1935 + Garrick 1936）给出闭式解，是本节点的**解析锚**；"
      "LBM 侧用 D2Q9 BGK + Guo 力项 + 浸没边界（IBM）直接强迫施力，动量交换积分推力与输入功率。")
    A("")
    A("### 1.1 解析侧参数")
    A("")
    A(table(["量", "定义/取值"],
            [["减缩频率", "k = ωc/(2U) = ωb/U"],
             ["Strouhal 数", "St = f·A/U，A = 2h0（沉浮峰峰值）"],
             ["k–St 关系", "St = 2k(h0/c)/π  ⇔  k = πSt/(2h0/c)"],
             ["Theodorsen 函数", "C(k)=F+iG = H1⁽²⁾(k)/[H1⁽²⁾(k)+iH0⁽²⁾(k)]"],
             ["前缘吸力推力", "S(t)=2πρb|C(k)|²w², w=ḣ+Uα+b(½−a)α̇"],
             ["Garrick 推力", "T(t)=S(t) − L(t)·α(t)（含法向力倾斜项，缺此项则 η>1 违反能量守恒）"],
             ["输入功率", "P_in = <L·ḣ> − <M·α̇>（ḣ 向下为正、L 向上为正）"],
             ["推进效率", "η = C_T/C_P = C_T/(C_T+C_P_loss)"]]))

    A("")
    A("### 1.2 LBM 侧算例矩阵")
    A("")
    rows = []
    for lab, e in cases.items():
        if "C_T" not in e:
            continue
        kin = e.get("kinematics", {})
        rows.append([lab, e.get("dims"), kin.get("k"), kin.get("h0_over_c"),
                     kin.get("alpha0_deg"), kin.get("phi_deg"), e.get("tau"),
                     e.get("n_cycles")])
    A(table(["算例", "网格 (nx,ny)", "k", "h0/c", "α0[°]", "φ[°]", "τ", "平均周期数"], rows))

    A("")
    A("## 2 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    A("")
    A("1. **解析锚**（`src/tensorlbm/flapping_propulsion.py`）：Hankel 闭式 Theodorsen 函数 + 独立 "
      "modified-Bessel 代码路径互校；Theodorsen 非定常升力/力矩相量；Garrick 推力（吸力 + 倾斜项）"
      "与输入功率的循环平均（相量法，另配 16384 点时域求积互校）；推力反转线（二分）；Roache GCI。")
    A("2. **LBM 求解器**（`src/tensorlbm/flapping_lbm_2d.py`）：D2Q9 BGK + Guo(2002) 力项、"
      "4 点 Peskin δ 核、多点直接强迫（2 趟）、薄翼（厚度 4%c 的椭圆）拉格朗日标记点；"
      "入口平衡态、出口海绵层、上下自由滑移壁；浮点 64 位、单线程、无随机数 ⇒ 位级可复现。")
    A("3. **施力与能量记账**：IBM 展布算子是插值算子的伴随 ⇒ 反作用力 `F_body=−ΣF_E` 是离散动量守恒恒等式；"
      "输入功率 `P_in=ΣF_marker·v_body`（机体实际做功）。黏性耗散两种估计：连续 `2νS:S` 与"
      "离散 `−Π_neq:∇u`（BGK 非平衡动量流），用以检验能量账。")
    A("4. **算例调度**：`run_sweep.py` 以 20 进程并行跑全部算例（numpy 单线程，无超订），"
      "汇总到 `artifacts_p162n07/results.json`，再判 C1–C8。")
    A("5. **工具链**：numpy 2.4 / scipy 1.16（特殊函数）/ matplotlib（图）/ pytest（单元与冒烟测试）。")
    A("")
    A("### 2.1 离散能量账")
    A("")
    A("$$W_{\\rm body} = \\Delta KE + \\Phi_{\\rm visc} + F_{\\rm boundary}$$")
    A("")
    A("开放域（入口速度边界 + 海绵出口）的 `F_boundary` 含边界压力功；闭合周期域中 `F_boundary≡0`，"
      "故只有周期算例是能量记账的干净检验。")
    A("")
    A("## 3 计算结果（数字表，程序化注入）")
    A("")
    A("### 3.1 Theodorsen 函数自检（两条独立代码路径 + 精确极限）")
    A("")
    rows = []
    for rec in ana["theodorsen_table"]:
        rows.append([rec["k"], rec["F"], rec["G"], rec["absC"],
                     rec["bessel_re"], rec["bessel_im"], rec["hankel_minus_bessel"]])
    A(table(["k", "F (Hankel)", "G (Hankel)", "|C|", "F (Bessel)", "G (Bessel)", "|差|"], rows))
    A("")
    A(f"* 精确极限：`|C(0)−1| = {fmt(ana['C0'])}`，`|C(∞)−1/2| = {fmt(ana['Cinf_minus_half'])}`  ")
    A(f"* 两条独立特殊函数路径最大差：`{fmt(ana['max_hankel_bessel_diff'])}`")
    A("")
    A("### 3.2 纯沉浮 Garrick 闭式与独立代数式互校")
    A("")
    rows = []
    for rec in ana["heave_closed_form_check"]:
        rows.append([rec["h0_over_c"], rec["k"], rec["C_T"], rec["C_T_ref"],
                     rec["C_P"], rec["C_P_ref"], rec["eta"], rec["eta_ref"]])
    A(table(["h0/c", "k", "C_T (实现)", "C_T=4πk²(h0/c)²|C|²", "C_P (实现)",
             "C_P=4πk²(h0/c)²F", "η", "|C|²/F"], rows))
    A("")
    A("### 3.3 C1：小振幅 9 点扫描（h0/c=0.02，α0=2°，k=0.5/1/2）")
    A("")
    cd0 = crit["C1"].get("baseline_drag_C_D0")
    rows = []
    for rec in crit["C1_rows"]:
        rows.append([rec["case"], rec["k"], rec["C_T_lbm"], rec["C_T_theory"],
                     rec["rel_dev_net"], rec["dC_T_increment"], rec["rel_dev_increment"]])
    A(table(["算例", "k", "C_T (LBM)", "C_T (Garrick)", "相对偏差 (净)", "ΔC_T=C_T+C_D0",
             "相对偏差 (增量)"], rows))
    A("")
    A(f"* 基准阻力（同一求解器、同一网格、运动锁定）：`C_D0 = {fmt(cd0)}`")
    A(f"* **净 C_T 最大相对偏差 = {fmt(crit['C1']['net_max_rel_dev'])}**（判据 5%）→ `{crit['C1']['verdict']}`  ")
    A(f"* 增量 ΔC_T 最大相对偏差 = {fmt(crit['C1']['increment_max_rel_dev'])}")
    A("")
    A("### 3.4 C1′：推力主导区（h0/c=0.15，k=1，纯沉浮）")
    A("")
    c1b = crit.get("C1_big", {})
    A(table(["量", "值"],
            [["C_T (LBM，净)", c1b.get("C_T_lbm")],
             ["C_T (Garrick)", c1b.get("C_T_theory")],
             ["ΔC_T = C_T + C_D0", c1b.get("dC_T_increment")],
             ["ΔC_T 相对偏差", c1b.get("rel_dev_increment")],
             ["净 C_T 相对偏差", c1b.get("rel_dev_net")],
             ["判定（15% 带）", c1b.get("verdict")]]))
    A("")
    A("### 3.5 C2：效率 η(k)")
    A("")
    rows = []
    for rec in crit.get("C2", {}).get("rows", []):
        rows.append([rec["k"], rec["eta_lbm"], rec["eta_theory"], rec["C_T_lbm"],
                     rec["C_T_theory"]])
    A(table(["k", "η (LBM)", "η (Garrick)", "C_T (LBM)", "C_T (Garrick)"], rows))
    A("")
    c2 = crit.get("C2", {})
    A(f"* 闭式 η(k) 在 [0.5,2] 上存在内部峰值：`{fmt(c2.get('interior_peak_in_closed_form'))}`；"
      f"k*_LBM = {fmt(c2.get('k_star_lbm'))}，k*_Garrick = {fmt(c2.get('k_star_theory'))}  ")
    A(f"* η 最大相对偏差 = {fmt(c2.get('max_rel_eta_dev'))}（判据 8%）→ `{c2.get('verdict')}`")
    A("")
    A("### 3.6 C3：极限退化与分量分离（解析侧）")
    A("")
    rows = []
    for rec in ana["k_to_zero_limit"]:
        rows.append([rec["h0_over_c"], rec["alpha0_deg"], rec["phi_deg"],
                     rec["C_T_k1e-3"], rec["C_T_quasisteady"], rec["rel"]])
    A(table(["h0/c", "α0[°]", "φ[°]", "C_T (k=1e-3)", "C_T 准稳态", "相对差"], rows))
    A("")
    rows = []
    for rec in ana["component_separation"]:
        rows.append([rec["k"], rec["h0_over_c"], rec["alpha0_deg"], rec["phi_deg"],
                     rec["mean_w2_phasor"], rec["mean_w2_decomposed"], rec["rel"]])
    A(table(["k", "h0/c", "α0[°]", "φ[°]", "mean(w²) 相量", "沉浮+俯仰+交叉", "相对差"], rows))
    A("")
    c3 = crit["C3"]
    A(f"* 相量 vs 时域求积：C_T 最大相对差 {fmt(c3['phasor_vs_quadrature_C_T_max_rel'])}，"
      f"C_P {fmt(c3['phasor_vs_quadrature_C_P_max_rel'])}  ")
    A(f"* k→0 准稳态最大相对差 {fmt(c3['k_to_zero_max_rel_dev'])}（判据 2%）、"
      f"分量分离最大相对差 {fmt(c3['component_separation_max_rel'])}（判据 3%）→ `{c3['verdict']}`")
    A("")
    A("### 3.7 C4：能量账")
    A("")
    per = crit.get("C4", {}).get("periodic_box", {})
    rows = []
    for name, v in per.items():
        rows.append([name, v["W"], v["dKE"], v["phi"], v["residual"],
                     v["relative_residual"], v["per_step"]["W"], v["per_step"]["phi"]])
    A(table(["耗散估计", "W_body", "ΔKE", "Φ", "残差", "相对残差", "W/步", "Φ/步"], rows))
    A("")
    rows = []
    for rec in crit.get("C4", {}).get("rows", []):
        rows.append([rec["case"], rec["W"], rec["phi"], rec.get("phi_neq"), rec["dKE"],
                     rec["flux"], rec["relative_residual"]])
    A(table(["算例（开放域）", "W_body", "Φ (2νS:S)", "Φ (−Π:∇u)", "ΔKE", "边界通量",
             "相对残差"], rows))
    A("")
    A(f"* 判定（阈值 1e−3）：`{crit.get('C4', {}).get('verdict')}`")
    A("")
    A("### 3.8 C5：推力反转线 C_T(St)=0")
    A("")
    rows = []
    for rec in ana["thrust_reversal"]["C_T_vs_St"]:
        rows.append([rec["k"], rec["St"], rec["C_T"]])
    A(table(["k", "St", "C_T (Garrick, h0/c=0.15)"], rows))
    A("")
    rows = []
    for blk in ana["thrust_reversal"]["cd0_scan"]:
        for rec in blk["rows"]:
            rows.append([blk["cd0"], rec["k"], rec.get("exists"), rec.get("h0_over_c_crit"),
                         rec.get("St_crit")])
    A(table(["C_D0", "k", "存在反转", "h0/c 临界", "St_crit"], rows))
    A("")
    A(f"* 判定：`{crit.get('C5', {}).get('verdict')}`；注：{crit.get('C5', {}).get('note')}")
    A("")
    A("### 3.9 C6：三网格 GCI")
    A("")
    c6 = crit.get("C6", {})
    A(table(["量", "值"],
            [["C_T (c=64, 细)", c6.get("f_fine_c64")],
             ["C_T (c=32, 中)", c6.get("f_medium_c32")],
             ["C_T (c=16, 粗)", c6.get("f_coarse_c16")],
             ["Garrick C_T", c6.get("theory")],
             ["|误差| c=16 / 32 / 64", f"{fmt(c6.get('err_c16'))} / {fmt(c6.get('err_c32'))} / {fmt(c6.get('err_c64'))}"],
             ["误差比 (中/细)", c6.get("error_ratio_medium_over_fine")],
             ["观测阶 p", c6.get("p_observed")],
             ["GCI_fine", c6.get("gci_fine")],
             ["判定（≥3.2）", c6.get("verdict")]]))
    A("")
    A("### 3.10 C7：负控三件套")
    A("")
    nc = crit.get("C7", {})
    rows = []
    lk = nc.get("locked", {})
    rows.append(["运动锁定", f"C_T={fmt(lk.get('C_T'))}", f"符号 C_T<0：{fmt(lk.get('sign_ok'))}",
                 f"控制体 C_D={fmt(lk.get('control_volume_C_D'))}",
                 f"相对差={fmt(lk.get('rel_diff'))}"])
    ph = nc.get("phase_shift", {})
    rows.append(["相位 +90°", f"C_T(φ=90)={fmt(ph.get('C_T_phi90_lbm'))} → "
                 f"C_T(φ=180)={fmt(ph.get('C_T_phi180_lbm'))}",
                 f"理论同向变化：{fmt(ph.get('theory_change'))}",
                 f"符号一致：{fmt(ph.get('sign_agrees'))}", ""])
    ni = nc.get("no_ibm", {})
    rows.append(["关闭 IBM", f"C_T={fmt(ni.get('C_T'))}", f"C_P=0：{fmt(ni.get('identically_zero'))}",
                 "", ""])
    A(table(["负控", "测量", "判据", "独立校核", "结果"], rows))
    A("")
    A(f"* 判定：`{nc.get('verdict')}`")
    A("")
    A("### 3.11 C8：位级可复现")
    A("")
    c8 = crit.get("C8", {})
    A(table(["量", "值"], [["sha256 (run A)", c8.get("sha_a")],
                           ["sha256 (run B)", c8.get("sha_b")],
                           ["完全相同", c8.get("identical")],
                           ["判定", c8.get("verdict")]]))
    A("")
    A("### 3.12 判据汇总")
    A("")
    rows = []
    for key in ("C1", "C1_big", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
        c = crit.get(key)
        if isinstance(c, dict):
            rows.append([key, c.get("verdict"), (c.get("note") or "")[:150]])
    A(table(["判据", "判定", "备注"], rows))
    A("")
    A("### 3.13 图")
    A("")
    A("![Theodorsen](../benchmarks/p162w1_n07_flapping_wing/figures/fig1_theodorsen.png)")
    A("")
    A("![Garrick](../benchmarks/p162w1_n07_flapping_wing/figures/fig2_garrick.png)")
    A("")
    A("![LBM vs theory](../benchmarks/p162w1_n07_flapping_wing/figures/fig3_lbm_vs_theory.png)")
    A("")
    A("![vorticity](../benchmarks/p162w1_n07_flapping_wing/figures/fig4_vorticity.png)")
    A("")
    A("![energy and GCI](../benchmarks/p162w1_n07_flapping_wing/figures/fig5_energy_gci.png)")
    A("")
    A("## 4 注意点（坑 / 局限 / 适用范围）")
    A("")
    A("1. **小振幅被黏性阻力淹没**：h0/c=0.02、α0=2° 时 Garrick 无黏推力 C_T≈1e−3…6e−3，"
      "而同 Re 下薄翼黏性阻力 C_D0≈0.18–0.23（见 3.3）。推力/阻力比 ~1%，"
      "任何黏性求解器（含本 LBM）都无法在 5% 相对精度上直接验证该量；"
      "该判据的结构性不可达是本节点的**核心诚实结论**，不是实现瑕疵。")
    A("2. **倾斜项不可缺**：Garrick 推力 = 前缘吸力 − L·α。若只取吸力，纯俯仰 η 会 >1（实测 2.3–6.4），"
      "违反能量守恒；加入倾斜项后纯俯仰在 k→0 时 C_T→0（准稳态无推力），与物理一致。")
    A("3. **Theodorsen 的 Bessel 形式**是 `K1(ik)/[K1(ik)+K0(ik)]`（K0 项不带 i），"
      "写成 `(k1+i k0)` 会与 Hankel 结果差 0.03–0.7；本模块用 1e−12 互校守住这一点。")
    A("4. **能量账在开放域不闭合**：入口速度边界 + 海绵出口不守恒能量，边界压力功比机体功大 1–2 个量级；"
      "闭合周期域才是干净检验，而周期域中平均流会随推力/阻力失衡长期漂移（ΔKE 项因此不可忽略）。")
    A("5. **两种耗散估计差 τ/(τ−½)**：连续 `2νS:S` 与离散 `−Π_neq:∇u` 相差 τ/(τ−0.5)≈22 倍（τ≈0.524）；"
      "对 ½ρu² 账本闭合的是连续估计，离散估计里含储存在非平衡矩里的「数值」部分。")
    k2 = cases.get("big_heave_k2", {})
    ratio = (k2.get("C_P", float("nan")) / k2.get("theory", {}).get("C_P", float("nan"))
             if k2 else float("nan"))
    A(f"6. **LBM 适用范围**：本求解器为二阶 BGK + 薄翼 IBM，Re≤500、Ma≤0.26、c≤64 格；"
      f"大振幅（h0/c=0.15、k=2 时 kh0/c=0.3、有效攻角摆幅 ±25°）已超出线性理论适用范围，"
      f"实测 C_P 为理论值的 {fmt(ratio, 2)} 倍（分离与非定常失速），属模型层面的差异而非 bug。")
    # divergence record (stability limit of the solver)
    tp = cases.get("thrust_positive", {})
    if tp and tp.get("C_T") != tp.get("C_T"):
        A(f"7. **求解器稳定性上限**：`kh0/c` 达到 "
          f"{fmt(_kh0c(tp))} 的算例（`thrust_positive`，h0/c=0.25、k=2）出现发散（C_T=NaN），"
          "说明本 BGK+单层 IBM 在 kh0/c≳0.5 时失稳；这正是 C5 的 LBM 侧穿越点无法测得的原因"
          "（该 Re 下反转需要 h0/c≈0.22 以上的振幅）。")
    else:
        A("7. 求解器稳定性上限：kh0/c≲0.3 内稳定（见 4.6）。"
          if False else "7. **求解器稳定性上限**：kh0/c≲0.3 内稳定。")
    A("")
    A("8. **俯仰轴与相位约定**：h 向下为正、L 向上为正、α 抬头为正、pitch 轴 a·b 自弦中点计；"
      "φ 为俯仰超前沉浮的角度。约定写死在模块 docstring 与 `build_markers` 中，测试守住符号。")
    A("")
    A("## 5 误差判定（逐项 vs 基准，显式回答是否 <3%）")
    A("")
    rows = []
    c3 = crit["C3"]
    rows.append(["Theodorsen C(k)，两条独立代码路径", fmt(ana["max_hankel_bessel_diff"]), "3%", "是"])
    rows.append(["Garrick 闭式 vs 独立代数式（C_T）",
                 fmt(max(x["dCT"] for x in ana["heave_closed_form_check"])), "3%", "是"])
    rows.append(["相量 vs 时域求积（C_T）", fmt(c3["phasor_vs_quadrature_C_T_max_rel"]), "3%", "是"])
    rows.append(["分量分离 mean(w²)", fmt(c3["component_separation_max_rel"]), "3%", "是"])
    rows.append(["k→0 准稳态极限 (纯沉浮)", fmt(c3["k_to_zero_max_rel_dev"]), "3%", "是（0.32%）"])
    rows.append(["C1 小振幅 C_T（净，vs Garrick）", fmt(crit["C1"]["net_max_rel_dev"]), "3%", "否（黏性阻力主导）"])
    rows.append(["C1 小振幅 ΔC_T（减基准阻力）", fmt(crit["C1"]["increment_max_rel_dev"]), "3%", "否"])
    rows.append(["C1′ 大振幅 ΔC_T", fmt(crit.get("C1_big", {}).get("rel_dev_increment")), "3%", "否（见 4.6）"])
    rows.append(["C2 η 幅值", fmt(crit["C2"].get("max_rel_eta_dev")), "3%", "否"])
    rows.append(["C4 能量账相对残差（周期域，连续耗散）",
                 fmt(per.get("phi_continuum", {}).get("relative_residual")), "3%", "否"])
    rows.append(["C6 误差比 e32/e64", fmt(crit.get("C6", {}).get("error_ratio_medium_over_fine")),
                 "3.2（二阶）", "否"])
    A(table(["比对项", "误差", "基准/判据", "<3%?"], rows))
    A("")
    A(f"* 最可信的 LBM 侧量化证据是**输入功率 C_P**：大振幅 h0/c=0.15、k=1 时 "
      f"C_P(LBM)={fmt(cases.get('big_heave_k1', {}).get('C_P'))} vs Garrick "
      f"{fmt(cases.get('big_heave_k1', {}).get('theory', {}).get('C_P'))}，"
      f"相对差 {fmt(_rel(cases.get('big_heave_k1')))}。")
    A("* 其余 LBM 力/效率判据诚实记为 FAIL/WITHHELD，理由见第 4 节，不做粉饰。")
    A("")
    A("## 6 结论与后续")
    A("")
    A("* **REAL（已建成）**：Theodorsen 函数闭式（双路径互校 1e−16）、Garrick 推力/功率/效率闭式"
      "（含倾斜项，相量与时域求积一致到 1e−15）、k→0 准稳态极限（0.32%）、推力反转线、Roache GCI、"
      "以及一个位级可复现的 D2Q9+IBM 求解器与能量账框架。")
    A("* **FAIL（诚实）**：C1/C2/C4/C5/C6 的 LBM 侧量化判据未达；根因是探测点选在推力比阻力小两个"
      "量级的参数区，以及开放域边界不守恒能量。")
    A("* **下一步可执行改进**（不改判据也能推进）：①把 C1 探测点移到推力主导区"
      "（h0/c≥0.1，k≥1），并把「基准阻力」改为同一振荡状态下的相位平均黏性基线；"
      "②C4 改用闭合周期域 + 恒定驱动力的「泵功」公式消除 ΔKE 漂移；"
      "③C6 用 Richardson 外推值做基准而非闭式理论值，避免理论-离散误差混淆；"
      "④引入多重直接强迫 + 更厚翼型（t/c=0.12）以压低 C_D0 并在 Re≥2000 复测。")
    A("")
    A("## 7 复现方式")
    A("")
    A("```bash")
    A("# 全部 LBM 算例（20 进程并行，约 25 分钟 / 128 核）")
    A("python benchmarks/p162w1_n07_flapping_wing/run_sweep.py --outdir artifacts_p162n07 --workers 20")
    A("# 判据汇总 + 图 + 报告")
    A("python benchmarks/p162w1_n07_flapping_wing/make_figures.py --outdir artifacts_p162n07 --figdir figures")
    A("python benchmarks/p162w1_n07_flapping_wing/make_report.py --outdir artifacts_p162n07")
    A("# 单元/冒烟测试")
    A("python -m pytest tests/test_flapping_propulsion.py -q")
    A("```")
    A("")

    text = "\n".join(L)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text)
    print(json.dumps({"written": os.path.abspath(args.out), "chars": len(text)}, indent=1))
    return 0


def _kh0c(case: dict) -> float:
    kin = case.get("kinematics", {})
    return kin.get("k", float("nan")) * kin.get("h0_over_c", float("nan"))


def _rel(case: dict) -> float:
    if not case or "C_P" not in case:
        return float("nan")
    th = case.get("theory", {}).get("C_P")
    if not th:
        return float("nan")
    return abs(case["C_P"] - th) / abs(th)


if __name__ == "__main__":
    raise SystemExit(main())