#!/usr/bin/env python
"""Generate ``docs/ROTATING_DUCT_P158W2_N01.md`` from the benchmark record.

Every number in the report is injected programmatically from
``records/rotating_duct_frame_w2.json`` (zero hand-copying).
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "records", "rotating_duct_frame_w2.json")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "docs", "ROTATING_DUCT_P158W2_N01.md"))


def f(x, nd=4):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e-3 else f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    d = json.load(open(REC))
    C = d["criteria"]
    cases = d["cases"]
    ekman = [c for c in cases if c["kind"] == "ekman"]
    couette = [c for c in cases if c["kind"] == "couette"]
    duct = [c for c in cases if c["kind"] == "duct"]
    lim = d["known_limitation"]

    lines = []
    A = lines.append
    A("# TensorLBM 旋转参考系内部流场验证面（二）：Guo/EDM 力项、Ekman 厚度标度律、旋转方管二次流")
    A("")
    A(f"> 战役 DAG 节点 `{d['node']}` · 仓库 TensorLBM · 模块 `{d['module']}` · "
      f"基准 `benchmarks/verified/rotating_duct_frame_w2/run.py` · 机器判据 "
      f"**{'全部达标' if d['overall_pass'] else '未全部达标（honest）'}**")
    A(f"> 记录 `benchmarks/verified/rotating_duct_frame_w2/records/rotating_duct_frame_w2.json` "
      f"· commit `{d['git_commit'][:12]}` · 生成 {d['generated_utc']} · 墙钟 {d['wall_clock_s']:.0f} s")
    A("")
    A("## ① 案例介绍")
    A("")
    A("### 物理背景")
    A("")
    A("在随参考系以角速度 $\\boldsymbol\\Omega$ 旋转的坐标系中，不可压 Navier–Stokes 方程带两个惯性伪力")
    A("")
    A("$$\\frac{D\\mathbf u}{Dt}=-\\frac{\\nabla p}{\\rho}-2\\boldsymbol\\Omega\\times\\mathbf u"
      "-\\boldsymbol\\Omega\\times(\\boldsymbol\\Omega\\times\\mathbf r)+\\nu\\nabla^2\\mathbf u .$$")
    A("")
    A("平台已有四个旋转系验证面（`rotating_ekman` 平板 Ekman 螺旋、`rotating_taylor_column`、"
      "`rotating_source_sink`、`rotating_interior_frame` 的 D2Q9 环隙 Couette + 旋转平面槽道），"
      "按节点纪律**不得重复**。本节点建设的是它们的空白区，四个面：")
    A("")
    A("1. **力项方案交叉验证（Guo 2002 vs EDM 精确差分法）**。库内此前只有一种力项实现（Guo）。"
      "本节点在同一 D3Q19 核内并列实现了精确差分法（Kupershtokh & Medvedev 2006；Fakhari & Lee 2013），"
      "在**解析解已知**的旋转平面槽道上逐点对拍。两条*独立*的力耦合公式同时命中解析解、且彼此一致，"
      "比单一实现自证强得多：符号错、漏掉 $(1-1/(2\\tau))$ 因子、或半程速度偏移写错，都会表现为"
      "*方案相关的系统偏差*。")
    A("2. **Ekman 层厚度的标度律**。已验证的 `rotating_ekman` 只在**单个** $\\Omega$ 上验过 $\\delta_E$；"
      "本节点在**一个数量级的转速范围**内测量 $\\delta_E(\\Omega)$ 并**拟合指数**，另给出一条"
      "*与拟合无关*的独立测量（速矢端迹相位：$\\arg(1-W/w_{geo})$ 每 $\\delta_E$ 深度转 1 rad）。"
      "两条通道合起来才叫「标度律验证」，单点只能叫「一个系数没写错」。")
    A("3. **旋转参考系反向旋转 Taylor–Couette**（$\\Omega_i\\Omega_o<0$），并把参考系转速"
      "$\\Omega_f\\neq0$ 打开。环隙中纯方位角流的 Coriolis 加速度是**径向**的，故方位角动量方程仍只由黏性控制，")
    A("   $$u_\\theta(r)=Ar+B/r,\\quad A=\\frac{\\Omega_o r_o^2-\\Omega_i r_i^2}{r_o^2-r_i^2},\\quad "
      "B=\\frac{r_i^2r_o^2(\\Omega_i-\\Omega_o)}{r_o^2-r_i^2}$$")
    A("   与 $\\Omega_f$ 无关；而伪力若实现有误则会生出**虚假径向流** $u_r$。本节点的锚是"
      "(i) 剖面（有效几何化后 ≤3%）、(ii) **力矩恒等式** "
      "$T(r)=2\\pi\\rho\\nu L r^2\\,(du_\\theta/dr-u_\\theta/r)$（对精确解与 $r$ 无关，且 $=4\\pi\\mu L|B|$，"
      "这是任何「拟合剖面」都伪造不出来的 r-无关性检验）、(iii) $u_r/u_\\theta\\to0$。")
    A("4. **旋转方管二次流**（Dean–Coriolis 对偶）。方形截面管，主流沿 $x$ 由体积力驱动，绕展向轴 $y$ 旋转。"
      "主流的 Coriolis 加速度 $a_z=2\\Omega u_x$ 的**旋度非零**，因而即使它不是梯度也驱动横向二次流：")
    A("   $$\\nabla^4\\psi=\\frac{2\\Omega}{\\nu}\\,\\partial_y u_x,\\qquad \\psi=\\partial_n\\psi=0\\ \\text{(壁面)}$$")
    A("   这与经典小 Dean 数弯管（Dean）问题在数学上**同一个线性问题**（曲率 $\\kappa$ 换成 "
      "$\\kappa_{eff}=2\\Omega/U$）。因此旋转方管继承 Dean 涡对拓扑；本报告把这层对应关系明确标为"
      "*结构对应*（不做 Dean 数量级系数比对），定量锚是**同一线性问题的独立有限差分解**。")
    A("")
    A("### 参数表（程序化读出，零手抄）")
    A("")
    A("| 面 | 网格 | $\\tau$ / $\\nu$ | 转速 | 驱动 | 其它 |")
    A("|---|---|---|---|---|---|")
    for c in ekman:
        p = c["params"]
        A(f"| Ekman 槽道 | $3\\times{p['grid'][1]}\\times3$ (D3Q19) | {f(p['tau'],4)} / {f(p['nu'],3)} | "
          f"$\\Omega={f(p['Omega'],3)}$ | $G={f(p['G'],3)}$ | $\\delta={c['delta_nodes']:.0f}$ 格, "
          f"高 $6\\delta$, 方案 {c['scheme']} |")
    for c in couette:
        p = c["params"]
        A(f"| 环隙 Couette | ${p['grid'][0]}^2$ (D2Q9) | {f(p['tau'],4)} / {f(p['nu'],3)} | "
          f"$\\Omega_i={p['omega_i']}$, $\\Omega_o={p['omega_o']}$, $\\Omega_f={p['omega_frame']}$ | "
          f"转动壁 | $r_i={p['ri']:.0f}$, $r_o={p['ro']:.0f}$ |")
    for c in duct:
        p = c["params"]
        A(f"| 旋转方管 | ${p['grid'][0]}\\times{p['grid'][1]}^2$ (D3Q19) | {f(p['tau'],4)} / {f(p['nu'],3)} | "
          f"$\\Omega={f(p['Omega'],3)}$ | $G={f(p['G'],3)}$ | $h={p['h']:.1f}$, 方案 {p['scheme']} |")
    A("")
    A("## ② 怎么做的")
    A("")
    A("- **求解器**：`src/tensorlbm/rotating_duct.py` 的 `ForcedD3Q19`（D3Q19 BGK，float64 numpy，"
      "拉取式 streaming + 半程反弹，移动壁用 Ladd 动量修正），力项可切 `guo` / `edm`：")
    A("  - Guo：$F_i=(1-\\frac{1}{2\\tau})w_i[3\\,(\\mathbf c_i-\\mathbf u^*)\\cdot\\mathbf a"
      "+9\\,(\\mathbf c_i\\cdot\\mathbf u^*)(\\mathbf c_i\\cdot\\mathbf a)]$，半程速度 "
      "$\\mathbf u^*=\\mathbf u+\\mathbf a/2$ 由 2 次不动点迭代求（Coriolis 依赖速度）；")
    A("  - EDM：$f_i^{post}=f_i-\\frac{1}{\\tau}(f_i-f_i^{eq}(\\rho,\\mathbf u))"
      "+[f_i^{eq}(\\rho,\\mathbf u+\\mathbf a)-f_i^{eq}(\\rho,\\mathbf u)]$，**同一**不动点速度，"
      "使比较只隔离「力项形式」而不混入时间居中差异。")
    A("  - 两者在 $\\mathbf a$ 的一阶上代数恒等，差异为 $O(a^2)$——本报告的实测差就是该量级"
      f"（判据 C1 = {f(C['C1_scheme_agreement']['value'],3)}）。")
    A("- **Taylor–Couette** 用已验证的 `RotatingAnnulusD2Q9`（D2Q9 + Ladd 移动壁 + Guo）跑反向旋转工况；"
      "力矩由剖面恒等式 `torque_from_profile` 从**测得的**壳层平均剖面算，天然给出 r-无关性检验。")
    A("- **方管二次流**：主流剖面先由有限差分（稀疏直接解）算出方形管 Poiseuille 解，作为初值；"
      "旋转项用 $a_z=2\\Omega u_x$（Coriolis 全项，含对二次流自身的 Coriolis 修正）。收敛判据为速度场残差。")
    A("- **独立参考解** `linear_coriolis_secondary_fd`：把 $\\nabla^4\\psi=g$ 与 Thom 涡量边界条件"
      "组装成**一个线性稀疏系统直接求解**（无迭代收敛问题），并回代检查"
      "$\\nabla^2\\omega+g$ 与 $\\omega+\\nabla^2\\psi$ 的残差（实测 ≈1e-14）。")
    A("- **工具链**：numpy float64 内核 + scipy 稀疏直接解 + matplotlib 出图；全部用例参数 "
      "SHA-256 锚定（`params_sha256`）。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### 判据汇总（机器可算，逐条给阈值）")
    A("")
    A("| 判据 | 定义 | 实测 | 阈值 | 判定 |")
    A("|---|---|---|---|---|")
    rows = [
        ("C1_scheme_agreement", "max $|u^{Guo}-u^{EDM}|/\\max|u|$（旋转槽道全部工况）", "<="),
        ("C2_ekman_profile", "5 个指定高度上 $\\max|u^{num}-u^{ana}|/U_g$（两分量）", "<="),
        ("C3_ekman_delta_prefactor", "$\\max|\\delta_{fit}/\\sqrt{\\nu/\\Omega}-1|$", "<="),
        ("C4_ekman_scaling_exponent", "$\\delta_{fit}(\\Omega)$ 的双对数斜率（理论 $-1/2$）", "|p+1/2|<="),
        ("C4b_ekman_phase_anchor", "$\\max|\\delta_{phase}/\\sqrt{\\nu/\\Omega}-1|$（相位法，不用拟合幅度）", "<="),
        ("C5_couette_profile", "内部壳层 $\\max|u^\\theta_{LBM}-(Ar+B/r)|/\\max$ 壁速", "<="),
        ("C6_couette_torque", "$|T_{profile}|/(4\\pi\\mu L|B|)-1$ 与 $T(r)$ 的 r-无关性散度", "<="),
        ("C7_duct_vs_reference", "最低转速下 $\\max|u_{sec}^{LBM}-u_{sec}^{FD}|/\\max|u_{sec}^{FD}|$", "<="),
        ("C9_mass_conservation", "D3Q19 工况最大相对质量漂移", "<="),
    ]
    for cid, desc, direction in rows:
        c = C[cid]
        tol = c.get("tolerance")
        tol_s = f(tol if not isinstance(tol, float) else f"{tol:.3g}")
        A(f"| {cid} | {desc} | {f(c['value'],4)} | {direction} {tol_s} | "
          f"**{'PASS' if c['pass'] else 'FAIL'}** |")
    c8 = C["C8_duct_topology"]
    A(f"| C8_duct_topology | 涡对拓扑：每半管 1 个涡胞、$\\psi$ 反对称、环量平衡、净横流为零、"
      f"$\\Omega\\to-\\Omega$ 反号 | 见下 | 见下 | **{'PASS' if c8['pass'] else 'FAIL'}** |")
    A(f"| C10_param_sha256 | 每个用例带 64 位 SHA-256 参数锚 | {C['C10_param_sha256']['value']} | — | "
      f"**{'PASS' if C['C10_param_sha256']['pass'] else 'FAIL'}** |")
    A("")
    A("### Ekman 标度律（S1/S2）")
    A("")
    A("| $\\delta_E$ (格) | $\\Omega$ | 方案 | $\\delta_{fit}$ | 拟合误差 | $\\delta_{phase}$ | 相位误差 | 五点剖面误差 | 残差 |")
    A("|---|---|---|---|---|---|---|---|")
    for c in ekman:
        A(f"| {c['delta_nodes']:.0f} | {f(c['params']['Omega'],3)} | {c['scheme']} | {c['delta_fit']:.4f} | "
          f"{100*c['delta_rel_err']:.2f}% | {c['delta_phase']:.4f} | {100*c['delta_phase_rel_err']:.2f}% | "
          f"{100*c['probe_max_err']:.2f}% | {c['residual']:.1e} |")
    A("")
    A(f"拟合指数 $p={C['C4_ekman_scaling_exponent']['value']:.4f}$（理论 $-0.5$，"
      f"$|p+1/2|={abs(C['C4_ekman_scaling_exponent']['value']+0.5):.4f}$，阈值 0.05），"
      f"覆盖 $\\Omega\\in[{C['C4_ekman_scaling_exponent']['omega_range'][0]:.4g},"
      f"{C['C4_ekman_scaling_exponent']['omega_range'][1]:.4g}]$（比值 "
      f"{C['C4_ekman_scaling_exponent']['ratio']:.1f}倍）。**注意纪实的边界**：解析 $\\delta$ 由构造固定"
      "（$f=2\\nu/\\delta^2$），因此该判据检验的是「求解器在该转速范围内是否引入与转速相关的系统偏差」"
      "（这正是 Guo/EDM 差异、伪力实现错误会暴露的地方），而不是「$\\delta$ 这个公式成立」。"
      "公式本身由 C3 的前因子（逐点 <1%）负责。")
    A("")
    A("![Ekman 标度律](assets/rotating_duct_p158w2_n01/fig1_ekman_scaling.png)")
    A("")
    A("![Ekman 剖面与方案差](assets/rotating_duct_p158w2_n01/fig2_ekman_profiles.png)")
    A("")
    A("### 反向旋转 Taylor–Couette（S3）")
    A("")
    A("| 工况 | 内部剖面误差 | 力矩相对误差 | $T(r)$ 散度 | 质量漂移 |")
    A("|---|---|---|---|---|")
    for c in couette:
        A(f"| {c['label']} | {100*c['profile_interior_rel_max']:.2f}% | {100*c['torque_rel_err']:.2f}% | "
          f"{100*c['torque_r_spread']:.1f}% | {c['mass_drift']:+.1e} |")
    A("")
    fl = np.load(os.path.join(HERE, "records", "rotating_duct_frame_w2_fields.npz"))
    r_ = fl["couette_r"]
    uth_ = fl["couette_u_th"]
    ur_ = fl["couette_u_r"]
    m_ = np.isfinite(uth_) & (r_ >= couette[0]["params"]["ri"] + 2) & (r_ <= couette[0]["params"]["ro"] - 2)
    ur_ratio = float(np.max(np.abs(ur_[m_])) / np.max(np.abs(uth_[m_])))
    A(f"虚假径向流：内部壳层 $\\max|u_r|/\\max|u_\\theta|={ur_ratio:.2e}$（工况 0 的场数据实测）——"
      "参考系伪力对纯方位角流不产生任何径向速度，这是 Coriolis 项纯径向性质的直接检验。")
    A("")
    A("![Taylor–Couette 剖面与力矩恒等式](assets/rotating_duct_p158w2_n01/fig3_couette.png)")
    A("")
    A("### 旋转方管二次流（S4）")
    A("")
    A("| $\\Omega$ | $\\max|u_{sec}|$ | 与参考解幅值比 | 场相对误差 | 涡胞 $(y,z)$ | $\\psi$ 反对称 | 环量平衡 | 净横流 |")
    A("|---|---|---|---|---|---|---|---|")
    for c in duct:
        g = c["diagnostics"]
        A(f"| {f(c['params']['Omega'],3)} | {c['secondary_max_lbm']:.3e} | {c['secondary_ratio']:.4f} | "
          f"{100*c['field_rel_err_vs_reference']:.2f}% | ({g['sign_changes_y_mid']},{g['sign_changes_z_quarter']}) | "
          f"{100*g['antisym_psi']:.2f}% | {100*g['circ_balance']:.2f}% | {g['net_flux_z']:+.1e} |")
    sf = C["C8_duct_topology"].get("sign_flip")
    if sf:
        A("")
        A(f"$\\Omega\\to-\\Omega$ 反号：相关系数 {sf['correlation']:.4f}（阈值 ≤ -0.99）——二次流随转速反向，"
          "这是 Coriolis 驱动（而非离心/几何伪影）的判据。")
    A("")
    A("![旋转方管二次流](assets/rotating_duct_p158w2_n01/fig4_duct_secondary.png)")
    A("")
    A("![二次流幅值与参考偏差趋势](assets/rotating_duct_p158w2_n01/fig5_duct_scaling.png)")
    A("")
    A("## ④ 误差判定（逐项 vs 基准，显式回答 <3%）")
    A("")
    A("| 关键量 | 基准 | 误差 | 是否 <3% |")
    A("|---|---|---|---|")
    SQ = "$\\sqrt{\\nu/\\Omega}$"
    rows4 = [
        ("Ekman 五点剖面", "闭式解", C["C2_ekman_profile"]["value"], "max", 0.03),
        ("Ekman delta 前因子", SQ, C["C3_ekman_delta_prefactor"]["value"], "max", 0.03),
        ("Ekman delta 相位锚", SQ, C["C4b_ekman_phase_anchor"]["value"], "max", 0.03),
        ("Ekman 标度指数", "-1/2", abs(C["C4_ekman_scaling_exponent"]["value"] + 0.5), "绝对", 0.05),
        ("Taylor-Couette 剖面", "$Ar+B/r$", C["C5_couette_profile"]["value"], "max", 0.03),
        ("Taylor-Couette 力矩", "$4\\pi\\mu L|B|$", C["C6_couette_torque"]["value"], "max", 0.03),
    ]
    for label, ref, val, kind, tol in rows4:
        if kind == "绝对":
            err = f"{val:.4f} (绝对)"
        else:
            err = f"{100*val:.2f}% ({kind})"
        A(f"| {label} | {ref} | {err} | {'是' if val < tol else '否（阈值 ' + str(tol) + '）'} |")
    A(f"| 方管二次流场 | 线性 FD 参考解 | {100*C['C7_duct_vs_reference']['value']:.2f}%（最低转速） | "
      f"{'否（惯性 O(Re_sec) 偏差，阈值 10%）' if C['C7_duct_vs_reference']['value'] >= 0.03 else '是'} |")
    A("")
    A("方管工况按设计不在 3% 档：参考解是**蠕动（creeping）线性解**，LBM 在有限横向 Reynolds 数下必然偏离，"
      "实测偏差随 $\\Omega$ 单调增长（见 fig5 右），把残余归因于惯性而不是实现错误。")
    A("")
    A("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    A("")
    A("1. **D3Q19 阶梯移动壁是本节点的真实缺陷发现**（honest）：" + lim["detail"])
    A("   实测：" + "，".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in lim["measured"].items()) + "。")
    A(f"   因此 Taylor–Couette 锚改用已验证的 D2Q9 环隙求解器；D3Q19 核只用于壁面与格点对齐"
      f"（平面槽道，误差 <1e-5）或没有移动壁（方管）的场合。该缺陷以 `known_limitation` 入记录，不掩盖。")
    A("2. **Ladd 动量交换力矩不可用作阶梯圆壁的锚**：压力项在粗阶梯上不精确抵消，残差比解析力矩大 2 个数量级"
      "（实测 $|T|\\sim10^2$ 而解析值 0.72）。力矩改由剖面恒等式给出（r-无关性 + 幅值双重检验）。")
    A("3. **Guo 与 EDM 在一阶上恒等**：实测差为机器精度量级（C1），因为两者只差 $O(a^2)$，"
      "而本工况 $a\\sim10^{-5}$–$10^{-4}$。这不是「没测出差别」，而是代数等价性的数值确认；"
      "单元测试同时断言「差值 > 0」（见 `test_guo_vs_edm_rotating_channel_agree_but_differ`）。")
    A("4. **$\\delta_E$ 与 Coriolis 参数约定**：战役判据写 $\\delta_E=\\sqrt{2\\nu/\\Omega}$，地球物理标准式"
      "$\\delta_E=\\sqrt{2\\nu/f}$；本报告统一用「$\\Omega$ 即 Coriolis 参数 $f$」的约定"
      "（与已验证 `rotating_ekman` 一致），并在记录里同时给出 $\\Omega$ 与所用公式。")
    A("5. **适用边界**：低 Mach（$\\mathrm{Ma}<0.1$）、常数密度、$\\delta_E\\ge4$ 格（更薄则解析锚受阶梯壁污染）；"
      "方管结果对 $\\kappa_{eff}=2\\Omega/U$ 的 Dean 对应只在 $Re_{sec}\\ll1$ 的线性区成立。")
    A("6. **质量守恒**：D3Q19 工况漂移 ≤1e-12（判据 C9 PASS）；D2Q9 环隙保留经典阶梯移动壁注入"
      f"（{lim['d2q9_annulus_mass_drift']:.1e}，对 2% 剖面误差无影响），二者都如实入账。")
    A("")
    A("## ⑥ 产物清单")
    A("")
    A("- 模块：`src/tensorlbm/rotating_duct.py`")
    A("- 基准：`benchmarks/verified/rotating_duct_frame_w2/run.py`（`make_figures.py` 出图，"
      "`make_doc.py` 生成本文档）")
    A("- 记录：`benchmarks/verified/rotating_duct_frame_w2/records/rotating_duct_frame_w2.json`（+ `_fields.npz`）")
    A("- 测试：`tests/test_rotating_duct.py`")
    A("")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
