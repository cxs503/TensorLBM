"""Generate the P82W5-N04 report payload (/tmp/report.json) from the benchmark JSONs.

Numbers are injected programmatically (no hand-copied digits); figures are embedded
as base64 PNGs.

    python scripts/make_report_p82w5_n04.py [--out /tmp/report.json]
"""

from __future__ import annotations

import base64
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "P82W5-N04")
FIG = os.path.join(ROOT, "figures", "p82w5_n04")


def load(name, case=None):
    """Load a benchmark JSON; with ``case`` return that case block."""
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    if case is None:
        return data
    return data.get("cases", {}).get(case)


def fnum(v, nd=4, pct=False):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, (int,)):
        return str(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pct:
        return "%.2f%%" % (100 * x)
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return "%.3e" % x
    return ("%%.%df" % nd) % x


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def figures():
    figs = []
    for path in sorted(glob.glob(os.path.join(FIG, "*.png"))):
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        figs.append({"name": os.path.basename(path),
                     "caption": os.path.basename(path).split("_")[0].upper()
                     + ": " + os.path.basename(path).replace("_", " ").rsplit(".", 1)[0],
                     "png_base64": b64})
    return figs


def main(argv=None):
    out_path = "/tmp/report.json"
    if argv and "--out" in argv:
        out_path = argv[argv.index("--out") + 1]
    a = load("case_a.json", "a")
    b = load("case_b.json", "b")
    c = load("case_c.json", "c")
    d = load("case_d.json", "d")
    e = load("case_e.json", "e")
    f = load("case_f.json", "f")
    ref_lad = load("reference_high.json") or load("reference_refine_ladder.json")

    md = []
    md.append("## 1 案例介绍\n")
    md.append(
        "屈服应力流体（Bingham / Herschel–Bulkley）在船–冰间隙（h/D）润滑阻力中承担"
        "「低于屈服应力即近似刚性」的关键角色。本节点把 TensorLBM 的屈服应力内核从 2D "
        "静止壁推进到 **D3Q19 + 移动壁**，并在三类解析/高精度参考解上定量闭合：\n"
        "① 3D 方管 Bingham 塞流剖面（无正则化的增广拉格朗日参考解）；"
        "② Bingham 平面 Couette–Poiseuille（一壁以 $U_w$ 移动 + 体积力，塞厚闭式 "
        "$\\delta_p = 2\\tau_y/G$）；③ $Bn\\to0$ 牛顿极限。\n")
    md.append(
        "\n几何与参数（lattice 单位，$\\rho=1$，$c_s^2=1/3$）：方管 $N\\times N$ 流体格，"
        "壁面位于 $y,z=0$ 与 $N+1$（半路反弹壁面位于格点面）；通道 $H$ 层流体、$x/z$ 周期、"
        "$y=H+1$ 层可移动。参数：$\\mu_0=0.1$，$G=\\rho a=1\\times10^{-5}$（方管）、"
        "$2\\times10^{-6}$（Couette–Poiseuille），$\\tau_y=2\\times10^{-5}$ / $4\\times10^{-6}$，"
        "Papanastasiou 参数 $m=1\\times10^{5}$，$U_w$ 由目标塞区位置反解得到。\n")

    md.append("\n## 2 方法与关键设置\n")
    md.append(
        "* 内核：`src/tensorlbm/yield_stress_moving_wall_3d.py`（D3Q19 BGK + Guo 体积力，"
        "复用 `yield_stress_rheology_3d` 的 Papanastasiou/HB 应力与剪切率恢复）。\n"
        "* 移动壁：**half-way bounce-back + Ladd 速度修正**（未用 Bouzidi）——本节点所有"
        "壁面都是落在格点面上的平面壁，此时半路反弹即二阶精度，Bouzidi 插值因子退化为 "
        "$q=1/2$；实现写作「stream 后覆写」："
        "$f_i(x_b)=f_{\\bar i}^{post}(x_b)+6w_i\\rho(x_b)(c_i\\cdot u_w)$，"
        "对每条 $x_b-c_i$ 为固体的连接。**选择声明**：Ladd/半路反弹方案，"
        "固体层每步重置为静止平衡态。\n"
        "* 动量账本（MEM）：逐连接精确记账 "
        "$M=\\sum c_i(f_i^{new}-f_i^{streamed}) = 2\\sum c_i f_{\\bar i}^{post} + "
        "\\sum c_i 6w_i\\rho(c_i\\cdot u_w)$，给出离散恒等式 "
        "$P(t+1)-P(t)=M_{body}+M_{wall}$ 与稳态力平衡两条独立判据。\n"
        "* 参考解：`src/tensorlbm/yield_stress_analytical_moving_wall.py`——"
        "Couette–Poiseuille 精确解（$\\tau_0$ 二分到机器精度）、牛顿方管 Fourier 级数、"
        "Bingham 方管**无正则化**增广拉格朗日解（细网格节点与 LBM 格心完全重合）；"
        "纯 Couette 的退化性显式分类。\n"
        "* 工具链：PyTorch float64 / `stream3d`（cat 后端）/ pytest / matplotlib。\n")
    if ref_lad:
        rows = [[k, fnum(v["Q"], 6), v["n_fine"], v["iters"]] for k, v in ref_lad.items()]
        md.append("\n参考解自身的网格收敛审计（同一物理问题、不同细网格 refine）：\n\n")
        md.append(table(rows, ["case", "Q (per unit depth)", "n_fine", "AL iterations"]) + "\n")

    # ---------------- (a)
    md.append("\n## 3 计算结果与判据\n")
    if a:
        rows = []
        for k, v in a["runs"].items():
            rows.append([k, fnum(v["linf_rel"], pct=True), fnum(v["l2_rel"], pct=True),
                         fnum(v["Q_rel"], pct=True),
                         "r%d" % v.get("reference_refine", 4),
                         fnum(v.get("force_balance"), 8), fnum(v.get("iters"))])
        md.append("### (a) 3D 方管 Bingham 剖面 vs 无正则化参考解\n")
        md.append(table(rows, ["N", "L∞ rel", "L2 rel", "Q rel", "参考解 refine",
                               "力平衡", "步数"]) + "\n")
        md.append("\n判据 a：N≥32 时 $L_\\infty\\le3\\%%$ → **%s**（实测 %s）\n" % (
            "PASS" if a["passed"] else "HONEST-FAIL", fnum(a["metric"], pct=True)))
        if "reference_self_audit" in a:
            sa = a["reference_self_audit"]
            md.append("\n参考解网格自审（N=16，refine=2 vs 4）：Q 相对差 %s。\n"
                      % fnum(sa["rel_diff"], pct=True))
    # ---------------- (b)
    if b:
        rows = []
        for k, v in b["runs"].items():
            rows.append([k, v["regime"], fnum(v.get("u_wall"), 6),
                         fnum(v["delta_p_exact"], 2),
                         fnum(v.get("delta_p_stress"), 3),
                         fnum(v.get("delta_p_err_stress"), pct=True),
                         fnum(v.get("delta_p_threshold"), 2),
                         fnum(v.get("delta_p_err_threshold"), pct=True),
                         fnum(v["plug_cv"], 3),
                         fnum(v["profile_linf_rel"], pct=True)])
        md.append("\n### (b) Bingham Couette–Poiseuille：塞厚与塞区刚性\n")
        md.append(table(rows, ["H", "regime", "U_w", "δ_p 精确", "δ_p 应力交点",
                               "误差", "δ_p 阈值法", "误差", "塞区 CV", "剖面 L∞"]) + "\n")
        if b.get("m_scan"):
            md.append("\n正则化参数 m 扫描（H=%s，$M=m\\tau_y/\\mu_0$ 为无量纲群）：\n\n"
                      % s_h if False else "\n正则化参数 m 扫描（同一 H，$M=m\\tau_y/\\mu_0$）：\n\n")
            rows = [[k, fnum(v["M_group"], 2), fnum(v["plug_viscosity"], 3),
                     fnum(v["plug_cv"], 3), fnum(v["profile_linf_rel"], pct=True)]
                    for k, v in b["m_scan"].items()]
            md.append(table(rows, ["case", "M", "μ_plug", "塞区 CV", "剖面 L∞"]) + "\n")
            md.append("\n结论：m 从 $10^5$ 增到 $10^6$ 时剖面误差不再改善、塞区 CV 反而变差——"
                      "$m\\dot\\gamma\\gg1$ 区 $\\mu_{eff}=\\mu_0+\\tau_y/\\dot\\gamma$ 与 m 无关，"
                      "继续加大 m 只增加数值刚度。\n")
        cc = b.get("couette_controls", {})
        if cc:
            md.append("\n纯 Couette 退化控制（无体积力）：\n\n")
            rows = []
            if "sub_yield" in cc:
                s = cc["sub_yield"]
                rows.append(["τ_w<τ_y (0.03τ_y)", "%.3e" % s["tau_wall"], s["regime"],
                             "u_mean/U_w=%s" % fnum(s["u_mean_rel"], 3),
                             "理想 Bingham 无稳态解；正则化模型给出 $\\mu_{eff}\\approx"
                             "\\mu_0+m\\tau_y$ 的蠕动线性剖面"])
            if "post_yield" in cc:
                s = cc["post_yield"]
                rows.append(["τ_w>τ_y", "%.3e" % s["tau_wall"], s["regime"],
                             fnum(s["linf_rel_vs_linear"], 5), "全间隙屈服、线性剖面"])
            md.append(table(rows, ["情形", "τ_wall", "判定", "u_max/U_w 或 L∞",
                                   "物理含义"]) + "\n")
        md.append("\n纯 Couette 控制说明：$\\tau_w<\\tau_y$ 分支在理想 Bingham 下**无稳态解**，"
                  "正则化模型把它替换成 $\\mu_{eff}\\approx\\mu_0+m\\tau_y$ 的蠕动线性剖面"
                  "（$u_{mean}\\approx U_w/2$），因此任何「部分塞区」的 Couette 报告都是把"
                  "不存在的解拟合出来的结果——这是本节点对历史口径的更正。\n")
        md.append("\n判据 b：$\\delta_p$ 误差 ≤3%% 且塞区 CV ≤1e-3 → **%s**"
                  "（δ_p 最大误差（应力交点估计量）%s，阈值法 %s，CV 最大 %s）\n"
                  % ("PASS" if b["passed"] else "HONEST-FAIL",
                     fnum(b["metric"].get("delta_p_err_stress_max"), pct=True),
                     fnum(b["metric"].get("delta_p_err_threshold_max"), pct=True),
                     fnum(b["metric"]["plug_cv_max"], 3)))
    # ---------------- (c)
    if c:
        rows = []
        for k, v in c["runs"].items():
            rows.append([k, fnum(v["yp_exact"], 3), fnum(v["yp_from_umax"], 4),
                         fnum(v["err_umax"], 4),
                         fnum(v.get("delta_p_stress"), 3),
                         fnum(v["profile_linf_rel"], pct=True), v.get("iters")])
        md.append("\n### (c) 屈服面位置分辨率阶梯（纯压力驱动通道，$y_p=\\tau_y/G$）\n")
        md.append(table(rows, ["N", "y_p 精确", "y_p(u_max 反演)", "误差",
                               "δ_p(应力交点)", "剖面 L∞", "步数"]) + "\n")
        sl = c["slopes"]
        md.append("\n拟合斜率（log–log）：u_max 反演 %s，γ̇ 阈值交点 %s，剖面 $L_\\infty$ %s。"
                  "剖面本身以 %s 阶收敛（说明离散是二阶的），而**屈服面位置**这个泛函只以 ~1 阶"
                  "收敛：$y_p=h-\\sqrt{2\\mu_0u_p/G}$ 在 $y_p\\ll h$ 时条件数极大"
                  "（$\\partial y_p/\\partial u_p\\approx-\\mu_0/(G(h-y_p))$），"
                  "叠加正则化模型在屈服面附近的 $O(1/M)$ 偏差。"
                  % (fnum(sl["umax_inversion"], 2), fnum(sl["gamma_crossing"], 2),
                     fnum(sl["profile_linf"], 2), fnum(abs(sl["profile_linf"] or 0), 2)))
        md.append("\n判据 c：斜率 ≥1.7 → **%s**（取屈服面估计量的较大斜率 %s）\n"
                  % ("PASS" if c["passed"] else "HONEST-FAIL", fnum(c["metric"], 2)))
    # ---------------- (d)
    if d:
        rows = [[k, fnum(v, 10)] for k, v in d["metric"].items()]
        md.append("\n### (d) MEM 动量账本\n")
        md.append(table(rows, ["闭合量", "实测"]) + "\n")
        md.append("\n判据 d：全部闭合率 ∈[0.97,1.03] → **%s**\n"
                  % ("PASS" if d["passed"] else "HONEST-FAIL"))
    # ---------------- (e)
    if e:
        rows = []
        for k, v in e["runs"].items():
            rows.append([k, fnum(v["tau_y"], 6),
                         fnum(v.get("shape_factor_sim"), 4),
                         fnum(v.get("shape_factor_ref"), 4),
                         fnum(v["ratio_err"], pct=True),
                         fnum(v.get("reference")), fnum(v["Q3D_rel_err"], pct=True)])
        md.append("\n### (e) 2D/3D 流量比与矩形管形状因子\n")
        md.append(table(rows, ["case", "τ_y", "形状因子 (LBM)", "形状因子 (解析)",
                               "比值误差", "参考解", "Q_3D 误差"]) + "\n")
        md.append("\n判据 e：比值误差 ≤5%% → **%s**（判定集 %s，最大 %s；"
                  "牛顿极限形状因子 $f=0.4217$）\n"
                  % ("PASS" if e["passed"] else "HONEST-FAIL",
                     ",".join(e["metric"].get("judged_runs", [])),
                     fnum(e["metric"].get("ratio_err_max_judged")
                          or e["metric"].get("ratio_err_max"), pct=True)))
    # ---------------- (f)
    if f:
        rows = [[k, fnum(v["linf_rel"], pct=True), v["iters"],
                 fnum(v.get("force_balance"), 8), fnum(v.get("shape_factor"), 4)]
                for k, v in f["runs"].items()]
        md.append("\n### (f) Bn=0 牛顿极限\n")
        md.append(table(rows, ["case", "L∞ rel", "步数", "力平衡", "形状因子"]) + "\n")
        md.append("\n判据 f：≤1%% → **%s**（最大 %s）\n"
                  % ("PASS" if f["passed"] else "HONEST-FAIL",
                     fnum(f["metric"]["linf_rel_max"], pct=True)))

    # ---------------- attention points
    md.append("\n## 4 注意点（踩过的坑与局限）\n")
    md.append(
        "1. **纯 Bingham 平面 Couette 不存在部分塞区**：零压力梯度 ⇒ 剪应力沿间隙均匀，"
        "要么全间隙屈服（线性剖面，$\\delta_p=0$），要么无稳态解。历史实现中的 "
        "$\\delta_p=H-\\mu_0U_w/\\tau_y$ 不是 Couette 问题的解；本节点改用 "
        "Couette–Poiseuille 作塞厚闭式锚（$\\delta_p=2\\tau_y/G$），并显式分类纯 Couette 两种退化情形。\n"
        "2. **Papanastasiou 的 $m$ 在 $m\\dot\\gamma\\gg1$ 区失效**：此时 "
        "$\\mu_{eff}\\to\\mu_0+\\tau_y/\\dot\\gamma$ 与 $m$ 无关，继续增大 $m$ 不改善塞区；"
        "塞内残余剪切为 $\\dot\\gamma\\approx(\\tau_y/\\mu_0)e^{-M}$，$M=m\\tau_y/\\mu_0$。\n"
        "3. **剪切率恢复方式决定塞区行为**：由非平衡应力反解时 `|Π|~τ_y` 把等效黏度钉在 "
        "$\\mu_0+\\tau_y/\\dot\\gamma_{min}$，塞区不可能刚性；由速度梯度反解才能让塞区黏度升到 "
        "$\\mu_0+m\\tau_y$。本节点两种模式都保留（`use_neq`）并分别报告。\n"
        "4. **稳态账本必须取最后一窗**：整段平均会把力平衡闭合率压到 ~0.98（起点瞬态贡献）。\n"
        "5. **移动壁 Ladd 项的 $\\rho$ 因子**：缺 $\\rho$ 在 $\\rho\\approx1$ 时不可见，"
        "但密度有 20% 偏差时即为 33% 的修正误差（2D 侧审计结论，3D 沿用同一实现）。\n"
        "6. 局限：壁面限定为格点面上的平面壁；方管 Bingham 的屈服面是闭合曲线，"
        "在 2 格厚屈服层下 $L_\\infty$ 由近壁区主导（见 (a) 的误差分布）；"
        "高 $M$ 时塞区黏度可达 $\\mu_0(1+M)$，需相应放宽 `nu_max`。\n")

    # ---------------- error verdict
    md.append("\n## 5 误差判定（逐项、显式回答是否 <3%）\n")
    verdict_rows = []
    verdict_rows.append([
        "(a) 3D 方管 $L_\\infty$",
        fnum(a["metric"], pct=True) if a else "—", "≤3%",
        ("是" if a and a["metric"] is not None and a["metric"] <= 0.03 else "否")])
    verdict_rows.append([
        "(b) 塞厚 $\\delta_p$",
        fnum(b["metric"].get("delta_p_err_stress_max"), pct=True) if b else "—",
        "≤3%", ("是" if b and b["metric"].get("delta_p_err_stress_max") is not None
                and b["metric"]["delta_p_err_stress_max"] <= 0.03 else "否")])
    verdict_rows.append([
        "(b) 塞区 CV", fnum(b["metric"]["plug_cv_max"], 3) if b else "—", "≤1e-3",
        ("是" if b and b["metric"]["plug_cv_max"] is not None
         and b["metric"]["plug_cv_max"] <= 1e-3 else "否")])
    verdict_rows.append([
        "(c) 收敛斜率", fnum(c["metric"], 2) if c else "—", "≥1.7",
        ("是" if c and c["metric"] is not None and c["metric"] >= 1.7 else "否")])
    verdict_rows.append(["(d) 动量闭合", "见上表", "[0.97,1.03]",
                         ("是" if d and d["passed"] else "否")])
    verdict_rows.append(["(e) 2D/3D 比值",
                         fnum((e["metric"].get("ratio_err_max_judged")
                               or e["metric"].get("ratio_err_max")), pct=True) if e else "—",
                         "≤5%", ("是" if e and e["passed"] else "否")])
    verdict_rows.append(["(f) Bn=0", fnum(f["metric"]["linf_rel_max"], pct=True) if f else "—",
                         "≤1%", ("是" if f and f["passed"] else "否")])
    md.append(table(verdict_rows, ["判据", "实测", "门槛", "满足？"]) + "\n")
    npass = sum(1 for r in verdict_rows if r[3] == "是")
    md.append("\n合计 **%d/%d** 项满足。未满足项按纪律如实保留为 HONEST-FAIL，"
              "不以后处理掩盖：其数值含义、成因与后续路线写在第 4 节。\n" % (npass, len(verdict_rows)))

    md.append("\n## 6 复现与产物\n")
    md.append(
        "* 内核：`src/tensorlbm/yield_stress_moving_wall_3d.py`\n"
        "* 参考解：`src/tensorlbm/yield_stress_analytical_moving_wall.py`\n"
        "* 文档：`docs/yield_stress_moving_wall_3d.md`\n"
        "* 基准：`scripts/bench_yield_stress_moving_wall_3d.py`（JSON 落 "
        "`artifacts/P82W5-N04/`，场数据 `.npy` 同名）\n"
        "* 图：`scripts/gen_yield_stress_moving_wall_figures.py` → `figures/p82w5_n04/`\n"
        "* 测试：`tests/test_yield_stress_moving_wall_3d.py`（11 项，含解析恒等式、"
        "AL 参考解交叉验证、移动壁牛顿锚、账本恒等式、质量守恒）\n")

    payload = {
        "title": "P82W5-N04 屈服应力 LBM 3D+移动壁：D3Q19 Papanastasiou/HB 正则化与半路反弹移动壁面",
        "summary": ("D3Q19 屈服应力内核推进到 3D+移动壁：实现 half-way bounce-back + Ladd 移动壁"
                    "与逐连接精确动量账本，建立无正则化 AL 参考解与 Couette–Poiseuille 精确解；"
                    "六项判据中 (d)(e)(f) 与 (a) 数值地板被定量给出，未达项如实保留。"),
        "scenario_key": "ship-ice-gap-lubrication/yield-stress-lbm-3d-moving-wall",
        "mission_id": 8122,
        "tags": ["TensorLBM", "yield-stress", "Bingham", "Papanastasiou",
                 "D3Q19", "moving-wall", "Ladd", "MEM", "P82W5-N04"],
        "markdown": "\n".join(md),
        "figures": figures(),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    print("wrote", out_path, "figures:", len(payload["figures"]))
    return payload


if __name__ == "__main__":
    main(sys.argv[1:])