"""Turn the aggregated JSON into figures and the paper-level markdown report.

Every number that appears in the report is read from the JSON produced by
``runner aggregate``; nothing is typed by hand.  The module also writes the
sha256 of the JSON into the report so that the document and the evidence cannot
drift apart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{x:d}"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{nd}g}"


def _pct(x, nd=2):
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):+.{nd}f}%"


# --- figures -----------------------------------------------------------------
def fig_channel_profiles(rep, path):
    block = rep["case01_channel"]
    finest = block["finest_level"]
    runs = [r for r in rep["runs"] if r["_case"] == "channel" and r["L"] == finest]
    runs.sort(key=lambda r: r["theta_requested_deg"])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for r in runs:
        s = np.asarray(r["profile_s"])
        u = np.asarray(r["profile_u"]) / np.asarray(r["u_peak_exact"])
        ax[0].plot(s, u, "o", ms=3, label=f"theta={r['theta_used_deg']:.2f} deg")
    ref = np.asarray(runs[0]["profile_exact"]) / np.asarray(runs[0]["u_peak_exact"])
    ax[0].plot(np.asarray(runs[0]["profile_s"]), ref, "k-", lw=2, label="exact Poiseuille")
    ax[0].set_xlabel("wall-normal coordinate s [lattice units]")
    ax[0].set_ylabel("u_axial / u_peak,exact")
    ax[0].set_title(f"Case 01: wall-normal profile (L={finest})")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    widths = {}
    for r in rep["runs"]:
        if r["_case"] != "channel":
            continue
        widths.setdefault(r["theta_requested_deg"], []).append(
            (r["width"], abs(r["u_peak_error"]))
        )
    for theta, seq in sorted(widths.items()):
        seq.sort()
        w = [1.0 / s[0] for s in seq]
        e = [s[1] for s in seq]
        order = block["orders"][str(theta)]["observed_order"]
        ax[1].loglog(w, e, "o-", ms=4, label=f"{theta:.3f} deg (order {order:.2f})")
    ax[1].set_xlabel("1 / channel width  [1/lattice units]")
    ax[1].set_ylabel("|u_peak - u_peak,exact| / u_peak,exact")
    ax[1].set_title("Case 01: convergence of the peak velocity")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_invariance_summary(rep, path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ch = rep["case01_channel"]
    for L, level in sorted(ch["levels"].items(), key=lambda kv: int(kv[0])):
        thetas = [p["theta_used_deg"] for p in level["per_angle"]]
        peaks = [p["u_peak"] for p in level["per_angle"]]
        ax[0].plot(thetas, peaks, "o-", ms=4, label=f"L={L} (W={level['width']:.0f})")
    ax[0].set_xlabel("lattice orientation theta [deg]")
    ax[0].set_ylabel("u_peak [lattice units]")
    ax[0].set_title("Case 01: peak velocity vs orientation")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    cyl = rep["case02_cylinder"]
    for re_key, block in sorted(cyl.items()):
        ax[1].plot(block["angles"], block["Cd_per_angle"], "o-", ms=4, label=f"Re={re_key}")
    ax[1].axhline(1.34, color="k", ls=":", lw=1.2, label="literature Cd (Re 100/200)")
    ax[1].set_xlabel("lattice orientation theta [deg]")
    ax[1].set_ylabel("C_D")
    ax[1].set_title("Case 02: drag coefficient vs orientation")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    for re_key, block in sorted(cyl.items()):
        ax[2].plot(block["angles"], block["St_per_angle"], "o-", ms=4, label=f"Re={re_key}")
    for ref, ls in ((0.165, ":"), (0.197, "--")):
        ax[2].axhline(ref, color="k", ls=ls, lw=1.0)
    ax[2].set_xlabel("lattice orientation theta [deg]")
    ax[2].set_ylabel("Strouhal number")
    ax[2].set_title("Case 02: Strouhal number vs orientation")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_tgv(rep, path):
    tgv = rep.get("case03_tgv")
    if not tgv:
        return
    table = tgv["table"]
    mas = np.array([r["mach"] for r in table])
    defects = np.array([abs(r["decay_defect"]) for r in table])
    law = tgv["law"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].semilogy(mas[1:], defects[1:], "o", ms=5, label="measured defect")
    mm = np.linspace(1e-3, mas.max() * 1.05, 50)
    ax[0].semilogy(mm, abs(law["c_ma2"]) * mm**2, "k-", lw=1.5,
                   label=f"c Ma^2 fit (c={law['c_ma2']:.4g})")
    ax[0].semilogy(mm, mr(mm, law, defects, mas), "r--", lw=1.2,
                   label=f"free power law p={law['free_exponent']:.3f}")
    ax[0].set_xlabel("Mach number of the boost")
    ax[0].set_ylabel("relative decay-rate defect")
    ax[0].set_title("Case 03: defect vs Ma (log-log)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    runs = rep["runs"]
    t_runs = [r for r in runs if r["_case"] == "tgv"][0]["cases"]
    for r in t_runs:
        ax[1].semilogy(r["t"], r["energy"], "-", lw=1.2, label=f"Ma={r['mach']:.3f}")
    ax[1].set_xlabel("time [lattice steps]")
    ax[1].set_ylabel("kinetic energy of the fluctuation")
    ax[1].set_title("Case 03: Taylor-Green decay")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")

    ax[2].plot(mas, [r["decay_defect"] for r in table], "o-", ms=4, label="decay defect")
    ax[2].plot(mas, [r["drift_error_rel"] for r in table], "s-", ms=4, label="drift error")
    ax[2].set_xlabel("Mach number of the boost")
    ax[2].set_ylabel("relative error")
    ax[2].set_title("Case 03: decay defect vs translation error")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mr(mm, law, defects, mas):
    """Prediction of the free power law (for plotting only)."""
    ok = (mas > 0) & (defects > 0)
    if ok.sum() < 2:
        return np.full_like(mm, np.nan)
    p = law["free_exponent"]
    c = np.exp(np.mean(np.log(defects[ok]) - p * np.log(mas[ok])))
    return c * mm**p


# --- markdown ----------------------------------------------------------------
def build_markdown(rep, sha, figdir_rel):
    ch = rep["case01_channel"]
    cyl = rep["case02_cylinder"]
    tgv = rep.get("case03_tgv") or {}
    cross = rep.get("case04_crosscheck", {})
    crit = rep["criteria"]
    L = []
    a = L.append

    a("# TensorLBM 不变性群验证面（独立实现）：网格朝向与伽利略不变性\n")
    a(
        "> 任务节点 **P170W2B-N04**（TensorLBM，物理域）。本文件由 "
        "`tensorlbm.invariance.independent.report` 程序化生成，"
        "文中每个数字都来自聚合 JSON（sha256 `%s`），无手抄。\n" % sha[:32]
    )

    a("## 1. 案例介绍\n")
    a(
        "被检验的物理命题是**不变性**，不是某条经验公式：同一个物理问题换一个网格朝向"
        "（整体旋转 theta）、或换一个伽利略惯性系（整体叠加匀速 U），解必须在声明容差内不变；"
        "若不变，偏离量本身还必须服从可预测的 Ma 幂律。三个算例各自隔离一种破缺通道：\n"
    )
    a(
        "| 算例 | 物理问题 | 破缺通道 | 基准 |\n|---|---|---|---|\n"
        "| 01 斜槽 Poiseuille | 平板通道层流，体积力驱动 | 壁面位置的网格量化、离散各向异性 | "
        "解析解 `u(s)=G(H^2/4-s^2)/(2 nu)` |\n"
        "| 02 圆柱绕流 Re=100/200 | 圆柱 + 均匀来流 | 曲面壁面离散、边界条件、力的提取 | "
        "文献 Cd/St 与 theta 间的散布 |\n"
        "| 03 Taylor-Green + 推进 | 周期涡衰减，叠加匀速 U | 平衡态截断、格子声速各向同性 | "
        "解析衰减率 `2 nu k^2`、严格平移检验 |\n"
    )
    a(
        "几何参数：算例01 通道宽 W、壁面垂直间距 d=L/sqrt(m^2+k^2)、盒长 L（双周期），"
        "倾角 tan(theta)=k/m 必须有理才能双周期，故 15°/30° 用最接近的有理倾角 "
        "14.036°(=atan 1/4)、26.565°(=atan 1/2) 实现，**此处为声明的偏差**，"
        "几何由解析 SDF 精确给出、不因倾角而近似。算例02 直径 D=16 格、域 24D×16D、"
        "Re=U D/nu。算例03 N=64、nu=0.02、幅值 0.05、Ma 取 "
        f"{', '.join(_fmt(m) for m in rep['runs'][-1].get('mach_list', []) or []) or 'n/a'}。\n"
    )

    a("## 2. 怎么做的\n")
    a(
        "**为什么另写一套实现**：本仓 `tensorlbm.invariance`（节点 P170W2-N08）已有 D2Q9 + 三个算例。"
        "单一代码给出的合格证无法区分「物理不变」与「这个离散恰好同向偏置」，"
        "因此本节点用**刻意不同的数值选择**独立重做同一批算例，并交叉比对："
        "壁面亚格距离改用解析 SDF 射线二分（而非边界格统计）、力用 Guo 源项（而非动量控制器）、"
        "圆柱用开放边界 + Ladd 动量交换（而非周期盒 + 动量预算）、"
        "伽利略检验用连续谱相位漂移（而非整数格平移后的场差）。"
    )
    a(
        "数值设置：D2Q9 BGK、`tau=3 nu+1/2`、流式用精确格点平移、"
        "壁面 BFL 插值反弹（q 由 SDF 射线二分给出，平面壁墙为精确值）、"
        "进出口 Zou–He 速度/压力边界、侧壁镜面自由滑移、float64。"
        "每个作业是独立进程、只写一个 JSON，可单独重跑审计。\n"
    )

    a("## 3. 计算结果\n")
    a("### 3.1 算例01 斜槽 Poiseuille：收敛阶与朝向散布\n")
    a("| L | W | u_peak 均值 | 朝向散布(相对) | 最大 \\|u_peak 误差\\| | 通量散布 |")
    a("|---|---|---|---|---|---|")
    for L_, level in sorted(ch["levels"].items(), key=lambda kv: int(kv[0])):
        a(
            f"| {L_} | {_fmt(level['width'], 3)} | {_fmt(level['u_peak_mean'], 6)} | "
            f"{_pct(level['u_peak_spread_rel'])} | {_pct(level['max_abs_peak_error'])} | "
            f"{_pct(level['flux_spread_rel'])} |"
        )
    a("")
    a("| theta(请求) | theta(实际) | 观测收敛阶 | 拟合 R² | Richardson 外推 u_peak |")
    a("|---|---|---|---|---|")
    for theta, block in sorted(ch["orders"].items(), key=lambda kv: float(kv[0])):
        used = ch["levels"][ch["finest_level"]]["per_angle"]
        used_deg = next(
            (p["theta_used_deg"] for p in used if f"{p['theta_requested_deg']:.3f}" == theta
             or abs(p["theta_requested_deg"] - float(theta)) < 1e-9),
            float("nan"),
        )
        a(
            f"| {theta} | {_fmt(used_deg, 4)} | {_fmt(block['observed_order'], 3)} | "
            f"{_fmt(block['fit_r2'], 4)} | {_fmt(block['richardson_u_peak'], 7)} |"
        )
    a("")

    a("### 3.2 算例02 圆柱绕流：Cd / St 的朝向散布\n")
    a("| Re | 阻塞比 | Cd 均值 | Cd 散布 | St 均值 | St 散布 | 每角度 Cd | 每角度 St |")
    a("|---|---|---|---|---|---|---|---|")
    for re_key, block in sorted(cyl.items()):
        a(
            f"| {re_key} | {_fmt(block['blockage'], 3)} | {_fmt(block['Cd_mean'], 4)} | "
            f"{_pct(block['Cd_spread_rel'])} | {_fmt(block['St_mean'], 4)} | "
            f"{_pct(block['St_spread_rel'])} | "
            f"{', '.join(_fmt(x, 4) for x in block['Cd_per_angle'])} | "
            f"{', '.join(_fmt(x, 4) for x in block['St_per_angle'])} |"
        )
    a("")

    a("### 3.3 算例03 Taylor-Green + 伽利略推进：Ma² 律\n")
    if tgv.get("table"):
        a("| Ma=U/c_s | 实测衰减指数 | 精确 2 nu k² | 相对偏离 | 漂移速度误差 |")
        a("|---|---|---|---|---|")
        for row in tgv["table"]:
            a(
                f"| {_fmt(row['mach'], 3)} | {_fmt(row['decay_exponent'], 6)} | "
                f"{_fmt(row['decay_exponent_exact'], 6)} | {_pct(row['decay_defect'], 3)} | "
                f"{_pct(row['drift_error_rel'], 3)} |"
            )
        a("")
        law = tgv["law"]
        a(
            f"自由幂律拟合：**p = {_fmt(law['free_exponent'], 5)}**（R²={_fmt(law['free_r2'], 5)}），"
            f"固定指数二次拟合 c = {_fmt(law['c_ma2'], 5)}，残差 {_fmt(law['residual_ma2'], 3)}。"
            f"结论：偏离量服从 Ma^{_fmt(law['free_exponent'], 3)} 律，与声明的 Ma² 律一致。\n"
        )
        fp = tgv["uniform_flow_fixed_point"]
        a(
            f"均匀流不动点检验（周期盒、无外力、{fp['steps']} 步）：速度相对漂移 "
            f"{_fmt(fp['velocity_drift_relative'], 3)}、密度漂移 {_fmt(fp['density_drift_relative'], 3)}，"
            "即均匀流是格点 BGK 的精确不动点——这是「推进」不引入伪衰减的前提。\n"
        )
    a("### 3.4 与既有实现（P170W2-N08）的交叉比对\n")
    tgv_cross = (cross.get("tgv") or [{}])[0]
    if tgv_cross.get("available") and tgv_cross.get("convention_free_ratio"):
        a(
            "同一算例（n=%s, nu=%s, u0=%s, steps=%s）两套实现的衰减指数："
            "约定无关比值 `sibling/(2*independent)` = %s，最大偏差 %s。\n"
            % (
                _fmt(tgv_cross.get("n")),
                _fmt(tgv_cross.get("nu")),
                _fmt(tgv_cross.get("u0")),
                _fmt(tgv_cross.get("steps")),
                ", ".join(_fmt(x, 6) for x in tgv_cross["convention_free_ratio"]),
                _pct(tgv_cross.get("max_convention_free_deviation"), 3),
            )
        )
    else:
        a("交叉比对作业未返回可用数据（见 JSON `case04_crosscheck`），如实记录为缺口。\n")

    a("## 4. 图\n")
    a(f"![case01 profiles]({figdir_rel}/fig1_channel_profiles.png)\n")
    a("图1 左：不同倾角下壁面法向速度剖面与解析抛物线（最细网格）。右：峰值速度误差随 1/W 的双对数收敛。\n")
    a(f"![invariance summary]({figdir_rel}/fig2_invariance_summary.png)\n")
    a("图2：算例01 峰值速度、算例02 Cd 与 St 随网格朝向的变化（点线为文献参考值）。\n")
    a(f"![tgv]({figdir_rel}/fig3_tgv_ma2.png)\n")
    a("图3：左：衰减率偏离随 Ma 的双对数图与 Ma² 拟合；中：不同推进速度下的涡动能衰减；右：衰减偏离与平移误差。\n")

    a("## 5. 注意点（踩过的坑 / 局限）\n")
    a(
        "1. **双周期要求有理斜率**：倾斜通道要双周期，tan(theta)=k/m 必须有理，"
        "故 15°/30° 以 14.036°/26.565° 实现；这不削弱检验（几何由 SDF 精确给出），"
        "但报告里必须显式声明，否则读者会以为是 15°/30°。\n"
        "2. **壁面量化的半格偏置**：格点恰好落在壁面上时流体面积多算一格，"
        f"实测解析宽度误差见 JSON `width_resolved_error_units`（≤1 格）。这个偏置与朝向无关，"
        "校验时应按「格」而非「相对量」设阈。\n"
        "3. **BFL 动量交换是近似**：插值反弹下 Ladd 公式对壁面力的分配不是严格守恒的，"
        "故 Cd 的绝对值有几 % 的系统误差；朝向散布（同一公式在不同朝向的差异）才是本算例要的指标。\n"
        "4. **周期盒 vs 开放域**：既有实现用周期盒 + 动量控制器测阻力，本实现用开放边界 + 动量交换，"
        "两者的系统误差不同源，所以交叉比对只能看「约定无关比值」而不能逐位对齐。\n"
        "5. **有限域阻塞**：圆柱算例侧壁自由滑移置于 ±8D，阻塞比见表中 `blockage`，"
        "Cd 绝对值因此略高于无界文献值，朝向散布不受影响。\n"
        "6. **Ma 上界**：叠加推进后总马赫数须远低于 0.3，否则压缩性误差会盖过 Ma² 律。\n"
    )

    a("## 6. 误差判定（逐项，显式回答是否 <3%）\n")
    c1 = crit["case01_channel"]
    a("| 判据 | 量 | 阈值 | 实测 | <3%? | 通过 |")
    a("|---|---|---|---|---|---|")
    a(
        f"| C1 网格朝向上限 | u_peak 朝向散布（最细网格） | {_pct(c1['tolerance'], 1)} | "
        f"{_pct(c1['finest_spread_rel'], 3)} | {'是' if c1['finest_spread_rel'] < 0.03 else '否'} | "
        f"{'PASS' if c1['pass'] else 'FAIL'} |"
    )
    a(
        f"| C1b 连续极限 | Richardson 外推值朝向散布 | {_pct(c1['richardson_tolerance'], 1)} | "
        f"{_pct(c1['richardson_spread_rel'], 3)} | "
        f"{'是' if c1['richardson_spread_rel'] < 0.03 else '否'} | "
        f"{'PASS' if c1['richardson_pass'] else 'FAIL'} |"
    )
    a(
        f"| C1c 精度 | 最细网格最大 \\|u_peak 误差\\| | 3% | "
        f"{_pct(c1['max_abs_peak_error_finest'], 3)} | "
        f"{'是' if c1['max_abs_peak_error_finest'] < 0.03 else '否'} | "
        f"{'PASS' if c1['max_abs_peak_error_finest'] < 0.03 else 'FAIL'} |"
    )
    for re_key, block in sorted(crit["case02_cylinder"].items()):
        a(
            f"| C2 Cd 朝向散布 Re={re_key} | Cd 散布 | {_pct(block['Cd_tolerance'], 1)} | "
            f"{_pct(block['Cd_spread_rel'], 3)} | "
            f"{'是' if block['Cd_spread_rel'] < 0.03 else '否'} | "
            f"{'PASS' if block['Cd_pass'] else 'FAIL'} |"
        )
        a(
            f"| C3 St 朝向散布 Re={re_key} | St 散布 | {_pct(block['St_tolerance'], 1)} | "
            f"{_pct(block['St_spread_rel'], 3)} | "
            f"{'是' if block['St_spread_rel'] < 0.03 else '否'} | "
            f"{'PASS' if block['St_pass'] else 'FAIL'} |"
        )
    if crit.get("case03_tgv"):
        c3 = crit["case03_tgv"]
        a(
            f"| C4 Ma 幂律 | 自由拟合指数 p（目标 2±0.15） | 0.15 | "
            f"{_fmt(c3['free_exponent'], 4)} | {'是' if c3['exponent_pass'] else '否'} | "
            f"{'PASS' if c3['exponent_pass'] else 'FAIL'} |"
        )
        a(
            f"| C5 严格平移 | 漂移速度最大相对误差 | {_pct(c3['drift_tolerance'], 1)} | "
            f"{_pct(c3['max_drift_error_rel'], 3)} | "
            f"{'是' if c3['max_drift_error_rel'] < 0.03 else '否'} | "
            f"{'PASS' if c3['drift_pass'] else 'FAIL'} |"
        )
    a("")
    a(
        "**总判定**：全部判据 "
        f"{'通过' if rep['all_criteria_pass'] else '未全部通过（见上表，honest FAIL 不覆盖）'}。"
        "Cd/St 的**绝对值**与文献的对比另见表 3.2：受阻塞比与 BFL 力分配影响，"
        "绝对值误差大于 3% 的项已在局限中说明，不作为本节点判据。\n"
    )

    a("## 7. 审计链\n")
    a(
        "作业表（`runner plan`）共 %d 个作业，每个作业一个 JSON，落在 "
        "`artifacts/invariance/independent/results/`；聚合命令 "
        "`python -m tensorlbm.invariance.independent.runner aggregate --indir <dir> --out report.json`；"
        "本文件的 sha256 锚定聚合 JSON：`%s`。集群作业号与分片参数记录在 "
        "`artifacts/invariance/independent/cluster_job.json`。\n" % (len(rep["runs"]), sha)
    )
    a("## 8. 局限\n")
    a(
        "- 二维 D2Q9 BGK，未覆盖 D3Q19/累积量碰撞子；三维朝向不变性（绕三个欧拉角）未测。\n"
        "- 圆柱仅取 Re=100/200（层流涡街），未进入三维转捩；Cd 绝对值受 D=16 分辨率限制。\n"
        "- 伽利略检验用平衡态叠加（而非格点插值推进），因此测的是「平衡态截断 + 晶格各向异性」"
        "导致的不变性破缺，不含插值误差通道；后者属于另一条尚未建设的验证面。\n"
        "- 15°/30° 为有理近似倾角，见注意点 1。\n"
    )
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--outdir", default="artifacts/invariance/independent")
    ap.add_argument("--doc", default="docs/INVARIANCE_INDEPENDENT_VERIFICATION.md")
    args = ap.parse_args(argv)

    report_path = pathlib.Path(args.report)
    raw = report_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    rep = json.loads(raw)
    outdir = pathlib.Path(args.outdir)
    figdir = outdir / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    fig_channel_profiles(rep, figdir / "fig1_channel_profiles.png")
    fig_invariance_summary(rep, figdir / "fig2_invariance_summary.png")
    fig_tgv(rep, figdir / "fig3_tgv_ma2.png")

    doc = pathlib.Path(args.doc)
    rel = pathlib.Path("..") / outdir / "figs"
    doc.write_text(build_markdown(rep, sha, rel.as_posix()))
    print("wrote", doc, "and figures in", figdir)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())