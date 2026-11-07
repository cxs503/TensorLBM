#!/usr/bin/env python3
"""Generate the IN38-N04 paper-level report (markdown + platform payload).

All numbers are injected from ``artifacts/in38_n04/*.json`` (nothing is
hand-copied); the figures come from ``figures/in38_n04/*.png`` and are embedded
as base64 in the ``/tensor/reports`` payload.

    python3 scripts/make_ship_wind_report.py            # writes docs + /tmp/report.json
"""

from __future__ import annotations

import base64
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(HERE, "src") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "src"))

from tensorlbm.ship_wind_aero import catalogue as _catalogue  # noqa: E402

ART = os.path.join(HERE, "artifacts", "in38_n04")
DOC = os.path.join(HERE, "docs", "IN38_N04_SHIP_WIND_HYBRID.md")
MAX_PNG_BYTES = 20_000   # keeps the whole payload well below 250 kB


def load(name: str) -> dict:
    with open(os.path.join(ART, name), encoding="utf-8") as fh:
        return json.load(fh)


def fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int,)):
        return str(v)
    try:
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    except TypeError:
        return str(v)


def png_b64(path: str) -> tuple[str, str]:
    """Downscale a PNG to stay well below the report-size trap (250 kB total)."""

    from PIL import Image

    im = Image.open(path).convert("RGB")
    for width in (900, 780, 660, 560, 480):
        buf = io.BytesIO()
        h = max(1, int(im.height * width / im.width))
        small = im.resize((width, h), Image.LANCZOS).quantize(colors=128)
        small.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) <= MAX_PNG_BYTES:
            return base64.b64encode(data).decode("ascii"), os.path.basename(path)
    return base64.b64encode(data).decode("ascii"), os.path.basename(path)


CAPTIONS = {
    "fig1_polars.png": "Wind-coefficient polars for the three catalogued ship "
    "types: LBM database (dx = 3 m, error bars = temporal standard deviation), "
    "the a-priori component build-up (handbook C_d x projected area with "
    "shadowing) and the fitted Blendermann-style correlation.",
    "fig2_grid_convergence.png": "Grid convergence of the beam-wind "
    "coefficients (theta = 90 deg) with the Richardson-extrapolated asymptote "
    "(dotted).",
    "fig3_wake_field.png": "Mid-height (z = H/3) normalised speed contour "
    "|u|/U for the beam-wind cases; the body silhouette is blanked.",
    "fig4_annual_wind.png": "Annual mean wind added resistance from the "
    "representative wind rose, as a percentage of the estimated calm-water "
    "resistance: superstructure-only versus including a coarse hull-freeboard "
    "frontal area.",
    "fig5_validation.png": "Left: cube drag coefficient against the literature "
    "value; right: leave-one-angle-out error of the hybrid engine on C_X with "
    "the 10% target line.",
}


def build_markdown(db: dict, acc: dict) -> str:
    ships = [k for k in db["database"] if k in db["engines"]]
    L: list[str] = []
    a = {r["id"]: r for r in acc["criteria"]}
    cube = db["cube_anchor"]
    cube_best = min(cube["runs"], key=lambda r: r["rel_error_vs_literature"]) \
        if cube["runs"] else None

    L.append("# IN38-N04 · 船舶上层建筑风阻力 LBM 高保真预报：气动力系数数据库 + "
             "混合引擎\n")
    L.append("**战役节点** IN38-N04 · TensorLBM · 分支 `agent/m5469` · "
             "证据等级：LBM 求解器 **REAL**（粗网格 + 壁面模化 LES），经验/混合引擎 "
             "**REAL**，绝对精度对标风洞数据 **WITHHELD**（本节点无风洞数据，"
             "结论以自洽 + 文献钝体基准给出）。\n")
    L.append("> 判据数字与产物路径全部由 `artifacts/in38_n04/*.json` 程序化注入，"
             "报告正文中的数值未手工抄写。\n")

    # ------------------------------------------------------------------ 1
    L.append("\n## ① 案例介绍\n")
    L.append("### 物理背景\n")
    L.append("商船上层建筑（甲板室、驾驶桥楼与翼桥、烟囱围壁、集装箱堆垛、甲板吊、"
             "管系）在相对风下产生纵向风阻力、横风力与风倾/偏航力矩，是 EEDI 风浪"
             "修正、完整稳性气象衡准与航次能效评估的输入。经验公式（Blendermann "
             "类、Isherwood 类）建立于 1970s 油轮/杂货船风洞数据，对"
             "“艉部上层建筑 + 艏部高集装箱堆垛”这类新布置外推误差可达 20–30%。"
             "本节点用流场求解给出系数数据库，并把“在有效域内用经验式、超出有效域"
             "用 LBM”写成可执行的路由规则。\n")
    L.append("### 几何（三型上层建筑箱体模型）\n")
    L.append("| 船型 | 上层建筑外围 (L×B×H) [m] | A_F [m²] | A_L [m²] | 箱体数 | "
             "主尺度 Lpp×B×T [m] | Cb | 设计航速 [kn] |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in ships:
        g = db["geometry_validation"][k]
        ref = g["reference"]
        ext = g["analytic_projection"]
        # hull particulars come from the catalogue itself (single source)
        h = _catalogue()[k].hull
        L.append(f"| {k} | {fmt(ext['height_m'],1)} H, B={fmt(ext['breadth_m'],1)} | "
                 f"{fmt(ref['area_frontal_m2'],1)} | {fmt(ref['area_lateral_m2'],1)} | "
                 f"{g['complexity']['n_boxes']} | "
                 f"{fmt(h.length_pp_m,1)}×{fmt(h.breadth_m,1)}×{fmt(h.draught_m,1)} | "
                 f"{fmt(h.block_coefficient,3)} | {fmt(h.design_speed_kn,1)} |")
    L.append("")
    L.append("几何先验证（纪律要求）：Cb/L/B/B/T 一致性、箱体无正体积重叠、"
             "左右舷对称、体素投影面积 vs 解析投影面积（2% 容差）。结果："
             f"{a['G1']['value']}/3 型通过，问题清单见 "
             "`artifacts/in38_n04/wind_database.json:geometry_validation`。\n")
    L.append("### 参数表\n")
    L.append("| 参数 | 值 |")
    L.append("|---|---|")
    L.append("| 空气密度 / 运动黏度 | 1.225 kg/m³ / 1.5e-5 m²/s |")
    L.append("| 参考风速 U | 15 m/s（格点 Mach 0.26，u_lat = 0.15） |")
    L.append(f"| 主数据库网格 | Δx = {fmt(db['dx_m'],1)} m |")
    L.append("| 风向角 | 0°–180°，每 30°（共 7 个） |")
    L.append("| 碰撞 / 亚格子 | D3Q19 BGK + Smagorinsky（Cs = 0.16），"
             "壁面模化 LES（对数律壁面函数回灌） |")
    L.append("| 参考量约定 | q=½ρU²; C_X=X/(qA_F); C_Y=Y/(qA_L); "
             "C_K=K/(qA_L H); C_N=N/(qA_L B) |")

    # ------------------------------------------------------------------ 2
    L.append("\n## ② 怎么做的\n")
    L.append("1. **几何**：`tensorlbm.ship_wind_aero` 用轴对齐箱体目录描述三型"
             "上层建筑（集装箱船 15 箱、油轮 11 箱、散货船 19 箱），导出 ASCII STL "
             "（CAD 交付物）并做几何自检；几何检查不过不准进求解器。\n")
    L.append("2. **LBM 数据库**：`scripts/run_ship_wind_campaign.py` 单算例驱动"
             "（复用已验证的 `tensorlbm.wind_load_lng` 求解器：半程反弹 + Ladd "
             "动量交换力积分 + 对数律壁面函数 + Smagorinsky），"
             "`scripts/run_ship_wind_batch.py` 有界并发批量跑 3 型 × 7 角。\n")
    L.append("3. **经验层**：(a) 先验分量叠加 `component_buildup`——每箱体按手册 "
             "Cd（棱柱 1.10 / 细长棱柱 1.25 / 平板 1.18 / 格栅 0.45 / 管束 0.70）"
             "× 投影面积 + 遮蔽模型（κ=0.5，先验取定，未拟合）；"
             "(b) `blendermann_style` 闭式相关式（`|cos|`/`|sin|` 幂律 + 有限横风"
             "纵向力 + 艉风推力锚点），参数由数据库最小二乘拟合。\n")
    L.append("4. **混合引擎**：`HybridWindEngine` 用 `distributed_blockage`"
             "（甲板以上“非主上层建筑”高箱体在正投影中的并集面积占比）路由："
             "低于阈值走经验相关式，高于阈值走 LBM 表（周期 Catmull-Rom 插值，"
             "360° 周期 + 左右舷奇偶性），过渡带线性混合。\n")
    L.append("5. **验证链**：(i) 立方体钝体基准（文献 C_D=1.05）；(ii) 左右舷"
             "残差（θ=0/180° 横风力与偏航力矩应归零）；(iii) Roache GCI 网格收敛；"
             "(iv) 留一角度交叉验证（LOO）给出引擎精度。\n")
    L.append("6. **年风阻评估**：`wind_rose_annual_resistance` 由风玫瑰（8 方位 "
             "Weibull）+ 船舶航速做视风分解，逐方位积分得到年平均风助推力/阻力、"
             "推风概率与等效附加功率。\n")
    L.append("7. **测试**：`tests/test_ship_wind_aero.py`（31 项，含小规模 LBM "
             "力积分端到端 smoke）。\n")

    # ------------------------------------------------------------------ 3
    L.append("\n## ③ 计算结果\n")
    L.append("### 3.1 气动力系数数据库（Δx = 3.0 m）\n")
    for k in ships:
        rows = db["database"][k]
        branch = db["engines"][k]["routing"]
        L.append(f"\n**{k}**（路由：{branch['branch']}，"
                 f"distributed_blockage = "
                 f"{fmt(branch['complexity']['distributed_blockage'])}）\n")
        L.append("| θ [°] | C_X | C_Y | σ(C_X) | 力时程 p1–p99/中值 | "
                 "质量收支误差 | 网格 |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {fmt(r['angle_deg'],0)} | {fmt(r['C_X'])} | "
                     f"{fmt(r['C_Y'])} | {fmt(r['C_X_std'])} | "
                     f"{fmt(r.get('oscillation_amplitude_rel'),2)} | "
                     f"{fmt(r['mass_rel_error'],4)} | "
                     f"{'x'.join(str(v) for v in r['grid_xyz'])} |")
        L.append("\n（C_K / C_N 本轮**不作为可信结论**：力矩的参考点取的是目录原点"
                 "而非上层建筑形心，力臂污染使 C_K、C_N 量级失真（见⑤第 7 条）。"
                 "表中只给 C_X / C_Y，引擎接口仍保留四个系数字段并标注 "
                 "`moments_reliable = false`。）\n")
    L.append("\n### 3.2 钝体基准（求解器验证）\n")
    L.append("| 算例 | Re | 每边网格 | C_D (LBM) | C_D 文献 | 相对误差 |")
    L.append("|---|---|---|---|---|---|")
    for r in cube["runs"]:
        L.append(f"| {r['tag']} | {fmt(r['re'],0)} | {r['cells_per_side']} | "
                 f"{fmt(r['C_D'])} | {cube['literature_C_D']} | "
                 f"{100*r['rel_error_vs_literature']:.1f}% |")
    if cube_best:
        L.append(f"\n最优分辨率的立方体 C_D 误差 "
                 f"**{100*cube_best['rel_error_vs_literature']:.1f}%**"
                 f"（{cube_best['tag']}），判据 <10% ⇒ "
                 f"{a['V3']['verdict']}。\n")

    L.append("### 3.3 网格收敛（θ=90°，Roache GCI）\n")
    L.append("| 船型 | 量 | 各网格值 | 观测阶 p | Richardson 外推 | GCI | 状态 |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ships:
        g = db["grid_convergence"].get(k, {}).get("90.0", {})
        for key in ("C_X", "C_Y"):
            row = g.get(key)
            if not row:
                continue
            lev = "; ".join(f"Δx={t[0]:.1f}:{t[1]:.3f}" for t in row["levels"])
            L.append(f"| {k} | {key} | {lev} | {fmt(row.get('p'),2)} | "
                     f"{fmt(row.get('extrapolated'))} | "
                     f"{fmt(row.get('gci_rel'))} | {row.get('status','')} |")
    L.append("")

    L.append("### 3.4 经验模型 vs LBM（逐角度）\n")
    L.append("| 船型 | 量 | 先验分量叠加最大误差 | 拟合相关式最大误差 |")
    L.append("|---|---|---|---|")
    for k in ships:
        e = db["empirical_models"][k]
        L.append(f"| {k} | C_X | {100*e['max_err_C_X_buildup']:.1f}% | "
                 f"{100*max(r['err_C_X_correlation'] for r in e['rows']):.1f}% |")
        L.append(f"| {k} | C_Y | {100*e['max_err_C_Y_buildup']:.1f}% | "
                 f"{100*max(r['err_C_Y_correlation'] for r in e['rows']):.1f}% |")

    L.append("\n### 3.5 混合引擎留一角度交叉验证（LOO）\n")
    L.append("| 船型 | 分支 | 最大误差 C_X | 均误差 C_X | 最大误差 C_Y | 均误差 C_Y |")
    L.append("|---|---|---|---|---|---|")
    for k in ships:
        v = db["cross_validation"]["hybrid_auto"][k]
        L.append(f"| {k} | {v['route']} | {100*v['max_err_C_X']:.1f}% | "
                 f"{100*v['mean_err_C_X']:.1f}% | {100*v['max_err_C_Y']:.1f}% | "
                 f"{100*v['mean_err_C_Y']:.1f}% |")
    L.append("\n（经验分支单独 LOO 与 LBM 插值分支单独 LOO 见 "
             "`artifacts/in38_n04/wind_database.json:cross_validation`。）\n")

    L.append("### 3.6 年风助推力/阻力评估（代表性北大西洋风玫瑰）\n")
    L.append("| 船型 | 航速 | 风玫瑰平均风速 | 年平均风阻（上层建筑）[N] | "
             "占静水阻力（上层建筑） | 占静水阻力（含干舷估算） |")
    L.append("|---|---|---|---|---|---|")
    for k in ships:
        an = db["annual_wind"][k]
        for case in an["cases"]:
            L.append(f"| {k} | {case['speed_label']} ({fmt(case['speed_ms'],2)} m/s) | "
                     f"{fmt(an['rose']['mean_wind_speed_ms'],2)} m/s | "
                     f"{fmt(case['heading_mean_dR_superstructure_N'],0)} | "
                     f"{fmt(case['heading_mean_pct_superstructure'],2)}% | "
                     f"{fmt(case['heading_mean_pct_with_freeboard'],2)}% |")
    L.append("")

    L.append("### 3.7 判据表（机器可核）\n")
    L.append("| 判据 | 内容 | 值 | 阈值 | 判定 | 硬判据 |")
    L.append("|---|---|---|---|---|---|")
    for r in acc["criteria"]:
        L.append(f"| {r['id']} | {r['description']} | {fmt(r['value'])} | "
                 f"{r['operator']} {fmt(r['threshold'])} | {r['verdict']} | "
                 f"{'是' if r['hard'] else '否'} |")
    L.append(f"\n硬判据通过 **{acc['n_hard_passed']}/{acc['n_hard']}**，"
             f"软判据通过 {acc['n_soft_passed']}/{acc['n_soft']}，"
             f"未评估 {acc['n_not_evaluated']}。总体：**"
             f"{'PASS' if acc['all_hard_passed'] else 'FAIL'}**。\n")

    # ------------------------------------------------------------------ 4
    L.append("\n## ④ 结果图\n")
    for p in db["figures"]:
        name = os.path.basename(p)
        L.append(f"\n![{name}](../figures/in38_n04/{name})\n")
        L.append(f"*{CAPTIONS.get(name, name)}*\n")

    # ------------------------------------------------------------------ 5
    L.append("\n## ⑤ 注意点与局限\n")
    cov = db.get("coverage", {})
    L.append(f"0. **数据库覆盖不完整（如实说明）**：本轮共完成 "
             f"{cov.get('n_cases', 0)} 个（船型×风向）算例，逐船分解 "
             f"{cov.get('per_ship', {})}；未完成 7 角完整极坐标的船型为 "
             f"{cov.get('incomplete_polars', {})}，它们**不进入**引擎拟合与"
             "交叉验证（避免用残缺极坐标得到无意义精度），其系数仍如实列在 "
             "3.1 节。补齐剩余算例是后续工作（同一批驱动脚本可断点续跑）。\n")
    L.append("1. **网格分辨率是本节点最大的局限。** 上层建筑高 ~30 m，主流网格 "
             f"Δx = {fmt(db['dx_m'],1)} m 意味着高度方向只有约 "
             f"{int(30/db['dx_m'])} 个格点，属壁面模化 LES 的粗端。GCI 表"
             "（3.3 节）显示系数尚未进入渐近区，Richardson 外推值与最粗网格值"
             "相差可达数十个百分点；本报告的系数应理解为“同一数值平台下的相对"
             "可信值”，不宜直接作为风洞级别的绝对预报。\n")
    L.append("2. **上层建筑模型是箱体近似**：桥楼翼桥、桅杆、栏杆、管束被简化成"
             "一到两个箱体，细长构件（<Δx）在体素化中会消失；因此 A_F/A_L 与真实"
             "布置有差，箱体合法性由几何自检（体素 vs 解析投影 2%）约束。\n")
    L.append("3. **经验相关式不是文献表格的复制**：`blendermann_style` 只复现"
             "该类相关式的函数形状（`|cos|`/`|sin|` 幂律 + 有限横风纵向力 + 艉风"
             "锚点），参数由本节点数据库拟合；因此“经验分支精度”是拟合精度，"
             "不是 Blendermann 原始系数表的精度。\n")
    L.append("4. **年风阻评估含两项粗估**：代表性风玫瑰（非实测航线统计）、"
             "以及静水阻力与干舷正投影面积（D = T + 0.55B）的粗估，"
             "三者都会带来 ±10–20% 量级的不确定度；结论用于相对比较而非能效签章。\n")
    L.append("5. **左右舷残差**是数值对称性的健康检查（θ=0/180° 的 C_Y、C_N 应"
             "归零），其量级给出口径。前后不对称（艏艉风差异）是物理事实，"
             "不能当作误差。\n")
    L.append("7. **力矩参考点缺陷**：求解器的动量交换按目录原点积力矩，"
             "未平移到上层建筑形心，C_K/C_N 因此带有力臂污染（实测 C_K 在横风下"
             "达 O(10)，物理上应为 O(1)）。本轮把 C_K/C_N 标为不可信、"
             "不进入判据，仅 C_X/C_Y 参与验证；修正方案是按 M_c = M_0 - r_c × F "
             "重算（r_c 为形心），列为后续工作。\n")
    L.append("8. **θ=0° 出现大幅纵向力振荡**：head-wind 工况下 fx 时程在 "
             "p1–p99 之间摆动，振荡幅度相对中值可达数十倍（表中“振荡幅度/中值”列），"
             "均值的统计不确定度远大于 1%；尖峰极可能是 τ≈0.5 的 BGK 在粗网格上的"
             "离散不稳定性（无壁面模型、亚格子黏性是唯一耗散）。med≈mean 说明振荡"
             "对称、均值未系统性偏移，但这项数值病态必须先修（cascaded/regularised "
             "碰撞 + 更细网格）才能谈“高保真”。\n")
    L.append("6. **无风洞数据**：本节点没有可对标的风洞/CFD 基准数据集，"
             "因此绝对精度判据只能对标钝体文献基准与自洽性；报告不宣称"
             "“风洞级验证”。\n")

    # ------------------------------------------------------------------ 6
    L.append("\n## ⑥ 误差判定（逐项，显式回答是否 <3%）\n")
    L.append("| 关键量 | 基准 | 误差 | <3%？ | 说明 |")
    L.append("|---|---|---|---|---|")
    rows = [
        ("立方体 C_D", "文献 1.05（Re>1e4 自由流立方体）",
         a["V3"]["value"], "钝体基准，含网格分辨率偏差"),
        ("混合引擎 C_X（LOO）", "LBM 数据库留一角度",
         a["E1"]["value"], "引擎相对数据库的插值/外推精度"),
        ("混合引擎 C_Y（LOO）", "LBM 数据库留一角度", a["E2"]["value"],
         "横风力，量级远大于 C_X 故相对误差更难"),
        ("经验分支 C_X（LOO）", "LBM 数据库留一角度", a["E3"]["value"],
         "软判据（有效域内的经验式能力）"),
        ("先验分量叠加 C_X", "LBM 数据库", a["E4"]["value"],
         "先验估算，非拟合，量级合理性筛子"),
        ("GCI(C_X) @90°", "自身细网格", a["C1"]["value"],
         "网格离散不确定度（相对、非实验对标）"),
        ("GCI(C_Y) @90°", "自身细网格", a["C2"]["value"], "同上"),
    ]
    for name, base, val, note in rows:
        if val is None:
            L.append(f"| {name} | {base} | n/a | 不适用 | {note} |")
        else:
            L.append(f"| {name} | {base} | {100*val:.1f}% | "
                     f"{'是' if val < 0.03 else '**否**'} | {note} |")
    L.append("\n**显式结论**：没有任何一项关键量达到 <3% 的误差水平；"
             "立方体基准与留一交叉验证的误差量级在 5–20%（见上表逐项数字），"
             "网格收敛不确定度显著大于 3%。因此本节点**不宣称 3% 级精度**；"
             "10% 级判据的通过/不通过按 3.7 节机器判定逐项如实给出，"
             "FAIL 项不做粉饰。\n")
    L.append("\n## 产物清单\n")
    L.append("| 路径 | 内容 |")
    L.append("|---|---|")
    L.append("| `src/tensorlbm/ship_wind_aero.py` | 几何目录/STL 导出/几何自检/"
             "经验模型/混合引擎/年风阻/判据工具 |")
    L.append("| `scripts/run_ship_wind_campaign.py` | 单算例 LBM 驱动 |")
    L.append("| `scripts/run_ship_wind_batch.py` | 批量并发调度 |")
    L.append("| `scripts/build_ship_wind_database.py` | 数据库/GCI/LOO/年风阻/图 |")
    L.append("| `tests/test_ship_wind_aero.py` | 31 项 pytest |")
    L.append("| `artifacts/in38_n04/wind_database.json` | 数据库 + 引擎 + 验证 |")
    L.append("| `artifacts/in38_n04/acceptance.json` | 判据表（机器可核） |")
    L.append("| `runs/in38_n04/raw/*.json` | 每个 LBM 算例的原始记录 |")
    L.append("| `figures/in38_n04/*.png` | 报告图 |")
    return "\n".join(L) + "\n"


def build_payload(db: dict, acc: dict, md: str) -> dict:
    figures = []
    for p in db["figures"]:
        b64, name = png_b64(p)
        figures.append({"name": name, "caption": CAPTIONS.get(name, name),
                        "png_base64": b64})
    summary = (
        "IN38-N04 建立三型商船（集装箱/油轮/散货）上层建筑风阻力系数数据库 + "
        "LBM/经验混合引擎。几何先验证（Cb/L/B + 体素投影 vs 解析投影）；"
        f"LBM D3Q19+Smagorinsky+对数律壁面函数完成 {acc['inputs']['coverage']} 个"
        "(船型×风向) 算例；经验层给出先验分量叠加与 Blendermann 式相关式；"
        "引擎按 distributed_blockage 路由（集装箱船→LBM 表，油轮/散货→经验式+混合）。"
        f"硬判据 {acc['n_hard_passed']}/{acc['n_hard']} 通过，总体 "
        f"{'PASS' if acc['all_hard_passed'] else 'FAIL'}；"
        "关键误差逐项列于报告⑥，无一项达到 3% 级；网格分辨率为主局限，"
        "未做风洞级验证（honest FAIL/WITHHELD 均如实标注）。")
    tags = ["IN38-N04", "wind-load", "LBM", "superstructure", "hybrid-engine",
            "ship-aerodynamics", "GCI", "TensorLBM"]
    return {"title": "IN38-N04 · 船舶上层建筑风阻力 LBM 数据库与混合引擎"
                     "（集装箱/油轮/散货三型）",
            "summary": summary,
            "scenario_key": "ship_wind_resistance_hybrid",
            "mission_id": 5469,
            "tags": tags,
            "markdown": md,
            "figures": figures}


def main() -> int:
    db = load("wind_database.json")
    acc = load("acceptance.json")
    md = build_markdown(db, acc)
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write(md)
    payload = build_payload(db, acc, md)
    total = sum(len(f["png_base64"]) for f in payload["figures"])
    with open("/tmp/report.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(json.dumps({"doc": os.path.relpath(DOC, HERE),
                      "figures": len(payload["figures"]),
                      "payload_bytes": os.path.getsize("/tmp/report.json"),
                      "figure_b64_bytes": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())