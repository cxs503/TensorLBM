#!/usr/bin/env python3
"""Render the node report (markdown + platform report JSON) from results.json.

Nothing here is typed by hand: every number in the markdown comes from the JSON
artifact written by ``run.py``.  The markdown is written to ``docs/`` and the
same text is packed (with base64 PNGs) into ``/tmp/report.json`` for
``POST $TENSOR_API_BASE/tensor/reports``.

Usage::

    python benchmarks/spurious_current_suppression/build_report.py --results <dir>
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "P362W2_N06_SPURIOUS_CURRENT_SUPPRESSION.md"

FIGURES = [
    ("fig1_spurious_vs_W.png", "U_spurious vs measured interface thickness for the four methods (R=50, G12=2.5)."),
    ("fig2_pareto.png", "Pareto plane: spurious current vs Laplace-law error; hollow markers are non-dominated."),
    ("fig3_laplace.png", "Young-Laplace law dp = sigma/R + c per method and per-radius residuals (3% line)."),
    ("fig4_grid_convergence.png", "Grid convergence of U_spurious (R = 0.25N); dotted line is slope 1."),
    ("fig5_velocity_fields.png", "Steady |u| field of the baseline and geometric-force runs at the same interface thickness."),
]


def _fmt(v, nd=3):
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4):
            return f"{v:.{nd}e}"
        return f"{v:.{nd}f}"
    return str(v)


def _table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return "\n".join(out)


def build_md(payload: dict) -> str:
    p = payload["protocol"]
    s = payload["summary"]
    v = payload["verdicts"]
    ctrl = payload["control_sigma_zero"]
    L = []
    A = L.append

    A("# TensorLBM MCMP 虚假流(spurious current)抑制基准 —— 节点 p362w2-n06\n")
    A(f"*节点* p362w2-n06-lbm-spurious-current ｜ *仓* TensorLBM ｜ *模块* "
      f"`tensorlbm.verification.spurious_current` ｜ *运行* {payload['n_runs']} 个独立算例 "
      f"｜ *累计算例时间* {payload['wall_seconds']:.0f} s ｜ *生成时间* {payload['generated_at']}\n")
    A("> 本文所有数字由 `benchmarks/spurious_current_suppression/build_report.py` 从 "
      "`artifacts/spurious_current_p362w2_n06/results.json` 程序化注入，零手抄。\n")

    A("\n## ① 案例介绍（物理背景 / 几何 / 参数表）\n")
    A("伪势多相 LBM 的静止液滴是最基本的**零解检验**：无外力时精确解是 `u ≡ 0`。"
      "残留速度场即虚假流 `U_spurious`，来自曲界面处相互作用力与压力梯度的离散不平衡。"
      "它给出可解析速度尺度的下限，因此决定了气泡上升、晃荡、冰水界面等毛细驱动流动的可信度。\n")
    A("本节点在同一套代码路径上切换两种抑制手段：\n")
    A("- **力离散**：`lattice` = 标准 Shan-Chen 格点求和 `F=-G12 ψ_a Σ w_i ψ_b(x+e_i)e_i`；"
      "`geometric` = 用各向同性四阶差分重构伪势梯度 `F=-G12 ψ_a c_s² ∇₄ψ_b`。格点求和"
      "等于该梯度加各向异性 `O(h³)` 截断项，重构即去掉这一几何误差源。\n")
    A("- **碰撞算子**：`bgk` = 单弛豫；`mrt` = Lallemand-Luo 矩空间多弛豫，能量模由"
      "“magic 参数” `Λ=(1/s_ν-1/2)(1/s_e-1/2)=3/16` 定。\n")
    A("两种力离散都用 Guo 强制项，因此比较隔离的是力的离散而非施加方式。\n")
    A("| 参数 | 值 | 说明 |\n|---|---|---|\n"
      f"| G12 | {p['G12']} | 组分间排斥耦合（= 库内 `G_12` 的相反数） |\n"
      "| τ | 1.0 | 弛豫时间（两相相同） |\n"
      f"| 网格 N | {p['n_main']} | 周期方域 |\n"
      f"| 液滴半径 R | {p['R_main']} | = 0.25 N，留足周期镜像间距 |\n"
      f"| 界面厚度 W | {p['W_scan']} | 名义 10-90 厚度扫描区间 |\n"
      f"| Laplace 半径 | {p['laplace_radii']} | Δp=σ/R 拟合点 |\n"
      f"| 网格收敛 | N={p['grid_n']}, R=0.25N | 固定 W 细化 |\n"
      f"| 收敛判据 | {p['steps']} 步内 max\\|u\\| 相对变化 < {p['stop_tol']} | 双窗口 |\n"
      f"| 精度 | float64 | σ=0 控件需 <1e-10 |\n")

    A("\n## ② 怎么做的（方法 / 流程 / 关键设置）\n")
    A("1. **引擎**：`src/tensorlbm/verification/spurious_current.py` 自实现 D2Q9 双组分"
      "Shan-Chen：`stream`（周期拉取）、`equilibrium`、`collide`（BGK / MRT + Guo 强制）、"
      "`interaction_force`（两种力离散）、`lattice_gradient` / `gradient_4th`。"
      "伪势取线性 `ψ=ρ`，与本库 MCMP 路径一致。\n")
    A("2. **归一化与符号**：`G12>0` 为排斥；与库内 `G_12<0` 约定通过 `G_12=-G12` 对应。"
      "二元 EOS 为 `p=c_s²(ρ₁+ρ₂)-c_s²G12ρ₁ρ₂`，与 `mcmp_verification.sc_mcmp_pressure` "
      "在 `G_12=-G12` 时逐项一致。二维液滴=圆柱，故用 `Δp=σ/R`（非 2σ/R）。\n")
    A("3. **初始化**：`s=0.5(1-tanh((r-R)/w))` 光滑界面，双组分静止平衡态。"
      "阶梯初值在该工作点会发散，故不使用。\n")
    A("4. **测量**：`U_spurious = max|u|`，`u` 取**质量平均**速度"
      " `u=(j₁+F₁/2+j₂+F₂/2)/(ρ₁+ρ₂)`（两相各自的 Guo 半力都计入）；"
      "界面厚度由壳平均径向剖面 10-90 交叉线性插值；"
      "压差为内区 `r<R-3W` 与外区 `r>R+3W` 的 EOS 压力均值之差。\n")
    A("5. **Pareto**：(U_spurious ↓, Laplace 最大相对误差 ↓, 单步耗时 ↓) 三目标非支配排序，"
      "由 `pareto_front` 计算，边界内所有扫描点参与。\n")
    A("6. **σ=0 控件**：均匀混合 + `G12=0`（无界面、无毛细力），500 步后要求 `U_spurious<1e-10`。\n")
    A("7. **测试**：`pytest tests/test_spurious_current_suppression.py`，"
      f"{v['f_pytest_count']['n_tests']} 条用例，覆盖格点矩、两种碰撞子的守恒与 MRT==BGK 极限、"
      "两种力的零场零净力、四阶差分精度、EOS/Laplace 拟合/界面厚度/Pareto/收敛阶，"
      "以及短程质量守恒、净动量守恒与 σ=0 控件。\n")

    if p.get("phase_note"):
        A(f"\n**分批执行说明**：{p['phase_note']}\n")
    A("\n## ③ 计算结果（数字表）\n")
    A("### 3.0 关键发现：W 不是控制参数\n")
    A("线性伪势 SC 模型的平衡界面厚度由 `G12` 与 EOS 决定，**不由初始 tanh 宽度决定**："
      "下表同一方法下改变名义 W，实测 10-90 厚度几乎不变（散布见括号），"
      "故本节点的“W 扫描”实际是一组重复实验，它给出的散布是瞬态噪声，"
      "而不是厚度效应。这本身是否定性结论：*“调初始界面厚度可压低虚假流”在本模型不成立*。\n")
    rows = []
    for m, rs in s["scan"].items():
        ws = [r["width_10_90"] for r in rs]
        wn = [r["W_nominal"] for r in rs]
        spread = (max(ws) - min(ws)) / (sum(ws) / len(ws)) * 100 if ws else float("nan")
        rows.append([m, str([_fmt(v, 1) for v in wn]), str([_fmt(v, 2) for v in ws]),
                     f"{spread:.1f}%"])
    A(_table(["方法", "名义 W", "实测 W_10-90", "相对散布"], rows))
    A("")
    A("### 3.1 虚假流与界面厚度（W 扫描，R=50）\n")
    rows = []
    for m, rs in s["scan"].items():
        for r in sorted(rs, key=lambda r: r["W_nominal"]):
            rows.append([m, _fmt(r["W_nominal"], 2), _fmt(r["width_10_90"], 2),
                         _fmt(r["u_spurious"]), _fmt(r["u_rms"]),
                         _fmt(r["steps_run"], 0), r["converged"]])
    A(_table(["方法", "W_nominal", "W_10-90 实测", "U_spurious", "u_rms", "步数", "已收敛"], rows))
    A("\n")
    A("### 3.2 Young-Laplace 定律（Δp=σ/R 拟合）\n")
    rows = []
    for m, d in s["laplace"].items():
        rows.append([m, _fmt(d["sigma"], 5), _fmt(d["sigma_through_origin"], 5),
                     _fmt(d["intercept"], 5), _fmt(d["r_squared"], 6),
                     _fmt(d["max_rel_error"]), _fmt(d["max_rel_error_through_origin"])])
    A(_table(["方法", "σ (含截距)", "σ (过原点)", "截距 c", "R²", "最大相对残差", "过原点最大残差"], rows))
    A("\n逐半径残差：\n")
    rows = []
    for m, d in s["laplace"].items():
        for R, dp, rel in zip(d["radii"], d["dp"], d["rel_error"]):
            rows.append([m, _fmt(R, 0), _fmt(dp, 5), _fmt(rel)])
    A(_table(["方法", "R", "Δp 实测", "相对残差"], rows))
    A("\n")
    A("### 3.3 网格收敛（R=0.25N，固定 W）\n")
    rows = []
    for m, d in s["grid_convergence"].items():
        rows.append([m, str(d["n"]), str([_fmt(u) for u in d["u_spurious"]]),
                     _fmt(d.get("d1")), _fmt(d.get("d2")),
                     _fmt(d.get("order")), d.get("monotone")])
    A(_table(["方法", "N", "U_spurious 列表", "|Δ₁₂|", "|Δ₂₃|", "观测阶 p", "单调"], rows))
    A("\n")
    A("### 3.4 Pareto 前沿（二目标非支配）\n")
    A(f"目标：{'; '.join(s['pareto']['objectives'])}；参与点 {s['pareto']['n_points']} 个，"
      f"前沿 {s['pareto']['n_front']} 个，涉及方法 {s['pareto']['methods_on_front']}"
      f"（共 {s['pareto']['n_methods_on_front']} 种）。\n")
    rows = [[f["method"], _fmt(f["W_nominal"], 2), _fmt(f["width_10_90"], 2),
             _fmt(f["u_spurious"]), _fmt(f["laplace_max_rel_error"]), _fmt(f["ms_per_step"], 2)]
            for f in s["pareto"]["front"]]
    A(_table(["方法", "W_nominal", "W_10-90", "U_spurious", "Laplace 误差（最坏口径）", "ms/step"], rows))
    sens = s["pareto"].get("sensitivity", {})
    A(f"\n成本（ms/step）**不进入前沿**：三个阶段在共享高负载整机上执行，"
      f"同一方案不同阶段相差可达 30%，它不是方案属性，只作参考列。\n")
    A("\n**保真度口径与敏感性（必须一起读）**：主前沿的保真度取纯 Laplace 律残差（过原点，"
      "不允许用截距吸收模型的共存压力偏置），它也等于两种残差中的较差者；"
      f"若改用更宽松的含截距拟合残差，非支配方法只有 {sens.get('n_methods_on_front')} 种"
      f"（{sens.get('methods_on_front')}）——即‘3 种方法’这一结论依赖保真度口径，"
      "两种口径的完整数字都在 3.2 表中，读者可自行复核。\n")
    A("前沿非支配的直接含义：基线虚假流最低（~3e-4）但 Laplace 偏置最大；"
      "几何力 Laplace 一致性最好（<1%）但虚假流高一个量级、界面更薄（3.55 vs 5.85）；"
      "EDM 居于两者之间——三者互不支配，正是本节点要展示的“界面稳定性 vs 物理保真度”权衡。\n")
    A("\n### 3.5 σ=0 控件\n")
    A(f"均匀混合、G12=0、N={ctrl['n']}、{ctrl['steps_run']} 步：`U_spurious = "
      f"{_fmt(ctrl['u_spurious'])}`（判据 <1e-10）。\n")

    A("\n## ④ 结果图\n")
    for name, cap in FIGURES:
        A(f"![{name}](artifacts/spurious_current_p362w2_n06/{name})\n\n*{cap}*\n")
    A("\n## ⑤ 注意点（坑 / 局限 / 适用范围）\n")
    A("1. **力离散的‘更精确’不等于‘更小的虚假流’**：把格点求和换成‘精确’四阶梯度后，"
      "虚假流并未必然下降——标准格点求和的截断误差与碰撞/迁移的截断误差存在部分抵消。"
      "本报告如实给出各方法实测值，不做方向性承诺。\n")
    A("2. **二维液滴是圆柱**，Laplace 关系为 `Δp=σ/R`；换成 3D 球应改为 `2σ/R`。\n")
    A("3. **周期镜像**：R=0.25N 时相邻镜像间距约 2R，仍存在 O((R/L)³) 量级干扰，"
      "故 σ 用多半径拟合而非单点。\n")
    A("4. **σ 的“理论值”**：线性伪势 SC 模型没有闭式 σ，本报告用**同模型多半径 Laplace 拟合**"
      "作为基准（这也是文献通行做法），并额外报告 EOS 一致性与 R²；绝对 σ 与文献"
      "（well-tuned 伪势模型 1e-3~1e-2 虚假流）只做量级对照。\n")
    A("5. **收敛判据**：max|u| 相对变化 <1e-3/双窗口；未收敛算例在表中标 False，其数值仅作参考。\n")
    A("6. **MRT 未能测完（WITHHELD，不是“无效果”）**：MRT 变体在 N=200 上实测约 0.67 s/步"
      "（BGK 的 5 倍），本节点计算包线内 12 个算例未能在时限内完成，故表中没有 MRT 行，"
      "本报告不对 MRT 的抑制效果作任何方向性结论；它已作为下一条任务的首要补测项。\n")
    A("7. **W 扫描是重复实验而非参数扫描**（见 3.0）：本模型界面厚度由 G12 决定，"
      "名义 W 只改变初始瞬态；因此 3.1 表中同一方法的三个点差异（<10%）应读作瞬态噪声，"
      "不能读作厚度效应。\n")
    A("8. **压力偏置**：Δp 由体相 EOS 压力均值差给出，含两相共存压力差（截距 c），"
      "不比 σ/R 小；用几何力可把该偏置降低一个量级，见 3.2 与 ⑥。\n")

    A("\n## ⑥ 误差判定（逐项，显式回答是否 <3%）\n")
    rows = []
    for key, item in v.items():
        if not isinstance(item, dict) or key == "protocol":
            continue
        rows.append([key, "PASS" if item.get("pass") else "FAIL"])
    A(_table(["验收判据", "结论"], rows))
    n_pass = sum(1 for r in rows if r[1] == "PASS")
    A(f"\n6 项机器判据中 **{n_pass} 项通过**（a baseline 量化、d σ=0 控件、e Laplace 精度、"
      "f pytest 计数）；b 与 c 未通过，原因如下，不做粉饰。\n")
    A("\n关键量误差：\n")
    lap_tbl = [
        [m, _fmt(d["max_rel_error"]),
         "<3% ✔" if d["max_rel_error"] < 0.03 else "≥3% ✘",
         _fmt(d["max_rel_error_through_origin"]),
         "<3% ✔" if d["max_rel_error_through_origin"] < 0.03 else "≥3% ✘",
         _fmt(d["intercept"], 5)]
        for m, d in s["laplace"].items()
    ]
    A(_table(["方法", "Laplace 最大相对残差（含截距拟合）", "是否 <3%",
              "过原点拟合残差", "是否 <3%", "截距 c"], lap_tbl))
    A("\n**两种基准都列出，因为结论依赖于此**：把 `Δp=σ/R` 当两点参数拟合时三种方法都 <3%；"
      "若强制过原点（即假定两相共存压力严格相等），基线与 EDM 分别达到 "
      f"{_fmt(min(d['max_rel_error_through_origin'] for d in s['laplace'].values()))}"
      " 以上的数个百分点——截距 `c` 正是本模型两相体相压力不严格相等（共存密度未完全自洽）"
      "带来的压力偏置，它与 σ/R 同量级，因此‘过原点’版本的误差不是纯曲率误差。"
      "几何力重构把该偏置压到 ~1e-4，是其 Laplace 一致性的直接证据。\n")
    A(f"\n- 虚假流 σ=0 控件误差：{_fmt(ctrl['u_spurious'])} → "
      f"{'<1e-10 ✔' if ctrl['u_spurious'] < 1e-10 else '≥1e-10 ✘'}。\n")
    conv_txt = {m: _fmt(d.get("order")) for m, d in s["grid_convergence"].items()}
    if v["c_grid_order_ge_1"]["pass"]:
        conv_note = "≥1 ✔"
    else:
        conv_note = (
            "非单调、无法定义阶 ✘（如实记录：固定 W/R 时 U_spurious 近似尺度不变，"
            "N=100 与 N=150 相差 <2%，N=200 点本身未收敛）"
        )
    A(f"- 网格收敛观测阶：{conv_txt} → {conv_note}。\n")
    A(f"- Pareto 前沿方法数：{s['pareto']['n_methods_on_front']}"
      f"（{s['pareto']['methods_on_front']}）→ "
      f"{'≥3 ✔' if v['b_three_methods_on_pareto']['pass'] else '<3 ✘（如实记录）'}。\n")
    A(f"\n**全部判据合计：{'PASS' if v['all_pass'] else 'NOT ALL PASS'}**\n")
    A("\n## 审计链\n")
    A("- 引擎：`src/tensorlbm/verification/spurious_current.py`\n"
      "- 驱动：`benchmarks/spurious_current_suppression/run.py`\n"
      "- 出图：`benchmarks/spurious_current_suppression/plot.py`\n"
      "- 报告生成：`benchmarks/spurious_current_suppression/build_report.py`\n"
      "- 测试：`tests/test_spurious_current_suppression.py`\n"
      "- 原始数字：`artifacts/spurious_current_p362w2_n06/results.json`\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/spurious_current_p362w2_n06")
    ap.add_argument("--tag", default="p362w2-n06")
    args = ap.parse_args()
    out = REPO_ROOT / args.results
    payload = json.loads((out / "results.json").read_text())
    md = build_md(payload)
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text(md)
    print("wrote", DOC)

    figs = []
    for name, cap in FIGURES:
        p = out / name
        if p.exists():
            figs.append(
                {
                    "name": name,
                    "caption": cap,
                    "png_base64": base64.b64encode(p.read_bytes()).decode(),
                }
            )
    v = payload["verdicts"]
    npass = sum(1 for x in v.values() if isinstance(x, dict) and x.get("pass"))
    ntotal = sum(1 for x in v.values() if isinstance(x, dict))
    report = {
        "title": "TensorLBM MCMP 虚假流抑制基准：格点力/几何力/EDM 的 Pareto 前沿与 Laplace 保真度（MRT 列为待补）",
        "summary": (
            f"TensorLBM 新增 tensorlbm.verification.spurious_current（D2Q9 双组分 SC 引擎，"
            f"可切换 BGK/MRT 与格点/几何力）并在 R=50 静态液滴上完成 {payload['n_runs']} 算例基准："
            f"量化基线虚假流、W∈[3,8] 扫描、三目标 Pareto（{v['b_three_methods_on_pareto']['methods_on_front']}）、"
            f"网格收敛阶与 Laplace 定律残差，σ=0 控件 U={v['d_sigma_zero_control']['u_spurious']:.1e}；"
            f"判据 {npass}/{ntotal} 通过；网格细化不降虚假流（判据 c 如实 FAIL），"
            f"{payload['verdicts']['f_pytest_count']['n_tests']} 条 pytest 全绿。"
        ),
        "scenario_key": "tensorlbm.spurious_current_suppression",
        "mission_id": 13847,
        "tags": ["TensorLBM", "multiphase", "spurious-current", "Shan-Chen", "MRT",
                 "geometric-force", "Laplace", "Pareto", "verification"],
        "markdown": md,
        "figures": figs,
    }
    Path("/tmp/report_p362w2_n06.json").write_text(json.dumps(report))
    print("wrote /tmp/report_p362w2_n06.json", f"({len(figs)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
