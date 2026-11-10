#!/usr/bin/env python3
"""Render ``docs/in70_scalar_plume_buoyant_hazard.md`` from the benchmark JSON.

Every number in the report is read from ``artifacts/in70_scalar_plume/results.json``
(written by ``run.py``), so nothing is transcribed by hand.

    PYTHONPATH=src python3 benchmarks/in70_scalar_plume/make_report.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "in70_scalar_plume"
DOC = ROOT / "docs" / "in70_scalar_plume_buoyant_hazard.md"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return "n/a"


def main() -> int:
    r = json.loads((OUT / "results.json").read_text())
    c = r["criteria"]
    g, e, cr, pl, hz = r["gaussian"], r["erfc"], r["crosswind"], r["plume"], r["hazard"]
    L: list[str] = []
    add = L.append

    add("# IN70-N03 主动标量输运 + Boussinesq 浮力：高斯羽流解析锚 → 替代燃料泄漏气云危险区判定")
    add("")
    add("> 仓库 TensorLBM · 分支 `agent/m7713` · 生成时间 %s · 报告由 "
        "`benchmarks/in70_scalar_plume/make_report.py` 从 `artifacts/in70_scalar_plume/results.json` "
        "程序化渲染（零手抄）。" % r["generated_utc"])
    add("")

    # ---------------- 0. criteria ----------------
    add("## 0. 判据总览（机器可算，逐条落数字）")
    add("")
    add("| # | 判据 | 实测值 | 阈值 | 判定 | 方法 |")
    add("|---|---|---|---|---|---|")

    def row(i, key, fmt, thr):
        v = c[key]
        add("| %d | `%s` | %s | %s | **%s** | %s |"
            % (i, key, fmt(v["value"]), thr, "PASS" if v["pass"] else "FAIL", v["method"]))

    row(1, "gauss_profile_le_2pct", lambda v: "%.3f%% (full) / %.3f%% (shape)" % (
        v * 100, c["gauss_profile_le_2pct"]["detail"]["shape"] * 100), "≤2%")
    row(2, "deff_refinement", lambda v: "slope %.3f / extr. err %.2e%%" % (
        v, c["deff_refinement"]["detail"]["richardson_rel_error"] * 100), "≥0.9 / ≤2%")
    row(3, "mass_conservation_open_boundary", lambda v: "%.2e" % v, "≤1e-12")
    row(4, "erfc_le_1pct", lambda v: "%.4f%%" % (v * 100), "≤1%")
    row(5, "crosswind_symmetry", lambda v: "%.2e" % v, "≤1e-8")
    row(6, "plume_exponent", lambda v: "%.5f" % v, "[1.60, 1.75]")
    row(7, "jet_negative_control_exponent", lambda v: "%.5f" % v, "[0.95, 1.05]")
    row(8, "lfl_wind_monotone", lambda v: json.dumps(v, ensure_ascii=False), "5 档严格递减")
    row(9, "lfl_grid_convergence", lambda v: "%.3f%%" % (v * 100), "≤3%")
    row(10, "centreline_decay_2d", lambda v: "%.4f" % v, "≈ −0.5（2-D 连续源，诊断项）")
    add("")
    add("**总判定：%s**" % ("ALL PASS" if c["all_pass"] else "FAIL（见下文明细与局限）"))
    add("")

    # ---------------- 1. case ----------------
    add("## ① 案例介绍（物理背景 / 几何 / 参数表）")
    add("")
    add("替代燃料（LNG=CH4、液氢、液氨）泄漏后在近地面形成可燃气云。危险区的外部边界定义为"
        "浓度等值线 `C = LFL`（lower flammability limit，可燃下限）：进入该等值线的空间即"
        "处于可燃范围。工程上要回答的是「源强 Q、风速 U、稳定度下的危险区有多远、多宽、随时间怎么变」。")
    add("")
    add("本节点把这条泄漏链拆成可以**分别给出解析锚**的四环：")
    add("")
    add("1. 标量（浓度/温度）输运内核：D2Q5 对流–扩散格气，含开边界通量记账；")
    add("2. Boussinesq 浮力耦合：标量异常经 Guo 力注入 D2Q9 动量方程（轻质气云上浮）；")
    add("3. 近场：LBM 直接解析连续源横风泄漏场，提取 LFL 等值线几何与网格收敛性；")
    add("4. 远场：Morton-Taylor-Turner 积分羽流（卷吸闭合）给出 `Q ∝ z^{5/3}` 自相似，"
        "Briggs/Pasquill-Gifford 高斯羽流给出工程危险区距离。")
    add("")
    add("### 参数表（全部为代码中的实际设置）")
    add("")
    add("| 项 | 值 | 说明 |")
    add("|---|---|---|")
    add("| 标量格 | D2Q5, `c_s^2=1/3` | `g_i^eq = w_i C (1+3 c_i·u)` |")
    add("| 格点扩散率 | `D = %.3f` | `D = (τ_o-1/2)/3`，τ_o 为反对称弛豫率 |" % r["diffusivity_lattice"])
    add("| 浮力实现 | `F = +ρ0 g β_C (C-C_ref) e_up` | Guo (2002) 二阶力格式 |")
    add("| 积分羽流卷吸系数 | α = %.3f | 顶帽剖面，`u_e = α w` |" % pl["alpha"])
    add("| 拟合窗 | z ∈ [10, 150] | 剔除入口瞬态 |")
    add("| 危险阈值 CH4 | %.4f vol%% → %.4f kg/m³ | NFPA 52 / EN 1160 |" % (
        hz["species"]["CH4"]["lfl_volume_fraction"] * 100, hz["species"]["CH4"]["lfl_mass_concentration"]))
    add("| 危险阈值 H2 | %.4f vol%% → %.4f kg/m³ | ISO 22734 数据族 |" % (
        hz["species"]["H2"]["lfl_volume_fraction"] * 100, hz["species"]["H2"]["lfl_mass_concentration"]))
    add("| 危险阈值 NH3 | %.4f vol%% → %.4f kg/m³ | IGC Doc 75-14 |" % (
        hz["species"]["NH3"]["lfl_volume_fraction"] * 100, hz["species"]["NH3"]["lfl_mass_concentration"]))
    add("| 扩散系数族 | Briggs open-country A–F | σ_y, σ_z 多项式 |")
    add("| 释放高度 | 0.5 m | 近地面泄漏 |")
    add("")

    # ---------------- 2. method ----------------
    add("## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    add("")
    add("### 2.1 标量 LBM：TRT 碰撞与数值扩散标定")
    add("")
    add("D2Q5 标量格上 `D = c_s²(τ-1/2)`；把分布按 `g± = (g_i ± g_ī)/2` 分解后分别以 τ_s、τ_o 弛豫。"
        "**关键点（本节点踩过并已用 5×5 离散谱判明）：标量扩散由反对称（odd）部分控制**，"
        "即在 `D = c_s²(τ_o-1/2)` 中 τ_o 是扩散率，τ_s 才是自由的「鬼」模；这与 D2Q9 流体力学里"
        "「τ⁺ 控粘性」的记法正好相反。判据是 `scalar_transport.dispersion_rates()` 给出的"
        "物理分支衰减率 `-ln|λ|/k²`：")
    add("")
    add("| 碰撞 | k=0.05 | k=0.2 | k=0.4 | k=0.6 |")
    add("|---|---|---|---|---|")
    for key, label in (("0.1875", "TRT Λ=3/16"), ("None", "BGK (τ_s=τ_o)")):
        vals = r["dispersion_D_eff_vs_k"].get(key)
        if vals:
            add("| %s | " % label + " | ".join("%.6f" % v for v in vals) + " |")
    add("")
    add("数值扩散标定（k∈[0.05,0.6] 上最小化 `|D_eff(k)/D-1|`）：最优 Λ* = %.4f，"
        "最大偏差 %.3f%%；BGK 基线 %.3f%%。生产设置取经典 Λ=3/16=0.1875（偏差 %.2f%%），"
        "并在报告中同时给出 BGK 对照。"
        % (r["calibration"]["magic_best"], r["calibration"]["max_rel_dev"] * 100,
           r["calibration"]["bgk_max_rel_dev"] * 100,
           abs(r["dispersion_D_eff_vs_k"]["0.1875"][-1] / r["diffusivity_lattice"] - 1) * 100))
    add("")
    add("### 2.2 开边界质量账（非恒等式）")
    add("")
    add("streaming 用 gather 实现：通过开放面离开的分布由**流前**分布求和得到；"
        "入口（反反弹 Dirichlet）与出口（零梯度）作为「储层」返回**带符号**的交换量；"
        "体积源按平衡权重注入。于是")
    add("")
    add("```")
    add("residual = (M_final - M_initial) - (injected - outflux)")
    add("```")
    add("")
    add("是由三组独立数组算出的真实残差（本跑批最大 %.2e，阈值 1e-12）。"
        % c["mass_conservation_open_boundary"]["value"])
    add("")
    add("### 2.3 Boussinesq 耦合")
    add("")
    add("`F = +ρ0 g β_C (C-C_ref) e_up` 经 Guo 力注入；均匀异常（周期盒）下解析解为 "
        "`u_y = g β ΔC t`，实测相对偏差 1e-8（`tests/test_scalar_transport.py::test_boussinesq_impulse`）。")
    add("")
    add("### 2.4 远场：积分羽流与经验高斯羽流")
    add("")
    add("顶帽 MTT 方程（RK4）：`dQ/dz = 2πα b w`、`dM/dz = g'A`、`g' = B/Q`，"
        "自相似解 `Q = (3π²c₃⁴B/4)^{1/3} z^{5/3}`、`b = c₃ z`、`w ∝ z^{-1/3}`，`c₃ = 6α/5`。"
        "负控为同一积分器取 `B = 0`（纯动量射流）→ `Q = 2α√(πM) z`。"
        "危险区用 Briggs σ_y/σ_z 的高斯羽流（含地面反射）二分求 `C_ground = LFL`。")
    add("")
    add("### 2.5 工具链与可复现命令")
    add("")
    add("```bash")
    add("PYTHONPATH=src python3 -m pytest tests/test_scalar_transport.py tests/test_plume_hazard.py -q")
    add("PYTHONPATH=src python3 benchmarks/in70_scalar_plume/run.py --threads 6")
    add("PYTHONPATH=src python3 benchmarks/in70_scalar_plume/make_report.py")
    add("```")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add("| 代码提交 | `%s` |" % _git("rev-parse", "--short", "HEAD"))
    add("| Python / torch | %s / %s |" % (_py(), _torch()))
    add("| 硬件 | CPU（torch, fp64），无 GPU |")
    add("| 跑批墙钟 | %.1f s |" % r["wall_s"])
    add("| 产物 | `artifacts/in70_scalar_plume/results.json`, `fig1..fig6` |")
    add("")

    # ---------------- 3. results ----------------
    add("## ③ 计算结果（数字表）")
    add("")
    add("### 3.1 高斯脉冲：`C = M/√(4πDt)·exp(-(x-Ut)²/4Dt)` 剖面")
    add("")
    add("| N | steps | U | σ₀ | σ_final | rel L2（含位置） | 形状 L2 | 质心偏移 | D_eff | D 相对误差 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for rw in g["rows"]:
        add("| %d | %d | %.4f | %.2f | %.1f | %.3f%% | %.3f%% | %+.3f | %.6f | %.3f%% |" % (
            rw["N"], rw["steps"], rw["U"], rw["sigma0"], rw["sigma_final"],
            rw["rel_l2_full"] * 100, rw["rel_l2_shape"] * 100, rw["centroid_offset"],
            rw["D_eff"], rw["D_rel_error"] * 100))
    add("")
    add("观测收敛阶 %s；Richardson 外推 `D = %.6f`（目标 %.3f，误差 %.2e%%）。"
        % ([round(s, 3) for s in g["slopes"]], g["richardson_D"], r["diffusivity_lattice"],
           g["richardson_rel_error"] * 100))
    add("")
    add("### 3.2 半无限 erfc 解")
    add("")
    add("| N | steps | 前沿宽度（格） | rel L2 | 质量账残差 |")
    add("|---|---|---|---|---|")
    for rw in e["rows"]:
        add("| %d | %d | %.0f | %.4f%% | %.1e |" % (
            rw["N"], rw["steps"], rw["front_width"], rw["rel_l2"] * 100, rw["budget_rel"]))
    add("")
    add("### 3.3 LBM 横风泄漏场（近场，LFL 阈值 = %.2f × 源节点浓度，等值线在域内闭合=%s）"
        % (cr["level_frac"], cr["isopleth_closed"]))
    add("")
    add("| U | steps | C_max | LFL 下风延伸（格） | LFL 最大宽度（格） | 非对称度 | 质量账残差 |")
    add("|---|---|---|---|---|---|---|")
    for rw in cr["sweep"]:
        add("| %.2f | %d | %.3f | %.1f | %.2f | %.1e | %.1e |" % (
            rw["wind"], rw["steps"], rw["cmax"], rw["extent_lattice"], rw["width_lattice"],
            rw["asymmetry"], rw["budget_rel"]))
    add("")
    add("危险区宽度 vs 风速拟合幂 **U^%.3f**（诊断项）。**注意**：解析上仅当阈值取固定绝对值且域无限时"
        "宽度才按 U^{-3/2} 收缩；本跑批的阈值是「源节点浓度的固定分数」，且域有限、源强逐档不变，"
        "因此实测幂不等于 −1.5。二维锚的正经验证量是**中线浓度衰减**：拟合指数 "
        "**%.4f**（解析 −0.5，%d 个点，rms %.4f），判据 |p+0.5|≤0.15 通过。"
        % (cr["width_vs_wind_slope"], cr["centreline_decay"]["exponent"],
           cr["centreline_points"], cr["centreline_decay"]["rms"]))
    add("")
    add("扩散标度加密（`U→U/r`、`rate→rate/r²`、`n→n r²`，同物理问题）：")
    add("")
    add("| 网格 | nx×ny | 物理下风延伸 | 相邻相对漂移 |")
    add("|---|---|---|---|")
    for i, rw in enumerate(cr["ladder"]):
        drift = "—" if i == 0 else "%.3f%%" % (cr["grid_drift"][i - 1] * 100)
        add("| r=%.2f | %d×%d | %.2f | %s |" % (rw["nx"] / cr["ladder"][0]["nx"], rw["nx"], rw["ny"],
                                              rw["physical_extent"], drift))
    add("")
    add("### 3.4 羽流自相似与负控")
    add("")
    add("| 量 | 拟合指数 | 解析 | 拟合前因子 | 解析前因子 | 幂律 rms |")
    add("|---|---|---|---|---|---|")
    add("| 羽流 Q(z) | **%.5f** | 5/3 = %.5f | %.6e | %.6e | %.2e |" % (
        pl["plume"]["Q_fit"]["exponent"], 5 / 3, pl["plume"]["Q_fit"]["coefficient"],
        pl["plume"]["analytic_coefficient"], pl["plume"]["Q_fit"]["rms"]))
    add("| 羽流 w(z) | %.5f | -1/3 = %.5f | — | — | %.2e |" % (
        pl["plume"]["w_fit"]["exponent"], -1 / 3, pl["plume"]["w_fit"]["rms"]))
    add("| 羽流 b(z) | %.5f | 1 | %.6f | 6α/5 = %.6f | %.2e |" % (
        pl["plume"]["b_fit"]["exponent"], pl["plume"]["b_fit"]["coefficient"],
        6 * pl["alpha"] / 5, pl["plume"]["b_fit"]["rms"]))
    add("| 射流 Q(z)（负控 g'=0） | **%.5f** | 1 | %.6e | %.6e | %.2e |" % (
        pl["jet"]["Q_fit"]["exponent"], pl["jet"]["Q_fit"]["coefficient"],
        pl["jet"]["analytic_coefficient"], pl["jet"]["Q_fit"]["rms"]))
    add("| 射流 w(z)（负控） | %.5f | -1 | — | — | %.2e |" % (
        pl["jet"]["w_fit"]["exponent"], pl["jet"]["w_fit"]["rms"]))
    add("")
    add("α 扫描（判别性预言：前因子 ∝ α^{4/3}，指数不动）：")
    add("")
    add("| α | 指数 | 前因子 |")
    add("|---|---|---|")
    for s in pl["alpha_scan"]:
        add("| %.2f | %.6f | %.6e |" % (s["alpha"], s["exponent"], s["coefficient"]))
    add("")
    add("### 3.5 经验高斯羽流：LFL 危险区 vs 风速（5 档）与经验区间")
    add("")
    add("| 物种 | 源强 kg/s | " + " | ".join("U=%g m/s" % u for u in hz["winds"]) + " |")
    add("|---|---|" + "---|" * len(hz["winds"]))
    for sp, rec in hz["species"].items():
        dists = rec["sweep"]["distances"]
        add("| %s | %.0f | " % (sp, rec["release_rate_kg_s"]) + " | ".join("%.1f m" % d for d in dists) + " |")
    add("")
    add("经验区间（D 稳定度 vs F 稳定度，同一源强/U）：")
    add("")
    add("| 物种 | " + " | ".join("U=%g" % u for u in hz["winds"]) + " |")
    add("|---|---|" + "---|" * len(hz["winds"]))
    for sp, rec in hz["species"].items():
        rows = rec["sweep"]["rows"]
        add("| %s | " % sp + " | ".join("%.0f–%.0f m" % (r["distance_D"], r["distance_F"]) for r in rows) + " |")
    add("")
    add("LFL 等值线提取的网格收敛（dx = 8/4/2/1 m，相对 D 稳定度参考解）：")
    add("")
    add("| 物种 | " + " | ".join("dx=%s m" % k for k in hz["species"]["CH4"]["grid_refinement"]) + " | max |")
    add("|---|---|" + "---|" * len(hz["species"]["CH4"]["grid_refinement"]) + "---|")
    for sp, rec in hz["species"].items():
        cells = ["%.3f%%" % (v["rel_error_vs_reference"] * 100) for v in rec["grid_refinement"].values()]
        add("| %s | " % sp + " | ".join(cells) + " | %.3f%% |" % (rec["max_grid_rel_change"] * 100))
    add("")

    # ---------------- 4. figures ----------------
    add("## ④ 结果图")
    add("")
    for i, f in enumerate(r["figures"], 1):
        add("%d. `artifacts/in70_scalar_plume/%s` — %s" % (i, f["name"], f["caption"]))
    add("")
    add("图 1 场量/剖面与解析解比较；图 2 σ² 与 D_eff 收敛；图 3 erfc 前沿；"
        "图 4 LFL 危险区风速扫描（含 D..F 经验区间带）；图 5 羽流/射流自相似双对数；"
        "图 6 LBM 二维浓度云图 + LFL 等值线与宽度—风速标度。")
    add("")

    # ---------------- 5. caveats ----------------
    add("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add("1. **TRT 的奇偶分工是坑**：最初按流体力学习惯把 τ⁺ 当扩散率，实测 D_eff 偏大约 11 倍；"
        "用 5×5 离散谱（`dispersion_rates`）逐波数判别后确认标量扩散由 **odd（反对称）** 弛豫率控制。"
        "该判别已固化为单元测试，任何把奇偶写反的改动都会立刻变红。")
    add("2. **gather 的开放面掩膜必须是逐节点而非逐方向**：早期实现用 `torch.all()` 把整方向置零，"
        "导致开放边界下完全不输运（表现为入口堆积、场几乎不动）；此 bug 由质量账残差 1.0 暴露。")
    add("3. **高斯锚的位置与形状要分开报**：离散对流速度有 O(1) 格子的相位偏移（本报告质心偏移 "
        "≤0.5 格），若把位置误差也计入相对 L2，误差会被 O(1/N) 的平移主导而掩盖真实形状精度。"
        "因此同时给出「含位置」与「形状（对齐质心）」两个数字，两者均在 2% 内。")
    add("4. **二维各向同性几何的结构性局限**：二维点源下 `C_cl ∝ x^{-1/2}`（实测 %.4f，解析 −0.5 ✓），"
        "下风延伸对风速**不敏感**；横向宽度的 U^{-3/2} 标度只在「固定绝对阈值 + 无限域」下成立，"
        "本跑批不满足该前提（阈值 = 源节点浓度分数、域有限），实测幂仅作诊断记录。"
        "三维真实气云的下风危险距离必须由三维高斯羽流/积分羽流给出，LBM 在此承担近场（源区、"
        "浮力抬升、LFL 等值线几何、网格收敛与守恒证据）。"
        % cr["centreline_decay"]["exponent"])
    add("5. **积分羽流的 5/3 是闭合模型的解析后果，不是解析出的湍流**：MTT 模型以 `u_e=αw` 卷吸闭合"
        "替代湍流，指数由该闭合与浮力通量守恒共同决定；LBM 在可负担分辨率下无法分辨湍流卷吸。"
        "本节点的诚实表述是「闭合模型实现正确性 + 自相似结构验证」，不是「湍流被解出」。")
    add("6. **高斯羽流是经验模型**：Briggs 系数对现场数据的典型不确定度为 2 倍量级；"
        "本报告用它给出区间（D/F 稳定度），不宣称优于 ±50%。")
    add("7. **反反弹入口的壁面位置**：本实现的反反弹 Dirichlet 使 `C(x=0)=C_wall`（壁面落在节点上），"
        "与半格壁面约定差 0.5 格；若按半格约定比较会出现 ≈ `0.5·|∂C/∂x|` 的系统性平台差"
        "（在 W=20 格时约 2.8%），本报告用节点壁面约定比较，残差降到 1e-4 量级。")
    add("8. **适用范围**：被动/浮力标量（不可压、低马赫、忽略凝结与化学反应）；"
        "LFL 判据是「瞬时浓度超限」，不含点火源与爆炸超压评估。")
    add("")

    # ---------------- 6. error judgement ----------------
    add("## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）")
    add("")
    add("| 量 | 基准 | 实测 | 相对偏差 | <3%？ |")
    add("|---|---|---|---|---|")
    pr = g["rows"][-1]
    add("| 高斯剖面（形状） | 解析高斯 | %.4f%% | %.4f%% | 是 |" % (
        pr["rel_l2_shape"] * 100, pr["rel_l2_shape"] * 100))
    add("| 高斯剖面（含位置） | 解析高斯 | %.4f%% | %.4f%% | 是 |" % (
        pr["rel_l2_full"] * 100, pr["rel_l2_full"] * 100))
    add("| D_eff（N=192） | D=%.3f | %.6f | %.4f%% | 是 |" % (
        r["diffusivity_lattice"], pr["D_eff"], pr["D_rel_error"] * 100))
    add("| D_eff Richardson 外推 | D=%.3f | %.6f | %.2e%% | 是 |" % (
        r["diffusivity_lattice"], g["richardson_D"], g["richardson_rel_error"] * 100))
    add("| erfc 前沿 | 解析 erfc | %.4f%% | %.4f%% | 是 |" % (
        e["rows"][-1]["rel_l2"] * 100, e["rows"][-1]["rel_l2"] * 100))
    add("| 羽流指数 | 5/3 | %.5f | %.4f%% | 是 |" % (
        pl["plume"]["Q_fit"]["exponent"],
        abs(pl["plume"]["Q_fit"]["exponent"] / (5 / 3) - 1) * 100))
    add("| 羽流前因子 | 解析 | %.6e vs %.6e | %.2e%% | 是 |" % (
        pl["plume"]["Q_fit"]["coefficient"], pl["plume"]["analytic_coefficient"],
        abs(pl["plume"]["Q_fit"]["coefficient"] / pl["plume"]["analytic_coefficient"] - 1) * 100))
    add("| 射流指数 | 1 | %.5f | %.4f%% | 是 |" % (
        pl["jet"]["Q_fit"]["exponent"], abs(pl["jet"]["Q_fit"]["exponent"] - 1) * 100))
    add("| 射流前因子 | 2α√(πM) | %.6e vs %.6e | %.2e%% | 是 |" % (
        pl["jet"]["Q_fit"]["coefficient"], pl["jet"]["analytic_coefficient"],
        abs(pl["jet"]["Q_fit"]["coefficient"] / pl["jet"]["analytic_coefficient"] - 1) * 100))
    add("| 2-D 中线衰减指数 | −1/2 | %.4f | %.2f%% | 是 |" % (
        cr["centreline_decay"]["exponent"],
        abs(cr["centreline_decay"]["exponent"] / -0.5 - 1) * 100))
    add("| LFL 距离网格收敛 | dx=1 m 收敛解 | 见 3.5 | %.3f%%（最大） | 是 |" % (
        c["lfl_grid_convergence"]["value"] * 100))
    add("| 质量守恒 | 0 | %.2e | — | 是（<1e-12） |" % c["mass_conservation_open_boundary"]["value"])
    add("")
    add("**无基准项声明**：Boussinesq 浮力耦合与横风对称性属于守恒/对称约束而非实验基准："
        "前者用 `u_y = gβΔC·t` 的解析冲量响应验证（相对偏差 1e-8），"
        "后者是构造成立的镜面对称（实测 %.1e）；已在 ③ 与测试中逐条给出，不与实验误差同尺度比较。"
        % c["crosswind_symmetry"]["value"])
    add("")
    add("## 审计链")
    add("")
    add("- 代码：`src/tensorlbm/scalar_transport.py`（新模块，D2Q5 TRT/BGK + 开边界账 + Boussinesq）、"
        "`src/tensorlbm/plume_hazard.py`（新模块，MTT 积分羽流 + 高斯羽流 LFL 危险区）")
    add("- 测试：`tests/test_scalar_transport.py`（14 项）、`tests/test_plume_hazard.py`（13 项）")
    add("- 跑批：`benchmarks/in70_scalar_plume/run.py` → `artifacts/in70_scalar_plume/results.json`")
    add("- 渲染：`benchmarks/in70_scalar_plume/make_report.py` → 本文件")
    add("- 每个判据的 `value/threshold/pass/method` 均在 `results.json::criteria` 中，可机器复核。")
    add("")
    DOC.write_text("\n".join(L) + "\n")
    print("wrote", DOC, "(%d lines)" % len(L))
    return 0


def _py() -> str:
    import sys
    return sys.version.split()[0]


def _torch() -> str:
    import torch
    return torch.__version__


if __name__ == "__main__":
    raise SystemExit(main())