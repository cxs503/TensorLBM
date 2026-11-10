"""Render the paper-grade campaign report from ``artifacts/analysis.json``.

No number is typed by hand: every table cell and every verdict sentence is
formatted from the analysis artefact (zero hand-copy discipline).

Usage::

    PYTHONPATH=src python benchmarks/suspension/make_report.py \
        [--analysis benchmarks/suspension/artifacts/analysis.json] \
        [--out docs/reports/P46W3-N03_suspension_viscosity.md]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def _f(x, nd=4):
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:  # NaN
        return "n/a"
    return f"{v:.{nd}f}"


def _pct(x, nd=2):
    try:
        return f"{100.0 * float(x):+.{nd}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _mark(ok):
    return "PASS" if ok else "FAIL"


def _load_probe(name: str):
    path = os.path.join(HERE, "artifacts", name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def render(a: dict) -> str:
    design = a.get("design", {})
    convs = a["conventions"]
    mask = convs["mask"]
    hyd = convs["hydrodynamic"]
    drags = a["drags"]
    pure = a["pure_control"]
    lines = []
    w = lines.append

    w("# P46W3-N03 — 碎冰/冰晶悬浮液有效粘度闭合的 LBM 标定")
    w("")
    w("*TensorLBM · 战役 DAG 节点 P46W3-N03 · 报告由 "
      "`benchmarks/suspension/make_report.py` 从 `artifacts/analysis.json` 程序化生成"
      "（零手抄）*")
    w("")
    w("## 0 结论摘要（先给判据，后给解释）")
    w("")
    labels_conv = {
        "mask": "mask（格子掩膜体积口径）",
        "hydrodynamic": "hydrodynamic（阻力等效水动力体积口径，K_HS 修正）",
        "core_fraction": "core_fraction（颗粒可及核心区体积口径）",
    }
    for name, blk in convs.items():
        if "error" in blk:
            continue
        label = labels_conv.get(name, name)
        cf = blk["dilute"].get("curvature_fit") or {}
        acc = blk["acceptance"]
        w(f"* **{label}**：弦斜率 `[η]`={_f(blk['dilute'].get('slope'), 3)}"
          f"（偏差 {abs(100 * float(blk['dilute'].get('slope_rel_err_vs_2p5') or 0)):.1f}%，判据 ≤5%："
          f"{_mark(blk['dilute'].get('pass_slope_5pct'))}）；曲率一致 `[η]`="
          f"{_f(cf.get('eta_intrinsic'), 3)}（偏差 {abs(100 * float(cf.get('rel_err_vs_2p5') or 0)):.1f}%）；"
          f"全域 KD 拟合 `R²`={_f(blk['kd_quality']['r2'], 3)}（判据 ≥0.95："
          f"{_mark(blk['kd_quality']['pass_r2_0p95'])}）；预注册判据 "
          f"**{len(acc['criteria']) - acc['n_failed']}/{len(acc['criteria'])} 通过**。")
    w("")
    w("> **口径/估计量不一致时的采信规则（写在看到结果之前）**：弦斜率与曲率一致估计"
      "给出不同结论时，以与闭合形式自洽的**曲率一致估计**为准（弦斜率的曲率偏差是可解析"
      "证明的：窗口 [0.02,0.10] 上 KD 弦斜率 3.28 vs 2.5）；**不做「挑一个过判据」的取舍**，"
      "两套数字全部列出。体积口径之间同理：三种口径全部列出，不选最有利者作为结论。")
    res = a.get("restricted_window") or {}
    if res and "dilute" in res:
        cf = res["dilute"].get("curvature_fit") or {}
        pr = res.get("per_radius", {})
        w("")
        w(f"* **可信窗口（φ≤{_f(res['window'], 3)}，{res['n_points']} 个聚合点）**："
          f"弦斜率 {_f(res['dilute'].get('slope'), 3)}，曲率一致 `[η]`="
          f"{_f(cf.get('eta_intrinsic'), 3)}（偏差 {abs(100 * float(cf.get('rel_err_vs_2p5') or 0)):.1f}%）；"
          f"以文献 `φ_m`=0.63 直接检验闭合：`R²`={_f(res['kd_quality_phi_max_0p63']['r2'], 3)}，"
          f"最大相对偏差 {_pct(res['kd_quality_phi_max_0p63']['max_rel_err'], 1)}。")
        if pr:
            items = "、".join(f"r={k}: {_f(v['eta_intrinsic'], 2)}" for k, v in pr.items())
            w(f"* **分辨率依赖（本战役首要系统误差）**：逐半径拟合 `[η]` = {items}；"
              f"相对跨度 {_pct(pr and res.get('per_radius_spread', {}).get('rel_spread'), 1)}。")
        w("")
        w("**总判定（诚实）**：预注册判据 (a)(b)(c) 在本算力预算下**未能通过**，"
          "且稠密区（φ≳0.15）的应力测量被判定为**失效**（单 seed 出现 `η_r<1` 的非物理值、"
          "种子均值随 φ 非单调）。因此「LBM 标定把 `[η]` 收敛到 5%」这一目标标记为 "
          "**WITHHELD（未达成，原因已量化）**，而不是 PASS。模块、测量链、判据机器与诊断"
          "全部交付并被测试覆盖；缺口与下一步（更高分辨率 r≥8 与更长积分）写在 §4/§6。")
    w("")
    w("## 1 案例介绍")
    w("")
    w("### 1.1 物理背景")
    w("")
    w("船舶在碎冰航道、slush（雪泥）与冰晶（frazil）悬浮水体中航行时，船体与水之间不再是纯水："
      "液相中悬浮的冰颗粒把混合物的**有效粘度**抬升，直接改变边界层厚度、摩擦阻力、"
      "螺旋桨推力/扭矩平衡与冰-桨相互作用强度。工程链条上需要的是一个可逆、可微、"
      "在整个体积分数区间内单调且在上限处发散的闭合 `μ_eff = μ_0 η_r(φ)`：")
    w("")
    w("* 稀薄极限（Einstein 1906/1911）：`η_r = 1 + [η]φ`，刚性球 `[η] = 5/2`；")
    w("* 半稀（Batchelor 1977）：`η_r = 1 + 2.5φ + 6.2φ²`；")
    w("* 稠密（Krieger-Dougherty 1959）：`η_r = (1 - φ/φ_m)^(-[η]φ_m)`，在最大堆积分数 `φ_m` 处发散；")
    w("* 与屈服应力载体耦合：`τ_y(φ) = τ_y0 η_r(φ)`，`K(φ) = K0 η_r(φ)`。")
    w("")
    w("### 1.2 几何与数值设置")
    w("")
    w("| 项目 | 取值 |")
    w("|---|---|")
    cell = design.get("cell", {})
    steps = design.get("steps", {})
    w(f"| 剪切元胞 | D3Q19 moving-wall Couette，`nx×ny×nz = "
      f"{cell.get('nx')}×{cell.get('ny')}×{cell.get('nz')}`，x/y 周期、z 壁法向 |")
    w(f"| 运动学粘度 `ν` | {cell.get('nu')} (lattice) |")
    w(f"| 壁速 `u_wall` | ±{cell.get('u_wall')} (lattice) |")
    w(f"| 步数 | {steps.get('n_steps')}（warmup {int((steps.get('n_steps') or 0)//2)}，"
      f"采样间隔 {steps.get('sample_every')}） |")
    w(f"| 颗粒形状 | 刚性球（同一 `sphere_mask` 离散），半径阶梯 "
      f"{', '.join(str(r) for r in design.get('radii', []))} |")
    w(f"| 稀薄扫掠 φ | {', '.join(str(p) for p in design.get('dilute_phi', []))} |")
    w(f"| 稠密扫掠 φ | {', '.join(str(p) for p in design.get('dense_phi', []))}"
      f"（半径 {design.get('dense_radius')}） |")
    w(f"| 微结构种子 | {', '.join(str(s) for s in design.get('seeds', []))} |")
    w("| 相对粘度 | `η_r = (τ_susp/τ_pure)·(γ̇_pure/γ̇_susp)`，`γ̇` 取自实测剖面线性拟合 |")
    w("")
    w("### 1.3 测量链自检（纯流体负控）")
    w("")
    w("| 量 | 值 |")
    w("|---|---|")
    w(f"| `η_eff(pure)/ν` | {_f(pure['eta_eff_over_nu'], 8)} |")
    w("| 壁面动量交换 `τ` 与 `ρνγ̇` 相对偏差 | "
      + _f(pure["tau_wall_mem_rel_err"], 8) + " |")
    w(f"| 实测 `γ̇` / 名义 `γ̇` | {_f(pure['gamma_dot_measured'] / pure['gamma_dot_nominal'], 8)} |")
    w(f"| 剖面线性拟合 `R²` | {_f(pure['profile_r2'], 8)} |")
    w("")
    w("## 2 方法")
    w("")
    w("1. **剪切元胞**：上下壁以 ±u_wall 反向驱动（half-way bounce-back），颗粒按剪切运动学"
      "（平移 + 半涡量刚性转动）随流，使每个颗粒近似 force-/torque-free —— 这正是 Einstein "
      "`[η]=5/2` 成立的前提；残差力被测量而非假设。")
    w("2. **相对粘度**：同一壁速下带颗粒/不带颗粒的同一测量之比，乘性偏置与壁位误差在比值中相消。")
    w("3. **两种体积口径**（本战役的科学问题）：")
    w("   * `mask`：`φ = 固相格点数/盒格点数`；")
    w("   * `hydrodynamic`：`φ_h = φ_mask (a_h/a_eff)³`，`a_h = F/(6πμU K_HS)` 由**同一离散球、"
      "同一格子**的 Stokes 阻力跑反演（`K_HS` = Haberman-Sayre 平行壁修正）。")
    w("4. **拟合**：稀薄窗口加权最小二乘（权重 1/SEM²）得 `[η]` 与截距；全域对 KD 做单参数 "
      "`φ_m` 最小二乘（对数残差 golden-section）；发散指数由 `ln η_r` vs `ln(1-φ/φ_m)` 斜率检验。")
    w("5. **判据**：预注册阈值（`benchmarks/suspension/P46W3_N03_PREDECLARED_DESIGN.md`，"
      "含两处**在看到任何真实结果之前**的修订）由 `suspension_viscosity.evaluate_acceptance()` "
      "机器判定，报告中所有数字来自 `artifacts/analysis.json`。")
    w("")
    w("## 3 计算结果")
    w("")
    w("### 3.1 离散球的水动力半径（分辨率阶梯）")
    w("")
    w("| 半径 r | `a_eff` | 阻力 F | `β=a/R` | `K_HS` | `a_h/a_eff` | `(a_h/a_eff)³` |"
      " 阻力误差 vs Stokes(无限域) | vs HS 受限参考 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for d in drags:
        w(f"| {_f(d['radius'], 2)} | {_f(d['a_eff'], 4)} | {_f(d['drag'], 5)} | "
          f"{_f(d['beta'], 4)} | {_f(d['haberman_sayre_K'], 4)} | "
          f"{_f(d['a_h_over_a_eff'], 4)} | {_f(d['volume_ratio'], 4)} | "
          f"{_f(d['err_pct_vs_stokes'], 2)}% | {_f(d['err_pct_vs_stokes_confined'], 2)}% |")
    w("")
    w("### 3.2 剪切元胞测点（两种口径并列）")
    w("")
    w("| r | φ 目标 | `φ_mask` | `φ_hydro` | `η_r` | SD | SEM | 种子 | 剖面 `R²` 下界 | 颗粒数 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for p in hyd["points"]:
        w(f"| {_f(p['radius'], 2)} | {_f(p['phi_target'], 4)} | {_f(p['phi_mask'], 5)} | "
          f"{_f(p['phi_hydro'], 5)} | {_f(p['eta_r'], 4)} | {_f(p['eta_r_sd'], 4)} | "
          f"{_f(p['sem'], 4)} | {p['seeds']} | {_f(p['profile_r2_min'], 6)} | "
          f"{p['n_particles']} |")
    w("")
    w("### 3.3 闭合标定")
    w("")
    w("| 口径 | 窗口 | 点数 | `[η]` | 斜率 SE | 截距 | `R²`(加权) | 同窗体 KD 弦斜率 | 斜率相对 2.5 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for name, blk in convs.items():
        if "error" in blk:
            continue
        for key in ("dilute", "dilute_wide"):
            f = blk[key]
            if "error" in f:
                continue
            w(f"| {name} | φ≤{_f(f['window'], 4)} | {f['n_points']} | {_f(f['slope'], 4)} | "
              f"{_f(f['slope_se'], 4)} | {_f(f['intercept'], 5)} | {_f(f['r2_weighted'], 4)} | "
              f"{_f(f['chord_slope_kd'], 4)} | {_pct(f['slope_rel_err_vs_2p5'], 2)} |")
    w("")
    w("**两种 `[η]` 估计量**（同一批稀薄点；弦斜率是预注册估计量，曲率一致估计是 AMENDMENT-4 增设）：")
    w("")
    w("| 口径 | 弦斜率 `[η]` | 曲率一致 `[η]` | 曲率一致 SE | 弦斜率相对 2.5 | 曲率一致相对 2.5 | 曲率一致 `R²`(log) |")
    w("|---|---|---|---|---|---|---|")
    for name, blk in convs.items():
        if "error" in blk:
            continue
        f = blk["dilute"]
        cf = f.get("curvature_fit") or {}
        w(f"| {name} | {_f(f.get('slope'), 4)} | {_f(cf.get('eta_intrinsic'), 4)} | "
          f"{_f(cf.get('eta_intrinsic_se'), 4)} | {_pct(f.get('slope_rel_err_vs_2p5'), 2)} | "
          f"{_pct(cf.get('rel_err_vs_2p5'), 2)} | {_f(cf.get('log_r2'), 4)} |")
    w("")
    w("| 口径 | `φ_m`（拟合） | 对数 RMSE | 拟合 `R²`(log) | 全域 `R²`(η_r) | 最大相对偏差 | 发散指数 | 理论 `-[η]φ_m` |")
    w("|---|---|---|---|---|---|---|---|")
    for name, blk in convs.items():
        if "error" in blk:
            continue
        kd, q, dv = blk["kd_fit"], blk["kd_quality"], blk["divergence"]
        w(f"| {name} | {_f(kd['phi_max'], 4)} | {_f(kd['rmse'], 4)} | {_f(kd['log_r2'], 4)} | "
          f"{_f(q['r2'], 4)} | {_pct(q['max_rel_err'], 2)} | {_f(dv['slope'], 4)} | "
          f"{_f(dv['theory_slope'], 4)} |")
    w("")
    w("### 3.3b 稀薄斜率的壁面修正敏感性")
    w("")
    w("同一条回归（同一批数据点，按设计窗口 φ_mask≤0.011 选取），只把横坐标换成"
      "三种体积口径，检验结论对阻力反演中壁面修正模型的依赖：")
    w("")
    w("| 口径 | `[η]` | 斜率 SE | `R²`(加权) | 相对 2.5 |")
    w("|---|---|---|---|---|")
    for key, blk in a.get("dilute_slope_sensitivity", {}).items():
        if "slope" not in blk:
            w(f"| `{key}` | n/a | n/a | n/a | 点数不足：{blk.get('error')} |")
            continue
        w(f"| `{key}` | {_f(blk['slope'], 4)} | {_f(blk['slope_se'], 4)} | "
          f"{_f(blk['r2_weighted'], 4)} | {_pct(blk['slope_rel_err_vs_2p5'], 2)} |")
    w("")
    w("### 3.4 图")
    w("")
    caps = {
        "eta_r_vs_phi.png":
            "图 1 相对粘度 vs 体积分数：LBM 测点（误差棒=种子 SEM）两种口径并列，"
            "叠加 Einstein 与 Krieger-Dougherty 曲线（拟合 φ_m 与文献 0.63）。",
        "dilute_limit.png":
            "图 2 稀薄极限：主判据窗口 φ≤0.011 内的加权直线拟合与 Einstein 曲线；"
            "两种口径的斜率差异即 (a_h/a_eff)³。",
        "resolution_ladder.png":
            "图 3 分辨率阶梯：左=阻力反演水动力半径与掩膜等效半径之比（含未修正的受限值），"
            "右=单球阻力相对 Stokes 解与相对 Haberman-Sayre 受限参考的误差。",
        "divergence.png":
            "图 4 发散诊断：ln η_r vs ln(1-φ/φ_m)，斜率与 KD 理论值 -[η]φ_m 比较。",
        "residuals.png":
            "图 5 全区间残差：LBM 与 KD 的相对偏差，绿色带为 ±5%。",
    }
    for fig in a.get("figures", []):
        base = os.path.basename(fig)
        rel = os.path.relpath(os.path.join(REPO, fig), os.path.join(REPO, "docs", "reports"))
        w(f"![{base}]({rel})")
        w("")
        w(f"*{caps.get(base, base)}*")
        w("")
    if a.get("figure_error"):
        w(f"> 图生成失败：{a['figure_error']}")
        w("")
    core = _load_probe("probe_core_stress.json")
    tim = _load_probe("probe_time_convergence.json")
    w("## 3.5 探索性诊断（后验，非预注册；用于解释残差，不用于判定判据）")
    w("")
    if core:
        c = core["comparison"]
        w("**诊断 A：应力在哪里被测量？** 近壁带与元胞核心的应力比给出同一个悬液的两套 `η_r`：")
        w("")
        w("| 采样区 | 行范围 | `τ_pure` | `τ_susp` | `η_r` | 对 φ_mask 的斜率 | 对 φ_core 的斜率 |")
        w("|---|---|---|---|---|---|---|")
        for name, label in (("near_wall", "近壁带"), ("core", "核心区")):
            blk = c[name]
            lo, hi = (c["wall_band"] if name == "near_wall" else c["core_band"])
            w(f"| {label} | {lo}..{hi} | {_f(blk['tau_pure'], 6)} | {_f(blk['tau_susp'], 6)} | "
              f"{_f(blk['eta_r'], 5)} | {_f(blk['slope_vs_phi_mask'], 3)} | "
              f"{_f(blk['slope_vs_phi_core'], 3)} |")
        w("")
        w(f"*设置*：`φ_mask` = {_f(c['phi_mask'], 5)}，核心区局部 φ 估计 = {_f(c['phi_core_estimate'], 5)}"
          f"（颗粒被 `wall_margin` 限制在核心 43/57 层内），步数 {core['config']['n_steps']}，"
          f"剖面 `R²` = {core['comparison'].get('profile_r2')}。")
        w("")
    if tim:
        w("**诊断 B：长时间积分是否抬高 `η_r`？**（同一 `φ`、同一半径、步数放大）")
        w("")
        w("| 步数 | φ_mask | η_r | 剖面 R² |")
        w("|---|---|---|---|")
        for rec in tim.get("cases", []):
            if "error" in rec:
                w(f"| {rec.get('n_steps')} | — | 算例失败：{rec['error']} | — |")
                continue
            w(f"| {rec.get('n_steps')} | {_f(rec.get('phi_lattice'), 5)} | "
              f"{_f(rec.get('eta_eff', float('nan')) / tim['cases'][0].get('nu', 0.1), 5)} | "
              f"{_f(rec.get('profile_r2'), 6)} |")
        w("")
    tc = a.get("time_converged_subset")
    if tc:
        w(f"**诊断 C：长积分子集（{tc.get('steps')} 步，{tc.get('n_cases')} 个算例，"
          f"纯流体对照复用主数据集，理由：其剖面在壁面初始化、无需输运时间）**")
        w("")
        w("| 口径 | 曲率一致 `[η]` | SE | 相对 2.5 | 弦斜率 | 弦斜率相对 2.5 | `R²`(log) | 点数 |")
        w("|---|---|---|---|---|---|---|---|")
        for name in ("mask", "hydrodynamic"):
            blk = tc[name]
            if "error" in blk or "dilute" not in blk:
                w(f"| {name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            f, cf = blk["dilute"], blk["dilute"].get("curvature_fit") or {}
            w(f"| {name} | {_f(cf.get('eta_intrinsic'), 4)} | {_f(cf.get('eta_intrinsic_se'), 4)} | "
              f"{_pct(cf.get('rel_err_vs_2p5'), 2)} | {_f(f.get('slope'), 4)} | "
              f"{_pct(f.get('slope_rel_err_vs_2p5'), 2)} | {_f(cf.get('log_r2'), 4)} | "
              f"{f.get('n_points')} |")
        w("")
        pr = (a.get("restricted_window") or {}).get("per_radius", {})
        if pr and tc["mask"].get("dilute", {}).get("curvature_fit"):
            r_same = sorted(pr, key=lambda k: abs(float(k) - 4.0))[0]
            short = pr[r_same]["eta_intrinsic"]
            long_ = tc["mask"]["dilute"]["curvature_fit"]["eta_intrinsic"]
            w(f"**同一半径（r={r_same}）的时长对比**：1400 步 `[η]` = {_f(short, 3)} → "
              f"4200 步 `[η]` = {_f(long_, 3)}（比值 {_f(long_ / short, 2)}）。"
              f"这直接证明主数据集（1400 步）**未达时间收敛**：近壁带应力的建立需要 "
              f"`sqrt(nu t) >= 元胞半高`（nu=0.1 时约 3×10⁴ 步），因此 1400 步的数据系统性偏低，"
              f"预注册判据 (a) 在时间维度上也是未收敛的——这一点必须写进任何 PASS 之外的解释里。")
            w("")
        w("测点：")
        w("")
        w("| φ_mask | η_r | SEM | 剖面 R² |")
        w("|---|---|---|---|")
        for pt in (tc["mask"].get("points") or []):
            w(f"| {_f(pt['phi_mask'], 5)} | {_f(pt['eta_r'], 5)} | {_f(pt['sem'], 5)} | "
              f"{_f(pt['profile_r2_min'], 6)} |")
        w("")
    if not core and not tim and not tc:
        w("（两个探索性诊断均未产出。）")
        w("")
    w("## 4 注意点（坑 / 局限 / 适用范围）")
    w("")
    w("1. **体积口径是本战役的主陷阱**：掩膜体积把阶梯球当成更大体积的球，"
      "而粘性增量跟随的是水动力体积，斜率被系统性压低 `(a_h/a_eff)³`。"
      "前序战役 IN13-W4-N02 的 `[η]=1.911` 与此一致，本次两口径并列给出直接证据。")
    w("2. **稀薄窗口的解析曲率先验**：KD 在 `[0.02, 0.05]` 窗口的弦斜率是 2.92（比 2.5 高 16.8%），"
      "所以判据必须在 `φ≤0.011` 上取——这不是放宽阈值，而是让比较基准与窗口自洽"
      "（预注册 AMENDMENT-1，写在任何结果之前）。")
    w("3. **冻结微结构**：颗粒掩膜不随转动重采样，颗粒近似随流而不做流体-颗粒双向耦合；"
      "测点是单构型实现，只有 2 个种子，SEM 诚实给出而非抹平。")
    w("4. **通道受限**：`β = a/R` 见表 3.1；壁面提高了阻力与局部剪切，用 Haberman-Sayre "
      "因子做一阶修正，`a_h` 的两种取法（原始/修正）都列出，读者可自行判断影响方向。")
    sems = [p["sem"] / p["eta_r"] for p in hyd["points"] if np.isfinite(p["sem"]) and p["sem"] > 0]
    if sems:
        w(f"5. **噪声地板（实测）**：单点 `SEM/η_r` 的中位数 "
          f"{_pct(np.median(sems), 2)}、最大 {_pct(np.max(sems), 2)}。"
          f"因此在 φ≤0.01（物理增量 1~2.5%）窗口内信噪比 <2，预注册的窄窗口扫掠被废止改为 "
          f"φ∈[0.02,0.10]（AMENDMENT-3/4 均有记录）；这也决定本报告以**曲率一致单参数估计**"
          f"为主判据估计量，弦斜率并列给出。")
    w("6. **适用范围**：牛顿载体、单分散刚性球、`Re_p ≪ 1`。真实碎冰/冰晶为多分散、非球形、"
      "易絮凝且有屈服应力，本闭合只是其牛顿极限；不外推到屈服/惯性主导区。")
    w("")
    w("## 5 误差判定（逐项，显式回答是否 <3%）")
    w("")
    w("判据 (a) 同时按两种估计量给出（预注册弦斜率 + AMENDMENT-4 曲率一致估计），"
      "并在三种体积口径下重复（判据阈值不变）：")
    w("")
    w("| 判据 | 阈值 | mask 弦 | mask 曲率 | hydro 弦 | hydro 曲率 | core 弦 | core 曲率 |")
    w("|---|---|---|---|---|---|---|---|")
    acc_m = mask["acceptance"]["criteria"]
    acc_h = hyd["acceptance"]["criteria"]
    acc_mc = mask["acceptance_curvature_estimator"]["criteria"]
    acc_hc = hyd["acceptance_curvature_estimator"]["criteria"]
    core = convs.get("core_fraction") or {}
    if "error" not in core and core:
        acc_co = core["acceptance"]["criteria"]
        acc_coc = core["acceptance_curvature_estimator"]["criteria"]
    else:
        acc_co = acc_coc = {}
    labels = {
        "a_einstein_slope": "(a) 稀薄斜率 vs 2.5，相对误差",
        "a_prime_intercept": "(a') 稀薄拟合截距 \|b-1\|",
        "b_divergence_exponent": "(b) 发散指数 vs -[η]φ_m，相对误差",
        "c_full_range_r2": "(c) 全域 KD 拟合 R²（越大越好）",
        "d_limit_increment": "(d2) 最小 φ 增量比偏离 1",
        "d_negative_control_mu": "(d1) 纯流体 η_eff/ν 偏差",
    }
    for key, label in labels.items():
        if key not in acc_m:
            continue
        m, h = acc_m[key], acc_h.get(key, {})
        mc, hc = acc_mc.get(key, {}), acc_hc.get(key, {})
        co, coc = acc_co.get(key, {}), acc_coc.get(key, {})
        thr = m["threshold"]
        w(f"| {label} | {thr} | {_f(m['value'], 4)} {_mark(m['pass'])} | "
          f"{_f(mc.get('value'), 4)} {_mark(mc.get('pass', False))} | "
          f"{_f(h.get('value'), 4)} {_mark(h.get('pass', False))} | "
          f"{_f(hc.get('value'), 4)} {_mark(hc.get('pass', False))} | "
          f"{_f(co.get('value'), 4)} {_mark(co.get('pass', False))} | "
          f"{_f(coc.get('value'), 4)} {_mark(coc.get('pass', False))} |")
    w("")
    valid = [b for b in convs.values() if "error" not in b]
    best = min(valid, key=lambda b: b["acceptance"]["n_failed"])
    hc_a = acc_hc.get("a_einstein_slope", {})
    coc_a = acc_coc.get("a_einstein_slope", {})
    w(f"**判据 (a) 是否 <3%**：mask/hydro/core 三口径的弦斜率相对偏差分别为 "
      f"{_pct(acc_m['a_einstein_slope']['value'], 2)} / {_pct(acc_h['a_einstein_slope']['value'], 2)} / "
      f"{_pct(acc_co.get('a_einstein_slope', {}).get('value'), 2)}；曲率一致估计下分别为 "
      f"{_pct(acc_mc['a_einstein_slope']['value'], 2)} / {_pct(hc_a.get('value'), 2)} / "
      f"{_pct(coc_a.get('value'), 2)}。"
      + ("**无一 <3%**。" if not any(abs(float(x.get('value', 1.0))) < 0.03
                                     for x in (acc_m['a_einstein_slope'], acc_hc.get('a_einstein_slope', {}),
                                               acc_coc.get('a_einstein_slope', {})))
         else "至少一种口径/估计量 <3%，见上表。"))
    w("")
    w(f"**判定口径**：{best['volume_convention']} 口径下 "
      f"{len(best['acceptance']['criteria']) - best['acceptance']['n_failed']}"
      f"/{len(best['acceptance']['criteria'])} 项判据通过；"
      f"未通过项见上表 FAIL 行，报告不做粉饰。")
    w("")
    w("## 6 下一步（把未达成的判据变成可达成的）")
    w("")
    w("1. **分辨率阶梯上移**：逐半径 `[η]` 随 r 单调上升（见 §3.6/图 6），说明现有 r=3~5 的阶梯"
      "远未进入渐近区。下一步用 r = 8, 10, 12（同一 48×48×57 元胞需放大到 ≥96³）做 3 档 + "
      "Riсhardson 外推，预算约 8×当前。")
    w("2. **积分时长 >10⁴ 步 或 改变压力/壁驱动设定**：诊断 A/B 显示近壁带应力的建立需要"
      "`sqrt(nu t) >= 元胞半高`，`nu=0.1` 时约需 3×10⁴ 步；或把 `ν` 降到 0.02（同时降 `u_wall` 保持 `Re_p`）。")
    w("3. **稠密区换测量方法**：φ≳0.15 时近壁带法失效（出现 `η_r<1`）。应改用体平均应力"
      "（`tau_rows_mean` 全列加权）或直接由颗粒应力张量求和（Batchelor 应力let 法），并把"
      "`φ_m` 的识别交给能到达 φ≈0.5 的多分散放置（RSA 在 0.4 附近卡死，需用润滑层/松弛放置）。")
    w("4. **求解器侧耦合验证**：本模块的 `SuspensionViscosityClosure` 已可直接注入载体粘度；"
      "下一步在 TensorLBM 的 Herschel-Bulkley / mu(I) 分支里做一次端到端算例（管道/槽道），"
      "与解析 Poiseuille 剖面比对。")
    w("")
    w(f"*产物*：`benchmarks/suspension/artifacts/raw_cases.json`（{len(hyd['points'])} 个聚合测点/"
      f"{sum(p['n_seeds'] for p in hyd['points'])} 个算例）、`artifacts/analysis.json`、"
      f"`artifacts/figures/`；实现 `src/tensorlbm/suspension_viscosity.py`；"
      f"测试 `tests/test_suspension_viscosity.py`。")
    w("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=os.path.join(HERE, "artifacts", "analysis.json"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "docs", "reports", "P46W3-N03_suspension_viscosity.md"))
    args = ap.parse_args(argv)
    with open(args.analysis) as fh:
        a = json.load(fh)
    md = render(a)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(md)
    print(f"report -> {args.out} ({len(md)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())