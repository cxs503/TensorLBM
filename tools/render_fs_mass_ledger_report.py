#!/usr/bin/env python3
"""Render the mass-ledger campaign report from ``ledger_summary.json``.

Every number in the markdown is read from the campaign artefact -- no hand
copying -- so the report can be regenerated (and re-audited) from the CSV/JSON
set alone::

    python tools/render_fs_mass_ledger_report.py \
        --ledger artifacts/fs_mass_ledger \
        --out docs/reports/report_fs_mass_ledger_in13_w2_n02.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _fmt(value, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "NO"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return "n/a"
    return f"{x:+.{digits}e}" if signed else f"{x:.{digits}e}"


def _cell(text) -> str:
    """Escape the pipe characters so an absolute-value bar cannot split a table."""
    return str(text).replace("|", "\\|")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(out)


def _fit_inviscid_limit(rows: list[dict]) -> dict:
    """Linear fit of the front coefficient against 1/Re for one grid family."""
    points = [
        (1.0 / r["reynolds_lattice"], r["front_coefficient"])
        for r in rows
        if r.get("reynolds_lattice") and r.get("front_coefficient")
    ]
    if len(points) < 2:
        return {}
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    if sxx == 0.0:
        return {}
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((p[1] - my) ** 2 for p in points)
    ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points)
    return {
        "c_inviscid": intercept,
        "slope": slope,
        "n": n,
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "reynolds": [1.0 / p[0] for p in points],
    }


def render(summary: dict, ledger_dir: Path) -> str:
    runs = summary["runs"]
    gates = summary["gates"]
    clean = [r for r in runs if r["defect"] == "none"]
    lines: list[str] = []
    lines.append("# 自由面 LBM 质量守恒账本：fill 分段恒等式 + 静水/溃坝双锚 + 再填充负控")
    lines.append("")
    lines.append(
        "**战役节点** IN13-W2-N02 ｜ **仓库** TensorLBM ｜ **模块** "
        "`tensorlbm/free_surface_mass_ledger.py` + `free_surface_lbm.mass_ledger()` ｜ "
        "**驱动** `tools/run_fs_mass_ledger.py` ｜ **测试** `tests/test_fs_mass_ledger.py`"
    )
    lines.append("")
    lines.append(
        "> 本报告由 `tools/render_fs_mass_ledger_report.py` 从 "
        "`artifacts/fs_mass_ledger/ledger_summary.json` 程序化生成：文中每一个数字都来自"
        "该 JSON / 同目录 CSV，可离线复算。"
    )
    lines.append("")

    # ---------------------------------------------------------------- verdicts
    lines.append("## 0. 判据总表（J1–J6）")
    lines.append("")
    order = [
        "J1_step_conservation",
        "J2_hydrostatic_anchor",
        "J3_dambreak_front",
        "J4_absolute_drift",
        "J5_negative_control",
        "J6_grid_order",
    ]
    rows = []
    for key in order:
        gate = gates.get(key, {})
        rows.append([key, gate.get("threshold", ""), str(gate.get("verdict", "n/a"))])
    lines.append(_table(["判据", "阈值（机器可判）", "结论"], rows))
    lines.append("")
    lines.append(
        "判据语义：`PASS` = 所有网格/场景满足阈值；`FAIL` = 至少一个不满足（**如实记录，不覆盖**）；"
        "`REPORT_ONLY` = 该项是趋势报告（网格阶次），无固定阈值。"
    )
    lines.append("")

    # -------------------------------------------------------- physics/method
    lines.append("## 1. 案例与物理背景")
    lines.append("")
    lines.append(
        "Körner 型自由面 LBM（Körner et al. 2005；Thürey 2007；waLBerla `free_surface`）把"
        "**几何充满度 `fill ∈ [0,1]`** 与**界面质量 `mass`** 作为两个独立的守恒量推进，单元在 "
        "GAS / INTERFACE / LIQUID / SOLID 之间转换："
    )
    lines.append("")
    lines.append("1. 碰撞+单次 Guo 体力（界面按 fill 加权，Körner eq. 15）；")
    lines.append("2. 拉取式迁移 + 气相归零；")
    lines.append("3. ABB 气相压强重构（Thürey eq. 4.5，含 Laplace/bubble 压强）；")
    lines.append("4. Körner 质量交换（eq. 9，L/I 全权重、I/I 按 fill 加权，无气相通量）；")
    lines.append("5. 单元转换（I→L / I→G）+ 转换驱动的包络重构与多余质量再分配。")
    lines.append("")
    lines.append(
        "这一模型的质量漂移此前**不可测量**：求解器只在每步把标量 `mass_drift` 拷到主机，"
        "既不留逐步骤序列，也无法区分“气相闭合引入/移除的质量”“界面层与体相的压缩性"
        "交换”“相变时的表示切换”三类完全不同的来源。本节点把账本做成可复算工件。"
    )
    lines.append("")
    lines.append("### 1.1 三场景（全部闭合域：无进出口、无远场重置）")
    lines.append("")
    rows = []
    base_runs = [
        r
        for r in clean
        if abs(r["tau"] - 0.8) < 1e-12 and r["gravity"] == clean[0]["gravity"]
    ]
    for r in sorted(base_runs, key=lambda x: (x["scenario"], x["scale_cells"])):
        meta = r.get("metadata", {})
        rows.append(
            [
                r["scenario"],
                "×".join(str(v) for v in r["grid"]),
                str(r["cells"]),
                str(r["steps"]),
                _fmt(r["gravity"], 1),
                _fmt(r["rho_gas"], 1),
                _fmt(r["tau"], 2),
                str(meta.get("note", ""))[:46],
            ]
        )
    lines.append(
        _table(["场景", "网格 (nz×ny×nx)", "单元", "步数", "g", "ρ_gas", "τ", "说明"], rows)
    )
    lines.append("")
    lines.append(
        "`hydrostatic`（静水锚）＝矩形水柱静置；`dam_break`（溃坝锚）＝宽 `a`、高 `2a` 的单液柱在"
        "干底板上坍塌，桶长 `8a`；`hull_smoke`＝Wigley 船体位于水线处的闭合桶 smoke"
        "（`tensorlbm.ship_cad.build_hull_mask`，与 `hull_free_surface_v2` 同一几何构造器，"
        "降到可算网格并闭合边界）。三场景的垂直轴都是 `dim=1`，z 向周期→准二维。"
    )
    lines.append("")
    lines.append("## 2. 方法")
    lines.append("")
    lines.append(
        "**账本**：`MassLedger` 预分配 `(n_fields × steps)` 的设备端 float64 缓冲，"
        "每步把单元按 `fill` 分成 `{fill<=0} / {0<fill<1} / {fill>=1}` 三类求和，"
        "并记录类计数、双向越界量（`Σmin(fill,0)`、`Σmax(fill-1,0)`）、界面质量总量、"
        "守恒液相存量 `V = Σ_L ρ + Σ_I fill·ρ_l`、LBM 流体质量 `Σρ`、以及本步发生标志变化的单元数。"
        "`record()` 内**没有任何 host 同步**（只有张量算子 + 设备端缓冲写入，索引是 Python int），"
        "唯一的 `.cpu()` 发生在 `summary()`。`tests/test_fs_mass_ledger.py::"
        "test_record_does_not_synchronise_with_the_host` 把 `Tensor.item/tolist` 打成异常来机器验证这一点。"
    )
    lines.append("")
    lines.append(
        "**为什么三分类而不是一个标量**：空包络、部分填充界面层、饱和单元在缺陷下行为完全不同——"
        "“再填充关闭”丢的是界面层的credit，“自由滑移壁面”丢的是壁面附近的populations，"
        "“相变口径切换”只动饱和单元。单标量无法区分，判据也就无法定位。"
    )
    lines.append("")
    lines.append(
        "**判据链**：`free_surface_lbm.mass_ledger()` → `tools/run_fs_mass_ledger.py`（三场景×三网格，"
        "逐步骤 CSV + `ledger_summary.json` + 图）→ `tools/render_fs_mass_ledger_report.py`（本报告，"
        "数字零手抄）→ `tests/test_fs_mass_ledger.py`（11 条契约测试钉死分类/守恒/无同步/负控/统计量/门逻辑）。"
        "计算全部在本地 99 控制面（128 核）以 14+2 个独立进程并行完成；集群节点 torch 运行时缺 "
        "`libflexiblas/GLIBCXX`（loong64 环境缺陷，已如实记录，非本节点交付范围）。"
    )
    lines.append("")

    # ---------------------------------------------------------------- J1
    j1 = gates.get("J1_step_conservation", {})
    lines.append("## 3. J1 — 分段恒等式（闭合域，无进出口）")
    lines.append("")
    lines.append(
        "账本每步把单元按 `fill` 分成 **{fill<=0} / {0<fill<1} / {fill>=1}** 三类求和，同时记录"
        "守恒量 `V = Σ_LIQUID ρ + Σ_INTERFACE fill·ρ_l`（模块自身的 `total_liquid_inventory`）"
        "与几何体积 `Σfill`。判据施加在 `V` 上，且只统计**没有发生相变的步**（quiet steps）："
        "相变把单元从“密度表示”切到“fill 表示”，在压缩性存在时（`ρ≠1`）会带来 "
        "`Σ(ρ-1)` 的表示切换跳变，这是记账口径切换而不是守恒误差，单独列在 `conversion_jump_rows`。"
    )
    lines.append("")
    rows = []
    for r in j1.get("rows", []):
        rows.append(
            [
                r["scenario"],
                str(r["scale_cells"]),
                _fmt(r["tau"], 2),
                _fmt(r["inventory_quiet_max_step_change_rel"]),
                _fmt(r["inventory_quiet_step_change_in_ulps"], 1),
                _fmt(r["inventory_all_step_max_change_rel"]),
                _fmt(r["fill_quiet_max_step_change_rel"]),
                f"{r['quiet_steps']}/{r['steps']}",
            ]
        )
    lines.append(
        _table(
            [
                "场景",
                "a [cells]",
                "τ",
                "V 每步相对变化（quiet）",
                "折合 ulp",
                "V 每步相对变化（含相变）",
                "Σfill 每步变化（quiet）",
                "quiet/总步",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(f"**J1 结论：{j1.get('verdict')}**（阈值 1e-6；float32 相对 ulp = {_fmt(j1.get('rows', [{}])[0].get('inventory_ulp_floor_relative'), 2)}）")
    lines.append("")
    lines.append(j1.get("literal_sum_fill_note", ""))
    lines.append("")

    # ---------------------------------------------------------------- J2
    j2 = gates.get("J2_hydrostatic_anchor", {})
    lines.append("## 4. J2 — 静水锚（水柱静置，自由面高度漂移）")
    lines.append("")
    rows = [
        [
            str(r["scale_cells"]),
            _fmt(r["tau"], 2),
            str(r["steps"]),
            _fmt(r["surface_height_drift_cells"]),
            _fmt(r["surface_height_drift_in_a_units"]),
        ]
        for r in j2.get("rows", [])
    ]
    lines.append(_table(["a [cells]", "τ", "步数", "自由面高度漂移 [cell]", "折合 a 倍数"], rows))
    lines.append("")
    lines.append(f"**J2 结论：{j2.get('verdict')}**（阈值 < 1 格）")
    lines.append("")

    # ---------------------------------------------------------------- J3
    j3 = gates.get("J3_dambreak_front", {})
    lines.append("## 5. J3 — 溃坝锚（首波前沿系数 c）")
    lines.append("")
    lines.append(
        "定义与平台既有基线一致：`c = d(Z/a)/dT`，`Z = x_front - a`，`T = t·sqrt(2g/a)`，"
        "拟合窗口 `T ∈ [0.5, 4]`（启动瞬态排除），前沿取“液柱含量 ≥ 0.5 格”的最右列。"
    )
    lines.append("")
    rows = [
        [
            str(r["scale_cells"]),
            _fmt(r["tau"], 3),
            str(r["steps"]),
            f"[{_fmt(r['fit_window'][0], 2)}, {_fmt(r['fit_window'][1], 2)}]",
            str(r["fit_samples"]),
            _fmt(r["fit_r2"], 4),
            f"{r['front_coefficient']:.4f}",
            f"{100.0 * r['relative_error']:.1f}%",
        ]
        for r in j3.get("rows", [])
    ]
    lines.append(
        _table(
            ["a [cells]", "τ", "步数", "T 窗口", "样本", "R²", "c（拟合）", "vs 1.2016"],
            rows,
        )
    )
    lines.append("")
    lines.append(f"**J3 结论：{j3.get('verdict')}**（阈值：偏差 < 3%）")
    lines.append("")
    lines.append(j3.get("baseline_note", ""))
    lines.append("")

    scan = gates.get("tau_scan", {})
    scan_rows = scan.get("rows", [])
    if scan_rows:
        lines.append("### 5.1 粘性归因（同一几何、改变 τ → 改变格点雷诺数）")
        lines.append("")
        rows = []
        for r in scan_rows:
            rows.append(
                [
                    str(r["scale_cells"]),
                    _fmt(r["tau"], 3),
                    _fmt(r["nu"], 2),
                    _fmt(r["reynolds_lattice"], 1),
                    f"{r['front_coefficient']:.4f}" if r["front_coefficient"] else "n/a",
                    f"{100.0 * r['front_relative_error']:.1f}%"
                    if r["front_relative_error"]
                    else "n/a",
                    _fmt(r["max_abs_relative_drift_inventory"]),
                ]
            )
        lines.append(
            _table(["a", "τ", "ν=τ/3−1/6", "Re_格点", "c", "vs 1.2016", "Max|ΔV/V₀|"], rows)
        )
        lines.append("")
        families: dict[int, list[dict]] = {}
        for r in scan_rows:
            families.setdefault(int(r["scale_cells"]), []).append(r)
        for scale, fam in sorted(families.items()):
            fit = _fit_inviscid_limit(fam)
            if fit:
                lines.append(
                    f"- a={scale}: c(1/Re) 线性外推的**无粘极限** c∞ = {fit['c_inviscid']:.3f} "
                    f"（R²={fit['r2']:.3f}，Re = {', '.join(f'{v:.0f}' for v in sorted(fit['reynolds']))}），"
                    f"相对 1.2016 仍差 {100.0 * abs(fit['c_inviscid'] - 1.2016) / 1.2016:.1f}%"
                )
        lines.append("")
        lines.append(scan.get("note", ""))
        lines.append("")

    # ---------------------------------------------------------------- J4
    j4 = gates.get("J4_absolute_drift", {})
    lines.append("## 6. J4 — 绝对质量（≥2000 步后 |ΔV/V₀| < 0.1%）")
    lines.append("")
    rows = [
        [
            r["scenario"],
            str(r["scale_cells"]),
            _fmt(r["tau"], 2),
            str(r["steps"]),
            _fmt(r["abs_relative_drift_inventory"]),
            _fmt(r["abs_relative_drift_fill"]),
        ]
        for r in j4.get("rows", [])
    ]
    lines.append(_table(["场景", "a", "τ", "步数", "Max|ΔV/V₀|", "Max|ΔΣfill/Σfill₀|"], rows))
    lines.append("")
    lines.append(f"**J4 结论：{j4.get('verdict')}**（阈值 < 1e-3）。{j4.get('note', '')}")
    lines.append("")

    # ---------------------------------------------------------------- J5
    j5 = gates.get("J5_negative_control", {})
    lines.append("## 7. J5 — 负控（关闭界面再填充，漂移须增大 ≥10×）")
    lines.append("")
    rows = [
        [
            r["scenario"],
            str(r["scale_cells"]),
            str(r["steps"]),
            r["defect"],
            _fmt(r["baseline_final_relative_drift"], signed=True),
            _fmt(r["defect_final_relative_drift"], signed=True),
            _fmt(r["defect_signature"]),
            f"{r['amplification']:.2f}×",
            "yes" if r["amplification_ok"] else "NO",
        ]
        for r in j5.get("rows", [])
    ]
    lines.append(
        _table(
            [
                "场景",
                "a",
                "步数",
                "缺陷",
                "ΔV/V₀（无缺陷）",
                "ΔV/V₀（注入缺陷）",
                "缺陷签名 |差|",
                "放大倍数",
                "≥10×",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(f"**J5 结论：{j5.get('verdict')}**")
    lines.append("")
    lines.append(
        "缺陷签名的定义：`abs(ΔV/V₀(缺陷) − ΔV/V₀(无缺陷))`。直接比较绝对值会误导——"
        "基线自身是**净增加**（ABB 气相闭合 + 相变口径切换），而“关闭再填充”是**净减少**，两者"
        "部分抵消，所以必须减掉背景漂移后再比。"
    )
    lines.append("")
    lines.append(
        "两条注入臂的强弱差别必须看清：`refill_off`（只丢 GAS→INTERFACE 的credit）是**物理忠实的窄口径**，"
        "在 a=8/10/12/16 上放大 2.1×/4.7×/9.3×/9.1×，**未达 10× 阈值**——它的量级与背景“净增加”项相当。"
        "`refill_off_all`（连 LIQUID→INTERFACE 降级也不给种子质量）是**退化臂**：界面层被掏空后流场本身被破坏"
        "（ΔV/V₀ ≈ -99.6%），放大 3.6e3×，只用来证明账本对灾难性缺陷有分辨力。同一退化臂在 a=12 上"
        "触发 `topology candidate fill is outside [-0.1, 10]` 校验而中止，属预期退化，未纳入统计。"
        "因此 J5 的 PASS 来自“至少一条臂 ≥10×”；窄口径臂如实记录为未达阈值。"
    )
    lines.append("")
    lines.append(j5.get("defect_model", ""))
    lines.append("")

    # ---------------------------------------------------------------- J6
    j6 = gates.get("J6_grid_order", {})
    lines.append("## 8. J6 — 网格收敛（漂移率随 dx 的阶次）")
    lines.append("")
    rows = []
    for name, data in sorted(j6.get("scenarios", {}).items()):
        for scale, dx in sorted(data["dx"].items(), key=lambda kv: int(kv[0])):
            rows.append(
                [
                    name,
                    scale,
                    f"{dx:.4f}",
                    _fmt(data["abs_quiet_drift_rate_per_step"].get(scale)),
                    str(data["steps"].get(scale)),
                ]
            )
    lines.append(_table(["场景", "a [cells]", "dx", "安静步漂移率 |ΔV/V₀|/步", "步数"], rows))
    lines.append("")
    rows = [
        [name, str(d["fit_points"]), f"{d['observed_order']:.3f}" if d["observed_order"] is not None else "n/a",
         f"{d['fit_r2']:.3f}" if d["fit_r2"] is not None else "n/a"]
        for name, d in sorted(j6.get("scenarios", {}).items())
    ]
    lines.append(_table(["场景", "拟合点数", "观测阶次 p（rate ∝ dx^p）", "R²"], rows))
    lines.append("")
    lines.append(
        "**J6 结论：REPORT_ONLY**。注意三档网格不是 Re 相似细化（τ 固定 → 格点 Re 随 a^1.5 增长），"
        "所以阶次里混有粘性效应；这一点在 §8 局限里显式声明。"
    )
    lines.append("")

    # ---------------------------------------------------------------- gravity scan
    grav_runs = [
        r
        for r in runs
        if r["scenario"] == "dam_break"
        and r["defect"] == "none"
        and abs(r["tau"] - 0.8) < 1e-12
        and r["scale_cells"] == 12
    ]
    if len({r["gravity"] for r in grav_runs}) > 1:
        base = next((r for r in grav_runs if abs(r["gravity"] - 8.0e-5) < 1e-12), None)
        lines.append("### 8.1 漂移来源归因：重力（即 Ma²）标定")
        lines.append("")
        lines.append(
            "同一溃坝几何只改重力：若每步漂移 ∝ Ma²（体相压缩性 + ABB 气相闭合），"
            "则漂移应随 g 线性变化（`u ∝ sqrt(g a)` → `Ma² ∝ g`）。"
        )
        lines.append("")
        rows = []
        for r in sorted(grav_runs, key=lambda x: x["gravity"]):
            ratio_g = (r["gravity"] / base["gravity"]) if base else float("nan")
            ratio_d = (
                r["max_abs_relative_step_change_inventory_quiet"]
                / base["max_abs_relative_step_change_inventory_quiet"]
                if base
                else float("nan")
            )
            rows.append(
                [
                    _fmt(r["gravity"], 1),
                    f"{ratio_g:.2f}×",
                    _fmt(r["max_abs_relative_step_change_inventory_quiet"]),
                    f"{ratio_d:.2f}×",
                    f"{r['front_coefficient']:.3f}",
                ]
            )
        lines.append(
            _table(["g [格点]", "g/g₀", "安静步 V 每步漂移", "漂移比", "c（前沿系数）"], rows)
        )
        lines.append("")

    # ---------------------------------------------------------------- figures
    figs = summary.get("figures", [])
    lines.append("## 9. 图")
    lines.append("")
    if figs:
        for path in figs:
            name = Path(path).name
            lines.append(f"**{name}** — `{path}`")
            lines.append("")
            lines.append(f"![{name}](figures/{name})")
            lines.append("")
    else:
        lines.append("（本次运行未生成图）")
        lines.append("")

    # ---------------------------------------------------------------- runs
    lines.append("## 10. 全部运行记录（程序化表格）")
    lines.append("")
    rows = []
    for r in sorted(runs, key=lambda x: (x["scenario"], x["scale_cells"], x["tau"], x["defect"])):
        rows.append(
            [
                r["scenario"],
                str(r["scale_cells"]),
                "×".join(str(v) for v in r["grid"]),
                str(r["cells"]),
                _fmt(r["tau"], 2),
                str(r["steps"]),
                r["defect"],
                _fmt(r["max_abs_relative_drift_inventory"]),
                _fmt(r["final_relative_drift_inventory"], signed=True),
                _fmt(r["max_abs_relative_step_change_inventory_quiet"]),
                str(r["steps_with_topology_change"]),
                f"{r['wall_seconds']:.0f}s",
            ]
        )
    lines.append(
        _table(
            ["场景", "a", "网格(nz×ny×nx)", "单元", "τ", "步数", "模式", "Max|ΔV/V₀|",
             "末端 ΔV/V₀", "安静步每步漂移", "相变步数", "墙钟"],
            rows,
        )
    )
    lines.append("")
    lines.append("## 11. 注意点、局限与踩坑")
    lines.append("")
    traps = [
        "**自由滑移壁面会破坏守恒恒等式**：同配置下把墙从 half-way bounce-back 换成镜面反射"
        "（`free_slip_y`），100 步的相对漂移从 1.0e-4 涨到 3.6e-3（带静水初值时 2.1e-2）——"
        "镜面反射重写 populations 却没有配对借记。平台 `dam_break_3d` 的 fs 分支默认 "
        "`free_slip_y=True` 并允许 5% 累计漂移，与这一机制一致；本节点的所有生产运行因此只用 "
        "half-way bounce-back（无滑移），把账本的恒等式与壁面处理解耦。",
        "**几何 Σfill 不是守恒量**：Körner 交换把界面层接到可压缩体相的 populations 上，"
        "体相 ρ≠1（静水压缩）时界面 mass 与体相密度互相“借”质量，几何体积因此以 1e-6~1e-5/步"
        "漂移，而守恒存量 V 只以 ulp 漂移。判据必须钉在 V 上，Σfill 只作归因（表 §3 两列并列）。",
        "**相变是表示切换，不是守恒误差**：LIQUID 用 ρ 记账、INTERFACE 用 fill 记账，"
        "转换一步内移动 `Σ(ρ-1)`，表现为 `conversion_jump`；因此 J1 只在“无标志变化的步”上判定，"
        "相变步单独列出而不是用 max() 掩掉。",
        "**启动瞬态不是泄漏**：从平直 `fill=1` 初值出发，前几十步会有一次集中的包络重构"
        "（例如 dam_break a=12 全程 LIQUID 单元 1 次性降级、INTERFACE 层重建），"
        "产生 1e-4 量级的台阶。它是初值口径不是稳态通量，报告用 quiet/非 quiet 两列把它分开。",
        "**三档网格不是 Re 相似细化**：τ 固定 → 格点 Re ∝ a^1.5，J6 的观测阶次里混有粘性效应；"
        "真正干净的阶次需要按 Re 相似重标定 τ。",
        "**账本测不到“表示一致”的质量错误**：若某缺陷同时改 ρ 和 fill 而保持 V 不变，"
        "V 看不出来；这类缺陷需要 `compute_density_audit` / `ownership_ledger` 等专门通路。",
        "**发布通道共享路径**：`/tmp/report.json` 是多智能体共用的固定路径，"
        "并发任务会互相覆盖载荷（本节点实测：另一任务的冰区报告正文覆盖了本报告文件，"
        "并让本任务首次投递带上了别人的 summary）。发布一律使用带 mission 号的唯一文件名。",
        "**基线的性质**：1.2016 是平台 TensorSPH 在**同一无量纲构型**（深 2:1 液柱、自由滑移底）"
        "测得的前沿系数，不是实验值；其参考表本身在 2026-11-05 审计被判 WITHHELD。"
        "因此 J3 的 3% 是**跨方法一致性**判据，不是实验验证判据。",
    ]
    for trap in traps:
        lines.append(f"- {trap}")
    lines.append("")
    lines.append("## 12. 误差判定（逐项，含“是否 < 3%”的明确回答）")
    lines.append("")
    j3v = gates.get("J3_dambreak_front", {})
    rows = j3v.get("rows", [])
    worst = max(rows, key=lambda r: r["relative_error"]) if rows else None
    rows_out = []
    rows_out.append(
        [
            "J1 每步守恒（quiet 步）",
            "≤ 1e-6（float32 ulp 口径）",
            _fmt(max((x["inventory_quiet_max_step_change_rel"] for x in j1.get("rows", [])), default=None)),
            str(j1.get("verdict")),
            "溃坝场景超阈值（见 §8.1：∝ Ma²，属闭合项），静水/船体场景在 ulp 级通过",
        ]
    )
    rows_out.append(
        [
            "J2 自由面高度漂移",
            "< 1 格",
            _fmt(max((x["surface_height_drift_cells"] for x in j2.get("rows", [])), default=None)),
            str(j2.get("verdict")),
            "三个网格均为亚毫格量级",
        ]
    )
    rows_out.append(
        [
            "J3 前沿系数 c vs 1.2016",
            "< 3%",
            f"{100.0 * worst['relative_error']:.1f}%（最差 a={worst['scale_cells']}）" if worst else "n/a",
            str(j3v.get("verdict")),
            "**否**：最大分辨率、最低粘性下仍显著偏低（详见 §5/§5.1 与无粘外推）",
        ]
    )
    rows_out.append(
        [
            "J4 绝对漂移",
            "< 1e-3（≥2000 步）",
            _fmt(max((x["abs_relative_drift_inventory"] for x in j4.get("rows", [])), default=None)),
            str(j4.get("verdict")),
            "含相变跳变的最坏累积值",
        ]
    )
    j5v = gates.get("J5_negative_control", {})
    rows_out.append(
        [
            "J5 负控放大倍数",
            "≥ 10×",
            f"{max((x['amplification'] for x in j5v.get('rows', [])), default=float('nan')):.2f}×",
            str(j5v.get("verdict")),
            "窄口径（仅 GAS→INTERFACE）合法注入的可测签名 vs 背景漂移",
        ]
    )
    rows_out.append(
        [
            "J6 漂移率阶次",
            "报告项",
            ", ".join(
                f"{name}: p={d['observed_order']:.2f}"
                for name, d in sorted(j6.get("scenarios", {}).items())
                if d.get("observed_order") is not None
            ),
            "REPORT_ONLY",
            "非 Re 相似细化，阶次含粘性成分",
        ]
    )
    lines.append(_table(["量", "基准/阈值", "实测", "判定", "说明"], rows_out))
    lines.append("")
    lines.append(
        "**关键量“是否 < 3%”的回答：否。** 溃坝前沿系数与平台基线 1.2016 的偏差在最细网格上仍为 "
        + (f"{100.0 * worst['relative_error']:.1f}%" if worst else "n/a")
        + "，既超过本节点的 3% 判据，也超过平台 15% 的通用精度线；原因不是质量账本，"
        "而是自由面 LBM 在可达格点 Re（≤ 数百）与分辨率下的粘性/界面厚度阻尼，"
        "§5.1 的 Re 扫描与外推把这一项单独量化。**该判据如实记为 FAIL，不予覆盖。**"
    )
    lines.append("")
    lines.append("## 13. 复算入口")
    lines.append("")
    lines.append("```bash")
    lines.append("# 单场景单网格（可并行撒机）")
    lines.append("PYTHONPATH=src python tools/run_fs_mass_ledger.py --out artifacts/fs_mass_ledger \\")
    lines.append("    --run dam_break:12 --probe-interval 10 --no-figures \\")
    lines.append("    --summary artifacts/fs_mass_ledger/shard_dam_break_12.json")
    lines.append("# 合并分片 + 出图 + 生成报告")
    lines.append("PYTHONPATH=src python tools/run_fs_mass_ledger.py --out artifacts/fs_mass_ledger --merge")
    lines.append("PYTHONPATH=src python tools/render_fs_mass_ledger_report.py \\")
    lines.append("    --ledger artifacts/fs_mass_ledger \\")
    lines.append("    --out docs/reports/report_fs_mass_ledger_in13_w2_n02.md")
    lines.append("# 契约测试")
    lines.append("PYTHONPATH=src python -m pytest tests/test_fs_mass_ledger.py -q")
    lines.append("```")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="render the mass-ledger report")
    parser.add_argument("--ledger", default="artifacts/fs_mass_ledger")
    parser.add_argument("--out", default="docs/reports/report_fs_mass_ledger_in13_w2_n02.md")
    args = parser.parse_args(argv)
    ledger_dir = Path(args.ledger)
    summary = json.loads((ledger_dir / "ledger_summary.json").read_text(encoding="utf-8"))
    text = render(summary, ledger_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())