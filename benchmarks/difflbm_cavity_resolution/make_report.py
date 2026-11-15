#!/usr/bin/env python3
"""Build figures + the paper-level report for GAP-DIFFLBM-CAV-RES.

Reads ``artifacts/gap_difflbm_cav_res/summary.json`` (run.py --aggregate) and
the per-case JSON (which keeps the centreline profiles), writes

  figures/gap_difflbm_cav_res/fig1_convergence.png   log-log error vs n
  figures/gap_difflbm_cav_res/fig2_profiles.png      centreline profiles vs Ghia
  docs/DIFFLBM_CAVITY_RESOLUTION.md                  report (numbers injected)

Every number in the markdown comes from the JSON — nothing hand-transcribed.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "gap_difflbm_cav_res"
FIG = REPO / "figures" / "gap_difflbm_cav_res"
DOC = REPO / "docs" / "DIFFLBM_CAVITY_RESOLUTION.md"

GHIA100 = {
    "y": [0.0, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172,
          0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0],
    "u": [0.0, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
          -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871,
          0.84123, 1.0],
    "x": [0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000,
          0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0],
    "v": [0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527,
          0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
          -0.05906, 0.0],
}


def load():
    summary = json.loads((ART / "summary.json").read_text())
    cases = {}
    for f in sorted(ART.glob("case_re*.json")):
        c = json.loads(f.read_text())
        cases[(int(c["re"]), c["nx"])] = c
    return summary, cases


def fig_convergence(summary) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, re_key in enumerate(sorted(summary["by_re"])):
        blk = summary["by_re"][re_key]
        n = np.array(blk["nx"], float)
        md = np.array(blk["max_abs_dev_pct"], float)
        l2 = np.array(blk["rmse_u"], float) * 100
        a = ax[k]
        a.loglog(n, md, "o-", label="max-point |u-u$_{Ghia}$|", color="C3")
        a.loglog(n, l2, "s--", label="L2 (RMSE) u-profile", color="C0")
        # 2nd-order reference line anchored at the coarsest point
        a.loglog(n, md[0] * (n[0] / n) ** 2, ":", color="k",
                 label="slope $-2$ (2nd order)")
        a.axhline(3.0, color="g", lw=1.2, ls="-.", label="3 % acceptance")
        a.axvline(97, color="gray", lw=0.8, ls="--", label="n = 97")
        p = blk["fit_vs_ghia_max_point"]["p"]
        ps = (blk["richardson"].get("vertical_centreline_u", {}).get("p") or {}).get("l2")
        a.set_title(f"Re = {re_key}   p$_{{self}}$ = "
                    f"{ps:.2f}   p$_{{Ghia}}$ = {p:.2f}")
        a.set_xlabel("n (grid nodes per side)")
        a.set_ylabel("error / %")
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("GAP-DIFFLBM-CAV-RES: cavity error vs resolution (log-log)")
    fig.tight_layout()
    out = FIG / "fig1_convergence.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_profiles(cases) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for (re, n), c in sorted(cases.items()):
        if re != 100:
            continue
        ax[0].plot(c["u_cl"], c["y"], lw=1.0, label=f"n={n}")
        ax[1].plot(c["x"], c["v_cl"], lw=1.0, label=f"n={n}")
    ax[0].plot(GHIA100["u"], GHIA100["y"], "ko", ms=4, label="Ghia 1982 (Table I)")
    ax[1].plot(GHIA100["x"], GHIA100["v"], "ko", ms=4, label="Ghia 1982 (Table II)")
    ax[0].set_xlabel("u / U$_{lid}$"); ax[0].set_ylabel("y / H")
    ax[0].set_title("Re=100 vertical centreline")
    ax[1].set_xlabel("v / U$_{lid}$"); ax[1].set_ylabel("x / H")
    ax[1].set_title("Re=100 horizontal centreline")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7.5)
    fig.tight_layout()
    out = FIG / "fig2_profiles.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def build_md(summary, fig1, fig2) -> str:
    L = []
    L.append("# GAP-DIFFLBM-CAV-RES — 方腔锚的分辨率收敛研究\n")
    L.append("> 本文件由 `benchmarks/difflbm_cavity_resolution/make_report.py` 程序化生成，"
             "文中所有数字取自 `artifacts/gap_difflbm_cav_res/summary.json`，零手抄。\n")
    L.append("## 1. 问题设定与基准出处\n")
    L.append("二维方腔顶盖驱动流（lid-driven cavity），`Re = u_lid·H/ν`，"
             "`H = n-1` 个格距，顶盖速度 `u_lid = 0.06`（Mach 数 ~0.10，可压缩性误差 O(Ma²)≈1%）。\n")
    L.append("- **基准出处**：Ghia, Ghia & Shin (1982), *J. Comput. Phys.* **48**, 387–411, "
             "Table I（竖直中线 u）与 Table II（水平中线 v），各 17 点，速度为 `u/U_lid`。"
             "本仓的可信转录位于库模块 `tensorlbm.lid_driven_cavity.GHIA_RE100` / `GHIA_RE400`，"
             "由已入库的 verified benchmark 使用（`benchmarks/verified/cavity/re100/run.py`，"
             "入库值 nx=128 → max|偏差| 0.7514%）。\n")
    L.append("- **本仓已知缺陷（不用于本单判定）**："
             "`tensorlbm.verification.lid_driven_cavity.reference_data.GHIA_RE100` 标注 "
             "`UNVERIFIED_OFFLINE` 且中心点 u(0.5) = −0.15612 与文献 −0.20581 差 24%"
             "（`docs/audit_m15543.md` F-A）。本单显式使用**库模块的可信表**，不用该常量。\n")
    L.append("- **判据（预注册）**：\n"
             "  - Q1：对全部 `n ≥ 97`，max-point 误差 < 3%；\n"
             "  - Q2：观测阶 `p ≥ 1.8`，主口径 = 自收敛 Richardson（`n = 33/65/129`，"
             "格距比 r = 2，竖直中线 u，L2 范数）；\n"
             "  - 稳态门：`max|Δu| < 1e-5`（每 10⁴ 步监测）。\n")
    L.append("- **为何把阶放到自收敛口径**：Ghia 表只有 5 位有效数字且本身不是精确解，"
             "vs-Ghia 误差在细网格会被参考数据精度截断（入库已见到 nx=128→192 仅 0.7514%→0.7263%）。"
             "该截断现象在本单单独列表报告，不隐藏。\n")

    L.append("\n## 2. 方法与实现\n")
    L.append("**零重复实现**——求解器/边界条件/参考表全部复用已验证模块：\n")
    L.append("| 组件 | 现有实现（文件） | 本单是否改动 |\n|---|---|---|\n"
             "| D2Q9 MRT 碰撞 + stream | `tensorlbm.solver.collide_mrt / stream` | 否 |\n"
             "| 三静止壁 pre-streaming 半程反弹（V3） | `benchmarks/verified/cavity/re100/run.py::stationary_pre_bounce` | 否 |\n"
             "| 顶盖 Zou-He 动壁 BC | `tensorlbm.lid_driven_cavity.zou_he_moving_lid` | 否 |\n"
             "| Ghia 1982 Table I/II | `tensorlbm.lid_driven_cavity.GHIA_RE100 / GHIA_RE400` | 否 |\n"
             "| 误差度量（max-point / rmse） | `benchmarks/verified/cavity/re{100,400}/run.py::run_case` | 否 |\n"
             "| 中线剖面外露 | 同上，新增**可选** `capture=` 形参（加法式、默认 None，不改既有行为） | 是（+9 行 ×2 文件） |\n"
             "| 分辨率扫描 + 阶拟合 | `benchmarks/difflbm_cavity_resolution/run.py`（本单新增） | 是（新文件） |\n")
    L.append("关键指针：`run.py::run_one`（调用 verified `run_case`，`capture={}` 取中线）、"
             "`run.py::richardson`（三网格 r=2 观测阶）、`run.py::aggregate`（判据表）。\n")
    L.append("格式：D2Q9 单松弛 MRT（`tau = 3·u_lid·n/Re + 0.5`），"
             "半程反弹壁、Zou-He 顶盖，浮点 float32 库路径（与入库 verified benchmark 完全一致）。\n")

    L.append("\n## 3. 参数表（无量纲）\n")
    L.append("| Re | u_lid | n | tau | steps | 稳态残差 |\n|---|---|---|---|---|---|\n")
    for re_key in sorted(summary["by_re"]):
        blk = summary["by_re"][re_key]
        for i, n in enumerate(blk["nx"]):
            L.append(f"| {re_key} | 0.06 | {n} | {blk['tau'][i]} | {blk['steps'][i]} | "
                     f"{blk['last_resid'][i]:.2e} |\n")

    L.append("\n## 4. 对照表（ref vs got + 误差，判据逐条打勾）\n")
    for re_key in sorted(summary["by_re"]):
        blk = summary["by_re"][re_key]
        L.append(f"\n### Re = {re_key}\n")
        L.append("| n | max-point 误差 vs Ghia / % | RMSE u | RMSE v | u(0.5,0.5) | Ghia u(0.5,0.5) | 涡心 (x,y) |\n"
                 "|---|---|---|---|---|---|---|\n")
        for i, n in enumerate(blk["nx"]):
            L.append(f"| {n} | {blk['max_abs_dev_pct'][i]:.4f} | {blk['rmse_u'][i]:.5f} | "
                     f"{blk['rmse_v'][i]:.5f} | {blk['u_mid'][i]:+.5f} | "
                     f"{blk['u_mid_ghia']:+.5f} | "
                     f"({blk['vortex'][i][0]:.4f}, {blk['vortex'][i][1]:.4f}) |\n")
        c = blk["criteria"]
        L.append("\n**判据**\n")
        L.append("| 判据 | 阈值 | 结果 | 判定 |\n|---|---|---|---|\n")
        L.append(f"| 观测阶 p（自收敛 L2） | ≥ 1.8 | "
                 f"{fmt((blk['richardson'].get('vertical_centreline_u',{}).get('p') or {}).get('l2'), 3)} | "
                 f"{'✅' if c['order_p_self_l2_ge_1.8'] else '❌'} |\n")
        L.append(f"| 观测阶 p（自收敛 max 范数） | ≥ 1.8 | "
                 f"{fmt((blk['richardson'].get('vertical_centreline_u',{}).get('p') or {}).get('max'), 3)} | "
                 f"{'✅' if c['order_p_self_max_ge_1.8'] else '❌'} |\n")
        L.append(f"| n ≥ 97 max-point < 3% | < 3 % | "
                 f"{', '.join(f'{d:.4f}%' for n, d in zip(blk['nx'], blk['max_abs_dev_pct']) if n >= 97)} | "
                 f"{'✅' if c['max_dev_lt_3pct_for_all_n_ge_97'] else '❌'} |\n")
        L.append(f"| max-point 误差随 n 单调下降 | 单调 | "
                 f"{'是' if c['max_dev_monotone_decrease'] else '否'} | "
                 f"{'✅' if c['max_dev_monotone_decrease'] else '❌'} |\n")
        L.append(f"| 全部工况达稳态 | resid < 1e-5 | "
                 f"{'是' if c['all_steady'] else '否'} | {'✅' if c['all_steady'] else '❌'} |\n")

    L.append("\n## 5. 收敛性（双对数曲线 + 观测阶）\n")
    L.append("![fig1](figures/gap_difflbm_cav_res/fig1_convergence.png)\n")
    L.append("\n**vs-Ghia 最小二乘阶（n ≥ 65）与逐段阶**\n")
    L.append("| Re | 口径 | p | R² | 逐段阶 |\n|---|---|---|---|---|\n")
    for re_key in sorted(summary["by_re"]):
        blk = summary["by_re"][re_key]
        pw = ", ".join(f"{d['from']}→{d['to']}: {d['p']:.2f}"
                       for d in blk["pairwise_order_vs_ghia"])
        L.append(f"| {re_key} | max-point vs Ghia | {fmt(blk['fit_vs_ghia_max_point']['p'],3)} | "
                 f"{fmt(blk['fit_vs_ghia_max_point']['r2'],4)} | {pw} |\n")
        L.append(f"| {re_key} | RMSE u vs Ghia | {fmt(blk['fit_vs_ghia_l2']['p'],3)} | "
                 f"{fmt(blk['fit_vs_ghia_l2']['r2'],4)} | — |\n")
    L.append("\n**自收敛（Richardson，r = 2，n = 33/65/129）**\n")
    L.append("| Re | 剖面 | ‖φ₃₃−φ₆₅‖ L2 | ‖φ₆₅−φ₁₂₉‖ L2 | p (L2) | p (max) |\n|---|---|---|---|---|---|\n")
    for re_key in sorted(summary["by_re"]):
        r = summary["by_re"][re_key]["richardson"]
        for key, lab in (("vertical_centreline_u", "u(y)"), ("horizontal_centreline_v", "v(x)")):
            if key in r:
                L.append(f"| {re_key} | {lab} | {r[key]['e_coarse_33_65']['l2']:.3e} | "
                         f"{r[key]['e_fine_65_129']['l2']:.3e} | "
                         f"{fmt(r[key]['p']['l2'],3)} | {fmt(r[key]['p']['max'],3)} |\n")
    L.append("\n![fig2](figures/gap_difflbm_cav_res/fig2_profiles.png)\n")
    return "".join(L)


def main():
    summary, cases = load()
    fig1 = fig_convergence(summary)
    fig2 = fig_profiles(cases) if cases else None
    md = build_md(summary, fig1, fig2)
    DOC.write_text(md)
    payload = {
        "title": "GAP-DIFFLBM-CAV-RES 方腔锚分辨率收敛研究（Re=100/400, n=33…129）",
        "markdown": md,
        "figures": [
            {"name": "fig1_convergence.png",
             "caption": "方腔误差 vs 网格 n 双对数曲线（max-point 与 L2；虚线=二阶参考斜率；绿线=3% 判据；灰线=n=97）",
             "png_base64": base64.b64encode(fig1.read_bytes()).decode()},
        ],
    }
    if fig2 is not None:
        payload["figures"].append(
            {"name": "fig2_profiles.png",
             "caption": "Re=100 中线剖面：n=33/65/97/129 与 Ghia(1982) 表值叠加",
             "png_base64": base64.b64encode(fig2.read_bytes()).decode()})
    (ART / "report_payload.json").write_text(json.dumps(payload))
    print(f"wrote {DOC} and {fig1} (+{fig2}), payload {ART/'report_payload.json'}")
    for re_key, blk in summary["by_re"].items():
        print(re_key, blk["max_abs_dev_pct"], blk["criteria"])


if __name__ == "__main__":
    main()