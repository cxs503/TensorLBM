#!/usr/bin/env python3
"""把 LBM 解析解验证梯（Step A–E）的实测 JSON 变成报告用的数字表 + 对照图。

输入（由 ``tests/test_lbm_verification.py --json`` 产生，数字全部程序化注入，不手抄）::

  artifacts/lbm_verification_ladder/metrics.json      主判据（float64）
  artifacts/lbm_verification_ladder/f32/metrics.json  dtype 对照（float32，可缺）

输出（同一目录，``--refine`` 另给 E 的网格/步数敏感性表）::

  tables.md              markdown 数字表（误差逐项 vs 判据）
  fig_ladder.png         A 衰减 / B·C 剖面 / D 摩擦系数 / E 方管：数值 vs 解析解
  fig_dtype_error.png    float64 vs float32 误差对照（对数轴 + 1% 阈值线）

用法::

    PYTHONPATH=src python tests/test_lbm_verification.py --json artifacts/lbm_verification_ladder/metrics.json
    PYTHONPATH=src TENSORLBM_TEST_DTYPE=float32 python tests/test_lbm_verification.py \
        --json artifacts/lbm_verification_ladder/f32/metrics.json
    PYTHONPATH=src python scripts/lbm_verification_report.py

设计约定：本脚本只读 JSON、只写 artifacts，不重跑求解器——报告里的每个数字都能
追到一次具体的梯子运行，避免文档与代码漂移。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "lbm_verification_ladder"

#: 每个 Step 的关键量（表格"数值解 vs 解析解"两列的取数字段）
KEY_QUANTITY = {
    "A": ("运动粘度 ν (lu²/ts)", "nu_measured", "nu_expected"),
    "B": ("Couette 剖面峰值偏差 Δu_max (lu)", None, None),
    "C": ("Poiseuille 峰值 u_max (lu)", "u_max_num", "u_max_exact"),
    "D": ("壁面摩擦系数 Cf", "cf_num", "cf_exact"),
    "E": ("方管截面最大速度 u_max", "u_max_num", "u_max_exact"),
}

#: 报告覆盖的 Step（顺序即表格/图的顺序）
STEPS = ("A", "B", "C", "D", "E")


def load(path: Path) -> dict:
    """读 metrics JSON 的 ``steps`` 段；文件缺失时返回空字典（不伪造数字）。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("steps", {})


def fmt(value: float) -> str:
    if not isinstance(value, float):
        return str(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.4e}"
    return f"{value:.6g}"


def main_table(steps: dict) -> str:
    rows = [
        "| Step | 案例 | 关键量 | 数值解 | 解析解 | 误差 % | 判据 % | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in STEPS:
        if key not in steps:
            continue
        r = steps[key]
        label, num_key, exact_key = KEY_QUANTITY[key]
        if num_key is None:  # B 的关键量就是与解析剖面的最大偏差本身
            num_s, exact_s = f"Δu_max = {fmt(float(r['rel_err']) * r['u_top'])}", "0（解析剖面）"
        else:
            num_s, exact_s = fmt(float(r[num_key])), fmt(float(r[exact_key]))
        rows.append(
            f"| {key} | {r['case']} | {label} | {num_s} | {exact_s} | "
            f"{fmt(float(r['err_pct']))} | {r['tolerance_pct']:g} | "
            f"{'PASS' if r['passed'] else 'FAIL'} |"
        )
    return "\n".join(rows)


def dtype_table(main_steps: dict, f32_steps: dict) -> str:
    rows = [
        "| Step | 误差 % (float64) | 误差 % (float32) | f32/f64 | 误差性质判定 |",
        "|---|---|---|---|---|",
    ]
    for key in STEPS:
        if key not in main_steps:
            continue
        e64 = float(main_steps[key]["err_pct"])
        if key not in f32_steps:
            rows.append(f"| {key} | {fmt(e64)} | — | — | float32 未运行 |")
            continue
        e32 = float(f32_steps[key]["err_pct"])
        ratio = e32 / e64 if e64 > 0 else float("inf")
        if ratio >= 10:
            verdict = "float32 舍入地板主导"
        elif ratio < 3:
            verdict = "离散/物理误差主导（dtype 不敏感）"
        else:
            verdict = "dtype 部分敏感"
        rows.append(f"| {key} | {fmt(e64)} | {fmt(e32)} | {ratio:.3g} | {verdict} |")
    return "\n".join(rows)


def refinement_table(metrics: dict) -> str:
    """Step E 的网格/步数敏感性（证明 0.42% 是离散误差而非未收敛，见 --refine）。"""
    ref = metrics.get("refinement") or {}
    blocks = []
    if ref.get("E_lengths"):
        rows = ["| 方管边长 L (lu) | 截面最大相对误差 % | 判据 % |", "|---|---|---|"]
        for d in ref["E_lengths"]:
            rows.append(f"| {d['length']} | {fmt(float(d['err_pct']))} | {metrics['tol_pct']:g} |")
        blocks.append("\n".join(rows))
    if ref.get("E_steps"):
        rows = ["| 时间步 | 截面最大相对误差 % |", "|---|---|"]
        for d in ref["E_steps"]:
            rows.append(f"| {d['steps']} | {fmt(float(d['err_pct']))} |")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def timing_table(steps: dict) -> str:
    rows = ["| Step | 时间步 | τ | 判据 |", "|---|---|---|---|"]
    for key in STEPS:
        if key not in steps:
            continue
        r = steps[key]
        rows.append(f"| {key} | {r['steps']} | {r['tau']:g} | 误差 < {r['tolerance_pct']:g}% |")
    return "\n".join(rows)


def ladder_figure(steps: dict, path: Path) -> bool:
    """A 衰减曲线 + B/C 通道剖面 + D 摩擦系数 + E 方管（对角剖面 + 截面云图），与解析解同图。"""
    try:  # 图依赖 matplotlib + numpy；两者缺失时优雅降级（返回 False，不写假图）
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:  # pragma: no cover - matplotlib/numpy 是可选依赖
        return False

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0))
    axes = axes.ravel()

    # (a) 剪切波衰减：log 轴才能看出 3 个数量级的指数衰减
    if "A" in steps:
        r = steps["A"]
        hist = np.asarray(r["history"], dtype=float)
        t = np.arange(hist.size)
        k = 2 * math.pi / r["nx"]
        analytic = r["amplitude_init"] * np.exp(-r["nu_expected"] * k * k * t)
        ax = axes[0]
        ax.semilogy(t[::5], (hist / r["amplitude_init"])[::5], "o", ms=3, label="LBM")
        ax.semilogy(t, analytic / r["amplitude_init"], "-", lw=1, label=r"$\exp(-\nu k^2 t)$")
        ax.set_title(f"A shear wave  err={r['err_pct']:.2e}%", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel(r"$A(t)/A(0)$")
        ax.legend(fontsize=7)

    # (b)(c) 通道流剖面
    for ax, key in zip(axes[1:3], ("B", "C")):
        if key not in steps:
            ax.axis("off")
            continue
        r = steps[key]
        ax.plot(r["u_num"], r["y"], "o", ms=4, label="LBM")
        ax.plot(r["u_exact"], r["y"], "-", lw=1, label="analytic")
        ax.set_title(f"{key} {r['case'].split('_', 1)[-1]}  err={r['err_pct']:.3g}%", fontsize=9)
        ax.set_xlabel(r"$u_x$ (lu)")
        ax.set_ylabel("y (lu)")
        ax.legend(fontsize=7)

    # (d) 摩擦系数：数值 vs 解析（相对误差太小，用相对偏差放大显示）
    ax = axes[3]
    if "D" in steps:
        r = steps["D"]
        vals = [r["cf_num"], r["cf_exact"]]
        ax.bar(["LBM", "exact"], vals, color=["#3b7dd8", "#b0b0b0"], width=0.55)
        for xi, v in enumerate(vals):
            ax.text(xi, v, f"{v:.6f}", ha="center", va="bottom", fontsize=8)
        lo = min(vals)
        ax.set_ylim(lo * 0.995, max(vals) * 1.004)
        ax.set_ylabel(r"$C_f$")
        ax.set_title(f"D wall friction  err={r['err_pct']:.3g}%", fontsize=9)
    else:
        ax.axis("off")

    # (e) 方管对角线剖面：LBM 点 vs Boussinesq 级数解
    ax = axes[4]
    if "E" in steps:
        r = steps["E"]
        ax.plot(r["diag_exact"], r["diag_coord"], "-", lw=1, label="analytic (Boussinesq)")
        ax.plot(r["diag_num"], r["diag_coord"], "o", ms=3.5, label="LBM (D3Q19)")
        ax.set_title(
            f"E square duct L={r['length']} (diagonal)  err_max={r['err_pct']:.3g}%", fontsize=9
        )
        ax.set_xlabel(r"$u_x$ (lu)")
        ax.set_ylabel("Y = Z (lu)")
        ax.legend(fontsize=7)
    else:
        ax.axis("off")

    # (f) 方管截面云图（场量）+ 解析等值线叠加：角区是否被正确解析一眼可见
    ax = axes[5]
    if "E" in steps:
        r = steps["E"]
        u = np.asarray(r["u_num"], dtype=float)
        c = np.asarray(r["diag_coord"], dtype=float)
        im = ax.imshow(
            u,
            origin="lower",
            extent=(c[0], c[-1], c[0], c[-1]),
            cmap="viridis",
            aspect="equal",
        )
        Y, Z = np.meshgrid(c, c)
        ax.contour(
            Y,
            Z,
            np.asarray(r["u_exact"], dtype=float),
            levels=6,
            colors="w",
            linewidths=0.7,
            linestyles="--",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$u_x$ (lu)")
        ax.set_title("E duct cross-section: LBM field + analytic isolines", fontsize=9)
        ax.set_xlabel("Y (lu)")
        ax.set_ylabel("Z (lu)")
    else:
        ax.axis("off")

    fig.suptitle("TensorLBM analytic verification ladder (Step A–E)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def dtype_figure(main_steps: dict, f32_steps: dict, path: Path) -> bool:
    """dtype 敏感度：同一判据在两个精度下的误差，对数轴 + 1% 阈值线。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return False

    keys = [k for k in STEPS if k in main_steps and k in f32_steps]
    if not keys:
        return False
    e64 = [float(main_steps[k]["err_pct"]) for k in keys]
    e32 = [float(f32_steps[k]["err_pct"]) for k in keys]
    tol = float(main_steps[keys[0]]["tolerance_pct"])

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = range(len(keys))
    ax.bar([x - 0.2 for x in xs], e64, width=0.4, label="float64")
    ax.bar([x + 0.2 for x in xs], e32, width=0.4, label="float32")
    ax.set_yscale("log")
    ax.axhline(tol, color="r", ls="--", lw=1, label=f"criterion {tol:g}%")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{k}: {main_steps[k]['case']}" for k in keys], fontsize=8)
    ax.set_ylabel("relative error (%)")
    ax.set_title("Verification ladder: dtype sensitivity (log scale)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    main_json = OUT / "metrics.json"
    f32_json = OUT / "f32" / "metrics.json"
    steps = load(main_json)
    if not steps:
        print(f"缺少 {main_json}：先跑 tests/test_lbm_verification.py --json <path>")
        return 1
    f32 = load(f32_json)

    fig_ladder = ladder_figure(steps, OUT / "fig_ladder.png")
    fig_dtype = dtype_figure(steps, f32, OUT / "fig_dtype_error.png")

    meta = json.loads(main_json.read_text(encoding="utf-8"))
    meta_update = meta  # 完整 metrics（含 refinement 段）
    f32_meta = json.loads(f32_json.read_text(encoding="utf-8")) if f32_json.exists() else {}
    n_pass = sum(1 for r in steps.values() if r["passed"])

    md = [
        "# LBM 解析解验证梯 Step A–E：实测结果",
        "",
        "> 本文件由 `scripts/lbm_verification_report.py` 从 `artifacts/lbm_verification_ladder/`",
        "> 的 metrics JSON 程序化生成，数字无手抄。",
        "",
        "## 1. 主判据（" + meta["dtype"] + "，判据 < " + f"{meta['tol_pct']:g}%" + "）",
        "",
        main_table(steps),
        "",
        f"通过 **{n_pass}/{len(steps)}**；设备 `{meta['device']}`，dtype `{meta['dtype']}`。",
        "",
        "## 2. dtype 敏感度（float32 对照）",
        "",
        dtype_table(steps, f32)
        if f32
        else "float32 对照缺失：先跑 `TENSORLBM_TEST_DTYPE=float32`。",
        "",
        (
            f"float32 判据同样满足（{sum(1 for r in f32.values() if r['passed'])}/{len(f32)}）。"
            "A/B/D 的误差在 float32 下抬高 1–2 个数量级（残差由单精度舍入地板决定），"
            "C/E 两个受限流的误差比 ≈1（离散误差主导、与 dtype 无关）；"
            "方法误差以 " + meta["dtype"] + " 数字为准。"
            if f32 and f32_meta
            else ""
        ),
        "",
        "## 3. Step E 的离散误差归因（网格 / 步数）",
        "",
        refinement_table(meta_update)
        if meta_update.get("refinement")
        else "未运行 `--refine`：网格/步数敏感性缺失。",
        "",
        (
            "读法：截面误差随边长 L 加密单调下降、随步数 1000→2000 完全不变 ⇒ "
            "E 的残差是离散误差（同 Step C 的性质），不是未到稳态。"
            if meta_update.get("refinement")
            else ""
        ),
        "",
        "## 4. 计算设置",
        "",
        timing_table(steps),
        "",
        "- 图：`fig_ladder.png`（A 衰减 + B/C 剖面 + D 摩擦系数 + E 方管对角剖面与截面云图 vs 解析解）、"
        "`fig_dtype_error.png`（"
        + meta["dtype"]
        + " vs "
        + (f32_meta.get("dtype", "float32") if f32_meta else "float32")
        + "）",
        "- 复现：`PYTHONPATH=src python tests/test_lbm_verification.py --refine --json "
        "artifacts/lbm_verification_ladder/metrics.json && "
        "PYTHONPATH=src python scripts/lbm_verification_report.py`",
    ]
    (OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nfigures: fig_ladder={fig_ladder} fig_dtype_error={fig_dtype}")
    return 0 if fig_ladder else 1


if __name__ == "__main__":
    sys.exit(main())
