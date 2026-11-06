#!/usr/bin/env python3
"""汇总 LBM 解析解验证梯（Step A–D）的实测结果 → 表格 + 图。

输入（由 ``tests/test_lbm_verification.py`` 产生，数值全部程序化注入，不手抄）:

  artifacts/lbm_verification_ladder/metrics.json        float64 主判据
  artifacts/lbm_verification_ladder/f32/metrics.json    float32 对照（dtype 敏感度）

输出:

  artifacts/lbm_verification_ladder/tables.md           报告用数字表
  artifacts/lbm_verification_ladder/fig_ladder.png      剖面/衰减 vs 解析解
  artifacts/lbm_verification_ladder/fig_dtype_error.png float64 vs float32 误差

用法::

    PYTHONPATH=src python scripts/lbm_verification_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

OUT = ROOT / "artifacts" / "lbm_verification_ladder"

#: 各 Step 的主判据量（报告里"关键量 vs 基准"逐项列出）
KEY_QUANTITY = {
    "A": ("运动粘度 ν (lu²/ts)", "nu_num", "nu_exact"),
    "B": ("Couette 剖面误差 max Δu/u_top", "err_pct", None),
    "C": ("Poiseuille 峰值 u_max (lu)", "u_max_num", "u_max_exact"),
    "D": ("壁面摩擦系数 Cf", "cf_num", "cf_exact"),
}


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("steps", {})


def fmt(value) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e5:
            return f"{value:.4e}"
        return f"{value:.6g}"
    return str(value)


def main_table(main: dict) -> str:
    rows = [
        "| Step | 案例 | 关键量 | 数值解 | 解析解 | 误差 % | 判据 % | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in ("A", "B", "C", "D"):
        if key not in main:
            continue
        r = main[key]
        label, num_key, exact_key = KEY_QUANTITY[key]
        if exact_key is None:  # B 的关键量就是剖面误差本身
            num_s = f"max err = {fmt(r['err_pct'])} %"
            exact_s = "0（解析剖面）"
        else:
            num_s = fmt(r[num_key])
            exact_s = fmt(r[exact_key])
        rows.append(
            f"| {key} | {r['case']} | {label} | {num_s} | {exact_s} | "
            f"{fmt(r['err_pct'])} | {r['tolerance_pct']:g} | "
            f"{'PASS' if r['passed'] else 'FAIL'} |"
        )
    return "\n".join(rows)


def dtype_table(main: dict, f32: dict) -> str:
    rows = [
        "| Step | 误差 % (float64) | 误差 % (float32) | float32/float64 倍数 | 结论 |",
        "|---|---|---|---|---|",
    ]
    for key in ("A", "B", "C", "D"):
        if key not in main:
            continue
        e64 = main[key]["err_pct"]
        if key not in f32:
            rows.append(f"| {key} | {fmt(e64)} | — | — | float32 未跑 |")
            continue
        e32 = f32[key]["err_pct"]
        ratio = e32 / e64 if e64 > 0 else float("inf")
        verdict = (
            "float32 舍入地板主导" if e32 > 100 * e64 else "物理/离散误差主导（dtype 不敏感）"
        )
        rows.append(f"| {key} | {fmt(e64)} | {fmt(e32)} | {ratio:.3g} | {verdict} |")
    return "\n".join(rows)


def timing_table(main: dict) -> str:
    rows = ["| Step | 时间步 | 判据 |", "|---|---|---|"]
    for key in ("A", "B", "C", "D"):
        if key in main:
            rows.append(f"| {key} | {main[key]['steps']} | 误差 < {main[key]['tolerance_pct']:g}% |")
    return "\n".join(rows)


def dtype_figure(main: dict, f32: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return False

    keys = [k for k in ("A", "B", "C", "D") if k in main and k in f32]
    if not keys:
        return False
    e64 = [main[k]["err_pct"] for k in keys]
    e32 = [f32[k]["err_pct"] for k in keys]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs = range(len(keys))
    ax.bar([x - 0.2 for x in xs], e64, width=0.4, label="float64")
    ax.bar([x + 0.2 for x in xs], e32, width=0.4, label="float32")
    ax.set_yscale("log")
    ax.axhline(1.0, color="r", ls="--", lw=1, label="阈值 1%")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{k}: {main[k]['case']}" for k in keys], fontsize=8)
    ax.set_ylabel("相对误差 %")
    ax.set_title("解析解验证梯：dtype 敏感度", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main() -> int:
    main_res = load(OUT / "metrics.json")
    f32_res = load(OUT / "f32" / "metrics.json")
    if not main_res:
        print(f"missing {OUT / 'metrics.json'} — 先跑 tests/test_lbm_verification.py")
        return 1

    import test_lbm_verification as ladder

    fig_ok = ladder.render_figure(main_res, OUT / "fig_ladder.png")
    dtype_ok = dtype_figure(main_res, f32_res, OUT / "fig_dtype_error.png")

    md = [
        "# LBM 解析解验证梯 Step A–D：实测结果",
        "",
        "> 本文件由 `scripts/lbm_verification_report.py` 程序化生成，数字直接取自",
        "> `artifacts/lbm_verification_ladder/metrics*.json`，无手抄。",
        "",
        "## 1. 主判据（float64）",
        "",
        main_table(main_res),
        "",
        f"通过 {sum(1 for r in main_res.values() if r['passed'])}/{len(main_res)}，"
        f"阈值 {ladder.TOL_PCT:g}%。",
        "",
        "## 2. dtype 敏感度（float32 对照）",
        "",
        dtype_table(main_res, f32_res),
        "",
        "## 3. 计算设置",
        "",
        timing_table(main_res),
        "",
        f"- 设备：`{json.loads((OUT / 'metrics.json').read_text())['device']}`；"
        f"dtype：`{json.loads((OUT / 'metrics.json').read_text())['dtype']}`",
        "- 图：`fig_ladder.png`（A 衰减曲线 + B/C 剖面 vs 解析解），"
        "`fig_dtype_error.png`（float64 vs float32）",
    ]
    (OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nfigures: ladder={fig_ok} dtype={dtype_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())