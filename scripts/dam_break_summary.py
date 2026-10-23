"""汇总 2D 溃坝四模型定量对比：对 Martin & Moyce (1952) 14 点实验数字化表。

参考表从 benchmarks/pending/dam_break_sc/result.json 键
reference_digitisation_lethe 读取（Lethe 官方后处理脚本同源）。
旧版本以线性趋势 X*=1+t* 充当 "Martin & Moyce" 参考并据此算 RMSE——
那不是实验数据（实验曲线非线性，T=2.96 处 X=3.67）。2026-10-23 修正，
详见 benchmarks/pending/dam_break_2d_multimodel/REPORT.md §4。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = Path("outputs/dam_break_benchmark")
REF_PATH = Path("benchmarks/pending/dam_break_sc/result.json")
MODELS = ["sc", "scmp", "cg", "fe"]


def load_reference() -> list[tuple[float, float]]:
    ref = json.loads(REF_PATH.read_text())["reference_digitisation_lethe"]
    return list(zip(ref["T"], ref["Z"]))


def read_front(csv_path: Path) -> list[tuple[float, float]]:
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((float(row["t_star"]), float(row["X_star"])))
    return rows


def interp(rows: list[tuple[float, float]], t: float) -> float | None:
    if t > rows[-1][0]:
        return None
    for i in range(len(rows) - 1):
        if rows[i][0] <= t <= rows[i + 1][0]:
            w = (t - rows[i][0]) / (rows[i + 1][0] - rows[i][0])
            return rows[i][1] * (1 - w) + rows[i + 1][1] * w
    return rows[-1][1]


def metrics(rows: list[tuple[float, float]], mm: list[tuple[float, float]]) -> dict:
    pts = [(t, interp(rows, t), x) for t, x in mm]
    pts = [(t, xi, xr) for t, xi, xr in pts if xi is not None]
    errs = [xi - xr for _, xi, xr in pts]
    xs = [x for _, x in rows]
    return {
        "n": len(pts),
        "t_final": rows[-1][0],
        "x_final": rows[-1][1],
        "x_ref_final": mm[-1][1],
        "abs_err_final": abs(errs[-1]) if errs else float("nan"),
        "rmse": math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else float("nan"),
        "mae": sum(abs(e) for e in errs) / len(errs) if errs else float("nan"),
        "monotonic": all(xs[i] <= xs[i + 1] + 1e-9 for i in range(len(xs) - 1)),
    }


def main():
    mm = load_reference()
    print("=" * 96)
    print("2D Dam-break — 多模型定量对比 vs Martin & Moyce 1952 实验 14 点数字化表")
    print(f"参考: {REF_PATH} reference_digitisation_lethe")
    print("=" * 96)
    print(
        f"{'模型':<6} {'点数':>4} {'t*_end':>8} {'X*_end':>8} {'X*_exp@end':>10} "
        f"{'|err|_end':>9} {'RMSE':>8} {'MAE':>8} {'单调':>5}"
    )
    print("-" * 96)

    all_rows = {}
    for m in MODELS:
        csv_path = BASE / m / "dam_break" / f"bench_dam_{m}_fast" / "front_position.csv"
        rows = read_front(csv_path)
        all_rows[m] = rows
        mt = metrics(rows, mm)
        print(
            f"{m:<6} {mt['n']:>4} {mt['t_final']:>8.3f} {mt['x_final']:>8.3f} "
            f"{mt['x_ref_final']:>10.3f} {mt['abs_err_final']:>9.3f} {mt['rmse']:>8.3f} "
            f"{mt['mae']:>8.3f} {'是' if mt['monotonic'] else '否':>5}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [t for t, _ in mm],
        [x for _, x in mm],
        "ko",
        ms=6,
        mfc="white",
        mew=1.8,
        label="Martin & Moyce 1952 实验",
        zorder=5,
    )
    style = {
        "sc": ("tab:blue", "-"),
        "scmp": ("tab:orange", "--"),
        "cg": ("tab:green", "-."),
        "fe": ("tab:red", ":"),
    }
    for m, rows in all_rows.items():
        ax.plot(
            [t for t, _ in rows],
            [x for _, x in rows],
            style[m][0],
            linestyle=style[m][1],
            label=m.upper(),
            lw=2,
        )
    ax.set_xlabel("t*")
    ax.set_ylabel("X*（前沿无量纲位置）")
    ax.set_title("2D 溃坝前沿演化 — 四模型 vs Martin & Moyce 1952 实验")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = BASE / "front_comparison.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n前沿对比图已保存: {out_png}")

    sc_dir = BASE / "sc" / "dam_break" / "bench_dam_sc_fast"
    snaps = sorted(sc_dir.glob("snapshot_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if snaps:
        frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in snaps]
        gif_path = BASE / "dam_break_sc_evolution.gif"
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True
        )
        print(f"sc 模型演化动画已保存: {gif_path}  (共 {len(frames)} 帧)")
    else:
        print("未找到 sc snapshot，跳过 GIF 生成")


if __name__ == "__main__":
    main()
