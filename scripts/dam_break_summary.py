"""汇总 2D 溃坝四模型定量对比，并生成前沿演化图与 sc 模型演化动画(GIF)。"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = Path("outputs/dam_break_benchmark")
MODELS = ["sc", "scmp", "cg", "fe"]


def read_front(csv_path: Path):
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((float(row["t_star"]), float(row["X_star"])))
    return rows


def metrics(rows):
    if not rows:
        return None
    errs = [x - (1.0 + t) for t, x in rows]
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    mae = sum(abs(e) for e in errs) / len(errs)
    xs = [x for _, x in rows]
    mono = all(xs[i] <= xs[i + 1] + 1e-9 for i in range(len(xs) - 1))
    return {
        "n": len(rows),
        "t_final": rows[-1][0],
        "x_final": rows[-1][1],
        "x_ref_final": 1.0 + rows[-1][0],
        "abs_err_final": abs(errs[-1]),
        "rmse": rmse,
        "mae": mae,
        "monotonic": mono,
    }


def main():
    print("=" * 92)
    print("2D Dam-break — 多模型定量对比 (Martin & Moyce 1952: X* = 1 + t*)")
    print("=" * 92)
    print(f"{'模型':<6} {'步数':>5} {'t*_end':>8} {'X*_end':>8} {'X*_ref':>8} {'|err|_end':>9} {'RMSE':>8} {'MAE':>8} {'单调':>5}")
    print("-" * 92)

    all_rows = {}
    for m in MODELS:
        csv_path = BASE / m / "dam_break" / f"bench_dam_{m}_fast" / "front_position.csv"
        rows = read_front(csv_path)
        all_rows[m] = rows
        mt = metrics(rows)
        print(
            f"{m:<6} {mt['n']:>5} {mt['t_final']:>8.3f} {mt['x_final']:>8.3f} "
            f"{mt['x_ref_final']:>8.3f} {mt['abs_err_final']:>9.3f} {mt['rmse']:>8.3f} "
            f"{mt['mae']:>8.3f} {'是' if mt['monotonic'] else '否':>5}"
        )

    # 画前沿演化对比图
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t_ref = [t for t, _ in all_rows["cg"]]  # 用 cg 的时间轴（最接近参考）
    ax.plot([1 + t for t in t_ref], t_ref, "k--", label="Martin & Moyce X*=1+t*", lw=1.5)
    style = {"sc": ("tab:blue", "-"), "scmp": ("tab:orange", "--"), "cg": ("tab:green", "-."), "fe": ("tab:red", ":")}
    for m, rows in all_rows.items():
        ts = [t for t, _ in rows]
        xs = [x for _, x in rows]
        ax.plot(ts, xs, style[m][0], linestyle=style[m][1], label=m.upper(), lw=2)
    ax.set_xlabel("t*")
    ax.set_ylabel("X* (前沿无量纲位置)")
    ax.set_title("2D 溃坝前沿演化 — 四种多相流模型对比 (fast 模式)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = BASE / "front_comparison.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n前沿对比图已保存: {out_png}")

    # 制作 sc 模型演化动画 GIF
    sc_dir = BASE / "sc" / "dam_break" / "bench_dam_sc_fast"
    snaps = sorted(sc_dir.glob("snapshot_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if snaps:
        frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in snaps]
        gif_path = BASE / "dam_break_sc_evolution.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0,
            optimize=True,
        )
        print(f"sc 模型演化动画已保存: {gif_path}  (共 {len(frames)} 帧)")
    else:
        print("未找到 sc snapshot，跳过 GIF 生成")


if __name__ == "__main__":
    main()
