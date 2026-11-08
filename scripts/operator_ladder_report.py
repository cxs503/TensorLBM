#!/usr/bin/env python3
"""算子不变量梯（Step F）：把 A/B/C/E 四个解析解算例在四种碰撞算子下的实测写成证据。

背景：`docs/d3q19-d3q27-mrt-consistency-audit.md` 只证明了 MRT 与 BGK **互相一致**，
并明确声明"不外推到物理解精度"。`docs/lbm_verification_ladder_r2.md` §10.6 也把
"MRT/正则化碰撞"列为解析解梯的未覆盖面。本脚本补上这一格：同一条解析解梯子
（ν / Couette 剖面 / Poiseuille 剖面 / 方管 Boussinesq 级数解）对

  bgk (SRT) · mrt (D3Q19 多弛豫) · trt (双弛豫 Λ=3/16) · rlbm (正则化 BGK)

各跑一遍，判据统一为相对误差 < 1%。

输入：直接调用 ``tests/test_lbm_verification.py`` 的 ``*_case(collision=...)``
（与 pytest 断言走同一段代码，因此 published 数字不会与断言漂移）。
输出（``artifacts/operator_ladder_20261108/``）::

  metrics.json          每个 (算子, 算例) 的关键量 / 误差 / 运行时长
  tables.md             报告用 markdown 数字表（含 < 1% / < 3% 判定列）
  fig_operator_ladder.png  各算例误差 vs 算子（对数轴 + 1% 阈值线）
  fig_poiseuille_error.png Poiseuille 剖面误差 u(y)-u_exact(y)：BGK 有斜线性滑移，
                           TRT(Λ=3/16) 压到 0 量级 —— 残差归属=壁面位置误差的直接证据

用法::

    PYTHONPATH=src python scripts/operator_ladder_report.py [--quick]

``--quick`` 把步数降到 1/4（仅用于脚本自检，产出不得作为证据发布）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "operator_ladder_20261108"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import torch  # noqa: E402

torch.set_num_threads(1)  # 小张量：多线程调度开销 > 计算本身（同梯子 pytest fixture）

import test_lbm_verification as TL  # noqa: E402

OPERATORS = TL.COLLISIONS  # ("bgk", "mrt", "trt", "rlbm")
STEPS = ("A_shear_wave", "B_couette", "C_poiseuille", "E_duct_poiseuille")
#: 每个算例报告的关键量（key, 数值解字段, 解析解字段, 单位）
KEY = {
    "A_shear_wave": ("nu", "nu_measured", "nu_expected", "lu2/ts"),
    "B_couette": ("du_max", None, None, "lu"),
    "C_poiseuille": ("u_max", "u_max_num", "u_max_exact", "lu"),
    "E_duct_poiseuille": ("u_max", "u_max_num", "u_max_exact", "lu"),
}


def run(collision: str, step: str, quick: bool = False) -> dict:
    kw = {}
    if quick:
        kw = {"steps": 250}
    t0 = time.time()
    if step == "A_shear_wave":
        r = TL.shear_wave_case(collision=collision, **kw)
    elif step == "B_couette":
        r = TL.couette_case(collision=collision, **kw)
    elif step == "C_poiseuille":
        r = TL.poiseuille_case(collision=collision, **kw)
    elif step == "E_duct_poiseuille":
        r = TL.duct_poiseuille_case(collision=collision, **( {"steps": 500} if quick else {} ))
    else:
        raise ValueError(step)
    r = dict(r)
    r["wall_time_s"] = round(time.time() - t0, 2)
    r["quick"] = quick
    return r


def collect(quick: bool = False) -> dict:
    torch.set_default_dtype(torch.float64)  # 裁决数值方法本身，不吃 float32 舍入地板
    data = {}
    for step in STEPS:
        for op in OPERATORS:
            r = run(op, step, quick=quick)
            data[f"{step}|{op}"] = r
            print(
                f"[{step:20s}] op={op:4s} rel_err={r['rel_err']:.3e} "
                f"({r['rel_err'] * 100:.4f}%)  {r['wall_time_s']:5.1f}s",
                flush=True,
            )
    return data


def _fmt(x: float, nd: int = 8) -> str:
    return f"{x:.{nd}g}"


def tables(data: dict) -> str:
    lines: list[str] = []
    lines.append("### T1 关键量：数值解 vs 解析解（判据：相对误差 < 1%）\n")
    lines.append("| 算例 | 算子 | 关键量 | 数值解 | 解析解 | 相对误差 | < 1% ? | 运行时 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for step in STEPS:
        name, num_k, exa_k, unit = KEY[step]
        for op in OPERATORS:
            r = data[f"{step}|{op}"]
            if step == "B_couette":
                num = r["u_num"]
                exa = r["u_exact"]
                num_s = f"max|u−u_ex|={max(abs(a - b) for a, b in zip(num, exa)):.3e}"
                exa_s = "0（线性剖面）"
            else:
                num_s = f"{_fmt(r[num_k])}"
                exa_s = f"{_fmt(r[exa_k])}"
            ok = "是" if r["rel_err"] < 0.01 else "**否**"
            lines.append(
                f"| {step} | {op} | {name} ({unit}) | {num_s} | {exa_s} | "
                f"{r['rel_err'] * 100:.4f}% | {ok} | {r['wall_time_s']:.1f} s |"
            )
    lines.append("")
    lines.append("### T2 同一算例跨算子横比（相对误差 %）\n")
    lines.append("| 算例 | " + " | ".join(OPERATORS) + " | 最好算子 |")
    lines.append("|---|" + "---|" * (len(OPERATORS) + 1))
    for step in STEPS:
        cells = [f"{data[f'{step}|{op}']['rel_err'] * 100:.4f}" for op in OPERATORS]
        best = min(OPERATORS, key=lambda o: data[f"{step}|{o}"]["rel_err"])
        lines.append(f"| {step} | " + " | ".join(cells) + f" | {best} |")
    lines.append("")
    lines.append("### T3 方管（Step E）积分量 R = gL²/(ν·ū) 对文献 f·Re=56.91（⇔ R=28.455）\n")
    lines.append("| 算子 | R_num | R（级数同批格点平均） | 对级数解误差 | 对文献误差 | < 3% ? |")
    lines.append("|---|---|---|---|---|---|")
    for op in OPERATORS:
        r = data[f"E_duct_poiseuille|{op}"]
        ok = "是" if r["resistance_err_literature"] < 0.03 else "**否**"
        lines.append(
            f"| {op} | {r['resistance_num']:.4f} | {r['resistance_exact_nodal']:.4f} | "
            f"{r['resistance_err_nodal'] * 100:.3f}% | {r['resistance_err_literature'] * 100:.3f}% | {ok} |"
        )
    lines.append("")
    return "\n".join(lines)


def figures(data: dict) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib 不可用，跳过图：{exc!r}")
        return []

    OUT.mkdir(parents=True, exist_ok=True)
    made = []

    # ---- 图 1：误差 vs 算子（对数轴 + 1% 阈值） ----
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    width = 0.8 / len(OPERATORS)
    for k, op in enumerate(OPERATORS):
        xs, ys = [], []
        for i, step in enumerate(STEPS):
            e = data[f"{step}|{op}"]["rel_err"]
            xs.append(i + (k - (len(OPERATORS) - 1) / 2) * width)
            ys.append(max(e * 100, 1e-8))  # 0 误差画在轴上 1e-8 处并单独标注
        ax.bar(xs, ys, width=width, label=op)
        for x, y, step in zip(xs, ys, STEPS):
            e = data[f"{step}|{op}"]["rel_err"] * 100
            ax.annotate(
                f"{e:.2g}" if e > 0 else "0",
                (x, y),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=6.5,
                rotation=90,
            )
    ax.set_yscale("log")
    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([s.split("_")[0] for s in STEPS])
    ax.set_ylabel("relative error vs exact solution  [%]")
    ax.set_title("Step F: analytic-anchor error per collision operator (D3Q19, float64)")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.0)
    ax.text(len(STEPS) - 0.5, 1.35, "1% acceptance gate", color="crimson", ha="right", fontsize=8)
    ax.legend(title="collision", fontsize=8, title_fontsize=8, ncol=4, loc="upper center")
    ax.set_ylim(1e-8, 1e2)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    p = OUT / "fig_operator_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # ---- 图 2：Poiseuille 剖面与其误差（残差归属） ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ref = data["C_poiseuille|bgk"]
    y = ref["y"]
    u_exact = ref["u_exact"]
    ax1.plot(u_exact, y, "k--", lw=1.2, label="exact (parabola)")
    for op in OPERATORS:
        r = data[f"C_poiseuille|{op}"]
        ax1.plot(r["u_num"], r["y"], marker="o", ms=3, lw=1.0, label=op)
    ax1.set_xlabel("u_x  [lu/ts]")
    ax1.set_ylabel("y  [lu]")
    ax1.set_title("Step C Poiseuille profile vs exact")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    for op in OPERATORS:
        r = data[f"C_poiseuille|{op}"]
        err = [a - b for a, b in zip(r["u_num"], r["u_exact"])]
        ax2.plot(err, r["y"], marker="o", ms=3, lw=1.0, label=op)
    ax2.axvline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("u_x - u_exact  [lu/ts]")
    ax2.set_ylabel("y  [lu]")
    ax2.set_title("profile error: BGK/RLBM/MRT slip vs TRT(Lambda=3/16)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    p = OUT / "fig_poiseuille_error.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="步数降到 1/4（仅自检，不作证据）")
    args = ap.parse_args(argv)

    data = collect(quick=args.quick)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    t = tables(data)
    (OUT / "tables.md").write_text(t)
    figs = figures(data)
    print("\n" + t)
    print("figures:", figs)
    print("out:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())