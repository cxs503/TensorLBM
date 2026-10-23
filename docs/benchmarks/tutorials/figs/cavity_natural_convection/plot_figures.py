#!/usr/bin/env python3
"""cavity_natural_convection 教程图件生成（m219 重算数据，零手抄）。

数据源全部来自本目录 data/（rerun_scan.py 本 mission 自跑产物）：
  data/scan_result.json   三档网格观测量 + Richardson 诊断
  data/result_n{64,128,256}.json  单档完整观测量与 Nu 发展史
  data/fields_n{64,128,256}.npz   稳态 T / |u| / ux / uy 场与中线剖面

输出 5 张图（docs/benchmarks/tutorials/figs/cavity_natural_convection/）：
  fig1_fields.png        温度场 + 速度幅值/流线（n=256 自跑场）
  fig2_profiles.png      中线速度剖面三档对比（u(y)、v(x)）
  fig3_grid_convergence.png  Nu/u_max/v_max 误差收敛 + Richardson 外推
  fig4_nu_wall.png       热/冷壁局部 Nu 分布三档对比
  fig5_nu_history.png    Nu 发展史（三档，稳态触发点标注）

用法：
  python plot_figures.py            # 读 data/，写本目录 *.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 150

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE
NU_REF = 2.243
U_MAX_REF = 16.178
V_MAX_REF = 19.617

GRID_COLORS = {64: "#7f7f7f", 128: "#1f77b4", 256: "#d62728"}


def load() -> tuple[dict, dict[int, dict]]:
    scan = json.loads((DATA / "scan_result.json").read_text(encoding="utf-8"))
    per = {
        g["nx"]: json.loads((DATA / f"result_n{g['nx']}.json").read_text(encoding="utf-8"))
        for g in scan["grids"]
    }
    return scan, per


def fig_fields() -> None:
    """图 1：n=256 稳态场 — 温度等值线填充 + 速度幅值与流线。"""
    z = np.load(DATA / "fields_n256.npz")
    T = z["T"]
    umag = z["umag"]
    ux, uy = z["ux"], z["uy"]
    x = z["x_norm"]
    y = z["y_norm"]
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    c0 = axes[0].contourf(X, Y, T, levels=24, cmap="inferno")
    axes[0].contour(X, Y, T, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors="w", linewidths=0.6)
    axes[0].set_title("(a) 温度场 T（n=256，Ra=1e4，Pr=0.71）", fontsize=11)
    axes[0].set_xlabel("x/H")
    axes[0].set_ylabel("y/H")
    axes[0].set_aspect("equal")
    fig.colorbar(c0, ax=axes[0], shrink=0.85, label="T*")

    c1 = axes[1].contourf(X, Y, umag, levels=24, cmap="viridis")
    skip = 8
    axes[1].streamplot(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        ux[::skip, ::skip],
        uy[::skip, ::skip],
        color="w",
        linewidth=0.5,
        density=1.1,
        arrowsize=0.7,
    )
    axes[1].set_title("(b) 速度幅值 |u|（扩散尺度）+ 流线", fontsize=11)
    axes[1].set_xlabel("x/H")
    axes[1].set_ylabel("y/H")
    axes[1].set_aspect("equal")
    fig.colorbar(c1, ax=axes[1], shrink=0.85, label="|u|·H/α")

    fig.suptitle("de Vahl Davis 方腔自然对流 — 本 mission 自跑稳态场（未做任何平滑）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_fields.png", bbox_inches="tight")
    plt.close(fig)


def fig_profiles(scan: dict) -> None:
    """图 2：中线速度剖面三档对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for g in scan["grids"]:
        nx = g["nx"]
        z = np.load(DATA / f"fields_n{nx}.npz")
        axes[0].plot(z["u_midwidth"], z["y_norm"], color=GRID_COLORS[nx], lw=1.6, label=f"n={nx}")
        axes[1].plot(z["x_norm"], z["v_midheight"], color=GRID_COLORS[nx], lw=1.6, label=f"n={nx}")

    axes[0].axvline(U_MAX_REF, color="k", ls="--", lw=1.0, label=f"DVD u_max={U_MAX_REF}")
    axes[1].axhline(V_MAX_REF, color="k", ls="--", lw=1.0, label=f"DVD v_max={V_MAX_REF}")
    axes[0].set_xlabel("u·H/α（竖直中线 x=H/2）")
    axes[0].set_ylabel("y/H")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("(a) 竖直中线水平速度 u(y)", fontsize=11)
    axes[1].set_xlabel("x/H")
    axes[1].set_ylabel("v·H/α（水平中线 y=H/2）")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title("(b) 水平中线竖直速度 v(x)", fontsize=11)
    fig.suptitle("中线速度剖面：三档网格（本 mission 自跑稳态场）+ de Vahl Davis 1983 峰值参考", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig_grid_convergence(scan: dict) -> None:
    """图 3：误差收敛（对数纵轴）+ Richardson 外推。"""
    nx = np.array([g["nx"] for g in scan["grids"]], dtype=float)
    nu_err = np.array([abs(g["nu_err_pct"]) for g in scan["grids"]])
    u_err = np.array([abs(g["u_max_err_pct"]) for g in scan["grids"]])
    v_err = np.array([abs(g["v_max_err_pct"]) for g in scan["grids"]])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    ax.semilogy(nx, nu_err, "o-", color="#d62728", lw=1.8, ms=7, label="|Nu 误差|")
    ax.semilogy(nx, u_err, "s-", color="#1f77b4", lw=1.6, ms=6, label="|u_max 误差|")
    ax.semilogy(nx, v_err, "^-", color="#2ca02c", lw=1.6, ms=6, label="|v_max 误差|")
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3% 验收门")
    ax.set_xscale("log", base=2)
    ax.set_xticks(nx)
    ax.set_xticklabels([f"{int(n)}" for n in nx])
    ax.set_xlabel("网格 n（H = n−1）")
    ax.set_ylabel("相对误差 [%]")
    ax.set_title("(a) 三观测量误差 vs 网格（全部严格单调下降）", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    nu = np.array([g["nu"] for g in scan["grids"]])
    ax.plot(nx, nu, "o", color="#d62728", ms=8, label="本 mission 自跑")
    ax.axhline(NU_REF, color="k", ls="--", lw=1.2, label=f"DVD 1983 参考 {NU_REF}")
    rich = scan["analysis"]["richardson"]
    if "nu_inf" in rich:
        ax.axhline(
            rich["nu_inf"],
            color="#1f77b4",
            ls=":",
            lw=1.6,
            label=f"Richardson 外推 {rich['nu_inf']:.4f}（p={rich['p_obs']:.2f}）",
        )
    ax.set_xticks(nx)
    ax.set_xticklabels([f"{int(n)}" for n in nx])
    ax.set_xlabel("网格 n")
    ax.set_ylabel("Nu")
    ax.set_title("(b) Nu 绝对值与网格外推（诊断项，非判据）", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_grid_convergence.png", bbox_inches="tight")
    plt.close(fig)


def fig_nu_wall(scan: dict) -> None:
    """图 4：热/冷壁局部 Nu 分布（三档网格叠图）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for g in scan["grids"]:
        nx = g["nx"]
        z = np.load(DATA / f"fields_n{nx}.npz")
        y = z["y_norm"]
        axes[0].plot(y, z["nu_hot_wall"], color=GRID_COLORS[nx], lw=1.4, label=f"n={nx}")
        axes[1].plot(y, z["nu_cold_wall"], color=GRID_COLORS[nx], lw=1.4, label=f"n={nx}")

    for ax, ttl in zip(axes, ["(a) 热壁（x=0，T=1）局部 Nu(y)", "(b) 冷壁（x=H，T=0）局部 Nu(y)"]):
        ax.axhline(NU_REF, color="k", ls="--", lw=1.0, label=f"整壁平均参考 {NU_REF}")
        ax.set_xlabel("y/H")
        ax.set_ylabel("局部 Nu = −(∂T/∂x)·H/ΔT")
        ax.set_title(ttl, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("壁面局部 Nu 分布（grad1 口径，端点含角点）：上下角点为导热主导区、中部为薄边界层高 Nu", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_nu_wall.png", bbox_inches="tight")
    plt.close(fig)


def fig_nu_history(scan: dict) -> None:
    """图 5：Nu 与 u_max 的发展史三档对比（每 1000 步采样）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for g in scan["grids"]:
        nx = g["nx"]
        hist = g["nu_history"]
        steps = np.array([h["step"] for h in hist], dtype=float)
        nu = np.array([h["nu"] for h in hist])
        umax = np.array([h["u_max_nd"] for h in hist])
        axes[0].plot(steps, nu, "-", color=GRID_COLORS[nx], lw=1.3, label=f"n={nx}")
        axes[1].plot(steps, umax, "-", color=GRID_COLORS[nx], lw=1.3, label=f"n={nx}")
        if g["t_stop"]:
            axes[0].axvline(g["t_stop"], color=GRID_COLORS[nx], ls=":", lw=1.0)
            axes[1].axvline(g["t_stop"], color=GRID_COLORS[nx], ls=":", lw=1.0)

    axes[0].axhline(NU_REF, color="k", ls="--", lw=1.0, label=f"DVD {NU_REF}")
    axes[1].axhline(U_MAX_REF, color="k", ls="--", lw=1.0, label=f"DVD {U_MAX_REF}")
    axes[0].set_xlabel("步数")
    axes[0].set_ylabel("Nu")
    axes[0].set_title("(a) Nu 发展史（虚线=稳态触发点）", fontsize=11)
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("步数")
    axes[1].set_ylabel("u_max·H/α")
    axes[1].set_title("(b) u_max 发展史", fontsize=11)
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("三档网格的时间发展（每 1000 步采样，本 mission 自跑日志）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_nu_history.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scan, _ = load()
    fig_fields()
    fig_profiles(scan)
    fig_grid_convergence(scan)
    fig_nu_wall(scan)
    fig_nu_history(scan)
    for p in sorted(OUT.glob("fig*.png")):
        print(f"wrote {p.name} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()