#!/usr/bin/env python3
"""sphere_re100_mem 图件生成脚本（零手抄：从历史数据工件生成 PNG）。

产出：
  figures/curve_cd_convergence.png      — Cd(t) 收敛史曲线（T1/T2 双档）
  figures/curve_error_tiers.png         — 档位误差曲线（2.7708%/2.5401%）
  figures/field_midplane_speed.png      — 球中剖面速度幅值云图（示意）
  figures/field_wake_structure.png      — 尾流结构云图（示意）

依赖：
  - ../sphere_re100/w8a/t1_d40big_12k.json（T1 档 Cd(t) 历史）
  - ../sphere_re100/w8a/t2_d60big_24k.json（T2 档 Cd(t) 历史）
  - ../sphere_re100/w8a/result.json（判决数字）

用法：
  python plot_figs.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE_DIR = Path(__file__).parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 数据路径（相对位置）
W8A_DIR = CASE_DIR.parent / "sphere_re100" / "w8a"
T1_PATH = W8A_DIR / "t1_d40big_12k.json"
T2_PATH = W8A_DIR / "t2_d60big_24k.json"
RESULT_PATH = W8A_DIR / "result.json"


def load_json(path):
    if not path.exists():
        sys.exit(f"ERROR: {path} 缺失")
    return json.loads(path.read_text())


def plot_cd_convergence(t1_data, t2_data, result_data):
    """Cd(t) 收敛史曲线（T1/T2 双档）。"""
    t1_hist = t1_data["history"]
    t2_hist = t2_data["history"]

    t1_steps = [h["step"] for h in t1_hist]
    t1_cd = [h["cd"] for h in t1_hist]

    t2_steps = [h["step"] for h in t2_hist]
    t2_cd = [h["cd"] for h in t2_hist]

    # 参考值
    cd_ref = result_data["reference"]["cd_ref"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # T1 档
    ax.plot(
        t1_steps, t1_cd, "o-", label="T1 (D=40, 12k steps)", markersize=3, linewidth=1.5, alpha=0.7
    )

    # T2 档
    ax.plot(
        t2_steps, t2_cd, "s-", label="T2 (D=60, 24k steps)", markersize=3, linewidth=1.5, alpha=0.7
    )

    # 参考线
    ax.axhline(
        cd_ref,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Schiller-Naumann ref: {cd_ref:.4f}",
    )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cd (drag coefficient)", fontsize=12)
    ax.set_title("Cd(t) Convergence History — Sphere Re=100 (BFL sparse kernel)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(t1_steps[-1], t2_steps[-1]) * 1.05)

    fig.tight_layout()
    out = FIG_DIR / "curve_cd_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size} bytes)")


def plot_error_tiers(result_data):
    """档位误差曲线（T1/T2 两档）。"""
    runs = result_data["runs"]

    tiers = []
    errors = []
    cd_values = []

    for run in runs:
        tier_name = run["tier"].replace("_d40big_12k", " (D=40)").replace("_d60big_24k", " (D=60)")
        tiers.append(tier_name)
        errors.append(run["err_pct"])
        cd_values.append(run["cd_window"])

    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = np.arange(len(tiers))
    bars = ax.bar(x_pos, errors, color=["#1f77b4", "#ff7f0e"], width=0.6, alpha=0.8)

    # 添加数值标签
    for bar, err, cd in zip(bars, errors, cd_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{err:.4f}%\nCd={cd:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 3% 门限线
    ax.axhline(3.0, color="red", linestyle="--", linewidth=2, label="Gate: 3% threshold")

    ax.set_xlabel("Grid Tier", fontsize=12)
    ax.set_ylabel("Relative Error (%)", fontsize=12)
    ax.set_title("Grid Convergence — Sphere Re=100 (BFL sparse kernel)", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tiers, fontsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(errors) * 1.3)

    # 添加单调收敛标注
    if errors[0] > errors[1]:
        ax.annotate(
            "Monotone decrease ✓",
            xy=(1, errors[1]),
            xytext=(1.3, errors[0] * 0.8),
            fontsize=10,
            color="green",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
        )

    fig.tight_layout()
    out = FIG_DIR / "curve_error_tiers.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size} bytes)")


def plot_field_midplane_speed_schematic():
    """球中剖面速度幅值云图（示意）。

    注：本案未 dump 完整 3D 场数据，此图为物理示意（基于 Re=100 绕球流典型结构）。
    真实场量可视化需重跑 run.py --dump 选项（未来工作）。
    """
    # 创建示意性速度场（基于 Re=100 绕球流物理特征）
    nx, ny = 200, 200
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y)

    # 球体位置（中心）
    R = 1.0
    r = np.sqrt(X**2 + Y**2)

    # 均匀来流 + 球体扰动（简化势流模型）
    u_inf = 1.0
    mask = r < R

    # 势流近似（偶极子）
    with np.errstate(divide="ignore", invalid="ignore"):
        u = u_inf * (1 - R**2 / r**2 * (1 - 2 * X**2 / r**2))
        v = -u_inf * R**2 / r**2 * (2 * X * Y / r**2)

    # 球内设为 0
    u[mask] = 0
    v[mask] = 0

    # 速度幅值
    speed = np.sqrt(u**2 + v**2)
    speed[mask] = np.nan

    fig, ax = plt.subplots(figsize=(8, 7))

    # 云图
    levels = np.linspace(0, 1.5, 30)
    im = ax.contourf(X, Y, speed, levels=levels, cmap="viridis", extend="max")
    cbar = fig.colorbar(im, ax=ax, label="Speed (U/U_inf)")

    # 球体轮廓
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(R * np.cos(theta), R * np.sin(theta), "k-", linewidth=2, label="Sphere")

    # 流线（稀疏）
    skip = 10
    ax.streamplot(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        u[::skip, ::skip],
        v[::skip, ::skip],
        color="white",
        linewidth=0.5,
        density=1.5,
    )

    ax.set_xlabel("x / R", fontsize=12)
    ax.set_ylabel("y / R", fontsize=12)
    ax.set_title("Midplane Speed Magnitude — Sphere Re=100 (schematic)", fontsize=13)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    fig.tight_layout()
    out = FIG_DIR / "field_midplane_speed.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size} bytes)")


def plot_wake_structure_schematic():
    """尾流结构云图（示意）。

    注：本案未 dump 完整 3D 场数据，此图为物理示意（Re=100 轴对称尾流）。
    真实场量可视化需重跑 run.py --dump 选项（未来工作）。
    """
    # 创建示意性尾流结构（x-z 平面，y=0 中剖面）
    nx, nz = 200, 200
    x = np.linspace(-1, 8, nx)  # 下游延伸
    z = np.linspace(-3, 3, nz)
    X, Z = np.meshgrid(x, z)

    # 球体位置（x=0, z=0）
    R = 1.0

    # 尾流衰减模型（简化）
    # 球后尾流速度亏损随下游距离衰减
    x_rel = X - 0  # 相对于球心
    r = np.sqrt(X**2 + Z**2)

    # 球内
    mask_sphere = r < R

    # 尾流区域（x > R）
    mask_wake = (X > R) & (np.abs(Z) < 2.0)

    # 速度亏损（高斯分布）
    sigma_z = 0.5 + 0.3 * (X - R)  # 尾流宽度随下游扩展
    deficit = np.exp(-(Z**2) / (2 * sigma_z**2)) * np.exp(-(X - R) / 3.0)
    deficit[X < R] = 0

    # 速度场
    u = 1.0 - deficit * 0.6  # 尾流速度亏损
    w = np.zeros_like(u)

    # 球内设为 0
    u[mask_sphere] = np.nan
    deficit[mask_sphere] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    # 云图（速度亏损）
    levels = np.linspace(0, 0.6, 25)
    im = ax.contourf(X, Z, deficit, levels=levels, cmap="inferno", extend="max")
    cbar = fig.colorbar(im, ax=ax, label="Wake velocity deficit (ΔU/U_inf)")

    # 球体轮廓
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(R * np.cos(theta), R * np.sin(theta), "k-", linewidth=2, label="Sphere")

    # 流线
    skip = 10
    ax.streamplot(
        X[::skip, ::skip],
        Z[::skip, ::skip],
        u[::skip, ::skip],
        w[::skip, ::skip],
        color="white",
        linewidth=0.5,
        density=1.2,
    )

    ax.set_xlabel("x / R (downstream)", fontsize=12)
    ax.set_ylabel("z / R", fontsize=12)
    ax.set_title("Wake Structure — Sphere Re=100 (schematic, y=0 midplane)", fontsize=13)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-1, 8)
    ax.set_ylim(-3, 3)

    # 标注尾流区域
    ax.annotate(
        "Axisymmetric wake",
        xy=(4, 0),
        xytext=(5, 1.5),
        fontsize=10,
        color="white",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
    )

    fig.tight_layout()
    out = FIG_DIR / "field_wake_structure.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size} bytes)")


def main():
    print("Loading data...")
    t1_data = load_json(T1_PATH)
    t2_data = load_json(T2_PATH)
    result_data = load_json(RESULT_PATH)

    print("Generating figures...")
    plot_cd_convergence(t1_data, t2_data, result_data)
    plot_error_tiers(result_data)
    plot_field_midplane_speed_schematic()
    plot_wake_structure_schematic()

    print("\nAll figures generated successfully!")
    print(f"Output directory: {FIG_DIR}")


if __name__ == "__main__":
    main()
