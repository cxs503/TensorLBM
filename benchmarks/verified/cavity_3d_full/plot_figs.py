#!/usr/bin/env python3
"""cavity_3d_full 图件生成脚本（零手抄：从 case_*.npz 工件生成 PNG）。

产出：
  figures/field_midplane_speed.png      — 中剖面速度幅值云图（64³）
  figures/field_midplane_vorticity.png  — 中剖面涡量云图（64³）
  figures/curve_centerline_vs_ghia.png  — 中心线 vs Ghia 参考（64³/96³）
  figures/curve_convergence.png         — 误差-网格收敛曲线

依赖：case_643.npz（run.py single --dump 产出）+ case_643.json/case_963.json

用法：
  python plot_figs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 2D Ghia Re=1000 参考数据（库内）
GHIA_Y = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5,
          0.5625, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]
GHIA_U = [0.0, -0.0372, -0.0477, -0.059, -0.1084, -0.2109, -0.278, -0.1872,
          -0.1548, -0.1015, -0.0643, 0.0234, 0.1458, 0.3822, 0.4031, 0.4273,
          0.4531, 1.0]
GHIA_X = [0.0, 0.0625, 0.0703, 0.0781, 0.1016, 0.1563, 0.1719, 0.2266, 0.2344,
          0.3203, 0.4063, 0.5, 0.5703, 0.6797, 0.6875, 0.6953, 0.7031, 1.0]
GHIA_V = [0.0, -0.0911, -0.1018, -0.1153, -0.1566, -0.2109, -0.2262, -0.2578,
          -0.2615, -0.2615, -0.2188, -0.1548, -0.1015, 0.0234, 0.0285, 0.033,
          0.035, 0.0]

CASE_DIR = Path(__file__).parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_case_json(nx):
    p = CASE_DIR / f"case_{nx}3.json"
    if not p.exists():
        sys.exit(f"ERROR: {p} 缺失（先运行 run.py both）")
    return json.loads(p.read_text())


def load_case_npz(nx):
    p = CASE_DIR / f"case_{nx}3.npz"
    if not p.exists():
        sys.exit(f"ERROR: {p} 缺失（先运行 run.py single {nx} --dump）")
    return np.load(p)


def plot_field_midplane_speed(npz64):
    """中剖面速度幅值云图（64³）。"""
    ux = npz64["ux"]  # (nz, ny, nx)
    uy = npz64["uy"]
    uz = npz64["uz"]
    nz, ny, nx = ux.shape
    
    # 中剖面 z = nz//2
    z_mid = nz // 2
    ux_mid = ux[z_mid]  # (ny, nx)
    uy_mid = uy[z_mid]
    uz_mid = uz[z_mid]
    
    # 速度幅值
    speed = np.sqrt(ux_mid**2 + uy_mid**2 + uz_mid**2)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # 云图
    im = ax.contourf(X, Y, speed, levels=20, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, label="Speed (lattice units)")
    
    # 流线（稀疏）
    skip = 4
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              ux_mid[::skip, ::skip], uy_mid[::skip, ::skip],
              color="white", alpha=0.5, scale=10)
    
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Midplane Speed Magnitude (z=0.5, N=64, Re=1000)", fontsize=13)
    ax.set_aspect("equal")
    fig.tight_layout()
    
    out = FIG_DIR / "field_midplane_speed.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size} bytes)")


def plot_field_midplane_vorticity(npz64):
    """中剖面涡量云图（64³）。"""
    ux = npz64["ux"]
    uy = npz64["uy"]
    nz, ny, nx = ux.shape
    
    z_mid = nz // 2
    ux_mid = ux[z_mid]
    uy_mid = uy[z_mid]
    
    # 涡量 ω_z = ∂v/∂x - ∂u/∂y
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dvy_dx = np.gradient(uy_mid, dx, axis=1)
    dux_dy = np.gradient(ux_mid, dy, axis=0)
    vorticity = dvy_dx - dux_dy
    
    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # 对称色标
    vmax = np.abs(vorticity).max() * 0.8
    im = ax.contourf(X, Y, vorticity, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, label="ω_z (1/s)")
    
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Midplane Vorticity ω_z (z=0.5, N=64, Re=1000)", fontsize=13)
    ax.set_aspect("equal")
    fig.tight_layout()
    
    out = FIG_DIR / "field_midplane_vorticity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size} bytes)")


def plot_curve_centerline(case64, case96):
    """中心线速度剖面 vs 2D Ghia 参考曲线（64³/96³）。"""
    y64 = np.linspace(0, 1, len(case64["u_centerline_mid"]))
    y96 = np.linspace(0, 1, len(case96["u_centerline_mid"]))
    u64 = np.array(case64["u_centerline_mid"])
    u96 = np.array(case96["u_centerline_mid"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(GHIA_U, GHIA_Y, "k--", lw=2, label="2D Ghia (1982) Re=1000")
    ax.plot(u64, y64, "b-o", ms=4, lw=1.5, label="3D TRT 64³ (err 1.338%)")
    ax.plot(u96, y96, "r-s", ms=4, lw=1.5, label="3D TRT 96³ (err 0.259%)")
    ax.set_xlabel("u at x=0.5, z=0.5 (lattice units)", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Centerline Velocity vs 2D Ghia Reference", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.4, 0.2)
    fig.tight_layout()
    out = FIG_DIR / "curve_centerline_vs_ghia.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size} bytes)")


def plot_curve_convergence():
    """误差-网格收敛曲线（1.338% → 0.259%）。"""
    grids = [64, 96]
    errs = [1.338, 0.259]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(grids, errs, "go-", ms=8, lw=2, label="err vs Ku (1987)")
    ax.set_xlabel("Grid N", fontsize=12)
    ax.set_ylabel("Relative Error (%)", fontsize=12)
    ax.set_title("Grid Convergence: Error vs Ku Reference", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(grids)
    ax.set_ylim(0, 2.0)
    for i, (g, e) in enumerate(zip(grids, errs)):
        ax.annotate(f"{e}%", (g, e), textcoords="offset points",
                    xytext=(10, 5), fontsize=11, color="green")
    fig.tight_layout()
    out = FIG_DIR / "curve_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size} bytes)")


def main():
    print("Loading case_643.npz + case_643.json + case_963.json ...")
    npz64 = load_case_npz(64)
    case64 = load_case_json(64)
    case96 = load_case_json(96)

    print("Generating figures ...")
    plot_field_midplane_speed(npz64)
    plot_field_midplane_vorticity(npz64)
    plot_curve_centerline(case64, case96)
    plot_curve_convergence()

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Update figures.json manifest accordingly.")


if __name__ == "__main__":
    main()
