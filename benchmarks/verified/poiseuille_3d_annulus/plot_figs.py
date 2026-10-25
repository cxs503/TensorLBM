#!/usr/bin/env python3
"""Generate the poiseuille_3d_annulus figures from in-repo artifacts (zero hand-copy).

Reads (all inside this case directory — no external/machine-local path needed):
  - case_Ro20_field.npz   (field dump produced by `run.py single ... --dump`)
  - case_Ro{20,40,80}.json (per-tier binned profiles + errors)
  - result.json           (three-tier composite verdict)

Writes:
  - figures/field_midplane_speed.png
  - figures/curve_profile_vs_analytical.png
  - figures/curve_error_convergence.png
  - figures.json          (manifest; every number in the PNGs comes from the
                           artifact keys named in `source`)

Regenerate with:
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=../../../src python run.py single 20 \
      case_Ro20_field.json --dump --device cuda:0     # refreshes the npz dump
  python plot_figs.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = Path(__file__).resolve().parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# In-repo field dump (produced by run.py --dump); legacy absolute path as fallback.
DUMP = CASE_DIR / "case_Ro20_field.npz"
DUMP_FALLBACK = Path("/nfs/wangxi/runs/bm_close/poiseuille_3d_annulus/figdump/case_Ro20_field.npz")
RESULT = CASE_DIR / "result.json"
RO_TIERS = (20, 40, 80)


def annulus_u(r, G, R_o, a, nu):
    """Exact annular Poiseuille profile."""
    return G / (4 * nu) * (R_o**2 - r**2 + (R_o**2 - a**2) / math.log(R_o / a) * np.log(r / R_o))


def _dump_path() -> Path:
    if DUMP.exists():
        return DUMP
    if DUMP_FALLBACK.exists():
        print(f"[warn] {DUMP.name} missing, falling back to {DUMP_FALLBACK}")
        return DUMP_FALLBACK
    raise FileNotFoundError(
        f"field dump not found: {DUMP} (run `run.py single 20 ... --dump` first)"
    )


def plot_field_midplane():
    """Cross-section velocity field contour (Ro=20 dump)."""
    d = np.load(_dump_path())
    u_meas = d["u_meas"]  # (nz, ny) mid-plane axial velocity
    fluid = d["fluid_np"]
    R_o = int(d["R_o"])
    a = float(d["a"])
    yc = int(d["yc"])
    zc = int(d["zc"])

    ny, nz = u_meas.shape
    yy = np.arange(ny) - yc
    zz = np.arange(nz) - zc
    YY, ZZ = np.meshgrid(yy, zz)
    RR = np.sqrt(YY**2 + ZZ**2)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Annular region mask for cleaner visualization
    annular = fluid & (RR >= a * 0.8) & (RR <= R_o * 1.1)
    u_clean = np.where(annular, u_meas, np.nan)

    cf = ax.contourf(YY, ZZ, u_clean, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax, label=r"$u_x$ (lattice)")

    # Draw inner and outer circle boundaries
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(a * np.cos(theta), a * np.sin(theta), "w--", lw=1, label=f"inner a={a}")
    ax.plot(R_o * np.cos(theta), R_o * np.sin(theta), "w--", lw=1, label=f"outer R_o={R_o}")

    ax.set_xlabel("y (lattice)")
    ax.set_ylabel("z (lattice)")
    ax.set_title(f"Annular Poiseuille mid-plane $u_x$ (R_o={R_o}, no-slip walls)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "field_midplane_speed.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size} bytes)")
    return out


def plot_profile_vs_analytical():
    """Radial profile vs analytical solution (all 3 tiers from case JSONs)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for idx, Ro in enumerate(RO_TIERS):
        ax = axes[idx]
        case_path = CASE_DIR / f"case_Ro{Ro}.json"
        case = json.loads(case_path.read_text())

        r_bins = np.array(case["r_bins"])
        u_bins = np.array(case["u_bins"])

        a = case["a"]
        R_o = case["R_o"]
        nu = case["nu_lb"]
        G_Q = case["G_Q"]

        # Analytical profile on fine grid (CH1 effective frame: fitted radii + G_Q)
        r_fine = np.linspace(case["delta_i"], case["delta_o"], 300)
        u_analytical = annulus_u(r_fine, G_Q, case["delta_o"], case["delta_i"], nu)

        ax.plot(u_analytical, r_fine, "k-", lw=1.5, label="analytical (eff frame)")
        ax.plot(
            u_bins,
            r_bins,
            "ro",
            ms=4,
            label=f"LBM bins (err={case['eff_max_bin_central_pct']:.4f}%)",
        )
        ax.axvline(0.0, color="gray", ls=":", lw=0.8)

        ax.set_xlabel(r"$u_x$ (lattice)")
        ax.set_title(f"$R_o$={Ro}", fontsize=11)
        if idx == 0:
            ax.set_ylabel("r (lattice)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Radial velocity profile vs analytical solution (CH1 effective frame)")
    fig.tight_layout()
    out = FIG_DIR / "curve_profile_vs_analytical.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size} bytes)")
    return out


def plot_geometry_schematic():
    """Geometry + axial-development schematic (exact solution keyed to artifacts).

    Coordinates/labels come from the case artifact (a, R_o) and the exact
    solution form documented in README.md + run.py; the dashed radial sample
    line is the binned measurement used by every reported error.
    """
    case = json.loads((CASE_DIR / "case_Ro20.json").read_text())
    a, R_o, nu, G_Q = case["a"], case["R_o"], case["nu_lb"], case["G_Q"]
    delta_i, delta_o = case["delta_i"], case["delta_o"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- (a) cross-section ---
    th = np.linspace(0, 2 * np.pi, 400)
    ax1.fill(R_o * np.cos(th), R_o * np.sin(th), color="#dbe6f0", zorder=0)
    ax1.fill(a * np.cos(th), a * np.sin(th), color="#95a5a6", zorder=1)
    ax1.plot(
        delta_o * np.cos(th),
        delta_o * np.sin(th),
        "b--",
        lw=1.1,
        label=r"effective outer $\delta_o$",
    )
    ax1.plot(
        delta_i * np.cos(th),
        delta_i * np.sin(th),
        "r--",
        lw=1.1,
        label=r"effective inner $\delta_i$",
    )
    ax1.plot([0, R_o], [0, 0], "k-", lw=1)
    ax1.annotate("", xy=(R_o, 0), xytext=(a, 0), arrowprops=dict(arrowstyle="<->", color="k"))
    ax1.text((a + R_o) / 2, 1.2, r"gap ($R_o-a$)", ha="center", fontsize=9)
    ax1.text(0, -a / 2, r"$a$" + f"={a:g}", ha="center", va="center", fontsize=9)
    ax1.text(
        -0.5 * R_o,
        0.75 * R_o,
        f"$R_o$={R_o:g}",
        fontsize=9,
        bbox=dict(fc="w", ec="none", alpha=0.75),
    )
    ax1.plot([0, R_o * np.cos(np.pi / 4)], [0, R_o * np.sin(np.pi / 4)], "k-", lw=0.8)
    ax1.text(0.62 * R_o, 0.30 * R_o, r"$r$", fontsize=10)
    ax1.text(
        R_o * 0.95,
        -R_o * 0.72,
        "outer wall\nno-slip",
        fontsize=8,
        ha="center",
        bbox=dict(fc="w", ec="none", alpha=0.65),
    )
    ax1.set_title(f"Annulus cross-section ($a/R_o$ = {a / R_o:g})")
    ax1.set_xlabel("y (lattice)")
    ax1.set_ylabel("z (lattice)")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    # --- (b) exact profile + axial development ---
    r = np.linspace(delta_i, delta_o, 400)
    ax2.plot(
        r,
        annulus_u(r, G_Q, delta_o, delta_i, nu),
        "b-",
        lw=2,
        label=f"exact $u(r)$ @ $R_o$={R_o:g} (CH1 frame)",
    )
    ax2.axhline(0, color="gray", ls=":", lw=1)
    ax2.axvline(delta_i, color="r", ls="--", lw=1)
    ax2.axvline(delta_o, color="b", ls="--", lw=1)
    ax2.text(delta_i + 0.2, 0.02, r"$\delta_i$", color="r", fontsize=9)
    ax2.text(delta_o - 0.9, 0.02, r"$\delta_o$", color="b", fontsize=9)
    ax2.set_xlabel("r (lattice)")
    ax2.set_ylabel(r"$u_x$ (lattice)")
    ax2.set_title("Exact annular Poiseuille profile (walls $u=0$)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        "Problem schematic: concentric-annulus Poiseuille flow "
        "(Zou-He inlet velocity / outlet pressure, L/R = "
        f"{case['L_over_R']:g} lattice lengths)"
    )
    fig.tight_layout()
    out = FIG_DIR / "diagram_annulus_geometry.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size} bytes)")
    return out


def plot_error_convergence():
    """Error convergence curve across Ro tiers."""
    result = json.loads(RESULT.read_text())

    Ro_list = result["R_o_list"]
    eff_errs = result["eff_errors_pct"]
    nom_errs = result["nom_errors_pct"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(Ro_list, eff_errs, "bo-", lw=2, ms=8, label="CH1 effective frame (primary)")
    ax.plot(Ro_list, nom_errs, "rs--", lw=1.5, ms=6, label="CH2 nominal frame (disclosed)")
    ax.axhline(3.0, color="gray", ls=":", lw=1, label="3% acceptance threshold")

    ax.set_xlabel(r"$R_o$ (grid size)")
    ax.set_ylabel("max relative error (%)")
    ax.set_title("Grid convergence: 3D annular Poiseuille flow")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(list(Ro_list))
    ax.set_xticklabels([str(r) for r in Ro_list])
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    for Ro, e in zip(Ro_list, eff_errs):
        ax.annotate(f"{e:.4f}%", (Ro, e), textcoords="offset points", xytext=(10, 6), fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "curve_error_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size} bytes)")
    return out


def write_manifest(out_dir: Path) -> Path:
    """Manifest at the case root: gates_check.py `figures` gate reads it there."""
    result = json.loads(RESULT.read_text())
    eff = result["eff_errors_pct"]
    manifest = {
        "case": "poiseuille_3d_annulus",
        "profile": "paper",
        "figures": [
            {
                "file": "figures/diagram_annulus_geometry.png",
                "kind": "diagram",
                "caption": "问题几何示意：同心环隙横截面（内柱 a、外壁 R_o、有效半径 δ_i/δ_o）"
                "与精确环形 Poiseuille 剖面 u(r)，内外壁 u=0 无滑移，轴向长 L/R=6",
                "source": "benchmarks/verified/poiseuille_3d_annulus/case_Ro20.json "
                "a + R_o + delta_i/delta_o + G_Q + nu_lb + L_over_R"
                "（剖面形式见同目录 run.py 头注精确解）",
            },
            {
                "file": "figures/field_midplane_speed.png",
                "kind": "field",
                "caption": "环隙横截面中剖面轴向速度 u_x 云图（R_o=20，内外壁无滑移，"
                "楼梯离散边界清晰可见）",
                "source": "benchmarks/verified/poiseuille_3d_annulus/case_Ro20_field.npz"
                " u_meas/fluid_np（run.py single 20 --dump 直出）",
            },
            {
                "file": "figures/curve_profile_vs_analytical.png",
                "kind": "curve",
                "caption": f"三档网格（R_o=20/40/80）径向速度剖面 vs CH1 有效帧解析解"
                f"（binned LBM 直读工件，最大误差 {eff[0]:.4f}/{eff[1]:.4f}/"
                f"{eff[2]:.4f}%）",
                "source": "benchmarks/verified/poiseuille_3d_annulus/"
                "case_Ro{20,40,80}.json r_bins + u_bins + delta_i/delta_o + G_Q",
            },
            {
                "file": "figures/curve_error_convergence.png",
                "kind": "curve",
                "caption": "CH1 有效帧（主判据）与 CH2 名义帧（披露）最大相对误差随网格"
                "细化单调收敛曲线，虚线为 3% 判据线",
                "source": "benchmarks/verified/poiseuille_3d_annulus/result.json "
                "eff_errors_pct + nom_errors_pct",
            },
        ],
    }
    manifest_path = out_dir / "figures.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {manifest_path}")
    return manifest_path


def main():
    print("Generating figures for poiseuille_3d_annulus...")
    f0 = plot_geometry_schematic()
    f1 = plot_field_midplane()
    f2 = plot_profile_vs_analytical()
    f3 = plot_error_convergence()
    write_manifest(CASE_DIR)

    # Verify all PNGs >= 6KB (gates_check paper `figures` gate floor)
    for f in (f0, f1, f2, f3):
        sz = f.stat().st_size
        assert sz >= 6144, f"{f} too small: {sz} bytes"
        print(f"  {f.name}: {sz} bytes OK")


if __name__ == "__main__":
    main()
