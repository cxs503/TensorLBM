#!/usr/bin/env python3
"""R2 出图：由 artifacts/r2_summary.json 程序化渲染（零手抄）。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def main() -> None:
    d = json.loads((ART / "r2_summary.json").read_text())
    fig_dir = ART / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- fig1: Δp vs 1/R (2D & 3D) + per-radius sigma ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for dim, mk, col in ((2, "o", "C0"), (3, "s", "C3")):
        rows = [r for r in d["droplets"] if r["dim"] == dim]
        inv_r = [1.0 / r["R_eq"] for r in rows]
        y = [r["dp"] / (dim - 1) for r in rows]
        sig = d["fit"][str(dim)]["sigma_mean_of_radii"]
        axes[0].plot(inv_r, y, mk, color=col, label=f"{dim}D 实测 (R_eq)")
        xs = [0.0, max(inv_r) * 1.08]
        axes[0].plot(xs, [sig * x for x in xs], "-", color=col, lw=1,
                     label=f"{dim}D σ={sig:.5f}")
        axes[1].plot(inv_r, [r["sigma"] for r in rows], mk, color=col, label=f"{dim}D σ_i")
        axes[1].axhline(sig, color=col, ls="--", lw=1)
    axes[0].set_xlabel("1/R_eq"); axes[0].set_ylabel("Δp/(dims-1)")
    axes[0].set_title("Young-Laplace: Δp/(d−1) vs 1/R")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("1/R_eq"); axes[1].set_ylabel("σ_i = Δp·R_eq/(d−1)")
    axes[1].set_title("per-radius surface tension"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_laplace_2d_3d.png", dpi=130); plt.close(fig)

    # ---- fig2: flat-slab profiles + EOS pressure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dim in zip(axes, (2, 3)):
        prof = d.get(f"profile_{dim}d")
        if not prof:
            ax.set_visible(False); continue
        n = len(prof)
        x = [i - n / 2.0 for i in range(n)]
        ax.plot(x, prof, lw=1.2, color="C0", label="ρ(n) 横向平均")
        p = [v / 3.0 - 5.0 / 6.0 * (1 - math.exp(-v)) ** 2 for v in prof]
        ax2 = ax.twinx()
        ax2.plot(x, p, lw=1.0, color="C1", ls="--", label="p_EOS(n)")
        ax2.set_ylabel("p_EOS", color="C1")
        fl = d["flat"][str(dim)]
        ax.axhline(fl["rho_l"], color="C0", ls=":", lw=0.8)
        ax.axhline(fl["rho_v"], color="C0", ls=":", lw=0.8)
        ax.set_title(
            f"{dim}D flat slab: p_l−p_v={fl['mech_gap']:.2e}  "
            f"σ_K={fl['sigma_K_tanh']:.5f}  w={fl['width_fit']:.1f}")
        ax.set_xlabel("normal coordinate n"); ax.set_ylabel("ρ")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center")
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_flat_slab.png", dpi=130); plt.close(fig)

    # ---- fig3: cross-dimension + Maxwell ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cd = d.get("cross_dim", {})
    if cd:
        labels = ["2D σ", "3D σ", "3D/2D 比值"]
        vals = [cd["sigma_2d"], cd["sigma_3d"], cd["ratio"]]
        axes[0].bar(labels[:2], vals[:2], color=["C0", "C3"])
        for i, v in enumerate(vals[:2]):
            axes[0].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
        axes[0].axhline(vals[0], color="C0", ls=":", lw=1)
        axes[0].set_ylabel("σ (lattice units)")
        axes[0].set_title(f"跨维 σ：3D/2D = {vals[2]:.4f}（κ3 预言 1.0，偏差 {cd['dev_from_prediction_1']*100:.2f}%）")
        axes[0].grid(alpha=0.3, axis="y")
    mw = d.get("maxwell", {})
    if mw and "flat" in d:
        keys = ["rho_l", "rho_v"]
        x = range(len(keys))
        w = 0.26
        axes[1].bar([i - w for i in x], [mw[k] for k in keys], width=w, label="Maxwell 解析", color="k")
        for j, dim in enumerate(("2", "3")):
            axes[1].bar([i + w * j for i in x], [d["flat"][dim][k] for k in keys], width=w,
                        label=f"{dim}D 离散", color=f"C{3*j}")
        axes[1].set_xticks(list(x)); axes[1].set_xticklabels([r"$\rho_l$", r"$\rho_v$"])
        axes[1].set_yscale("log"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
        axes[1].set_title("共存密度：离散（2D/3D）vs Maxwell 解析")
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_crossdim_maxwell.png", dpi=130); plt.close(fig)
    print("figures →", fig_dir)


if __name__ == "__main__":
    main()
