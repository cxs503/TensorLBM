"""Figures for p447w2-N06 (L_cav error attribution).

Reads ``artifacts/p447w2_n06/attribution.json`` plus the saved time-averaged
fields and writes the report figures to ``docs/figures/``.  Every annotation is
taken from the JSON (no hand-typed numbers).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p447w2_n06"
FIG = ROOT / "docs" / "figures"
DPI = 110

CRIT_LABEL = {
    "isobaric": r"isobaric $p<p_v$",
    "alpha_v>=0.5": r"$\alpha_v\geq0.5$",
    "L_eq": r"$L_{eq}=\int\alpha_v dV/(t\,c)$",
}


def _load():
    data = json.loads((ART / "attribution.json").read_text())
    return data


def fig_matrix(data) -> Path:
    runs = list(data["runs"].items())
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.6))
    if len(runs) == 1:
        axes = [axes]
    for ax, (tag, r) in zip(axes, runs):
        cells = [str(c) for c in r["cells"]]
        crits = list(r["matrix"][str(r["cells"][0])].keys())
        m = np.array([[r["matrix"][c][k] for k in crits] for c in cells], dtype=float)
        im = ax.imshow(m, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(crits)))
        ax.set_xticklabels([CRIT_LABEL.get(k, k) for k in crits], rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([f"cpc={c}" for c in cells], fontsize=8)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                e = r["errors"][cells[i]][crits[j]]
                ax.text(j, i, f"{m[i, j]:.3f}\n({100*e:.0f}%)", ha="center", va="center",
                        color="w", fontsize=7)
        ax.set_title(f"{tag}: $L_{{cav}}/c$ vs ref {r['reference']['l_cav_exp']:.2f}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("4 grids x 3 criterion calibers (12-point matrix; % = error vs Brennen 1994)", fontsize=10)
    fig.tight_layout()
    out = FIG / "p447w2_n06_matrix_heatmap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_convergence(data) -> Path:
    runs = list(data["runs"].items())
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 3.6))
    if len(runs) == 1:
        axes = [axes]
    for ax, (tag, r) in zip(axes, runs):
        cells = r["cells"]
        h = [1.0 / c for c in cells]
        for crit in r["matrix"][str(cells[0])].keys():
            ys = [r["matrix"][str(c)][crit] for c in cells]
            ax.plot(h, ys, "o-", label=CRIT_LABEL.get(crit, crit), ms=4, lw=1.2)
            for seq in r["gci"][crit]:
                if seq["valid"] and 0.0 <= seq["f_extrap"] <= 1.2:
                    ax.plot([0.0], [seq["f_extrap"]], marker="*", ms=9,
                            color=ax.lines[-1].get_color())
        ax.axhline(r["reference"]["l_cav_exp"], color="k", ls="--", lw=1.1,
                   label=f"Brennen 1994 ({r['reference']['l_cav_exp']:.2f})")
        ax.set_xlabel("h / c"), ax.set_ylabel(r"$L_{cav}/c$")
        ax.set_title(tag, fontsize=9)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle(r"$L_{cav}$ vs grid spacing (stars = admissible Richardson $h\to0$ extrapolates)",
                 fontsize=10)
    fig.tight_layout()
    out = FIG / "p447w2_n06_convergence.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_fields(data) -> Path:
    tags = list(data["runs"].keys())
    fig, axes = plt.subplots(2, len(tags), figsize=(4.1 * len(tags), 5.4), squeeze=False)
    for col, tag in enumerate(tags):
        r = data["runs"][tag]
        finest = max(r["cells"])
        npz = ART / f"mean_fields_{finest}_sigma{r['sigma']:g}.npz"
        z = np.load(npz)
        p, a, solid, h = z["p_mean"], z["alpha_mean"], z["solid"], float(z["h"])
        solid = solid.astype(bool)
        extent = [0.0, p.shape[0] * h, 0.0, p.shape[1] * h]
        pv = -0.5 * r["sigma"]
        pm = np.ma.masked_where(solid.T, p.T)
        ax = axes[0, col]
        im = ax.imshow(pm, origin="lower", extent=extent, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.contour(np.where(solid.T, np.nan, p.T), levels=[pv], colors="k", linewidths=0.7,
                   extent=extent)
        ax.set_title(f"{tag}: mean pressure (cpc={finest})\ncontour: $p=p_v$", fontsize=8)
        ax.set_xlabel("x/c"), ax.set_ylabel("y/c")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        am = np.ma.masked_where(solid.T, a.T)
        ax = axes[1, col]
        im = ax.imshow(am, origin="lower", extent=extent, cmap="magma",
                       vmin=0.0, vmax=max(1e-6, float(a.max())))
        ax.set_title(f"mean vapour fraction (max={float(a.max()):.2e})", fontsize=8)
        ax.set_xlabel("x/c"), ax.set_ylabel("y/c")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    out = FIG / "p447w2_n06_fields.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_variance(data) -> Path:
    runs = list(data["runs"].items())
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    labels = [t for t, _ in runs]
    keys = ["pct_criterion", "pct_grid", "pct_interaction", "pct_residual"]
    names = ["criterion caliber", "grid", "interaction", "residual (unsteady/unmodelled)"]
    ax = axes[0]
    bottom = np.zeros(len(runs))
    for k, nm in zip(keys, names):
        vals = np.array([100.0 * 0.0 + r["anova"][k] for _, r in runs])
        ax.bar(labels, vals, bottom=bottom, label=nm)
        bottom += vals
    ax.axhline(50.0, color="k", ls=":", lw=1)
    ax.set_ylabel("% of total variance")
    ax.set_title("two-way ANOVA of the relative error\n(criterion x grid, 6 block replicates)", fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    ax = axes[1]
    bottom = np.zeros(len(runs))
    for k, nm in (("pct_criterion", "Delta(criterion caliber)"),
                  ("pct_discretisation", "Delta(discretisation)"),
                  ("pct_residual", "Delta(unmodelled residual)")):
        vals = np.array([r["budget"][k] for _, r in runs])
        ax.bar(labels, vals, bottom=bottom, label=nm)
        bottom += vals
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("% of the total error gap")
    ax.set_title("telescoping error budget vs Brennen 1994", fontsize=9)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    out = FIG / "p447w2_n06_variance.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_sensitivity(data) -> Path:
    runs = list(data["runs"].items())
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    for tag, r in runs:
        g = r["verdicts"]["g_alpha_sensitivity"]["relative_lengths"]
        for c, vals in g.items():
            ax.plot([0.1, 0.5, 0.9], [vals["0.1"], vals["0.5"], vals["0.9"]], "o-", ms=3, lw=1.0,
                    label=f"{tag} cpc={c}")
    ax.set_xlabel(r"threshold level $\tau/\max(\alpha_v)$")
    ax.set_ylabel(r"$L_{cav}/c$")
    ax.set_title("alpha_v criterion: sensitivity to its own threshold", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5, ncol=2)
    ax = axes[1]
    for tag, r in runs:
        gs = r["grid_summary"]
        finest = max(gs, key=lambda d: d["cells_per_chord"])
        t = [s["t"] for s in finest["series"]]
        ax.plot(t, [s["l_isobaric_inst"] for s in finest["series"]], lw=0.8,
                label=f"{tag} isobaric")
        ax.plot(t, [s["alpha_max"] for s in finest["series"]], lw=0.8, ls="--",
                label=f"{tag} max alpha_v")
    ax.set_xlabel("t U/c"), ax.set_ylabel("instantaneous value")
    ax.set_title(f"finest grid (cpc={data['runs'][runs[0][0]]['cells'][-1]}): unsteady signal", fontsize=9)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    out = FIG / "p447w2_n06_sensitivity.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    data = _load()
    made = [fig_matrix(data), fig_convergence(data), fig_fields(data),
            fig_variance(data), fig_sensitivity(data)]
    for p in made:
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())