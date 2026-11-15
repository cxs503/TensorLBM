"""Figures for the Peskin tensor-product stencil + MDF cross-validation (R2)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 9})


def physics() -> dict:
    out = {}
    for path in sorted(ART.glob("physics_*.json")):
        d = json.loads(path.read_text())
        out[d["tag"]] = d
    return out


def fig_fields(data: dict) -> None:
    npzs = sorted(ART.glob("field_*.npz"))
    if not npzs:
        return
    fig, axes = plt.subplots(1, len(npzs), figsize=(3.4 * len(npzs), 3.2))
    axes = np.atleast_1d(axes)
    for ax, path in zip(axes, npzs):
        z = np.load(path)
        ux, uz, D = z["ux"], z["uz"], float(z["diameter"])
        ny, nx = ux.shape
        omega = np.gradient(ux, axis=1) - np.gradient(uz, axis=0)
        lim = np.percentile(np.abs(ux), 99)
        im = ax.imshow(
            omega.T, origin="lower", cmap="RdBu_r", extent=[0, ny, 0, nx],
            vmin=-np.percentile(np.abs(omega), 99), vmax=np.percentile(np.abs(omega), 99),
        )
        th = np.linspace(0, 2 * math.pi, 200)
        c = z["center"]
        ax.plot(c[1] + D / 2 * np.cos(th), c[0] + D / 2 * np.sin(th), "k-", lw=0.8)
        ax.set_title(path.stem.replace("field_", ""), fontsize=7)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\omega_y$")
    fig.suptitle("mid-plane spanwise vorticity, steady sphere wake (periodic box)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_wake_vorticity.png")
    plt.close(fig)


def fig_convergence(data: dict) -> None:
    if not data:
        return
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for tag, d in data.items():
        h = d["history"]
        steps = [r["step"] for r in h]
        ax[0].plot(steps, [r["cd_drive"] for r in h], lw=1.0, label=f"{tag} drive")
        ax[0].plot(steps, [r["cd_marker"] for r in h], lw=0.8, ls="--", label=f"{tag} marker")
        ax[1].plot(steps, [r["closure_rel"] for r in h], lw=0.9, label=tag)
    ax[0].set_xlabel("LBM step")
    ax[0].set_ylabel(r"$C_d$")
    ax[0].set_title("drag, momentum-balance vs marker reaction")
    ax[0].legend(fontsize=5)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(0.0, color="k", lw=0.6)
    ax[1].axhspan(-0.15, 0.15, color="g", alpha=0.1, label="+-15% band")
    ax[1].set_xlabel("LBM step")
    ax[1].set_ylabel(r"$(C_{d,marker}-C_{d,drive})/C_{d,drive}$")
    ax[1].set_title("steady-state closure gate")
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_convergence_closure.png")
    plt.close(fig)


def fig_extrapolation(summary: dict) -> None:
    rows = [r for r in summary["rows"] if r["n_iter"] == 5]
    if not rows:
        return
    x = np.array([(math.pi / 6) ** (1 / 3) * r["d_over_h"] for r in rows])
    y = np.array([r["cd_drive"] for r in rows])
    e = np.array([r["cd_drive_std"] for r in rows])
    order = np.argsort(x)
    x, y, e = x[order], y[order], e[order]
    ex = summary["extrapolation"].get("all", {})
    re_mean = ex.get("reynolds_mean", rows[0]["reynolds_measured"])
    sn = ex.get("schiller_naumann", rows[0]["cd_schiller_naumann"])
    cw = ex.get("clift", sn)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(x, y, yerr=e, fmt="o-", color="C3", label="LBM (momentum balance)")
    if ex:
        xx = np.linspace(0, x.max() * 1.05, 50)
        ax.plot(xx, ex["cd_inf"] + ex["slope_b"] * xx, "C3--", lw=1.0,
                label="linear fit in $\\phi^{1/3}$")
        ax.errorbar([0], [ex["cd_inf"]], yerr=[ex["cd_inf_sigma"] if np.isfinite(ex["cd_inf_sigma"]) else 0],
                    fmt="s", color="C3", label=f"extrapolated $C_d^\\infty$={ex['cd_inf']:.3f}")
    ax.axhline(sn, color="k", lw=1.2, label=f"Schiller-Naumann {sn:.3f}")
    ax.axhline(cw, color="C0", lw=1.0, ls=":", label=f"Clift {cw:.3f}")
    ax.axhspan(sn * 0.85, sn * 1.15, color="g", alpha=0.12, label="+-15% acceptance")
    for xi, yi, r in zip(x, y, [r for r in sorted(rows, key=lambda r: r["d_over_h"])]):
        ax.annotate(f"D/H=1/{1 / r['d_over_h']:.0f}\nRe={r['reynolds_measured']:.0f}",
                    (xi, yi), textcoords="offset points", xytext=(4, 6), fontsize=6)
    ax.set_xlabel(r"$(\pi/6)^{1/3}\,D/H$   ($\\phi^{1/3}$)")
    ax.set_ylabel(r"$C_d$")
    ax.set_title("sphere drag vs box size, Re~%.0f" % re_mean)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_box_extrapolation.png")
    plt.close(fig)


def fig_mdf() -> None:
    path = ART / "operator_r2.json"
    if not path.exists():
        return
    op = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for n_iter, series in op["mdf"]["series"].items():
        res = series["residual_series"]
        ax.semilogy(range(1, len(res) + 1), res, "o-", ms=3, lw=0.9, label=f"n_iter={n_iter}")
    cases = {c["kernel"]: c for c in op["cases"]}
    ax.set_xlabel("multi-direct-forcing iteration")
    ax.set_ylabel(r"$\max|\mathbf{u}_t-\mathbf{u}_m|/\|\mathbf{u}_m\|_\infty$")
    ax.set_title("MDF slip residual (sphere stencil, float64)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_mdf_residual.png")
    plt.close(fig)


def main() -> None:
    data = physics()
    fig_fields(data)
    fig_convergence(data)
    summary_path = ART / "xval_summary_r2.json"
    if summary_path.exists():
        fig_extrapolation(json.loads(summary_path.read_text()))
    fig_mdf()
    for p in sorted(FIG.glob("*.png")):
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()