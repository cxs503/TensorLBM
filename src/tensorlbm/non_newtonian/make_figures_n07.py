"""Figures for the P304W2-N07 non-Newtonian LBM verification report.

Reads ``runs/n07/*.npz`` (raw fields) and writes PNGs into
``artifacts/n07/``.  Every figure is regenerated from disk data, so no number
in the report is hand-copied.

    python -m tensorlbm.non_newtonian.make_figures_n07
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .verify_n07 import GHIA_RE100_U  # noqa: E402

U_LID = 0.1


def fig_cavity(runs: Path, out: Path) -> list[str]:
    made = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    styles = {0.5: "-", 1.0: "-", 1.5: "-"}
    for n in (0.5, 1.0, 1.5):
        p = runs / f"cavity_n{n}_N65.npz"
        if not p.exists():
            continue
        d = np.load(p)
        axes[0].plot(d["u_vertical"] / U_LID, d["y"], styles[n], lw=1.6,
                     label=f"LBM power-law n={n} (N=65)")
        axes[1].plot(d["x"], d["v_horizontal"] / U_LID, styles[n], lw=1.6,
                     label=f"n={n}")
    yg = np.array(GHIA_RE100_U["y"])
    ug = np.array(GHIA_RE100_U["u"])
    axes[0].plot(ug, yg, "ko", ms=4, mfc="none", label="Ghia et al. (1982), Re=100")
    axes[0].set_xlabel("u / u_lid")
    axes[0].set_ylabel("y / L")
    axes[0].set_title("Lid-driven cavity, vertical centreline")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("x / L")
    axes[1].set_ylabel("v / u_lid")
    axes[1].set_title("Horizontal centreline (LBM only)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    f = out / "fig1_cavity_centerlines.png"
    fig.savefig(f, dpi=140)
    plt.close(fig)
    made.append(str(f))
    return made


def fig_pipe(runs: Path, out: Path) -> list[str]:
    made = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    panels = [
        ("pipe_n1.0_ny129.npz", "power law n=1.0 (Newtonian)"),
        ("pipe_n1.5_ny129.npz", "power law n=1.5 (shear thickening)"),
        ("pipe_cy_n0.5_ny129.npz", "Carreau-Yasuda n=0.5 (shear thinning)"),
    ]
    for ax, (fn, title) in zip(axes, panels):
        p = runs / fn
        if not p.exists():
            p = runs / fn.replace("ny129", "ny65")
            title += "  [ny=65]"
        if not p.exists():
            ax.set_visible(False)
            continue
        d = np.load(p)
        y, u, ua = d["y"], d["u"], d["ua"]
        l2 = np.linalg.norm(u - ua) / np.linalg.norm(ua)
        ax.plot(u * 1e3, y, "-", lw=1.8, label="LBM")
        ax.plot(ua * 1e3, y, "k--", lw=1.2, label="reference (exact)")
        ax.set_title(f"{title}\nL2 = {l2 * 100:.2f}%", fontsize=9)
        ax.set_xlabel("u [1e-3 lattice]")
        ax.set_ylabel("y [lattice]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    f = out / "fig2_channel_profiles.png"
    fig.savefig(f, dpi=140)
    plt.close(fig)
    made.append(str(f))

    # viscosity / relaxation-time field of the Carreau-Yasuda case
    p = runs / "pipe_cy_n0.5_ny65.npz"
    if p.exists():
        d = np.load(p)
        tau = d["tau"][:, d["tau"].shape[1] // 2]
        y = d["y"]
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot(tau, y, "-o", ms=3)
        ax.set_xlabel(r"local relaxation time $\tau(y)$")
        ax.set_ylabel("y [lattice]")
        ax.set_title("Carreau-Yasuda n=0.5: $\\tau$ field")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        f = out / "fig3_tau_field_cy.png"
        fig.savefig(f, dpi=140)
        plt.close(fig)
        made.append(str(f))
    return made


def fig_convergence(runs: Path, out: Path) -> list[str]:
    made = []
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for p in sorted(runs.glob("result_pipe_*.json")):
        d = json.loads(p.read_text())
        hist = d.get("flow_history") or []
        if hist:
            ax.semilogy(np.arange(1, len(hist) + 1), hist, lw=1.2, label=p.stem)
    if ax.lines:
        ax.axhline(1e-6, color="k", ls=":", lw=1)
        ax.set_xlabel("Picard block")
        ax.set_ylabel(r"$\max|\Delta u| / u_{ref}$")
        ax.set_title("Flow residual history (channel cases)")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        f = out / "fig4_convergence.png"
        fig.savefig(f, dpi=140)
        made.append(str(f))
    plt.close(fig)
    return made


def fig_contraction(runs: Path, out: Path) -> list[str]:
    made = []
    ps = sorted(runs.glob("contraction_*.npz"))
    if not ps:
        return made
    fig, axes = plt.subplots(1, len(ps), figsize=(4.4 * len(ps), 3.4))
    if len(ps) == 1:
        axes = [axes]
    for ax, p in zip(axes, ps):
        d = np.load(p)
        ux = d["ux"]
        ux = np.where(d["wall_mask"], np.nan, ux)
        im = ax.imshow(ux, origin="lower", cmap="RdBu_r", aspect="auto")
        ax.set_title(p.stem, fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    f = out / "fig5_contraction_fields.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    made.append(str(f))
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/n07")
    ap.add_argument("--out", default="artifacts/n07")
    a = ap.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    made += fig_cavity(runs, out)
    made += fig_pipe(runs, out)
    made += fig_convergence(runs, out)
    made += fig_contraction(runs, out)
    print("\n".join(made))
    (out / "figures.json").write_text(json.dumps(made, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())