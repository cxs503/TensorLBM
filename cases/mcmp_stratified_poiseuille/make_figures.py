"""Figures for the MCMP stratified-Poiseuille reparation report.

Everything is drawn from the JSON records (``artifacts/mcmp_stratified_poiseuille``)
— no number is typed into this file.

Usage::

    PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/make_figures.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORDS = os.path.join(ROOT, "artifacts", "mcmp_stratified_poiseuille", "records")
ANALYSIS = os.path.join(ROOT, "artifacts", "mcmp_stratified_poiseuille", "analysis.json")
OUT = os.path.join(ROOT, "artifacts", "mcmp_stratified_poiseuille", "figures")


def load(tag: str) -> dict:
    return json.load(open(os.path.join(RECORDS, f"{tag}.json")))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ana = json.load(open(ANALYSIS))
    by_tag = {row["tag"]: row for row in ana["flow"]}

    # ---- fig 1: velocity profiles, ny = 33 ---------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for tag, label, style in (
        ("cp_ny33_self_shift", "u_eq=self, force=shift (A0)", "-"),
        ("cp_ny33_viscous_guo", "u_eq=viscous, force=guo (repair)", "-"),
    ):
        rec = load(tag)
        ny = rec["config"]["ny"]
        ax.plot(rec["u_profile_mix"], [j - 0.5 for j in range(ny)], style,
                label=label + " (sim)")
    ny = 33
    for wall_tag, rule, ls in (("halfway", "component", "--"), ("cellcentre", "component", ":")):
        ref = by_tag["cp_ny33_self_shift"]["references"][f"{wall_tag}/{rule}"]
        ax.plot(ref["profile"], [j - 0.5 for j in range(ny)], ls,
                color="k", alpha=0.6,
                label=f"reference {wall_tag}/{rule}")
    ax.set_xlabel("$u_{mix}$")
    ax.set_ylabel("y (wall plane at 0.5 and ny-1.5)")
    ax.set_title("Stratified Poiseuille, ny=33, $G_{12}=-2.5$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_profiles_ny33.png"), dpi=140)
    plt.close(fig)

    # ---- fig 2: phase field, mixing vs stratification ----------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for tag, label in (("tp_ny25_g2.5_shift", "$G_{12}=-2.5$ (stratified)"),
                       ("tp_ny25_g1_shift", "$G_{12}=-1.0$ (mixed)")):
        rec = load(tag)
        ax.plot([j - 0.5 for j in range(rec["config"]["ny"])], rec["alpha_profile"],
                "o-", ms=3, label=f"{label}, $\\phi$={rec['phi_contrast']:.2f}")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xlabel("y")
    ax.set_ylabel(r"$\alpha=(\rho_1-\rho_l)/(\rho_h-\rho_l)$")
    ax.set_title("Phase indicator: the audit's $G_{12}=-1.0$ set does not stratify")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_phase_field.png"), dpi=140)
    plt.close(fig)

    # ---- fig 3: wall convention calibration -------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for key, label, marker in (("H_ny_minus_1", "H = ny-1 (cell-centre walls)", "s"),
                               ("H_ny_minus_2", "H = ny-2 (half-way walls)", "o")):
        for tau, ls in ((1.0, "-"), (0.75, "--")):
            xs, ys = [], []
            for w in ana["wall_calibration"]:
                if abs(w["tau"] - tau) < 1e-9:
                    xs.append(w["ny"])
                    ys.append(100 * w[key]["u_max_error_rel"])
            ax.plot(xs, ys, ls, marker=marker, ms=4, label=f"{label}, tau={tau}")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("ny")
    ax.set_ylabel("u_max error vs single-phase Poiseuille [%]")
    ax.set_title("Which wall convention does this code path use?")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_wall_calibration.png"), dpi=140)
    plt.close(fig)

    # ---- fig 4: flow-rate error by configuration --------------------------
    rows = [r for r in ana["flow"]
            if r.get("finite") and r["ny"] == 25 and abs(r["G_12"] + 2.5) < 1e-9]
    rows.sort(key=lambda r: (r["force_scheme"], r["u_eq"]))
    labels = [f"{r['u_eq']}\n{r['force_scheme']}" for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    width = 0.2
    for i, (key, lbl) in enumerate((("halfway/component", "halfway walls, component nu"),
                                    ("halfway/mixture", "halfway walls, mixture nu"),
                                    ("cellcentre/component", "cell-centre walls, component nu"),
                                    ("cellcentre/mixture", "cell-centre walls, mixture nu"))):
        ax.bar([x + (i - 1.5) * width for x in range(len(rows))],
               [100 * r["references"][key]["err_Q_mix"] for r in rows],
               width, label=lbl)
    ax.axhline(10, color="r", lw=0.8, ls="--")
    ax.axhline(-10, color="r", lw=0.8, ls="--")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$(Q_{mix}-Q_{ref})/Q_{ref}$ [%]")
    ax.set_title("Two-phase flow-rate error, ny=25, $G_{12}=-2.5$ (diverged configs omitted)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_error_matrix.png"), dpi=140)
    plt.close(fig)

    # ---- fig 5: Young-Laplace (mcmp_verify_bench results, if present) -----
    bench = os.path.join(ROOT, "artifacts", "mcmp_verify_bench", "results.json")
    if os.path.exists(bench):
        data = json.load(open(bench))
        cases = data.get("young_laplace", {}).get("cases") or []
        if cases:
            fig, ax = plt.subplots(figsize=(5.4, 4.0))
            levels = sorted({c["G_12"] for c in cases})
            for lv in levels:
                cs = [c for c in cases if c["G_12"] == lv]
                xs = [2.0 / c["R"] for c in cs]
                ys = [c["dp"] for c in cs]
                ax.plot(xs, ys, "o", label=f"$G_{{12}}$={lv}")
                sigma = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
                xx = [0, max(xs) * 1.05]
                ax.plot(xx, [sigma * v for v in xx], "-", lw=0.8)
            ax.set_xlabel("2/R")
            ax.set_ylabel(r"$\Delta p$")
            ax.set_title("Young-Laplace (mcmp_verify_bench)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "fig5_laplace.png"), dpi=140)
            plt.close(fig)

    print("figures written to", OUT)
    for f in sorted(glob.glob(os.path.join(OUT, "*.png"))):
        print(" ", os.path.basename(f), os.path.getsize(f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())