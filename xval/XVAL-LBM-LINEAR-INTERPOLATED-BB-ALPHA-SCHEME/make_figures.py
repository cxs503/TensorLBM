"""Figures for XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME from artifacts/*/results.json."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def load(name):
    return json.loads((ART / name / "results.json").read_text())["cases"]


def fig_accuracy(cases):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {"halfway": ("s", "#888888"), "bfl": ("o", "#1f77b4"),
              "bfl_par": ("^", "#d62728"), "cli": ("v", "#2ca02c")}
    carriers = ["bgk", "trt_d2", "trt_c"]
    for k, carrier in enumerate(carriers):
        for closure, (mk, col) in styles.items():
            pts = sorted((c["delta"], 100 * c["l2_rel"]) for c in cases
                         if c["carrier"] == carrier and c["closure"] == closure)
            if not pts:
                continue
            d, e = zip(*pts)
            ax.plot(d, e, marker=mk, ls="--" if k else "-", color=col, ms=6,
                    alpha=0.85, label=f"{carrier} / {closure}")
    ax.axhline(15.0, color="k", lw=1.2, ls=":")
    ax.text(0.26, 17.0, "acceptance 15% (owner)", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel(r"sub-grid wall distance $\delta$ (cells)")
    ax.set_ylabel(r"$L_2$ profile error vs exact Poiseuille  (%)")
    ax.set_title("Interpolated BB alpha-scheme: accuracy vs carrier\n(N=16, nu=0.1, F/2 shift applied)", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(HERE / "xval_alpha_accuracy.png", dpi=130)
    return HERE / "xval_alpha_accuracy.png"


def fig_lambda(cases):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for delta, col in zip((0.25, 0.5, 0.75), ("#d62728", "#1f77b4", "#2ca02c")):
        pts = sorted((c["lambda_product"], c["w_bot"]) for c in cases
                     if abs(c["delta"] - delta) < 1e-12)
        if not pts:
            continue
        lam, w = zip(*pts)
        ax.plot(lam, w, "o-", color=col, ms=5, label=rf"$\delta$={delta} (target $w$={delta})")
        ax.axhline(delta, color=col, lw=1, ls=":")
        ax.axvline(0.5 * delta**2, color=col, lw=1, ls="-.", alpha=0.7)
        ax.axvline(3 / 16, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel(r"TRT product $\Lambda^+\Lambda^-$")
    ax.set_ylabel(r"effective wall distance $w_{bot}$ (cells)")
    ax.set_title(r"Wall placement vs TRT magic product (bfl_par closure, $K$-correction on)" "\n"
                 r"dash-dot: OSS rule $\delta^2/2$; dotted: classical $3/16$", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(HERE / "xval_alpha_lambda_scan.png", dpi=130)
    return HERE / "xval_alpha_lambda_scan.png"


def fig_profile(cases):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    pick = [("bgk", "bfl_par", "BGK + bfl_par", "#d62728"),
            ("trt_d2", "bfl_par", r"TRT($\Lambda^+\Lambda^-=\delta^2/2$) + bfl_par", "#1f77b4"),
            ("bgk", "halfway", "BGK + halfway", "#888888")]
    d = 0.25
    for carrier, closure, label, col in pick:
        cand = [c for c in cases if c["carrier"] == carrier and c["closure"] == closure
                and abs(c["delta"] - d) < 1e-12]
        if not cand:
            continue
        c = cand[0]
        ax.plot(c["u"], c["y"], "o-", color=col, ms=4, lw=1, label=label)
        ax.plot(c["u_ref"], c["y"], "k--", lw=1.6, label="exact Poiseuille" if col != "#1f77b4" else None)
    ax.set_xlabel(r"$u_x$ (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title(rf"Steady profile, $\delta$={d}, N=16 (walls at y=0 and y=H)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(HERE / "xval_alpha_profile.png", dpi=130)
    return HERE / "xval_alpha_profile.png"


def main():
    main_cases = load("main")
    figs = [fig_accuracy(main_cases), fig_profile(main_cases)]
    try:
        figs.append(fig_lambda(load("lambda_scan")))
    except FileNotFoundError:
        pass
    out = {}
    for f in figs:
        out[f.name] = base64.b64encode(f.read_bytes()).decode()
    (ART / "figures_b64.json").write_text(json.dumps(out))
    print("figures:", [f.name for f in figs])


if __name__ == "__main__":
    main()
