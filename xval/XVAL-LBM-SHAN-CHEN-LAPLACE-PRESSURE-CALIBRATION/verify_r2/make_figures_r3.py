#!/usr/bin/env python3
"""R3 figures — rendered only from artifacts (no hand-entered numbers).

    python make_figures_r3.py

Inputs : ``artifacts/r3_summary.json`` (+ optional ``artifacts/r3_parts/field_2d.json``)
Outputs: ``artifacts/fig_r3_crossdim.png``, ``artifacts/fig_r3_flat.png``,
         ``artifacts/fig_r3_field.png`` (the field one is skipped while the
         snapshot leg has not been fetched back).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "artifacts" / "r3_summary.json"
FIELD = HERE / "artifacts" / "r3_parts" / "field_2d.json"
OUT = HERE / "artifacts"


def _load() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def fig_crossdim(d: dict) -> Path | None:
    rows = [r for r in d.get("droplets") or []]
    if not rows:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {2: "tab:blue", 3: "tab:red"}
    for dim in (2, 3):
        sub = sorted([r for r in rows if r["dim"] == dim], key=lambda r: r["R_init"])
        if not sub:
            continue
        x = np.array([1.0 / r["R_eq"] for r in sub])
        y = np.array([r["dp"] / (dim - 1) for r in sub])
        fit = (d.get("fit") or {}).get(str(dim)) or {}
        sigma = fit.get("sigma_through_origin")
        ax[0].plot(x, y, "o", color=colors[dim], label=f"{dim}D  Δp/{dim - 1} (R={[r['R_init'] for r in sub]})")
        if sigma:
            xs = np.linspace(0, x.max() * 1.1, 10)
            ax[0].plot(xs, sigma * xs, "-", color=colors[dim], lw=1.2,
                       label=f"{dim}D fit σ={sigma:.5f}")
        for r in sub:
            ax[0].annotate(f"R_eq={r['R_eq']:.2f}", (1.0 / r["R_eq"], r["dp"] / (dim - 1)),
                           textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax[0].set_xlabel("1 / R_eq  [1/Δx]")
    ax[0].set_ylabel(r"$\Delta p/(d-1)$  →  $\sigma$ slope")
    ax[0].set_title("Young–Laplace: σ from the slope (2D & 3D)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    cd = d.get("cross_dim") or {}
    if cd:
        s2, s3 = cd["sigma_2d"], cd["sigma_3d"]
        ax[1].bar([0, 1], [s2, s3], color=[colors[2], colors[3]], width=0.5)
        ax[1].axhline(s2, color="k", ls=":", lw=1, label="σ_2D reference")
        for frac, lab in ((0.03, "±3%"), (0.15, "±15%")):
            ax[1].axhspan(s2 * (1 - frac), s2 * (1 + frac), color="gray", alpha=0.15, label=lab)
        ax[1].set_xticks([0, 1], ["σ_2D", "σ_3D"])
        ax[1].set_ylabel("σ (mean of radii)")
        ax[1].set_title(f"cross-dim ratio σ_3D/σ_2D = {cd['ratio']:.4f}\n"
                        f"(κ3 prediction 1.0; dev {100 * cd['dev_from_prediction_1']:.2f}%)")
        ax[1].grid(alpha=0.3, axis="y")
        ax[1].legend(fontsize=7)
    fig.tight_layout()
    out = OUT / "fig_r3_crossdim.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_flat(d: dict) -> Path | None:
    flat = d.get("flat") or {}
    mw = d.get("maxwell") or {}
    have = [dim for dim in ("2", "3") if flat.get(dim)]
    if not have:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    labels, rl, rv = [], [], []
    for dim in have:
        labels.append(f"{dim}D slab")
        rl.append(flat[dim]["rho_l"])
        rv.append(flat[dim]["rho_v"])
    x = np.arange(len(labels))
    ax[0].bar(x - 0.18, rl, 0.34, label="ρ_l (lattice)")
    ax[0].bar(x + 0.18, rv, 0.34, label="ρ_v (lattice)")
    if mw:
        ax[0].axhline(mw["rho_l"], color="C0", ls="--", lw=1.2, label="Maxwell ρ_l")
        ax[0].axhline(mw["rho_v"], color="C1", ls="--", lw=1.2, label="Maxwell ρ_v")
    ax[0].set_xticks(x, labels)
    ax[0].set_ylabel("density")
    ax[0].set_title("Planar film: coexistence vs Maxwell construction")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, axis="y")

    gaps = [flat[dim]["mech_gap"] for dim in have]
    sk = [flat[dim]["sigma_K_tanh"] for dim in have]
    sl = []
    for dim in have:
        fit = (d.get("fit") or {}).get(dim) or {}
        sl.append(fit.get("sigma_mean_of_radii", np.nan))
    ax[1].bar(x - 0.18, gaps, 0.34, label="mechanical gap |p_l-p_v|/p_v")
    ax2 = ax[1].twinx()
    ax2.bar(x + 0.18, sk, 0.34, color="C3", label="σ_K (tanh fit)")
    ax2.plot(x, sl, "kx", ms=8, label="σ_Laplace (droplets)")
    ax[1].set_xticks(x, labels)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("mech gap (log)")
    ax2.set_ylabel("σ")
    ax[1].set_title("Mechanical balance & Korteweg anchor vs Laplace σ")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=7)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = OUT / "fig_r3_flat.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_field() -> Path | None:
    if not FIELD.exists():
        return None
    d = json.loads(FIELD.read_text(encoding="utf-8"))
    rho = np.asarray(d["rho"], dtype=float)
    L = rho.shape[0]
    m = d.get("measure") or {}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    im = ax[0].imshow(rho, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax[0], label="ρ (lattice units)")
    ax[0].set_title(f"2D SCMP droplet ρ field (L={L}, R_init={d['meta']['R_init']}, "
                    f"steps={d['meta']['steps']})")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    c = L // 2
    r = np.arange(L) - c
    ax[1].plot(r, rho[c, :], "o-", ms=2, label="ρ along y=center")
    if m.get("rho_in") is not None:
        ax[1].axhline(m["rho_in"], color="r", ls="--", lw=1, label=f"ρ_in={m['rho_in']:.4f}")
        ax[1].axhline(m["rho_out"], color="b", ls="--", lw=1, label=f"ρ_out={m['rho_out']:.4f}")
    ax[1].set_xlim(-L / 2, L / 2)
    ax[1].set_xlabel("r [Δx]")
    ax[1].set_ylabel("ρ")
    ax[1].set_title(f"radial cut: Δp·R_eq = σ = {m.get('dp', float('nan')) * m.get('R_eq', float('nan')):.5f}"
                    if m else "radial cut")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_r3_field.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    d = _load()
    made = [p for p in (fig_crossdim(d), fig_flat(d), fig_field()) if p]
    for p in made:
        print("wrote", p)
    if not made:
        print("no artifacts yet — nothing to plot")


if __name__ == "__main__":
    main()