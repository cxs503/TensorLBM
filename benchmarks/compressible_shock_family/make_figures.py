#!/usr/bin/env python3
"""Figures for the W18-N07 compressible-shock verification report.

Reads the JSON/NPZ produced by ``run.py`` and writes PNGs into the same
directory.  Nothing here recomputes physics: every plotted number comes from the
saved run outputs, so a figure cannot disagree with the JSON.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

OUT = os.path.abspath(
    os.environ.get(
        "SHOCK_OUT",
        os.path.join(
            os.path.dirname(__file__), "..", "..", "artifacts", "compressible_shock_family"
        ),
    )
)


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fig_normal_shock():
    data = load("normal_shock.json")
    if not data:
        return None
    recs = [r for r in data["records"] if r["variant"] == "riemann"]
    machs = sorted({r["mach"] for r in recs})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, ax, label in (
        (("rho_ratio",), axes[0], r"$\rho_2/\rho_1$"),
        (("p_ratio",), axes[1], r"$p_2/p_1$"),
    ):
        exact = [recs[[r["mach"] for r in recs].index(m)]["exact"]["rho_ratio"] for m in machs]
        if key[0] == "p_ratio":
            exact = [recs[[r["mach"] for r in recs].index(m)]["exact"]["p_ratio"] for m in machs]
        ax.plot(machs, exact, "k-", lw=2, label="Rankine-Hugoniot (exact)")
        for n, style in zip(sorted({r["n"] for r in recs}), ("o", "s", "^")):
            vals = [r["measured"][key[0]] for r in recs if r["n"] == n]
            ms = [r["mach"] for r in recs if r["n"] == n]
            order = np.argsort(ms)
            ax.plot(
                np.array(ms)[order],
                np.array(vals)[order],
                style,
                ms=5,
                label=f"kinetic solver, n={n}",
            )
        ax.set_xlabel("shock Mach number $M_s$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("Post-shock ratios vs exact R-H (riemann variant)")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_normal_shock_ratios.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)

    # error bars per Mach number and grid
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant, style in (("riemann", "o-"), ("jump", "s--")):
        sub = [r for r in data["records"] if r["variant"] == variant]
        for n, alpha in zip(sorted({r["n"] for r in sub}), (0.45, 0.7, 1.0)):
            ss = sorted([r for r in sub if r["n"] == n], key=lambda r: r["mach"])
            ax.semilogy(
                [r["mach"] for r in ss],
                [100 * r["max_error"] for r in ss],
                style,
                alpha=alpha,
                label=f"{variant}, n={n}",
            )
    ax.axhline(3.0, color="r", ls=":", label="3% acceptance")
    ax.set_xlabel("shock Mach number $M_s$")
    ax.set_ylabel("max relative error [%]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Normal-shock scan: worst-case error per case")
    fig.tight_layout()
    path2 = os.path.join(OUT, "fig_normal_shock_error.png")
    fig.savefig(path2, dpi=160)
    plt.close(fig)
    return [path, path2]


def _field_name(prefix, rec):
    return f"{prefix}_{rec['nx']}x{rec['ny']}.npz"


def fig_dmr():
    data = load("dmr.json")
    if not data:
        return None
    recs = data["records"]
    paths = []
    n = len(recs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), squeeze=False)
    for ax, rec in zip(axes[:, 0], recs):
        f = os.path.join(OUT, _field_name("dmr_field", rec))
        if not os.path.exists(f):
            continue
        d = np.load(f)
        rho = d["rho"]
        X, Y = np.meshgrid(d["xc"], d["yc"])
        levels = np.linspace(1.5, 22.0, 40)
        cs = ax.contour(X, Y, rho, levels=levels, cmap="viridis", linewidths=0.7)
        ax.contourf(X, Y, rho, levels=levels, cmap="viridis", alpha=0.35)
        ax.plot(
            rec["front"]["profile_y"],
            rec["front"]["profile_x"],
            "r.",
            ms=2,
            label="detected leading front",
        )
        tp = rec["triple_point"]
        if tp["triple_point_y"] == tp["triple_point_y"]:
            ax.plot(
                [tp["stem_x_at_contact"]], [tp["triple_point_y"]], "wo", ms=5, label="triple point"
            )
        ax.set_xlim(1.2, 4.0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("y")
        ax.set_title(
            f"density contours, {rec['nx']}x{rec['ny']}, t={rec['t_end']:.2f} "
            f"(stem h={tp['stem_height']:.3f})",
            fontsize=9,
        )
        ax.legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("x")
    fig.tight_layout()
    p1 = os.path.join(OUT, "fig_dmr_density.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # front profiles + stem height convergence
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for rec in recs:
        axes[0].plot(
            rec["front"]["profile_x"], rec["front"]["profile_y"], label=f"{rec['nx']}x{rec['ny']}"
        )
    axes[0].plot(
        [1 / 6 + 11.5475 * recs[0]["t_end"] + y / np.sqrt(3.0) for y in np.linspace(0, 1, 20)],
        np.linspace(0, 1, 20),
        "k--",
        lw=1,
        label="exact incident shock",
    )
    axes[0].set_xlabel("leading front x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Detected leading front vs grid")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    hs = [r["triple_point"]["stem_height"] for r in recs]
    dxs = [r["dx"] for r in recs]
    axes[1].plot(dxs, hs, "o-")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("dx")
    axes[1].set_ylabel("Mach-stem height at t=0.2")
    axes[1].set_title("Stem height (grid convergence)")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p2 = os.path.join(OUT, "fig_dmr_front.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return [p1, p2]


def fig_ffs():
    data = load("ffs.json")
    if not data:
        return None
    recs = data["records"]
    fig, axes = plt.subplots(1, len(recs), figsize=(5.2 * len(recs), 3.6), squeeze=False)
    for ax, rec in zip(axes[0], recs):
        f = os.path.join(OUT, _field_name("ffs_field", rec))
        if not os.path.exists(f):
            continue
        d = np.load(f)
        rho = d["rho"]
        X, Y = np.meshgrid(d["xc"], d["yc"])
        levels = np.linspace(1.3, 6.0, 40)
        ax.contourf(X, Y, np.log10(rho), levels=40, cmap="inferno")
        ax.contour(X, Y, rho, levels=levels, colors="w", linewidths=0.4, alpha=0.6)
        ax.set_xlim(0.3, 2.2)
        ax.set_ylim(0, 1)
        ax.set_title(f"log10(rho), {rec['nx']}x{rec['ny']}, t={rec['t_end']:.1f}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    p1 = os.path.join(OUT, "fig_ffs_density.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # centreline profile through the step face
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for rec in recs:
        f = os.path.join(OUT, _field_name("ffs_field", rec))
        if not os.path.exists(f):
            continue
        d = np.load(f)
        rho = d["rho"]
        y = rec["sample_row_y"]
        j = int(np.argmin(np.abs(d["yc"] - y)))
        ax.plot(d["xc"], rho[j], label=f"{rec['nx']}x{rec['ny']} (y={y:.3f})")
    ax.axvline(0.6, color="k", ls=":", label="step face")
    ax.set_xlim(0.35, 0.65)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$ along the face centreline")
    ax.set_title("Bow-shock stand-off ahead of the step face")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(OUT, "fig_ffs_standoff.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return [p1, p2]


def main():
    made = []
    for fn in (fig_normal_shock, fig_dmr, fig_ffs):
        out = fn()
        if out:
            made.extend(out)
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
