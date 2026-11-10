#!/usr/bin/env python3
"""Figures for the frazil power-law shear campaign (read from the artifact).

Everything is read back from ``artifacts/frazil_powerlaw_shear/campaign.json`` --
no number in a figure is typed by hand.  Writes PNGs next to the artifact and a
``figures.json`` index (name, caption, path) that the report generator embeds.

Usage::

    python3 benchmarks/frazil_powerlaw_shear/make_figures.py [--artifact ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_ART = os.path.join(ROOT, "artifacts", "frazil_powerlaw_shear", "campaign.json")
KRIEGER_PHI_MAX = 0.60
KRIEGER_ETA = 2.5


def _coarse_tag(art):
    """The production/coarse resolution tag of *this* artifact (scale-agnostic)."""
    tags = {}
    for key in art["flow_curves"]:
        tag = key.split("|")[0]
        tags[tag] = tags.get(tag, 0) + 1
    if not tags:
        return None
    return sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _curves(art, res=None):
    res = res or _coarse_tag(art)
    out = {}
    for key, val in art["flow_curves"].items():
        r, fam, phi = key.split("|")
        if r != res:
            continue
        out[(fam, float(phi))] = val
    return out


def fig_flow_curves(art, outdir):
    cur = _curves(art)
    if not cur:
        return None
    fams = sorted({k[0] for k in cur})
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4.2), sharey=False)
    if len(fams) == 1:
        axes = [axes]
    for ax, fam in zip(axes, fams):
        for (f, phi), val in sorted(cur.items()):
            if f != fam:
                continue
            g = np.array(val["gamma_dots"])
            t = np.array(val["taus"])
            ax.loglog(g, t, "o", ms=5, label=f"c={phi:.2f} (LBM)")
            gg = np.logspace(np.log10(g.min()) - 0.1, np.log10(g.max()) + 0.1, 40)
            ax.loglog(gg, val["K"] * gg ** val["n"], "-", lw=1.2,
                      label=f"fit n={val['n']:.3f}, K={val['K']:.4g}")
        ax.set_xlabel(r"shear rate $\dot\gamma$ [lattice]")
        ax.set_ylabel(r"shear stress $\tau$ [lattice]")
        ax.set_title(f"{fam} family (D3Q19 cell, nu={art['plan']['nu']})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_flow_curves.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_relative_viscosity(art, outdir):
    cur = _curves(art)
    if not cur:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    phis = np.linspace(1e-3, 0.55, 200)
    ein = 1.0 + 2.5 * phis
    kd = (1.0 - phis / KRIEGER_PHI_MAX) ** (-KRIEGER_ETA * KRIEGER_PHI_MAX)
    ax.plot(phis, ein, "--", color="tab:gray", label=r"Einstein $1+2.5c$")
    ax.plot(phis, kd, "-", color="tab:red", label=r"Krieger-Dougherty ($\phi_m=0.6$)")
    for fam, marker, colour in (("dispersed", "o", "tab:blue"),
                               ("flocculated", "s", "tab:green")):
        pts = sorted((phi, v) for (f, phi), v in cur.items() if f == fam)
        if not pts:
            continue
        x = [p[1].get("phi_lattice") or p[0] for p in pts]
        y = [p[1]["eta_r_ref"] for p in pts]
        ax.plot(x, y, marker, color=colour, ms=7, label=f"{fam} (LBM, lowest rate)")
        for xi, yi, (phi_nom, v) in zip(x, y, pts):
            ax.annotate(f"c={phi_nom:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(6, -10), fontsize=7, color=colour)
    ax.set_xlabel("ice volume fraction $c$ (lattice mask)")
    ax.set_ylabel(r"relative viscosity $\eta_{eff}/\mu_0$")
    ax.set_yscale("log")
    ax.set_xlim(0, 0.55)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_relative_viscosity.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_shear_thinning(art, outdir):
    cur = _curves(art)
    fams = sorted({k[0] for k in cur})
    if not fams:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for fam, marker in (("dispersed", "o"), ("flocculated", "s")):
        for (f, phi), val in sorted(cur.items()):
            if f != fam:
                continue
            ax.loglog(val["gamma_dots"], val["eta_r_points"], marker + "-", ms=5,
                      label=f"{fam} c={phi:.2f}")
    ax.set_xlabel(r"shear rate $\dot\gamma$ [lattice]")
    ax.set_ylabel(r"$\eta_{eff}/\mu_0$")
    ax.set_title("structure-frozen vs structure-following suspension")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_shear_thinning.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_exponent(art, outdir):
    cur = _curves(art)
    if not cur:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for fam, marker in (("dispersed", "o"), ("flocculated", "s")):
        pts = sorted((phi, v) for (f, phi), v in cur.items() if f == fam)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1]["n"] for p in pts], marker + "-", ms=7,
                label=f"{fam} family")
    ax.axhline(1.0, color="tab:gray", ls="--", lw=1, label="Newtonian n=1")
    ax.axhline(0.95, color="tab:red", ls=":", lw=1, label="shear-thinning criterion n<0.95")
    ax.set_xlabel("ice volume fraction $c$")
    ax.set_ylabel("power-law index $n$")
    ax.set_ylim(0.5, 1.15)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_powerlaw_index.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_structure(art, outdir):
    """Measured floc size in the cell versus the shear-breakup law."""
    coarse = _coarse_tag(art)
    fields = [c for c in art["per_case"]
              if c.get("family") == "flocculated" and c.get("seed") == "mean"
              and c.get("res") == coarse]
    if not fields:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    anchor = art["plan"]["anchor"]
    g = np.logspace(math.log10(min(art["plan"]["gamma_rates"])) - 0.05,
                    math.log10(max(art["plan"]["gamma_rates"])) + 0.05, 60)
    rr = np.sqrt(anchor["bond_force_lattice"] / (anchor["mu"] * g))
    ax.loglog(g, (rr / anchor["radius"]) ** anchor["df"], "-", color="tab:gray",
              label=r"$N_{eq}=(R_{eq}/a)^{d_f}$, $R_{eq}=\sqrt{F_b/\mu\dot\gamma}$")
    for phi in sorted({c["phi"] for c in fields}):
        pts = [c for c in fields if c["phi"] == phi]
        ax.loglog([p["gamma_dot"] for p in pts], [p["mean_n_primary"] for p in pts],
                  "s", ms=7, label=f"measured, c={phi:.2f}")
    ax.set_xlabel(r"shear rate $\dot\gamma$ [lattice]")
    ax.set_ylabel("primaries per aggregate $N$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "fig5_structure.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


CAPTIONS = {
    "fig1_flow_curves.png":
        "Measured flow curves tau(gamma_dot) from the D3Q19 frozen-structure shear cell "
        "for the dispersed (left) and flocculated (right) families, with the "
        "least-squares power-law fits tau = K gamma_dot^n drawn through the data points.",
    "fig2_relative_viscosity.png":
        "Relative viscosity versus ice volume fraction from the shear cell (lowest rate), "
        "against Einstein 1+2.5c and Krieger-Dougherty (phi_max = 0.6, [eta] = 2.5). "
        "The flocculated points lie above the hard-sphere closures because a fractal "
        "aggregate traps water.",
    "fig3_shear_thinning.png":
        "Relative viscosity versus shear rate: the structure-frozen (dispersed) family is "
        "flat (Newtonian), while the structure-following (flocculated) family falls as the "
        "aggregates break up.",
    "fig4_powerlaw_index.png":
        "Power-law index n versus concentration for both families, with the n = 1 "
        "(Newtonian) and n = 0.95 (shear-thinning criterion) lines.",
    "fig5_structure.png":
        "Primaries per aggregate actually placed in the cell versus the shear-breakup law "
        "that selected them (the law's parameters are anchored, not fitted).",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=DEFAULT_ART)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)
    with open(args.artifact) as fh:
        art = json.load(fh)
    outdir = args.outdir or os.path.dirname(args.artifact)
    figs = []
    for fn in (fig_flow_curves, fig_relative_viscosity, fig_shear_thinning,
               fig_exponent, fig_structure):
        try:
            path = fn(art, outdir)
        except Exception as exc:  # noqa: BLE001 - a missing panel must not hide the rest
            print(f"WARN {fn.__name__}: {exc!r}", file=sys.stderr)
            path = None
        if path:
            figs.append({"name": os.path.basename(path),
                         "path": os.path.relpath(path, ROOT),
                         "caption": CAPTIONS.get(os.path.basename(path), "")})
    index = os.path.join(outdir, "figures.json")
    with open(index, "w") as fh:
        json.dump(figs, fh, indent=1)
    print(json.dumps(figs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())