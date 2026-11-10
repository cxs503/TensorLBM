#!/usr/bin/env python3
"""Figures for the stand-off resolvability study.

Every number drawn comes from a JSON/NPZ record produced by
``scripts/run_resolvability_study.py`` (grid runs, ladder collect, Mach scan) --
nothing is hard-coded, so a figure can never disagree with the report tables.

    python scripts/make_resolvability_figures.py \
        --ladder runs/resolvability/ma3_ladder.json \
        --machscan runs/resolvability/mach_scan.json \
        --budget runs/resolvability/mach_scan.json \
        --fields runs/resolvability/fields_ma3_G4.npz \
        --outdir figures/resolvability
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

STATE_COLOR = {"PASS": "#1b7f3a", "UNRESOLVED": "#d98200", "WITHHELD": "#8b1a1a"}


def load(path):
    with open(path) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ figures
def fig_convergence(summary, outdir):
    """Stand-off vs cell size for both QoIs + the Richardson limit."""
    grids = summary["grids"]
    rows = [(g["dx"], g["standoff"], g.get("standoff_mass"), g["label"],
             (g.get("verdict") or {}).get("state"))
            for g in grids.values() if g.get("standoff_mass")]
    if not rows:
        return None
    rows.sort(reverse=True)
    h = np.array([r[0] for r in rows])
    iso = np.array([r[1] for r in rows])
    mass = np.array([r[2] for r in rows])
    d_ext = summary.get("standoff_extrapolated")

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.loglog(h, mass, "o-", color="#20478b", label="$\\delta_{mass}$ (QoI, mass deficit)")
    ax.loglog(h, iso, "s--", color="#7a7a7a", mfc="none", ms=7,
              label="$\\delta_{iso}$ (0.5-jump crossing, diagnostic)")
    for (hh, dd, mm, lab, st) in rows:
        ax.plot(hh, mm, "o", ms=10, mfc=STATE_COLOR.get(st, "grey"), mec="k", mew=0.6, zorder=5)
    if d_ext:
        ax.axhline(d_ext, color="#1b7f3a", ls="--", lw=1.2,
                   label=f"Richardson limit {d_ext:.5f}")
        ax.fill_between([h.min() * 0.8, h.max() * 1.3], d_ext * 0.95, d_ext * 1.05,
                        color="#1b7f3a", alpha=0.12, label="$\\pm$5 % band")
    a2 = summary.get("A2") or {}
    ps = ", ".join(f"{p:.2f}" for p in (a2.get("p") or []) if p is not None)
    note = f"observed p = [{ps}]" if ps else "no convergent triplet"
    if a2.get("gci_21") is not None:
        note += f",  GCI$_{{21}}$ = {a2['gci_21'] * 100:.2f} %"
    ax.set_xlabel("cell size $h=\\Delta x=\\Delta y$ [-]")
    ax.set_ylabel("stand-off $\\delta$ [-]")
    t_end = float(summary.get("t_end", 2.0))
    ax.set_title(f"grid convergence at Ma={summary['mach']:.1f}, t={t_end:.1f}\n{note}",
                 fontsize=10)
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    out = os.path.join(outdir, f"convergence_ma{summary['mach']:.0f}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_richardson(summary, outdir):
    """Observed order per triplet + GCI, i.e. the A2 evidence on one axis."""
    trio = summary.get("richardson") or {}
    triplets = [t for t in trio.get("triplets", []) if t.get("ok")]
    if not triplets:
        return None
    labels = [t["label"].replace("_", " ") for t in triplets]
    ps = [t["p"] for t in triplets]
    gcis = [t.get("gci_fine", float("nan")) for t in triplets]
    x = np.arange(len(triplets))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax.bar(x, ps, 0.5, color="#20478b")
    ax.axhspan(1.5, 2.5, color="#1b7f3a", alpha=0.15, label="A2 band [1.5, 2.5]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("observed order $p$")
    ax.set_title("observed order per triplet", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax2.bar(x, gcis, 0.5, color="#d98200")
    ax2.axhline(0.05, color="#8b1a1a", ls="--", lw=1.0, label="A2 limit 5 %")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("GCI (fine pair) [-]")
    ax2.set_title("grid-convergence index", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(outdir, "richardson_diagnostics.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_resolvability_map(summaries, outdir):
    """Relative error vs resolvability ratio R, coloured by the three-state verdict."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    seen = set()
    for s in summaries:
        for g in s["grids"].values():
            err = g.get("rel_err_vs_extrapolated")
            R = g.get("R")
            st = (g.get("verdict") or {}).get("state", "WITHHELD")
            if err is None or R is None or not np.isfinite(err) or not np.isfinite(R):
                continue
            lab = st if st not in seen else None
            ax.plot(R, max(err, 1e-6), "o", ms=8, mfc=STATE_COLOR.get(st, "grey"),
                    mec="k", mew=0.6, label=lab)
            seen.add(st)
            ax.annotate(g["label"].split("_")[0], (R, max(err, 1e-6)),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axvline(2.0, color="#8b1a1a", ls="--", lw=1.2, label="$R_{hard}=2$ (never PASS)")
    ax.axvline(4.0, color="#1b7f3a", ls="--", lw=1.2, label="$R_{cert}=4$ (certifiable)")
    ax.axhline(0.05, color="k", ls=":", lw=1.0, label="5 % error band")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("resolvability ratio $R=\\delta/\\Delta x$ [-]")
    ax.set_ylabel("$|\\delta-\\delta_{ext}|/\\delta_{ext}$ [-]")
    ax.set_title("three-state verdict map", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, "resolvability_map.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_mach_trend(machscan, outdir):
    """delta / step height vs Mach with the out-of-validity correlation."""
    from tensorlbm.compressible_shock.resolvability import billig_standoff

    rows = (machscan.get("A4") or {}).get("rows") or []
    if not rows:
        return None
    machs = np.array([r["mach"] for r in rows])
    ratios = np.array([r["standoff_over_step_h"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(machs, ratios, "o-", color="#20478b", label="measured $\\delta_{mass}/h_{step}$")
    mm = np.linspace(machs.min(), machs.max(), 200)
    ax.plot(mm, [billig_standoff(m, radius=0.1) / 0.2 for m in mm], "--",
            color="#8b1a1a", label="Billig (1967), $R_{nose}=h/2$ (axisymmetric)")
    a4 = machscan.get("A4") or {}
    ax.text(0.03, 0.05,
            "correlation check: " + ("WITHHELD (outside validity)"
                                     if a4.get("correlation_withheld") else "within 20 %")
            + f"\ntrend: {'monotone decreasing (correct direction)' if a4.get('direction_ok') else 'NOT monotone'}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(fc="white", alpha=0.8, ec="#999999"))
    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("$\\delta / h_{step}$ [-]")
    ax.set_title("Mach trend of the step stand-off (A4)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, "mach_trend.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_budget(budget, outdir):
    """Minimal cell count to certify vs Mach, split into the two requirements."""
    rows = (budget.get("budget") or budget).get("rows") or []
    if not rows:
        return None
    machs = np.array([r["mach"] for r in rows])
    n_acc = np.array([r["N_min_accuracy"] for r in rows])
    n_res = np.array([r["N_min_resolvability"] for r in rows])
    n_min = np.array([r["N_min"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(machs, n_acc, "s--", color="#20478b", label="$N$ for < 5 % accuracy")
    ax.loglog(machs, n_res, "^:", color="#d98200", label="$N$ for $R\\geq 4$")
    ax.loglog(machs, n_min, "o-", color="#1b7f3a", label="$N_{min}$ (governing)")
    ax.set_xlabel("Mach number [-]")
    ax.set_ylabel("cells $N=N_xN_y$ (square cells) [-]")
    ax.set_title("grid budget law: cells needed to certify the stand-off", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, "grid_budget.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_mass_audit(records, outdir):
    """Naive vs relative-caliber mass residual, per grid and for the cut corner."""
    labels, naive, rel = [], [], []
    for rec in records:
        a = rec.get("blockage_audit") or {}
        if not a.get("ok"):
            continue
        labels.append(rec["label"])
        naive.append(max(a["residual_naive"], 1e-18))
        rel.append(max(a["residual_relative"], 1e-18))
    if not labels:
        return None
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(x - 0.2, naive, 0.4, label="naive (donor cells, corner dropped)", color="#8b1a1a")
    ax.bar(x + 0.2, rel, 0.4, label="relative caliber (interface fluxes, corner kept)",
           color="#1b7f3a")
    ax.axhline(1e-6, color="k", ls=":", lw=1.0, label="A3 threshold $10^{-6}$")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace("_", " ") for lab in labels], rotation=15, fontsize=8)
    ax.set_ylabel("relative mass residual per step [-]")
    ax.set_title("blockage-corner mass bookkeeping (A3)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, "corner_mass_audit.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_field(npz_path, outdir, zoom=(0.15, 0.95, 0.0, 0.6)):
    """Density + numerical schlieren of a stored grid with the measurement rows."""
    d = np.load(npz_path)
    rho = d["rho"]
    xc, yc = d["xc"], d["yc"]
    step_x, step_h = float(d["step_x"]), float(d["step_h"])
    mach = float(d["mach"])
    label = str(d["label"])
    solid = d["solid"] if "solid" in d else np.zeros_like(rho, dtype=bool)
    rho_plot = np.where(solid, np.nan, rho)
    grads = np.gradient(rho_plot)
    schlieren = np.log1p(200.0 * np.hypot(grads[1] / float(d["dx"]), grads[0] / float(d["dy"])))
    extent = [xc[0] - 0.5 * float(d["dx"]), xc[-1] + 0.5 * float(d["dx"]),
              yc[0] - 0.5 * float(d["dy"]), yc[-1] + 0.5 * float(d["dy"])]

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6))
    ax = axes[0]
    im = ax.imshow(rho_plot, origin="lower", extent=extent, cmap="inferno", aspect="auto")
    for frac in (0.35, 0.5, 0.65):
        ax.axhline(frac * step_h, color="c", lw=0.6, alpha=0.8)
    ax.axvline(step_x, color="w", ls="--", lw=0.8)
    ax.set_ylabel("$y$ [-]")
    ax.set_title(f"density, forward-facing step Ma={mach:g}, {label}, t={float(d['t']):.2f}",
                 fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01, label="$\\rho$")

    ax2 = axes[1]
    im2 = ax2.imshow(schlieren, origin="lower", extent=extent, cmap="Greys",
                     vmin=0, vmax=np.nanmax(schlieren), aspect="auto")
    ax2.set_xlim(zoom[0], zoom[1])
    ax2.set_ylim(zoom[2], zoom[3])
    ax2.axvline(step_x, color="r", ls="--", lw=0.9, label="step face $x_{step}$")
    ax2.set_xlabel("$x$ [-]")
    ax2.set_ylabel("$y$ [-]")
    ax2.set_title("numerical schlieren (zoom on the bow shock and the corner)", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    fig.colorbar(im2, ax=ax2, pad=0.01, label="$\\log(1+200|\\nabla\\rho|)$")
    fig.tight_layout()
    out = os.path.join(outdir, "field_ma%d_%s.png" % (round(mach), label))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", required=True, help="ladder summary JSON (collect)")
    ap.add_argument("--machscan", default=None, help="Mach-scan JSON (mach-scan)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional grid records for the audit/verdict panels")
    ap.add_argument("--fields", nargs="*", default=[], help=".npz field snapshots")
    ap.add_argument("--outdir", default="figures/resolvability")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    ladder = load(args.ladder)
    made = [fig_convergence(ladder, args.outdir),
            fig_richardson(ladder, args.outdir),
            fig_resolvability_map([ladder], args.outdir)]
    records = list(ladder.get("records", []))
    for p in args.extra:
        records.append(load(p))
    made.append(fig_mass_audit(records, args.outdir))
    if args.machscan:
        scan = load(args.machscan)
        made.append(fig_mach_trend(scan, args.outdir))
        made.append(fig_budget(scan, args.outdir))
    for p in args.fields:
        made.append(fig_field(p, args.outdir))
    for m in made:
        if m:
            print("wrote", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())