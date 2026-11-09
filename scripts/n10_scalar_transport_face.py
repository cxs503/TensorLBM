#!/usr/bin/env python
"""N10 face driver: merge shard outputs, emit figures, tables and markdown.

Usage
-----
    python scripts/n10_scalar_transport_face.py --out artifacts/scalar_transport_n10 \
        --merges runs/n10/out0,runs/n10/out1,... [--figure-dir figures/scalar_transport_n10]

The heavy work (the actual lattice runs) is done by
``python -m tensorlbm.scalar_transport_verification --only <ids>`` shards; this
script only post-processes: it re-evaluates the gates on the merged item set,
writes ``report.json``/``tables.md``, and draws the figures.  No number in the
markdown is typed by hand -- every value comes from the JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.scalar_transport_verification import (
    ARIS_K_SHEAR,
    GATE_THRESHOLDS,
    assemble_report,
    evaluate_gates,
    load_item_dirs,
)
from tensorlbm.taylor_aris import ARIS_K_CHANNEL, ARIS_K_PIPE

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True, "grid.alpha": 0.3})


def _fig_erf(items: dict, path: Path) -> str:
    erf = items["erf"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    finest = erf["levels"][-1]
    x = np.asarray(finest["x"])
    ax1.plot(x, finest["c_ref"], "k-", lw=1.6, label="exact $\\mathrm{erfc}(x/\\sqrt{4Dt})/2$")
    ax1.plot(
        x[::2], np.asarray(finest["c_num"])[::2], "o", ms=2.4, mfc="none", color="tab:red",
        label=f"lattice D1Q3, n={finest['n']}",
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("c")
    ax1.set_title("(a) semi-infinite diffusion, $t=1$, $D=1$")
    ax1.legend(loc="lower left", fontsize=7)
    ax2.loglog(erf["dxs"], erf["l2_rel_errors"], "s-", color="tab:blue", label="L2 relative error")
    dx = np.asarray(erf["dxs"], dtype=float)
    ax2.loglog(dx, erf["l2_rel_errors"][-1] * (dx / dx[-1]) ** 2, "k--", lw=1, label="slope 2")
    ax2.set_xlabel("$\\Delta x$ (diffusive scaling, $D_{lat}$ fixed)")
    ax2.set_ylabel("$\\|c_h-c\\|_2/\\|c\\|_2$")
    ax2.set_title(
        f"(b) order p={erf['order_fit']:.3f} (R$^2$={erf['order_fit_r2']:.5f}), "
        f"$p_{{2finest}}$={erf['order_two_finest']:.2f}"
    )
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _fig_taylor(items: dict, path: Path) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    chan = {r["item_id"]: r for r in items.values() if r.get("kind") == "channel"}
    shear = {r["item_id"]: r for r in items.values() if r.get("kind") == "shear"}

    # (a) D_eff/D vs Pe for the reference grid, both estimators
    ref = sorted(
        [r for k, r in chan.items() if r.get("grid") == {"ny": 13, "nx": 1024, "diffusivity": 0.01}],
        key=lambda r: r["pe_r"],
    )
    pe = np.linspace(0.01, 45, 200)
    ax1.plot(pe, 1.0 + ARIS_K_CHANNEL * pe**2, "k-", lw=1.4, label="closed form $1+\\frac{2}{105}Pe^2$")
    ax1.plot([r["pe_r"] for r in ref], [r["ratio_over_d"] for r in ref], "o", color="tab:blue",
             label="channel, variance (moment)")
    modes = sorted([r for r in chan.values() if r.get("method") == "mode"], key=lambda r: r["pe_r"])
    ax1.plot([r["pe_r"] for r in modes], [r["ratio_over_d"] for r in modes], "x", color="tab:blue",
             ms=7, mew=1.6, label="channel, Fourier-mode decay")
    she = sorted(shear.values(), key=lambda r: r["pe_r"])
    ax1.plot(pe, 1.0 + ARIS_K_SHEAR * pe**2, "k--", lw=1.2, label="closed form $1+\\frac{2}{15}Pe_\\gamma^2$")
    ax1.plot([r["pe_r"] for r in she], [r["ratio_over_d"] for r in she], "s", color="tab:orange",
             label="linear shear $u=\\gamma y$, variance")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("$Pe$")
    ax1.set_ylabel("$D_{eff}/D$")
    ax1.set_title("(a) Taylor-Aris dispersion vs closed forms")
    ax1.legend(fontsize=6.4, loc="upper left")

    # (b) relative error vs Pe per cross-stream resolution
    for ny, style, color in ((9, "o-", "tab:red"), (13, "s-", "tab:blue"), (17, "^--", "tab:green")):
        rows = sorted(
            [r for r in chan.values() if r.get("method") == "variance" and r.get("ny") == ny
             and r["nx"] == max(rr["nx"] for rr in chan.values() if rr.get("ny") == ny)],
            key=lambda r: r["pe_r"],
        )
        if not rows:
            continue
        ax2.plot([r["pe_r"] for r in rows], [100 * r["rel_err_analytic"] for r in rows], style,
                 color=color, label=f"ny={ny} (nx={rows[0]['nx']}, D={rows[0]['diffusivity']})")
    ax2.axhspan(-100 * GATE_THRESHOLDS["G2_rel_err_max"], 100 * GATE_THRESHOLDS["G2_rel_err_max"],
                color="0.85", label="±5 % gate")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("$Pe_r$")
    ax2.set_ylabel("$D_{eff}$ error vs closed form  [%]")
    ax2.set_title("(b) cross-stream resolution effect (moment method)")
    ax2.legend(fontsize=6.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _fig_g8(gates: dict, path: Path) -> str:
    rows = gates["G8"]["rows"]
    nys = sorted({r["ny"] for r in rows})
    pes = sorted({r["pe_r"] for r in rows})
    grid = np.full((len(nys), len(pes)), np.nan)
    valid = np.zeros_like(grid, dtype=bool)
    for r in rows:
        i, j = nys.index(r["ny"]), pes.index(r["pe_r"])
        s = r.get("sigma_over_nx")
        ok = s is not None and np.isfinite(s) and s <= 1.0 / 6.0
        grid[i, j] = 100 * r["rel_err"]
        valid[i, j] = ok
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
    for i in range(len(nys)):
        for j in range(len(pes)):
            if np.isnan(grid[i, j]):
                continue
            txt = f"{grid[i, j]:+.1f}%" + ("" if valid[i, j] else "\n(σ>nx/6)")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="w" if abs(grid[i, j]) > 5 else "k")
    ax.set_xticks(range(len(pes)), [f"{p:g}" for p in pes])
    ax.set_yticks(range(len(nys)), [str(n) for n in nys])
    ax.set_xlabel("$Pe_r$")
    ax.set_ylabel("cross-stream nodes $n_y$")
    ax.set_title("$D_{eff}$ error vs closed form, best streamwise grid per $n_y$")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _fig_g4(items: dict, path: Path) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    cols = [r for r in items.values() if "G4" in r.get("tags", []) and r.get("kind") == "channel"]
    by_d: dict[float, list[float]] = {}
    for r in cols:
        by_d.setdefault(r["diffusivity"], []).append(r["k_meas"])
    ds = sorted(by_d)
    ax1.axhline(ARIS_K_CHANNEL, color="k", ls="--", lw=1, label="$K=2/105$")
    ax1.plot(ds, [float(np.mean(by_d[d])) for d in ds], "o", color="tab:purple")
    ax1.set_xscale("log")
    ax1.set_xlabel("molecular diffusivity $D$ (=> $Sc=\\nu/D$ range)")
    ax1.set_ylabel("$K_{meas}=(D_{eff}/D-1)Pe_r^{-2}$")
    ax1.set_title("(a) dispersion constant collapses across $D$")
    ax1.legend(fontsize=7)

    taus = sorted([r for r in items.values() if r.get("kind") == "tau_m"], key=lambda r: r["sc"])
    if taus:
        sc = [r["sc"] for r in taus]
        ax2.plot(sc, [r["ratio_over_d"] for r in taus], "s-", color="tab:green",
                 label="$D_{eff}/D$ (moment method)")
        ax2.plot(sc, [r["discrete_aris"] / r["diffusivity"] for r in taus], "k:", lw=1,
                 label="discrete Aris reference")
        ax2.set_xscale("log")
        ax2.set_xlabel("$Sc=\\nu/D$ of the momentum field ($\\tau_m$ sweep, $\\tau_d$ fixed)")
        ax2.set_ylabel("$D_{eff}/D$")
        ax2.set_title("(b) scalar does not see the momentum relaxation")
        ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _fig_mass(items: dict, hist_path: Path | None, path: Path) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for tag, color in (("mass_f64", "tab:blue"), ("mass_f32", "tab:red")):
        r = items.get(tag)
        if not r:
            continue
        ax1.semilogy(r["checkpoints"], r["drift"], "o-", color=color, label=r["dtype"])
    ax1.axhline(GATE_THRESHOLDS["G5_mass_drift_max"] * 1e-1, color="k", ls="--", lw=1,
                label="gate $10^{-10}$/1e4 steps")
    ax1.set_xlabel("step")
    ax1.set_ylabel("$|M(t)-M(0)|/M(0)$")
    ax1.set_title("(a) total-scalar drift (zero-flux walls)")
    ax1.legend(fontsize=7)

    ref = None
    for r in items.values():
        if r.get("kind") == "channel" and r.get("ny") == 13 and r.get("pe_r") == 20.0 and r.get("method") == "variance" and r.get("nx") == 1024:
            ref = r
            break
    if ref is not None:
        t = np.asarray(ref.get("history_t", []))
        q = np.asarray(ref.get("history_q", []))
        if t.size > 2:
            ax2.plot(t, q, "o", ms=1.6, color="tab:blue", label="$\\sigma^2(t)$ (moment history)")
            i0 = int(ref["fit_t0"] / t[-1] * (t.size - 1))
            ax2.plot(t[i0:], q[i0:], "r-", lw=1.2, label="fit window")
            ax2.set_xlabel("t")
            ax2.set_ylabel("plume variance $\\sigma^2$")
            ax2.set_title(f"(b) variance growth, ny=13 Pe=20, $R^2$={ref['r2']:.5f}")
            ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _tables_md(report: dict) -> str:
    g = report["gates"]
    lines = ["# N10 gate summary", "", "| gate | pass | key numbers |", "|---|---|---|"]
    g1 = g.get("G1", {})
    lines.append(
        f"| G1 | {g1.get('pass')} | finest L2 = {g1.get('finest_l2_rel_error', float('nan')):.2e} "
        f"(<= {g1.get('threshold')}), order = {g1.get('observed_order', float('nan')):.3f} |"
    )
    g2 = g.get("G2", {})
    if g2:
        w = g2["worst"]
        lines.append(
            f"| G2 | {g2['pass']} | worst {w['case']} Pe={w['pe_r']}: "
            f"{w['rel_err']:+.2%} (limit {g2['threshold']:.0%}) |"
        )
    g3 = g.get("G3", {})
    if g3:
        w = g3["worst"]
        lines.append(
            f"| G3 | {g3['pass']} | Pe={w['pe_r']}: D_eff/D = {w['d_eff_over_d']:.5f}, "
            f"dev {w['rel_dev']:+.3%} |"
        )
    g4 = g.get("G4", {})
    if g4:
        lines.append(
            f"| G4 | {g4.get('pass')} | collapse divergence "
            f"{g4.get('collapse', {}).get('divergence', float('nan')):.3%}, tau_m divergence "
            f"{g4.get('tau_m', {}).get('divergence', float('nan')):.3%} |"
        )
    g5 = g.get("G5", {})
    if g5:
        w = g5["worst"]
        lines.append(
            f"| G5 | {g5['pass']} | drift {w['drift_per_1e4_steps']:.2e} / 1e4 steps "
            f"({w['dtype']}) |"
        )
    g6 = g.get("G6", {})
    if g6:
        e = g6["entries"][0]
        lines.append(f"| G6 | {g6['pass']} | u=0: D_eff/D = {e['d_eff_over_d']:.5f} |")
    g7 = g.get("G7", {})
    if g7:
        lines.append(
            f"| G7 | {g7.get('pass')} | $R^2$ = {g7.get('order_fit_r2', float('nan')):.5f}, "
            f"monotone = {g7.get('monotone')} |"
        )
    g8 = g.get("G8", {})
    if g8:
        lines.append(
            f"| G8 | {g8.get('pass')} | {g8.get('n_within_threshold')}/{g8.get('n_cells')} cells "
            f"within {g8.get('threshold'):.0%} |"
        )
    lines += ["", "## G8 accuracy-domain table", "",
              "| geometry | Pe | ny | nx | D | steps | sigma/nx | D_eff | closed form | rel err | valid |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in g8.get("rows", []):
        s = r.get("sigma_over_nx")
        valid = "yes" if (s is not None and np.isfinite(s) and s <= 1 / 6) else "no (sigma>nx/6)"
        lines.append(
            f"| {r.get('geometry', 'channel')} | {r['pe_r']:g} | {r['ny']} | {r['nx']} | "
            f"{items_D(r)} | {r['steps']} | {s:.3f} | {r['d_eff']:.5f} | "
            f"{r['d_eff_closed_form']:.5f} | {r['rel_err']:+.2%} | {valid} |"
        )
    return "\n".join(lines) + "\n"


def items_D(row: dict) -> str:
    return f"{row.get('diffusivity', float('nan')):g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/scalar_transport_n10")
    ap.add_argument("--merges", required=True)
    ap.add_argument("--figure-dir", default="figures/scalar_transport_n10")
    ap.add_argument("--mode", default="full")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [Path(p) for p in args.merges.split(",")]
    items = load_item_dirs(dirs)
    gates = evaluate_gates(items)
    report = assemble_report(items, gates, mode=args.mode, outdir=out)
    # keep the raw per-item records (they hold the erf profiles / moment history)
    (out / "items_merged.json").write_text(json.dumps(items))

    figdir = Path(args.figure_dir)
    figdir.mkdir(parents=True, exist_ok=True)
    figs = []
    figs.append({"name": "n10_erf_diffusion.png", "path": _fig_erf(items, figdir / "n10_erf_diffusion.png")})
    figs.append({"name": "n10_taylor_aris.png", "path": _fig_taylor(items, figdir / "n10_taylor_aris.png")})
    figs.append({"name": "n10_accuracy_domain.png", "path": _fig_g8(gates, figdir / "n10_accuracy_domain.png")})
    if any(r.get("kind") == "tau_m" for r in items.values()):
        figs.append({"name": "n10_sc_independence.png", "path": _fig_g4(items, figdir / "n10_sc_independence.png")})
    figs.append({"name": "n10_mass_and_moments.png", "path": _fig_mass(items, None, figdir / "n10_mass_and_moments.png")})
    report["figures"] = figs
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "tables.md").write_text(_tables_md(report))
    print(json.dumps({k: report["gates"][k]["pass"] for k in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8") if k in report["gates"]}))
    print("verdict", report["verdict"], "figures", [f["name"] for f in figs])
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())