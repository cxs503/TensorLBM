#!/usr/bin/env python3
"""Figures for the Re=200 square-cylinder case (CASE-LBM-SQUARE-CYL-NORBERG).

Reads only the run artifacts written by ``run.py`` (case_D*.json, series_D*.npz)
and the graded reference file, so every plotted number is traceable to a file.
Outputs PNGs plus a machine-readable figure index (sha256 per file).

Usage:
    python benchmarks/verified/square_cylinder_re200/make_figures.py \
        --runs runs/m16410_square_cyl_re200 --out figures/square_cylinder_re200
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_case(dirpath: Path) -> tuple[dict, dict[str, dict]]:
    summary = json.loads((dirpath / "result.json").read_text())
    return summary, summary["grids"]


def fig_wake(dirpath: Path, out: Path, D: str, re: str) -> Path | None:
    f = dirpath / f"series_D{D}.npz"
    if not f.exists():
        return None
    vort = np.load(f)["vorticity"]
    ny, nx = vort.shape
    Dn = int(D)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    extent = [0, nx / Dn, 0, ny / Dn]
    vmax = np.percentile(np.abs(vort), 99.5) or 1.0
    im = ax.imshow(vort, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_title(f"Re={re} square cylinder, D={D}: spanwise vorticity (snapshot at end of run)")
    ax.axhline(0.0, color="k", lw=0.0)
    fig.colorbar(im, ax=ax, label=r"$\omega_z$ (lattice units)")
    p = out / f"wake_re{re}_D{D}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_signals(dirpath: Path, out: Path, D: str, re: str, rec: dict) -> Path | None:
    f = dirpath / f"series_D{D}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    cd, cl = z["cd"], z["cl"]
    w0 = int(rec["analyze_from"])
    t = np.arange(cd.size)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.plot(t, cd, lw=0.6, color="C3")
    a1.plot(t[w0:], np.full(cd.size - w0, rec["cd"]["mean"]), "k--", lw=1.0,
            label=f"mean over window = {rec['cd']['mean']:.4f}")
    a1.axvline(w0, color="0.6", ls=":", label="analysis window start")
    a1.set_ylabel(r"$C_d$")
    a1.legend(loc="upper right", fontsize=8)
    a1.set_title(f"Re={re}, D={D}: force history (seed period visible at the left)")
    a2.plot(t, cl, lw=0.6, color="C0")
    a2.axvline(w0, color="0.6", ls=":")
    a2.set_ylabel(r"$C_l$")
    a2.set_xlabel("lattice step")
    p = out / f"signals_re{re}_D{D}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_convergence(cases: dict[Path, tuple[dict, dict[str, dict]]], out: Path,
                    reference: dict) -> Path:
    """Cd and St vs grid resolution for every Re, with verified reference rungs."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = {(r["re"], r["metric"]): r for r in reference["rows"]}
    colors = {150.0: "C0", 200.0: "C1"}
    for dirpath, (summary, grids) in cases.items():
        re = float(summary["re"])
        Ds = sorted(int(k) for k in grids)
        cds = [grids[str(D)]["cd"]["mean"] for D in Ds]
        sts = [grids[str(D)]["st"]["value"] for D in Ds]
        c = colors.get(re, "C2")
        a1.plot(Ds, cds, "o-", color=c, label=f"Re={re:g} (measured)")
        a2.plot(Ds, sts, "o-", color=c, label=f"Re={re:g} (measured)")
        for ax, vals in ((a1, cds), (a2, sts)):
            ax.annotate(f"{vals[-1]:.4f}", (Ds[-1], vals[-1]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=c)
        for ax, metric in ((a1, "cd"), (a2, "st")):
            row = rows.get((re, metric))
            if row and row.get("ref") is not None:
                ax.axhline(row["ref"], color=c, ls="--", lw=1.0,
                           label=f"Re={re:g} ref {row['ref']} ({row['source'].split()[0]})")
    a1.set_xlabel("D (lattice nodes per body side)")
    a1.set_ylabel(r"mean $C_d$")
    a2.set_ylabel("St = fD/U")
    a1.set_title("drag: measured vs verified reference rungs")
    a2.set_title("Strouhal: measured vs verified reference rungs")
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    p = out / "convergence_cd_st.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_ladder(cases: dict[Path, tuple[dict, dict[str, dict]]], out: Path,
               reference: dict) -> Path:
    """Error bars vs tolerance: verified rungs (referee) and the anchor Re=200 (no referee)."""
    rows = {(r["re"], r["metric"]): r for r in reference["rows"]}
    labels, errs, tols, colors = [], [], [], []
    for re in (150.0, 200.0):
        for metric, tol_default in (("cd", 10.0), ("st", 8.0)):
            vals = []
            for dirpath, (summary, grids) in cases.items():
                if float(summary["re"]) != re:
                    continue
                key = metric if metric == "cd" else metric
                vals.extend(
                    g["cd"]["mean"] if metric == "cd" else g["st"]["value"] for g in grids.values()
                )
            if not vals:
                continue
            row = rows.get((re, metric), {})
            if row.get("ref") is None:
                labels.append(f"Re={re:g} {metric.upper()}\n(WITHHELD referee)")
                errs.append(0.0)
                tols.append(row.get("tol_pct", tol_default))
                colors.append("0.7")
                continue
            err = np.mean([(v - row["ref"]) / row["ref"] * 100 for v in vals])
            labels.append(f"Re={re:g} {metric.upper()}")
            errs.append(err)
            tols.append(row.get("tol_pct", tol_default))
            colors.append("C0" if abs(err) <= tols[-1] else "C3")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, errs, color=colors, width=0.55)
    for i, (e, t) in enumerate(zip(errs, tols)):
        ax.plot([i - 0.35, i + 0.35], [t, t], "k--", lw=1.0)
        ax.plot([i - 0.35, i + 0.35], [-t, -t], "k--", lw=1.0)
        if colors[i] != "0.7":
            ax.annotate(f"{e:+.2f}%", (i, e), textcoords="offset points",
                        xytext=(0, 4 if e >= 0 else -12), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("error vs reference centre (%)")
    ax.set_title("reference ladder: measured error vs tolerance (dashed = tolerance band)")
    ax.grid(axis="y", alpha=0.3)
    p = out / "reference_ladder.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory holding re*_D*/result.json")
    ap.add_argument("--out", default=str(CASE_DIR.parents[2] / "figures" / "square_cylinder_re200"))
    args = ap.parse_args()

    runs = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads((CASE_DIR / "reference.json").read_text())

    cases: dict[Path, tuple[dict, dict[str, dict]]] = {}
    produced: list[dict] = []
    for sub in sorted(runs.glob("re*_D*")):
        if not (sub / "result.json").exists():
            continue
        summary, grids = _load_case(sub)
        cases[sub] = (summary, grids)
        re = f"{float(summary['re']):g}"
        for D, rec in grids.items():
            p = fig_wake(sub, out, D, re)
            if p:
                produced.append({"name": p.name, "kind": "field",
                                 "caption": f"Re={re} D={D} vorticity snapshot"})
            p = fig_signals(sub, out, D, re, rec)
            if p:
                produced.append({"name": p.name, "kind": "timeseries",
                                 "caption": f"Re={re} D={D} Cd/Cl history"})
    if cases:
        p = fig_convergence(cases, out, reference)
        produced.append({"name": p.name, "kind": "comparison",
                         "caption": "Cd and St vs grid; dashed = verified reference rung"})
        p = fig_ladder(cases, out, reference)
        produced.append({"name": p.name, "kind": "comparison",
                         "caption": "measured error vs tolerance per reference row"})

    index = {
        "figures": [{**f, "sha256": _sha(out / f["name"])} for f in produced],
        "runs_dir": str(runs),
        "reference_file": str(CASE_DIR / "reference.json"),
    }
    (out / "figure_index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps([f["name"] for f in produced], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())