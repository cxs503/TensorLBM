#!/usr/bin/env python3
"""Analysis + figures for CASE-CAVITY-RE5000 (Ghia 1982, Re = 5000).

Reads the runner's ``result.json`` artefacts (nothing is re-computed here, and no
number is hand-copied: every table printed below is derived from the JSON) and
writes

  figures/cavity_re5000_centerline.png   u(0.5, y), v(x, 0.5) vs Ghia + 8 % band
  figures/cavity_re5000_error.png        per-row error with the 8 % tolerance line
  figures/cavity_re5000_convergence.png  max|du| / max error vs step
  figures/cavity_re5000_convention.png   node- vs wall-coordinate re-scoring
  figures/cavity_re5000_gridconv.png     coarse -> fine delta of u(0.5, y)

and prints a markdown summary on stdout for programmatic injection into a report.

Usage
-----
    python benchmarks/verified/cavity/re5000/analyze_convergence.py \
        --runs benchmarks/verified/cavity/re5000/out_local/re5000_n256_long.json \
               benchmarks/verified/cavity/re5000/out_local/re5000_n320_long.json \
        --outdir benchmarks/verified/cavity/re5000/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOL_PCT = 8.0


def load(path):
    return json.loads(Path(path).read_text())


def score(rec, ref, convention="node"):
    """Re-score a stored grid record on either coordinate convention."""
    # legacy artefacts (pre-2026-11) carry only the node coordinate as y_centerline
    ykey = f"y_centerline_{convention}"
    xkey = f"x_centerline_{convention}"
    legacy = rec.get("coord")                       # pre-2026-11 artefacts
    y = np.asarray(rec.get(ykey) or rec.get("y_centerline") or legacy, dtype=float)
    u = np.asarray(rec["u_centerline"], dtype=float)
    x = np.asarray(rec.get(xkey) or rec.get("x_centerline") or legacy, dtype=float)
    v = np.asarray(rec["v_centerline"], dtype=float)
    u_ref = np.asarray(ref["u_re5000"], dtype=float)
    v_ref = np.asarray(ref["v_re5000"], dtype=float)
    u_got = np.interp(u_ref[:, 0], y, u)
    v_got = np.interp(v_ref[:, 0], x, v)
    du = (u_got - u_ref[:, 1]) * 100.0
    dv = (v_got - v_ref[:, 1]) * 100.0
    return {
        "y_ref": u_ref[:, 0], "u_ref": u_ref[:, 1], "u_got": u_got, "du": du,
        "x_ref": v_ref[:, 0], "v_ref": v_ref[:, 1], "v_got": v_got, "dv": dv,
        "max_abs_err_pct": float(max(np.abs(du).max(), np.abs(dv).max())),
        "rmse_pct": float(np.sqrt(np.mean(np.concatenate([du, dv]) ** 2))),
        "n_u": int(np.abs(du).argmax()), "n_v": int(np.abs(dv).argmax()),
    }


def plot_centerline(runs, ref, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for label, rec, style in runs:
        s = score(rec, ref)
        axes[0].plot(s["u_got"], s["y_ref"], style, lw=1.3, ms=3, label=f"{label}")
        axes[1].plot(s["x_ref"], s["v_got"], style, lw=1.3, ms=3, label=f"{label}")
    s = score(runs[0][1], ref)
    axes[0].plot(s["u_ref"], s["y_ref"], "ko", ms=4, mfc="none", label="Ghia 1982 (Table III)")
    axes[1].plot(s["x_ref"], s["v_ref"], "ko", ms=4, mfc="none", label="Ghia 1982 (Table IV)")
    for ax, tol, ylabel, xlabel, title in (
        (axes[0], TOL_PCT / 100.0, "y/L", "u/U_lid", "u along x/L = 0.5"),
        (axes[1], TOL_PCT / 100.0, "v/U_lid", "x/L", "v along y/L = 0.5"),
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title + f"  (±{tol*100:.0f}% band)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    if runs:
        s = score(runs[0][1], ref)
        axes[0].fill_betweenx(s["y_ref"], s["u_ref"] - 0.08, s["u_ref"] + 0.08,
                              color="grey", alpha=0.15)
        axes[1].fill_between(s["x_ref"], s["v_ref"] - 0.08, s["v_ref"] + 0.08,
                             color="grey", alpha=0.15)
    fig.suptitle("CASE-CAVITY-RE5000: centreline profiles vs Ghia (1982), Re = 5000")
    fig.tight_layout()
    p = Path(outdir) / "cavity_re5000_centerline.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_errors(runs, ref, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, rec, style in runs:
        s = score(rec, ref)
        axes[0].plot(s["y_ref"], s["du"], style, lw=1.1, ms=3, label=label)
        axes[1].plot(s["x_ref"], s["dv"], style, lw=1.1, ms=3, label=label)
    for ax, title in ((axes[0], "u error (x/L = 0.5)"), (axes[1], "v error (y/L = 0.5)")):
        ax.axhline(TOL_PCT, color="r", ls="--", lw=1)
        ax.axhline(-TOL_PCT, color="r", ls="--", lw=1, label=f"±{TOL_PCT:.0f}% tolerance")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("tabular coordinate (-)")
        ax.set_ylabel("(got - ref) [% of U_lid]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Pointwise error on the Ghia (1982) tabulated rows")
    fig.tight_layout()
    p = Path(outdir) / "cavity_re5000_error.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def resid_history(rec):
    """Normalise the runner's history entries (new: list, legacy: dict)."""
    out = []
    for h in rec.get("resid_history") or []:
        if isinstance(h, dict):
            out.append((h.get("step"), h.get("resid"), h.get("resid_rel"),
                        h.get("err_pct"), h.get("u_min"), h.get("v_max")))
        else:
            out.append(tuple(h) + (None,) * (6 - len(h)))
    return out


def plot_convergence(runs, outdir):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, rec, style in runs:
        hist = [h for h in resid_history(rec) if h[1] is not None]
        if not hist:
            continue
        steps = np.array([h[0] for h in hist], dtype=float)
        resid = np.array([h[1] for h in hist], dtype=float)
        ax.semilogy(steps, resid, style, lw=1.2, ms=3, label=f"{label}: max|Δu|")
    ax.set_xlabel("step")
    ax.set_ylabel("max|Δu| over monitor interval [lattice units]")
    ax.set_title("Convergence: residual history (u_lid = 0.1)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = Path(outdir) / "cavity_re5000_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_convention(rec, ref, outdir):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for conv, style in (("node", "o-"), ("wall", "s--")):
        s = score(rec, ref, conv)
        ax.plot(s["y_ref"], s["du"], style, lw=1.1, ms=3, label=f"u err, {conv} coords")
        ax.plot(s["x_ref"], s["dv"], style, lw=0.9, ms=3, alpha=0.6,
                label=f"v err, {conv} coords")
    ax.axhline(TOL_PCT, color="r", ls="--", lw=1)
    ax.axhline(-TOL_PCT, color="r", ls="--", lw=1)
    ax.set_xlabel("tabular coordinate (-)")
    ax.set_ylabel("(got - ref) [% of U_lid]")
    ax.set_title("Coordinate-convention sensitivity of the near-wall error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    p = Path(outdir) / "cavity_re5000_convention.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_gridconv(coarse, fine, outdir):
    y = np.asarray(fine["y_centerline_node"], dtype=float)
    uc = np.interp(y, np.asarray(coarse["y_centerline_node"], dtype=float),
                   np.asarray(coarse["u_centerline"], dtype=float))
    d = (np.asarray(fine["u_centerline"], dtype=float) - uc) * 100.0
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(d, y, "o-", lw=1.2, ms=3)
    ax.set_xlabel(f"u({fine['n']}²) - u({coarse['n']}²)  [% of U_lid]")
    ax.set_ylabel("y/L")
    ax.set_title(f"Grid convergence {coarse['n']}² → {fine['n']}²")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = Path(outdir) / "cavity_re5000_gridconv.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def markdown_table(runs, ref, outdir):
    lines = ["| run | n | steps | final `max|Δu|` | status | max abs err [%] | RMSE [%] | vs 8 % |",
             "|---|---|---|---|---|---|---|---|"]
    for label, rec, _ in runs:
        s = score(rec, ref)
        fr = rec.get("final_resid")
        frs = "n/a" if fr is None else f"{fr:.2e}"
        lines.append(f"| {label} | {rec['n']} | {rec['steps']} | {frs} | {rec['status']} | "
                     f"{s['max_abs_err_pct']:.2f} | {s['rmse_pct']:.2f} | "
                     f"{'PASS' if s['max_abs_err_pct'] <= TOL_PCT else 'FAIL'} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--ref",
                    default=str(Path(__file__).parent / "reference" / "ghia_re5000_reference.json"))
    ap.add_argument("--outdir", default=str(Path(__file__).parent / "figures"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ref = load(args.ref)
    labels = args.labels or [Path(p).stem for p in args.runs]
    styles = ["b-", "g--", "m-.", "c:"]

    runs = []
    for label, path in zip(labels, args.runs):
        payload = load(path)
        for key, rec in payload["grids"].items():
            if not rec.get("u_centerline"):
                continue
            runs.append((f"{label} n={rec['n']}", rec, styles[len(runs) % len(styles)]))

    if not runs:
        raise SystemExit("no usable grid records in the given artefacts")

    print("## Scored runs (re-scored from stored profiles, node convention)\n")
    print(markdown_table(runs, ref, outdir))

    print("\n## Error at the worst rows (node vs wall coordinate convention)\n")
    print("| run | max err node [%] | max err wall [%] | worst row u (y/L, err%) | worst row v (x/L, err%) |")
    print("|---|---|---|---|---|")
    conv = {}
    for label, rec, _ in runs:
        sn = score(rec, ref, "node")
        sw = score(rec, ref, "wall")
        conv[label] = (sn, sw)
        print(f"| {label} | {sn['max_abs_err_pct']:.2f} | {sw['max_abs_err_pct']:.2f} | "
              f"({sn['y_ref'][sn['n_u']]:.4f}, {sn['du'][sn['n_u']]:+.1f}) | "
              f"({sn['x_ref'][sn['n_v']]:.4f}, {sn['dv'][sn['n_v']]:+.1f}) |")

    figs = [plot_centerline(runs, ref, outdir), plot_errors(runs, ref, outdir),
            plot_convergence(runs, outdir),
            plot_convention(runs[0][1], ref, outdir)]
    if len(runs) >= 2:
        figs.append(plot_gridconv(runs[1][1], runs[0][1], outdir))
    print("\n## Figures\n")
    for p in figs:
        print(f"* `{p}`")


if __name__ == "__main__":
    main()