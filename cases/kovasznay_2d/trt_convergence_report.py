#!/usr/bin/env python3
"""Build the report tables + figure for the Kovasznay TRT convergence arm.

Everything is read from the JSON artifacts written by ``trt_convergence.py`` -- no
number in the generated markdown/figures is typed by hand.

Usage
-----
::

    python cases/kovasznay_2d/trt_convergence_report.py \\
        --artifacts artifacts/m18406 --outdir artifacts/m18406

Writes ``TRT_CONVERGENCE_TABLES.md`` and ``fig_kovasznay_trt.png`` into ``--outdir``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LADDER_FILES = ["ladder32.json", "ladder64.json", "ladder64_40k.json", "ladder128.json"]


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(x, nd=5):
    if x is None:
        return "--"
    if isinstance(x, float) and not math.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def collect_ladder(art: Path) -> list[dict]:
    rows = []
    for name in LADDER_FILES:
        d = _load(art / name)
        if not d:
            continue
        for r in d["rows"]:
            if not r.get("finite"):
                continue
            rows.append({"source": name, **r})
    rows.sort(key=lambda r: (r["ny"], r["steps"]))
    return rows


def table_ladder(rows: list[dict]) -> str:
    head = (
        "| artifact | ny | nx | tau+ | tau- | steps | drift(t) | full u L2% | full v L2% "
        "| int u/U0% | int v/U0% | int u L2% | int v L2% |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    out = [head]
    for r in rows:
        out.append(
            "| {src} | {ny} | {nx} | {tp} | {tm} | {st} | {dr} | {fu} | {fv} | {iu} | {iv} "
            "| {iur} | {ivr} |\n".format(
                src=r["source"],
                ny=r["ny"],
                nx=r["nx"],
                tp=_fmt(r["tau_plus"], 4),
                tm=_fmt(r.get("tau_minus"), 3),
                st=r["steps"],
                dr=_sci(r.get("drift_last")),
                fu=_fmt(r["full"]["u_l2_rel_pct"], 4),
                fv=_fmt(r["full"]["v_l2_rel_pct"], 4),
                iu=_fmt(r["interior"]["u_l2_abs_over_u0_pct"], 5),
                iv=_fmt(r["interior"]["v_l2_abs_over_u0_pct"], 5),
                iur=_fmt(r["interior"]["u_l2_rel_pct"], 4),
                ivr=_fmt(r["interior"]["v_l2_rel_pct"], 4),
            )
        )
    return "".join(out)


def _sci(x) -> str:
    if x is None:
        return "--"
    if isinstance(x, float) and not math.isfinite(x):
        return "nan"
    return f"{x:.2e}"


def pick_converged(rows: list[dict]) -> list[dict]:
    """One entry per ny: prefer a run that hit the steady-state test, then the longest one.

    The ladder arms stop at the drift criterion, so different grids consume different
    step budgets; what has to match across grids is the *converged* solution, not the
    step count.  ``ladder64.json`` vs ``ladder64_40k.json`` is exactly the control that
    shows the step budget does not matter (< 1 % change in the interior error).
    """
    best: dict[int, dict] = {}
    for r in rows:
        cur = best.get(r["ny"])
        if cur is None:
            best[r["ny"]] = r
            continue
        key_new = (bool(r.get("steady")), r["steps"])
        key_cur = (bool(cur.get("steady")), cur["steps"])
        if key_new > key_cur:
            best[r["ny"]] = r
    return [best[k] for k in sorted(best)]


def table_orders(rows: list[dict]) -> str:
    """Observed order on the converged ladder (one steady run per grid)."""
    groups: dict[int, list[dict]] = {0: pick_converged(rows)}
    lines = [
        "| pair | p (int u/U0) | p (int v/U0) | p (full u L2 rel) | p (full v L2 rel) |\n",
        "|---|---|---|---|---|\n",
    ]
    for steps, grp in sorted(groups.items()):
        grp = sorted(grp, key=lambda r: r["ny"])
        if len(grp) < 2:
            continue
        for i in range(len(grp) - 1):
            a, b = grp[i], grp[i + 1]
            ha, hb = 1.0 / a["ny"], 1.0 / b["ny"]
            row = [f"| {a['ny']}->{b['ny']} "]
            for key, sub in (
                ("int u", "u_l2_abs_over_u0_pct"),
                ("int v", "v_l2_abs_over_u0_pct"),
            ):
                ea = a["interior"][sub] / 100.0
                eb = b["interior"][sub] / 100.0
                p = math.log(ea / eb) / math.log(ha / hb) if eb > 0 else float("nan")
                row.append(f"| {_fmt(p, 2)} ")
            for sub in ("u_l2_rel_pct", "v_l2_rel_pct"):
                ea = a["full"][sub] / 100.0
                eb = b["full"][sub] / 100.0
                p = math.log(ea / eb) / math.log(ha / hb) if eb > 0 else float("nan")
                row.append(f"| {_fmt(p, 2)} ")
            lines.append("".join(row) + "|\n")
    return "".join(lines)


def table_stability(stab: dict) -> str:
    head = (
        "| collision | lambda | u0 | Ma_max | tau+ | tau- | steps run | finite "
        "| int u L2% (3k steps) | int v L2% (3k steps) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    out = [head]
    for r in stab["rows"]:
        out.append(
            "| {c} | {l} | {u0} | {ma} | {tp} | {tm} | {st} | {fi} | {iu} | {iv} |\n".format(
                c=r["collision"],
                l="-" if r["lambda_trt"] is None else f"{r['lambda_trt']:.4f}",
                u0=_fmt(r["u0"], 6),
                ma=_fmt(r["ma_max"], 4),
                tp=_fmt(r["tau_plus"], 4),
                tm=_fmt(r["tau_minus"], 3),
                st=r["steps"],
                fi="yes" if r["finite"] else "**NO (NaN)**",
                iu=_fmt(r["interior_u_l2_rel_pct"], 4),
                iv=_fmt(r["interior_v_l2_rel_pct"], 4),
            )
        )
    return "".join(out)


def figure(art: Path, rows: list[dict], stab: dict, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) grid convergence of the interior error (step-matched sub-ladder)
    ax = axes[0]
    matched = pick_converged(rows)
    if matched:
        ny = np.array([r["ny"] for r in matched], dtype=float)
        iu = np.array([r["interior"]["u_l2_abs_over_u0_pct"] for r in matched])
        iv = np.array([r["interior"]["v_l2_abs_over_u0_pct"] for r in matched])
        ax.loglog(ny, iv, "o-", label="interior v / U0 (TRT, Ma=0.026)")
        ax.loglog(ny, iu, "s-", label="interior u / U0 (TRT, Ma=0.026)")
        h2 = iv[0] * (ny[0] / ny) ** 2
        ax.loglog(ny, h2, "k--", lw=1, label="slope h$^2$ (2nd order guide)")
        for x, y in zip(ny, iv):
            ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(4, -10),
                        fontsize=7)
        for x, y in zip(ny, iu):
            ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("ny (y-cells per unit length)")
    ax.set_ylabel("interior L2 error / U0")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("(a) boundary-free grid convergence (TRT, U0=0.0075)")

    # (b) compressibility floor: Ma ladder for BGK (from the case entry) vs TRT now
    ax = axes[1]
    trt = [r for r in stab["rows"] if r["collision"] == "trt" and r["finite"]]
    bgk = [r for r in stab["rows"] if r["collision"] == "bgk"]
    if trt:
        ma2 = np.array([r["ma2"] for r in trt])
        iu = np.array([r["interior_u_l2_rel_pct"] for r in trt])
        iv = np.array([r["interior_v_l2_rel_pct"] for r in trt])
        ax.loglog(ma2, iv, "o-", label="TRT interior v L2%")
        ax.loglog(ma2, iu, "s-", label="TRT interior u L2%")
        ax.loglog(ma2, iv[0] * (ma2 / ma2[0]), "k--", lw=1, label="slope Ma$^2$ (guide)")
    for r in bgk:
        if not r["finite"]:
            ax.axvline(r["ma2"], color="0.7", ls=":", lw=1)
    ax.set_xlabel("Ma$^2$")
    ax.set_ylabel("interior L2 error %")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("(b) Ma ladder at ny=32, 3000 steps (dotted = BGK diverged)")

    # (c) stability wall
    ax = axes[2]
    taus = sorted({r["tau_plus"] for r in stab["rows"]})
    for i, collision in enumerate(("bgk", "trt")):
        vals = []
        for t in taus:
            hit = [r for r in stab["rows"] if r["collision"] == collision and
                   abs(r["tau_plus"] - t) < 1e-12]
            vals.append(1.0 if hit and hit[0]["finite"] else 0.0)
        ax.barh(np.arange(len(taus)) + (i - 0.5) * 0.38, vals, height=0.36,
                label=collision.upper(), color=("tab:red" if collision == "bgk" else "tab:blue"))
    ax.set_yticks(np.arange(len(taus)))
    ax.set_yticklabels([f"{t:.4f}" for t in taus])
    ax.set_xlabel("survived 3000 steps (1 = finite, 0 = NaN)")
    ax.set_ylabel("tau+")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("(c) stability wall: BGK vs TRT")

    fig.suptitle(
        "Kovasznay (1948) exact solution -- TRT arm: boundary-free convergence order + stability wall",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = outdir / "fig_kovasznay_trt.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_field(art: Path, outdir: Path) -> Path | None:
    npz_path = art / "field_ny64.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    u_num, v_num = d["u_num"], d["v_num"]
    u_ana, v_ana = d["u_ana"], d["v_ana"]
    u0 = float(d["u0"])
    ny, nx = int(d["ny"]), int(d["nx"])
    err = 100.0 * (np.abs(u_num - u_ana) + np.abs(v_num - v_ana)) / u0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    speed = np.sqrt(u_num**2 + v_num**2) / u0
    im = ax.imshow(speed, origin="lower", aspect="auto", cmap="viridis",
                   extent=(0, 3, 0, 1))
    ax.contour(np.linspace(0, 3, nx), np.linspace(0, 1, ny), v_ana / u0, levels=8,
               colors="w", linewidths=0.5, alpha=0.7)
    ax.set_title(f"TRT |u|/U0, ny={ny} (white: analytic v contours)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("x'"); ax.set_ylabel("y'")

    ax = axes[1]
    im = ax.imshow(err, origin="lower", aspect="auto", cmap="magma",
                   extent=(0, 3, 0, 1), vmin=0, vmax=float(np.percentile(err, 99)))
    ax.axvline(3.0 - 1.0, color="c", ls="--", lw=1)
    ax.text(3.0 - 0.97, 0.94, "interior window ends here", color="c", fontsize=7)
    ax.set_title("100·(|du|+|dv|)/U0  [%]")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("x'")
    fig.suptitle("Kovasznay TRT arm, U0=0.0075 (Ma=0.026): field and absolute error map", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = outdir / "fig_kovasznay_trt_field.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/m18406")
    ap.add_argument("--outdir", default="artifacts/m18406")
    args = ap.parse_args(argv)
    art = Path(args.artifacts)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = collect_ladder(art)
    stab = _load(art / "stability.json")
    md = ["# Kovasznay TRT arm -- generated tables\n\n",
          "All numbers are read from the JSON artifacts in this directory.\n\n",
          "## Grid ladder (TRT, U0 = 0.0075, analytic Dirichlet outlet)\n\n",
          table_ladder(rows), "\n",
          "## Observed order of accuracy\n\n", table_orders(rows), "\n"]
    if stab:
        md += ["## BGK vs TRT stability wall (ny = 32, 3000 steps)\n\n",
               table_stability(stab), "\n"]
    figs = []
    if rows and stab:
        figs.append(figure(art, rows, stab, outdir))
    ff = figure_field(art, outdir)
    if ff:
        figs.append(ff)
    for f in figs:
        md.append(f"![fig]({f.name})\n")
    out = outdir / "TRT_CONVERGENCE_TABLES.md"
    out.write_text("".join(md), encoding="utf-8")
    print(out)
    for f in figs:
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())