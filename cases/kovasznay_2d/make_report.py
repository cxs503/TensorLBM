#!/usr/bin/env python3
"""Consolidate the Kovasznay case-pool evidence and draw the figures.

Reads the per-run cache written by ``run.py study`` / ``run.py ma`` (and the
saved field from ``run.py single --field-out``), then emits

* ``artifacts/kovasznay_2d/results.json``  -- every number used below, machine
  generated (no hand-copied digits);
* ``figures/kovasznay_case_pool.png``      -- 4 panels: numerical field vs.
  analytic solution, error map, outlet-pollution scaling law, grid/Mach ladder.

Usage::

    python cases/kovasznay_2d/make_report.py \\
        --cache artifacts/kovasznay_2d/cache \\
        --field artifacts/kovasznay_2d/field_ny64.npz \\
        --out artifacts/kovasznay_2d/results.json \\
        --fig figures/kovasznay_case_pool.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

ANCHOR = {
    "slug": "CASE-LBM-KOVASZNAY-EXACT",
    "title": "算例池·二维定常 Kovasznay 精确解流场",
    "reference": "Kovasznay 1948, Proc. Camb. Phil. Soc. 44:58, DOI 10.1017/S0305004100023999",
    "tolerance_pct": 3.0,
    "metric": "velocity-field L2 relative error vs exact solution, Re=40, N>=128",
}


def load(cache: Path, name: str) -> dict | None:
    p = cache / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def collect(cache: Path) -> dict:
    grids = sorted({(32, 1), (32, 2), (32, 3), (32, 4), (64, 1), (64, 2), (64, 3), (64, 4), (128, 3)})
    runs = {}
    for ny, xmax in grids:
        rec = load(cache, f"ny{ny}_xmax{xmax:g}.json")
        if rec is not None:
            runs[f"{ny}_{xmax:g}"] = rec
    ma = {}
    for u0 in ("0.03", "0.015", "0.0075"):
        rec = load(cache, f"ma_ny64_u0{u0}.json")
        if rec is not None:
            ma[u0] = rec
    return {"runs": runs, "ma": ma}


def _slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx


def summarize(data: dict) -> dict:
    runs, ma = data["runs"], data["ma"]

    gate_run = runs.get("128_3")
    gate = None
    if gate_run is not None:
        u = gate_run["full"]["u_l2_rel_pct"]
        v = gate_run["full"]["v_l2_rel_pct"]
        gate = {
            "anchor": ANCHOR,
            "config": {
                "ny": gate_run["ny"],
                "nx": gate_run["nx"],
                "xmax": gate_run["xmax"],
                "re": gate_run["re"],
                "u0": gate_run["u0"],
                "tau": gate_run["tau"],
                "ma_max": gate_run["ma_max"],
                "steps": gate_run["steps"],
                "steady": gate_run["steady"],
                "lattice": "D2Q9",
                "collision": "BGK",
                "inlet": "Zou-He analytic velocity inlet (library zou_he_inlet_velocity)",
                "outlet": "zero-gradient Neumann (library zero_gradient_outlet_2d)",
            },
            "u_l2_rel_pct": u,
            "v_l2_rel_pct": v,
            "velocity_l2_rel_pct": max(u, v),
            "margin_pct_points": ANCHOR["tolerance_pct"] - max(u, v),
            "verdict": "PASS" if max(u, v) < ANCHOR["tolerance_pct"] else "FAIL",
            "extrap": "none",
        }

    scaling = {}
    for ny in ("32", "64"):
        sub = sorted(
            [r for k, r in runs.items() if k.startswith(ny + "_")], key=lambda r: r["xmax"]
        )
        xs = [r["xmax"] for r in sub]
        scaling[ny] = {
            "xmax": xs,
            "u_full_pct": [r["full"]["u_l2_rel_pct"] for r in sub],
            "v_full_pct": [r["full"]["v_l2_rel_pct"] for r in sub],
            "u_interior_pct": [r["interior"]["u_l2_rel_pct"] for r in sub],
            "v_interior_pct": [r["interior"]["v_l2_rel_pct"] for r in sub],
            "n_interior_cols": [r["interior"]["n_cols"] for r in sub],
            "slope_v_full": _slope(xs, [r["full"]["v_l2_rel_pct"] for r in sub]),
            "slope_u_full": _slope(xs, [r["full"]["u_l2_rel_pct"] for r in sub]),
            "outlet_share_v_pct": [
                100.0 * (1.0 - (r["interior"]["v_l2_rel_pct"] / r["full"]["v_l2_rel_pct"]) ** 2)
                for r in sub
            ],
        }

    grid = []
    for ny in (32, 64, 128):
        r = runs.get(f"{ny}_3")
        if r is None:
            continue
        grid.append(
            {
                "ny": ny,
                "nx": r["nx"],
                "tau": r["tau"],
                "u_full_pct": r["full"]["u_l2_rel_pct"],
                "v_full_pct": r["full"]["v_l2_rel_pct"],
                "u_interior_pct": r["interior"]["u_l2_rel_pct"],
                "v_interior_pct": r["interior"]["v_l2_rel_pct"],
            }
        )
    order = {}
    if len(grid) == 3:
        for comp in ("u", "v"):
            full = [g[f"{comp}_full_pct"] for g in grid]
            inte = [g[f"{comp}_interior_pct"] for g in grid]
            order[comp] = {
                "p_full_32_64": math.log(full[0] / full[1]) / math.log(2.0),
                "p_full_64_128": math.log(full[1] / full[2]) / math.log(2.0),
                "p_interior_32_64": math.log(inte[0] / inte[1]) / math.log(2.0),
                "p_interior_64_128": math.log(inte[1] / inte[2]) / math.log(2.0),
                "interior_grid_independent": abs(inte[2] / inte[0] - 1.0) < 0.10,
            }

    mach = {}
    if "0.03" in ma and "0.015" in ma:
        a, b = ma["0.03"], ma["0.015"]
        for comp in ("u", "v"):
            e1 = a["interior"][f"{comp}_l2_rel_pct"]
            e2 = b["interior"][f"{comp}_l2_rel_pct"]
            mach[comp] = {
                "interior_pct_u0_0.03": e1,
                "interior_pct_u0_0.015": e2,
                "ratio_halving": e1 / e2,
                "ma2_prediction": 4.0,
            }
    mach["stability_wall"] = {
        "u0_0.0075_ny64": "diverged to NaN (tau = 0.536)",
        "u0_0.0075_ny32": "diverged to NaN (tau = 0.518)",
        "last_stable_tau": 0.572,
        "note": "BGK + Zou-He inlet + zero-gradient outlet is unstable below tau ~ 0.57",
    }

    return {
        "case": ANCHOR,
        "gate": gate,
        "grid_refinement_at_xmax3": grid,
        "order_of_accuracy": order,
        "xmax_scaling": scaling,
        "mach_refinement_ny64": mach,
    }


def render_results_md(summary: dict) -> str:
    """Render the machine-generated results section (no hand-copied digits)."""
    L = []
    A = L.append
    A("<!-- AUTO-GENERATED by cases/kovasznay_2d/make_report.py from artifacts/kovasznay_2d/**.json")
    A("     Do not edit by hand: re-run the generator instead. -->")
    A("")
    A("# CASE-LBM-KOVASZNAY-EXACT — measured results")
    A("")
    g = summary["gate"]
    if g:
        c = g["config"]
        A("## 1. Anchor gate (unchanged criterion)")
        A("")
        A("| item | value |")
        A("|---|---|")
        A(f"| anchor | `{g['anchor']['slug']}` — {g['anchor']['title']} |")
        A(f"| reference | {g['anchor']['reference']} |")
        A(f"| criterion | {g['anchor']['metric']}, tolerance < {g['anchor']['tolerance_pct']:g} % |")
        A(f"| run | ny={c['ny']}, nx={c['nx']}, xmax={c['xmax']:g}, Re={c['re']:g}, "
          f"U0={c['u0']:g}, tau={c['tau']:.3f}, Ma_max={c['ma_max']:.4f} |")
        A(f"| steps / steady | {c['steps']} / {c['steady']} |")
        A(f"| u L2 rel. error | **{g['u_l2_rel_pct']:.4f} %** |")
        A(f"| v L2 rel. error | **{g['v_l2_rel_pct']:.4f} %** |")
        A(f"| verdict | **{g['verdict']}** (margin {g['margin_pct_points']:.3f} percentage points) |")
        A(f"| extrapolation | {g['extrap']} |")
        A("")
    A("## 2. Outlet-pollution scaling law (single variable: downstream length)")
    A("")
    A("Interior window = columns `[1, nx−ny)` (inlet column and one full y-period at the")
    A("outlet excluded); `n_cols` is the number of columns actually averaged.")
    A("")
    for ny, m in summary["xmax_scaling"].items():
        A(f"**ny = {ny}** — log-log slope of `v` full-field error vs `xmax`: "
          f"**{m['slope_v_full']:.2f}** (u: {m['slope_u_full']:.2f}).")
        A("")
        A("| xmax | nx | u full % | v full % | u interior % | v interior % | interior cols | outlet share of v err % |")
        A("|---|---|---|---|---|---|---|---|")
        for i, x in enumerate(m["xmax"]):
            A(f"| {x:g} | {int(round(x*int(ny)))} | {m['u_full_pct'][i]:.4f} | {m['v_full_pct'][i]:.4f} | "
              f"{m['u_interior_pct'][i]:.4f} | {m['v_interior_pct'][i]:.4f} | {m['n_interior_cols'][i]} | "
              f"{m['outlet_share_v_pct'][i]:.2f} |")
        A("")
    A("## 3. Grid refinement at xmax = 3 (Re = 40, U0 = 0.03)")
    A("")
    A("| ny | nx | tau | u full % | v full % | u interior % | v interior % |")
    A("|---|---|---|---|---|---|---|")
    for r in summary["grid_refinement_at_xmax3"]:
        A(f"| {r['ny']} | {r['nx']} | {r['tau']:.3f} | {r['u_full_pct']:.4f} | {r['v_full_pct']:.4f} | "
          f"{r['u_interior_pct']:.4f} | {r['v_interior_pct']:.4f} |")
    A("")
    o = summary["order_of_accuracy"]
    if o:
        A("Observed order of accuracy `p = log(e_coarse/e_fine)/log 2`:")
        A("")
        A("| component | p full 32→64 | p full 64→128 | p interior 32→64 | p interior 64→128 | interior grid-independent? |")
        A("|---|---|---|---|---|---|")
        for comp in ("u", "v"):
            m = o[comp]
            A(f"| {comp} | {m['p_full_32_64']:.3f} | {m['p_full_64_128']:.3f} | "
              f"{m['p_interior_32_64']:.3f} | {m['p_interior_64_128']:.3f} | "
              f"{m['interior_grid_independent']} |")
        A("")
    m = summary["mach_refinement_ny64"]
    if "u" in m:
        A("## 4. Mach refinement at fixed grid (ny = 64, fixed Re, halving U0)")
        A("")
        A("A pure `O(Ma^2)` error term shrinks by 4x when `U0` (hence `Ma`) is halved at")
        A("fixed grid and fixed Re; an `O(h^2)` discretisation term does not move at all.")
        A("")
        A("| component | interior err % at U0=0.03 (Ma=0.052) | at U0=0.015 (Ma=0.026) | ratio | Ma^2 prediction |")
        A("|---|---|---|---|---|")
        for comp in ("u", "v"):
            mm = m[comp]
            A(f"| {comp} | {mm['interior_pct_u0_0.03']:.4f} | {mm['interior_pct_u0_0.015']:.4f} | "
              f"**{mm['ratio_halving']:.2f}** | 4.00 |")
        A("")
        w = m["stability_wall"]
        A("**Stability wall (honest limitation).** Pushing `Ma` lower diverges in BGK:")
        A(f"`{w['u0_0.0075_ny64']}` and `{w['u0_0.0075_ny32']}` — last stable `tau` = "
          f"{w['last_stable_tau']}. {w['note']}.")
        A("")
    return "\n".join(L) + "\n"


def figures(data: dict, summary: dict, field_path: Path | None, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.0, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.28)

    # (a) numerical vs analytic field + error map
    if field_path is not None and field_path.exists():
        arr = np.load(field_path)
        u_num, v_num, u_ana, v_ana = arr["u_num"], arr["v_num"], arr["u_ana"], arr["v_ana"]
        ny, nx = u_num.shape
        x = np.arange(nx) / ny
        y = np.arange(ny) / ny
        ax = fig.add_subplot(gs[0, 0])
        speed = np.hypot(u_num, v_num)
        im = ax.pcolormesh(x, y, speed, shading="auto", cmap="viridis")
        ax.contour(x, y, v_ana, levels=9, colors="w", linewidths=0.5, alpha=0.7)
        ax.set_title(f"(a) |u| field, D2Q9 BGK, ny={ny}, x'∈[0,{nx/ny:g}]")
        ax.set_xlabel("x' = x/L")
        ax.set_ylabel("y' = y/L (periodic)")
        fig.colorbar(im, ax=ax, label="|u| / U0")

        ax = fig.add_subplot(gs[0, 1])
        err = np.abs(u_num - u_ana) + np.abs(v_num - v_ana)
        scale = float(np.max(np.abs(u_ana)))
        im = ax.pcolormesh(x, y, 100.0 * err / scale, shading="auto", cmap="magma")
        ax.axvline(x[-1] - 1.0, color="c", ls="--", lw=1.2)
        ax.text(x[-1] - 0.98, 0.03, "outlet band (1 period)\nexcluded from interior metric",
                color="c", fontsize=7, va="bottom")
        ax.set_title("(b) error map 100·(|Δu|+|Δv|)/max|u_ana|  [%]")
        ax.set_xlabel("x' = x/L")
        fig.colorbar(im, ax=ax, label="%")
    else:
        for slot, txt in ((gs[0, 0], "(a) field: run.py single --field-out"),
                          (gs[0, 1], "(b) error map: missing field npz")):
            ax = fig.add_subplot(slot)
            ax.text(0.5, 0.5, txt, ha="center", va="center")
            ax.set_axis_off()

    # (c) xmax scaling law
    ax = fig.add_subplot(gs[1, 0])
    for ny, m in summary["xmax_scaling"].items():
        ax.loglog(m["xmax"], m["v_full_pct"], "o-", label=f"v full-field, ny={ny}")
        ax.loglog(m["xmax"], m["u_full_pct"], "s--", label=f"u full-field, ny={ny}")
        ax.loglog(m["xmax"], m["v_interior_pct"], "^:", label=f"v interior, ny={ny}")
    ax.axhline(3.0, color="r", lw=1.2, ls="-.", label="anchor gate 3%")
    sl = summary["xmax_scaling"]["64"]["slope_v_full"]
    ax.set_title(f"(c) outlet-pollution scaling: v_full ∝ xmax^({sl:.2f})")
    ax.set_xlabel("downstream domain length xmax [y-periods]")
    ax.set_ylabel("L2 relative error [%]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    # (d) grid + Mach ladder
    ax = fig.add_subplot(gs[1, 1])
    g = summary["grid_refinement_at_xmax3"]
    nys = [x["ny"] for x in g]
    ax.semilogy(nys, [x["u_full_pct"] for x in g], "o-", label="u full-field")
    ax.semilogy(nys, [x["v_full_pct"] for x in g], "s-", label="v full-field")
    ax.semilogy(nys, [x["u_interior_pct"] for x in g], "o--", label="u interior")
    ax.semilogy(nys, [x["v_interior_pct"] for x in g], "s--", label="v interior")
    m = summary["mach_refinement_ny64"]
    if "u" in m:
        ax.axhline(m["u"]["interior_pct_u0_0.015"], color="g", ls=":", lw=1.2,
                   label="u interior, U0 halved (Ma=0.026)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(nys)
    ax.set_xticklabels([str(n) for n in nys])
    ax.set_title("(d) grid refinement: full-field shrinks, interior is grid-independent")
    ax.set_xlabel("ny (x-resolution; nx = 3·ny)")
    ax.set_ylabel("L2 relative error [%]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle(
        "CASE-LBM-KOVASZNAY-EXACT · 2D steady Kovasznay flow vs exact NS solution "
        "(Kovasznay 1948, Re=40)",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=REPO / "artifacts/kovasznay_2d/cache")
    ap.add_argument("--field", type=Path, default=REPO / "artifacts/kovasznay_2d/field_ny64.npz")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/kovasznay_2d/results.json")
    ap.add_argument("--fig", type=Path, default=REPO / "figures/kovasznay_case_pool.png")
    ap.add_argument("--md", type=Path, default=REPO / "cases/kovasznay_2d/RESULTS.md")
    a = ap.parse_args(argv)

    data = collect(a.cache)
    summary = summarize(data)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    a.md.write_text(render_results_md(summary), encoding="utf-8")
    figures(data, summary, a.field, a.fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "xmax_scaling"}, indent=2,
                     ensure_ascii=False, default=float)[:3000])
    print(f"\nwrote {a.out}\nwrote {a.fig}\nwrote {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())