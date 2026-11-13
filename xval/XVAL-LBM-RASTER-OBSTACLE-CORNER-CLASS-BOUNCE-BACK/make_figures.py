"""Figures for XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK.

    python make_figures.py --results results.json --root runs --figdir figures

Produces (all numbers come from ``results.json`` / the run npz files, nothing is
typed by hand):

* ``fig1_fields.png``   u_x field of both arms + their difference (baseline C1)
* ``fig2_corner_links.png``  the link-level difference map on one rectangle
* ``fig3_convergence.png``   K(t) of both arms + K/b^2 with the error bars of the ladder
* ``fig4_anchor.png``        C0 velocity profile vs the discrete Poiseuille solution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from raster_case import (  # noqa: E402
    Case,
    link_difference,
    oss_rect_corner_list,
    raster_solid_mask,
)
from run_xval import CONFIGS  # noqa: E402


def _load(root: Path, cfg: str, arm: str):
    with np.load(root / f"{cfg}_{arm}" / "fields.npz") as z:
        return dict(ux=z["ux"], rho=z["rho"], solid=z["solid"], history=z["history"])


def fig_fields(root: Path, figdir: Path, cfg: str = "C1") -> Path:
    o = _load(root, cfg, "oss")
    m = _load(root, cfg, "mask")
    fluid = ~m["solid"]
    vmax = max(np.abs(o["ux"][fluid]).max(), np.abs(m["ux"][fluid]).max())
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.6), sharex=True, sharey=True)
    im0 = axes[0].pcolormesh(o["ux"].T, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
    axes[0].set_title(f"OSS arm (corner-class table), {cfg}: u_x")
    fig.colorbar(im0, ax=axes[0], label="u_x")
    im1 = axes[1].pcolormesh(m["ux"].T, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
    axes[1].set_title("TensorLBM arm (exact closed-link mask): u_x")
    fig.colorbar(im1, ax=axes[1], label="u_x")
    d = np.where(fluid, m["ux"] - o["ux"], np.nan)
    lim = max(np.nanmax(np.abs(d)), 1e-30)
    im2 = axes[2].pcolormesh(d.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    axes[2].set_title("difference  u_x(mask) - u_x(OSS)  [same colour scale on purpose]")
    fig.colorbar(im2, ax=axes[2], label="delta u_x")
    for ax in axes:
        ax.set_ylabel("y [lattice]")
    axes[-1].set_xlabel("x [lattice, periodic]")
    p = figdir / "fig1_fields.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_corner_links(figdir: Path, cfg: str = "C1") -> Path:
    case = Case(**CONFIGS[cfg])
    rects = oss_rect_corner_list(case.nx, case.ny, case.b)
    solid = raster_solid_mask(case.nx, case.ny, rects)
    d = link_difference(solid, rects)
    (x0, y0), (x1, y1) = rects[0]
    xlo, xhi = x0 - 2, x1 + 3
    ylo, yhi = y0 - 2, y1 + 3
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1,
                               facecolor="0.75", edgecolor="0.3", label="raster solid cells"))
    styles = {True: dict(color="tab:red", lw=1.6, alpha=0.9), False: dict(color="tab:green", lw=1.0, alpha=0.9)}
    from tensorlbm.bitflag_neighbour_mask import D2Q9_C as _C
    from raster_case import D2Q9_C  # OSS ordering (1=E,2=N,...)

    n_exact = n_spur = 0
    for x in range(xlo, xhi + 1):
        for y in range(ylo, yhi + 1):
            if solid[x, y]:
                continue
            for i in range(1, 9):
                bit = 1 << (i - 1)
                is_exact = int(d["exact_mask"][x, y]) & bit
                is_spur = int(d["spurious_mask"][x, y]) & bit
                if not (is_exact or is_spur):
                    continue
                cx, cy = D2Q9_C[i]
                spurious = bool(is_spur)
                n_spur += int(spurious)
                n_exact += int(bool(is_exact))
                ax.annotate("", xy=(x + 0.5 * cx, y + 0.5 * cy), xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", **styles[spurious]))
    ax.plot([], [], color="tab:green", label=f"exact closed link ({n_exact} shown)")
    ax.plot([], [], color="tab:red", label=f"link bounced only by the OSS table ({n_spur} shown)")
    ax.set_xlim(xlo - 1, xhi + 1)
    ax.set_ylim(ylo - 1, yhi + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [lattice, periodic]")
    ax.set_ylabel("y [lattice]")
    ax.set_title(f"Corner-class table vs exact closed links, one {case.b}x{case.b} block\n"
                 f"table = {d['table_links']} links/geometry, exact = {d['exact_links']}, "
                 f"spurious = {d['spurious_links']}, missed = {d['missed_links']}")
    ax.legend(loc="upper left", fontsize=8)
    p = figdir / "fig2_corner_links.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_convergence(root: Path, results: dict, figdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for arm, style in (("oss", dict(color="tab:red", label="OSS corner-class table")),
                       ("mask", dict(color="tab:blue", label="TensorLBM exact mask"))):
        h = _load(root, "C1", arm)["history"]
        axes[0].plot(h[:, 0], h[:, 2], **style)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("K [lattice units]")
    axes[0].set_title("permeability history, C1")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    rows = [r for r in results["rows"] if r["status"] == "ok"]
    labels = [f"{r['config']}\nb={r['b']}, tau={r['tau']}, g={r['g']:g}" for r in rows]
    ks = [r["k_mask_norm"] for r in rows]
    ko = [r["k_oss_norm"] for r in rows]
    xs = np.arange(len(rows))
    axes[1].bar(xs - 0.18, ko, width=0.36, color="tab:red", label="OSS")
    axes[1].bar(xs + 0.18, ks, width=0.36, color="tab:blue", label="TensorLBM")
    for i, r in enumerate(rows):
        axes[1].annotate(f"{r['err_k_pct']:.1f}%", (i, max(ko[i], ks[i])), ha="center",
                         va="bottom", fontsize=8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("K / b^2  [-]")
    axes[1].set_title("ladder: porosity-normalised permeability, label = |delta|/OSS")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    p = figdir / "fig3_convergence.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_anchor(results: dict, figdir: Path) -> Path:
    a = results["anchor"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    any_arm = next(v for v in a.values() if isinstance(v, dict))
    ax.plot(any_arm["u_profile_analytic"], np.arange(len(any_arm["u_profile_analytic"])),
            "k-", lw=2, label="discrete Poiseuille  g/(2 nu) y(ny-y)")
    for arm, style in (("oss", dict(color="tab:red", ls="--", label="OSS arm")),
                       ("mask", dict(color="tab:blue", ls=":", label="TensorLBM arm"))):
        if not a.get(arm):
            continue
        u = np.array(a[arm]["u_profile_sim"])
        ax.plot(u, np.arange(len(u)), **style)
    ax.set_xlabel("u_x [lattice]")
    ax.set_ylabel("y [lattice]")
    txt = "\n".join(
        f"{arm}: K err {a[arm]['err_k_pct']:.3f} %, profile err {a[arm]['err_profile_max_pct']:.3f} %"
        for arm in ("oss", "mask") if a.get(arm)
    )
    ax.set_title("C0 analytic anchor (no obstacles)\n" + txt)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = figdir / "fig4_anchor.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results.json"))
    ap.add_argument("--root", default=str(HERE / "runs"))
    ap.add_argument("--figdir", default=str(HERE / "figures"))
    args = ap.parse_args(argv)
    results = json.loads(Path(args.results).read_text())
    root, figdir = Path(args.root), Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    made = []
    try:
        made.append(str(fig_fields(root, figdir)))
    except Exception as exc:
        print(f"[fig1] skipped: {exc!r}")
    try:
        made.append(str(fig_corner_links(figdir)))
    except Exception as exc:
        print(f"[fig2] skipped: {exc!r}")
    try:
        made.append(str(fig_convergence(root, results, figdir)))
    except Exception as exc:
        print(f"[fig3] skipped: {exc!r}")
    try:
        made.append(str(fig_anchor(results, figdir)))
    except Exception as exc:
        print(f"[fig4] skipped: {exc!r}")
    print("[figures] " + ", ".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())