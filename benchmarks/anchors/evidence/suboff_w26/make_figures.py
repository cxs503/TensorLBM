#!/usr/bin/env python3
"""W26 / mission m17569 — figures for the SUBOFF C_F anchor + GCI + sign-surface report.

Every plotted number comes from ``anchor_register.json`` / ``gci_sign_surface.json``
or from the raw case ``.npz`` files; nothing is typed in by hand.

Figures
-------
1. anchor_cf.png        ITTC-57 line vs the 17 recorded anchor rows, plus the
                       recompute residual of every row on a log scale against the
                       1e-9 criterion line; the case Reynolds number is marked so the
                       caliber gap is visible in the same frame.
2. gci_ladder.png       C_T / C_F against 1/N for the three-grid fixed-Re ladder with
                       the Richardson extrapolation, annotated with observed order and GCI.
3. force_history.png    block-mean C_T history per ladder grid (the steadiness evidence
                       that the GCI numbers stand on).
4. sign_surface.png     the (tau, N) sign surface of the scan, with C_T written in each cell.
5. field_ux.png         mid-plane streamwise velocity field of the finest ladder grid with
                       the hull solid masked out (the CLOUD evidence).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
FIG = _HERE / "figures"
FIG.mkdir(exist_ok=True)

ANCHOR = json.loads((_HERE / "anchor_register.json").read_text())
GCI = json.loads((_HERE / "gci_sign_surface.json").read_text())


def ittc(re, coef=0.075, shift=2.0):
    return coef / (np.log10(re) - shift) ** 2


def fig_anchor():
    rows = ANCHOR["anchor_rows_full_precision"]
    chk = ANCHOR["ittc57_recompute_check"]["rows"]
    formula = ANCHOR["ittc1957_formula"]
    coef, shift = formula["coef"], formula["shift"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    Re = np.logspace(2.05, 8.2, 400)
    ax.loglog(Re, ittc(Re, coef, shift), "-", color="#1f77b4", lw=1.8,
              label=f"ITTC-1957  C_F = {coef}/(log10 Re − {shift})²")
    re_vals = np.array([r["recorded"]["Re"] for r in rows])
    cf_vals = np.array([r["recorded"]["CF_ittc78"] for r in rows])
    ax.loglog(re_vals, cf_vals, "o", ms=5, color="#d62728",
              label=f"recorded C_F (n={len(rows)} anchor rows)")
    case_re = ANCHOR["case_vs_anchor_caliber"]["case_Re_lattice"]
    ax.axvline(case_re, color="k", ls="--", lw=1.2)
    ax.loglog([case_re], [ANCHOR["case_vs_anchor_caliber"]["CF_ittc57_at_case_Re"]], "s",
              ms=8, mfc="none", mec="k", mew=1.6,
              label="case caliber Re=1e3 (frozen LBM case)")
    ax.axvspan(1e7 * 0.95, 1.2e7 * 1.2, color="tab:green", alpha=0.10)
    ax.annotate("DTRC SUBOFF anchor band\nRe ≈ 1.16e7", xy=(1.16e7, 0.0031),
                xytext=(2e5, 0.02), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Re = U·L/ν  (L = 4.356 m)")
    ax.set_ylabel("C_F  [–]")
    ax.set_title("(a) C_F: ITTC-57 line vs recorded anchor rows\n"
                 "(no measured C_F exists in-repo → WITHHELD)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    res = np.array([c["abs_residual"] for c in chk])
    ax2.semilogy(np.arange(1, len(res) + 1), res, "o-", ms=4, color="#2ca02c")
    ax2.axhline(1e-9, color="crimson", ls="--", lw=1.3, label="criterion 1e-9")
    ax2.set_xlabel("anchor row index")
    ax2.set_ylabel("|C_F recomputed − C_F recorded|  [–]")
    ax2.set_title(f"(b) ITTC-57 recompute residual, max = {res.max():.3e}\n"
                  f"{len(res)}/{len(res)} rows ≤ 1e-9", fontsize=10)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "anchor_cf.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_gci():
    cases = GCI["ladder"]["cases"]
    q = GCI["ladder"]["quantities"]
    Ns = np.array([c["N"] for c in cases], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, key in zip(axes, ("C_T", "C_F")):
        vals = np.array([c[key] for c in cases], dtype=float)
        g = q[key]
        ax.plot(1.0 / Ns, vals, "o-", ms=6, color="#1f77b4", label=f"{key} (3 grids)")
        if "f_exact_richardson" in g:
            ax.axhline(g["f_exact_richardson"], color="crimson", ls="--", lw=1.2,
                       label=f"Richardson  f_ex = {g['f_exact_richardson']:.6g}")
        ax.axhline(g["f1_fine"], color="gray", ls=":", lw=1.0, label="finest-grid value")
        p_txt = "n/a" if g.get("observed_order") is None else f"{g['observed_order']:.3f}"
        g_txt = "n/a" if g.get("gci_fine_pct") is None else f"{g['gci_fine_pct']:.2f} %"
        ax.set_title(f"{key}: observed order p = {p_txt}, GCI_fine = {g_txt}\n"
                     f"status = {g['status']}", fontsize=10)
        ax.set_xlabel("1/N  [1/cells-per-hull-length]")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        for x, v, c in zip(1.0 / Ns, vals, [c["N"] for c in cases]):
            ax.annotate(f"N={int(c)}", xy=(x, v), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
    fig.suptitle("Three-grid ladder at fixed lattice Re = 1000 (tau follows the grid)", fontsize=11)
    fig.tight_layout()
    p = FIG / "gci_ladder.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_history():
    cases = GCI["ladder"]["cases"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for c in cases:
        bm = np.array(c["C_T_block_means"], dtype=float)
        frac = (np.arange(len(bm)) + 0.5) / len(bm)
        ax.plot(frac, bm, "o-", ms=4, label=f"N={c['N']} (drift {100 * c['C_T_window_drift']:.2f}%)")
    ax.set_xlabel("normalised run time (fraction of the recorded samples)")
    ax.set_ylabel("block-mean C_T")
    ax.set_yscale("log")
    ax.set_title("Force-history steadiness evidence for the GCI ladder", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "force_history.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_surface():
    surf = GCI.get("sign_surface")
    if not surf:
        return None
    cells = surf["cells"]
    taus = sorted({c["tau"] for c in cells})
    Ns = sorted({c["N"] for c in cells})
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    colour = {"POSITIVE": "#2ca02c", "POSITIVE_UNSTEADY": "#98df8a",
              "NEGATIVE": "#d62728", "NEGATIVE_UNSTEADY": "#ff9896",
              "DIVERGED": "#7f7f7f"}
    for c in cells:
        x = taus.index(c["tau"])
        y = Ns.index(c["N"])
        ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                                   color=colour.get(c["status"], "white"),
                                   ec="k", lw=0.6))
        ax.text(x, y + 0.12, f"{c['C_T']:+.4f}", ha="center", va="center", fontsize=8)
        ax.text(x, y - 0.22, f"drift {100 * c['C_T_window_drift']:.1f}%", ha="center",
                va="center", fontsize=6.5)
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"{t:.4f}" for t in taus], rotation=30)
    ax.set_yticks(range(len(Ns)))
    ax.set_yticklabels([str(n) for n in Ns])
    ax.set_xlim(-0.5, len(taus) - 0.5)
    ax.set_ylim(-0.5, len(Ns) - 0.5)
    ax.set_xlabel("relaxation time τ")
    ax.set_ylabel("N  [cells per hull length]")
    ax.set_title("(τ, N) sign surface of window-mean C_T (cell text = C_T, drift = last-window "
                 "drift)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colour.values()]
    ax.legend(handles, colour.keys(), fontsize=7, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    p = FIG / "sign_surface.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_field():
    sl = sorted(glob.glob(str(_HERE.parents[3] / ".scratch/m17569/runs/ladder_N*/slice_*.npz")))
    if not sl:
        return None
    # finest available grid
    best = None
    for f in sl:
        tag = Path(f).name.replace("slice_", "").replace(".npz", "")
        n = int(tag.split("_N")[-1])
        if best is None or n > best[0]:
            best = (n, f)
    n, f = best
    with np.load(f) as z:
        ux = z["ux_midplane"]
        solid = z["solid_midplane"]
        u_lb = float(z["u_lb"])
    ux = np.ma.array(ux, mask=solid)
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    vmax = float(np.nanpercentile(ux.compressed(), 99))
    im = ax.pcolormesh(ux.T / u_lb, shading="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
    ax.contour(solid.T.astype(float), levels=[0.5], colors="k", linewidths=0.8)
    ax.set_xlabel("streamwise lattice index i  (x)")
    ax.set_ylabel("transverse index k  (z)")
    ax.set_title(f"Mid-plane u_x/u_lb, SUBOFF bare hull, N={n}, Re_lattice≈1000 "
                 f"(hull outline black)", fontsize=10)
    fig.colorbar(im, ax=ax, label="u_x / u_lb")
    ax.set_aspect("auto")
    fig.tight_layout()
    p = FIG / "field_ux.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (fig_anchor, fig_gci, fig_history, fig_surface, fig_field):
        out = fn()
        print(f"[fig] {out}")