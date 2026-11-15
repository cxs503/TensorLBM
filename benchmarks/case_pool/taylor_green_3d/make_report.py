#!/usr/bin/env python3
"""CASE-TAYLOR-GREEN-3D -- artifacts/summary.json -> tables + PNGs + summary.md.

Every number written into the markdown comes from the artifact; nothing is typed
by hand.  The only literal text is the interpretation, which quotes numbers via
``fmt`` calls on the loaded dict.

    PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent
ART = CASE_DIR / "artifacts"
FIG = ART / "figures"


def _load() -> dict:
    return json.loads((ART / "summary.json").read_text())


def fig_decay(s: dict) -> Path:
    h = s["legs"]["primary_history"]
    t = [r["eddy_time"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for a, key_l, key_r, lab in (
        (ax[0], "energy_ratio_lbm", "energy_ratio_series", "kinetic energy  E(t)/E(0)"),
        (ax[1], "enstrophy_ratio_lbm", "enstrophy_ratio_series", "enstrophy  Ω(t)/Ω(0)"),
    ):
        a.plot(t, [r[key_r] for r in h], "k-", lw=3.0, alpha=0.35,
               label="exact time-Taylor series")
        a.plot(t, [r[key_l] for r in h], "o-", ms=3.5, lw=1.2, label="D3Q27 BGK (float64)")
        a.set_xlabel("eddy time  $A k t$")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    p = s["legs"]["primary"]
    fig.suptitle(
        f"Taylor-Green 3-D, N={p['n']}³, Re={p['reynolds']:.0f}, ν={p['nu']:.4g}: "
        f"ΔE={p['energy_ratio_rel_err_pct']:+.2f}%, ΔΩ={p['enstrophy_ratio_rel_err_pct']:+.2f}%",
        fontsize=10,
    )
    out = FIG / "decay_vs_series.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_grid(s: dict) -> Path:
    g = s["legs"]["grid"]
    n = np.array(g["grids"], dtype=float)
    e = np.array(g["energy_ratio_rel_err_pct"])
    o = np.array(g["enstrophy_ratio_rel_err_pct"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(n, e, "o-", label="energy decay ratio")
    ax.loglog(n, o, "s-", label="enstrophy ratio")
    # reference slopes through the finest point
    nn = np.array([n[-2], n[-1]])
    for arr, style in ((e, "k:"), (o, "k:")):
        ax.loglog(nn, arr[-1] * (nn / n[-1]) ** -2.0, style, lw=1.0, alpha=0.6)
    ax.axhline(8.0, color="tab:red", ls="--", lw=1.0, label="pool band 8 %")
    ax.set_xlabel("grid N (fixed Re = 160)")
    ax.set_ylabel("|relative error| vs exact series  [%]")
    ax.set_title(f"grid convergence, observed order = {g['observed_order']:.2f}\n"
                 f"(dotted guide: slope -2)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    out = FIG / "grid_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_field() -> Path | None:
    f = ART / "field_primary.npz"
    if not f.is_file():
        return None
    d = np.load(f)
    lbm, ser = d["wz_lbm_midplane"], d["wz_series_midplane"]
    diff = lbm - ser
    vmax = float(np.max(np.abs(ser))) or 1.0
    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.9))
    panels = ((ser, "series (exact)", -vmax, vmax),
              (lbm, "D3Q27 LBM", -vmax, vmax),
              (diff, "LBM - series", -0.15 * vmax, 0.15 * vmax))
    for a, (arr, title, lo, hi) in zip(ax, panels):
        im = a.imshow(arr, origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi)
        a.set_title(title, fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"ω_z mid-plane at T={float(d['eddy_time']):.2f} eddy times "
        f"(N={int(d['n'])}³); difference panel scaled ×0.15", fontsize=10)
    out = FIG / "vorticity_midplane.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_reference(s: dict) -> Path | None:
    rc = s["legs"].get("reference_convergence")
    if not rc:
        return None
    rows = rc["variants"]
    labels = [f"order {r['order']}\nnfft {r['nfft']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(x, [r["energy_ratio"] for r in rows], "o-", label="E(T)/E(0)")
    ax2 = ax.twinx()
    ax2.plot(x, [r["enstrophy_ratio"] for r in rows], "s-", color="tab:red",
             label="Ω(T)/Ω(0)")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("reference E(T)/E(0)")
    ax2.set_ylabel("reference Ω(T)/Ω(0)", color="tab:red")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax2.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_title(
        f"reference self-convergence: order spread {rc['order_convergence_rel']:.2e}, "
        f"work-grid spread {rc['workgrid_invariance_rel']:.2e}", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "reference_convergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def write_markdown(s: dict, figs: dict[str, Path | None]) -> Path:
    p = s["protocol"]
    leg = s["legs"]
    prim = leg["primary"]
    g = leg["grid"]
    rows = "\n".join(
        f"| {q['id']} | {q['ref']:.6g} | {q['got']:.6g} | {q['direction']} {q['tolerance']:g} "
        f"| {'PASS' if q['pass'] else '**FAIL**'} |"
        for q in s["quantities"]
    )
    ladder = "\n".join(
        f"| {n} | {e:.3f} | {o:.3f} | {u:.3f} |"
        for n, e, o, u in zip(g["grids"], g["energy_ratio_rel_err_pct"],
                              g["enstrophy_ratio_rel_err_pct"], g["velocity_l2_rel_err_pct"])
    )
    fig_lines = "\n".join(
        f"* `{name}` — {path.name if path else 'not generated'}"
        for name, path in figs.items()
    )
    md = f"""# CASE-TAYLOR-GREEN-3D — case-pool acceptance (generated)

Generated by `make_report.py` from `artifacts/summary.json` / `field_primary.npz`.
Reference: {"exact time-Taylor series (级数精确解)"}.

## Protocol

| item | value |
|---|---|
| primary grid | {p['grid_primary']}³ |
| ladder | {p['grids_ladder']} |
| T | {p['eddy_times']} eddy time(s), lattice steps {prim['steps']} |
| amplitude A / Ma | {p['amplitude']} / {prim['ma']:.4f} |
| Re = A·N/ν | {p['reynolds']:.0f} |
| ν / τ | {prim['nu']:.4g} / {prim['tau']:.4g} |
| series order | {p['series_order']} |
| pool band | < {p['tol_pct']} % |

## Verdict

**{'PASS' if s['verdict']['pass'] else 'HONEST FAIL'}**
({'all ' + str(s['verdict']['n_quantities']) + ' predeclared quantities inside band'
  if s['verdict']['pass'] else 'failed: ' + ', '.join(s['verdict']['failed'])})

| quantity | reference | measured | bound | verdict |
|---|---|---|---|---|
{rows}

## Grid ladder (fixed Re)

| N | \\|ΔE\\|/E [%] | \\|ΔΩ\\|/Ω [%] | velocity L2 [%] |
|---|---|---|---|
{ladder}

observed order (energy ratio) = {g['observed_order']:.2f}; monotone decreasing = {g['monotone_decreasing']}

## Primary state at T

| quantity | LBM | series |
|---|---|---|
| E(T)/E(0) | {prim['energy_ratio_lbm']:.6f} | {prim['energy_ratio_series']:.6f} |
| Ω(T)/Ω(0) | {prim['enstrophy_ratio_lbm']:.6f} | {prim['enstrophy_ratio_series']:.6f} |
| velocity L2 (abs) | {prim['u_l2_abs']:.4g} | rms u = {prim['u_l2_ref']:.4g} |
| ρ deviation from 1 | {prim['rho_deviation']:.3g} | expected O(Ma²) = {prim['ma'] ** 2:.2e} |

## Figures

{fig_lines}
"""
    out = ART / "summary.md"
    out.write_text(md)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    s = _load()
    figs = {
        "decay": fig_decay(s),
        "grid": fig_grid(s),
        "vorticity": fig_field(),
        "reference": fig_reference(s),
    }
    md = write_markdown(s, figs)
    print(f"wrote {md}")
    for k, v in figs.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())