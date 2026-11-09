"""Figures for IN101-N01: velocity profiles, form-drag scaling, drag band.

All plotting is driven by the campaign records / verdict JSON, so a figure can
never disagree with the numbers in the artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_profiles(records: dict[str, dict[str, Any]], out: Path) -> Path:
    """Mean streamwise profile in wall units for the smooth ladder + a keel case."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    smooth = sorted((r for r in records.values() if r["family"] == "smooth"),
                    key=lambda r: r["grid"]["nz"])
    for r in smooth:
        prof = r["profile"]
        yp = np.asarray(prof["y_plus"], float)
        up = np.asarray(prof["u_plus"], float)
        m = (yp > 0.3) & np.isfinite(up)
        ax.semilogx(yp[m], up[m], marker=".", ms=3, lw=1,
                    label=f"smooth {r['grid']['nz']} cells (Cf={r['cf']:.3g})")
    keels = sorted((r for r in records.values() if r["family"] == "keel"),
                   key=lambda r: r["tags"]["h_keel"])
    if keels:
        r = keels[-1]
        prof = r["profile"]
        yp = np.asarray(prof["y_plus"], float)
        up = np.asarray(prof["u_plus"], float)
        m = (yp > 0.3) & np.isfinite(up)
        ax.semilogx(yp[m], up[m], "k-", marker="s", ms=3, lw=1.2,
                    label=f"keel h={r['tags']['h_keel']:.0f} lam={r['tags']['lam']:.2f}")
    yp = np.logspace(0.3, np.log10(60), 60)
    ax.semilogx(yp, np.log(yp) / 0.41 + 5.2, "r--", lw=1,
                label="log law $U^+=\\ln y^+/\\kappa+B$")
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("$U^+$")
    ax.set_title("Under-ice channel mean profile (wall units)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    path = out / "fig_profiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_form_scaling(verdict_b: dict[str, Any], out: Path) -> Path:
    """Measured C_d^form against the analytic frontal-area estimate."""
    plt = _plt()
    rows = verdict_b.get("rows") or []
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    if rows:
        s = np.array([row["S_fr_over_plan"] for row in rows])
        cdf = np.array([row["cd_form"] for row in rows])
        ax.loglog(s, cdf, "o", ms=6, label="LBM form drag")
        fit = verdict_b.get("fit") or {}
        if fit.get("ok"):
            ss = np.logspace(np.log10(s.min()) - 0.1, np.log10(s.max()) + 0.1, 50)
            ax.loglog(ss, fit["A"] * ss ** fit["exponent"], "-",
                      label=f"fit $C_d^{{form}}={fit['A']:.3g}\\,S^{{{fit['exponent']:.2f}}}$ "
                            f"($R^2$={fit['r2']:.3f})")
        x = np.linspace(s.min(), s.max(), 20)
        # analytic expectation: C_d^form = 0.5 C_D,keel S with C_D,keel from the
        # constant of the fit (the *exponent* is the prediction with no free
        # parameter, which is what the criterion tests)
        ax.set_xlabel("$S_{fr}/S_{plan}$ (measured x-normal area ratio)")
        ax.set_ylabel("$C_d^{form}$")
        ax.set_title(f"Form-drag scaling: exponent {fit.get('exponent', float('nan')):.2f} "
                     f"vs analytic 1.00")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "fig_form_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_drag_band(verdict_d: dict[str, Any], smooth_cd: float, out: Path) -> Path:
    """C_d of every swept morphology against the observational band."""
    plt = _plt()
    rows = verdict_d.get("rows") or []
    lo, hi = verdict_d.get("band", [0.004, 0.02])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.axhspan(lo, hi, color="tab:green", alpha=0.15,
               label=f"drift-inversion band {lo}-{hi}")
    if rows:
        cd = np.array([row["cd_total"] for row in rows])
        cdf = np.array([row["cd_form"] for row in rows])
        x = np.arange(len(rows))
        ax.bar(x, cd, 0.6, color="tab:blue", label="C_d total")
        ax.bar(x, cdf, 0.6, color="tab:orange", label="C_d form")
        ax.set_xticks(x)
        ax.set_xticklabels([row["case_id"].replace("keel_", "") for row in rows],
                           rotation=45, ha="right", fontsize=7)
    ax.axhline(smooth_cd, color="k", ls=":", lw=1,
               label=f"smooth reference C_d={smooth_cd:.3g}")
    ax.set_yscale("log")
    ax.set_ylabel("$C_d$")
    ax.set_title("Swept $C_d$ vs observed ice-water drag band")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "fig_drag_band.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_geometry(records: dict[str, dict[str, Any]], out: Path) -> Path:
    """Frontal-area ratio and normalised form drag along the depth sweep."""
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.8))
    keels = [r for r in records.values() if r["family"] == "keel"]
    lams = sorted({r["tags"]["lam"] for r in keels})
    for lam in lams:
        sel = sorted((r for r in keels if abs(r["tags"]["lam"] - lam) < 1e-9),
                     key=lambda r: r["tags"]["h_keel"])
        h = [r["tags"]["h_keel"] for r in sel]
        ax1.plot(h, [r["S_fr_over_plan"] for r in sel], "o-", label=f"measured lam={lam}")
        ax1.plot(h, [r["analytic_S_fr_over_plan"] for r in sel], "--",
                 label=f"analytic lam={lam}")
        ax2.plot(h, [r["cd_form"] / r["S_fr_over_plan"] for r in sel], "s-",
                 label=f"lam={lam}")
    ax1.set_xlabel("$h_{keel}$ (lattice cells)")
    ax1.set_ylabel("$S_{fr}/S_{plan}$")
    ax1.set_title("Frontal-area ratio: measured vs $0.5\\lambda h_{keel}/h_{eff}$")
    ax2.set_xlabel("$h_{keel}$ (lattice cells)")
    ax2.set_ylabel("$C_d^{form}/(S_{fr}/S_{plan})$")
    ax2.set_title("Normalised form drag")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "fig_geometry.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all(records: dict[str, dict[str, Any]], verdicts: dict[str, Any],
             out_dir: str | Path) -> list[tuple[str, Path]]:
    """Render every figure; returns ``[(name, path), ...]``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    smooth = sorted((r for r in records.values() if r["family"] == "smooth"),
                    key=lambda r: r["grid"]["nz"])
    smooth_cd = float(smooth[-1]["cd_total"]) if smooth else float("nan")
    crit = verdicts["criteria"]
    made = [
        ("profiles", fig_profiles(records, out)),
        ("form_scaling", fig_form_scaling(crit["b_form_scaling"], out)),
        ("drag_band", fig_drag_band(crit["d_observed_band"], smooth_cd, out)),
        ("geometry", fig_geometry(records, out)),
    ]
    return made