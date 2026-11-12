#!/usr/bin/env python3
"""Figures for board audit m16351 (CASE-LBM-SQUARE-CYL-NORBERG, TensorLBM).

All numbers come from the byte copies under
``artifacts/audit_m16351/bytecopies/`` (sha256 in ``manifest.json``), so the
figures are regenerable without the mission work tree.

    fig1_force_traces.png    Cd / Cl time histories (D=8) + analysis window
    fig2_cl_spectrum.png     Cl spectra (D=8, D=10) with St peak markers
    fig3_ref_vs_got.png      got vs registered referee + tracesable 2D bracket
    fig4_vorticity.png       instantaneous vorticity field (D=8, final state)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/audit_m16351"
OUT = ROOT / "docs/board_audit_m16351/figs"
OUT.mkdir(parents=True, exist_ok=True)
U = 0.05


def spectra():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    peaks = {}
    for tag, D, wu, color in (("D8", 8, 0.5, "#1f77b4"), ("D10", 10, 0.25, "#d62728")):
        z = np.load(ART / f"bytecopies/series_{tag}.npz")
        cd, cl = z["cd"], z["cl"]
        w0 = int(wu * cd.size)
        w = cl[w0:] - cl[w0:].mean()
        f = np.fft.rfftfreq(w.size, d=1.0)  # cycles / step
        spec = np.abs(np.fft.rfft(w * np.hanning(w.size))) / max(w.size, 1)
        st = f * D / U
        ax.semilogy(st[1:], spec[1:], color=color, lw=1.4, label=f"D={D} (n={w.size})")
        k = int(np.argmax(spec[1:])) + 1
        # parabolic refinement (same estimator as the audited post-processing)
        a_, b_, c_ = spec[k - 1], spec[k], spec[k + 1]
        den = a_ - 2 * b_ + c_
        dlt = 0.5 * (a_ - c_) / den if abs(den) > 1e-30 else 0.0
        st_int = float((k + dlt) / w.size * D / U)
        peaks[tag] = {"bin": round(float(st[k]), 4), "interpolated": round(st_int, 4)}
        ax.axvline(st_int, color=color, ls="--", lw=1.0)
        ax.annotate(f"St={st_int:.4f} (bin {st[k]:.3f})", xy=(st_int, spec[k]),
                    xytext=(6, 4), textcoords="offset points", color=color, fontsize=9)
    ax.axvline(0.13, color="k", ls=":", lw=1.6)
    ax.annotate("registered referee St=0.13", xy=(0.13, ax.get_ylim()[0] * 3),
                xytext=(4, 0), textcoords="offset points", fontsize=8, rotation=90)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("St = f·D/U")
    ax.set_ylabel("|Cl| amplitude spectrum")
    ax.set_title("Cl spectrum — vortex-shedding peak (audit recompute)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cl_spectrum.png", dpi=150)
    plt.close(fig)
    (ART / "fig_peaks.json").write_text(json.dumps(peaks, indent=2))
    return peaks


def traces():
    fig, ax = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    z = np.load(ART / "bytecopies/series_D8.npz")
    cd, cl = z["cd"], z["cl"]
    n = cd.size
    t = np.arange(n)
    ax[0].plot(t, cd, color="#1f77b4", lw=0.8)
    ax[0].axvspan(0, 0.5 * n, color="0.85", alpha=0.6)
    ax[0].axhline(cd[int(0.5 * n):].mean(), color="k", ls="--", lw=1)
    ax[0].annotate(f"<Cd>_win = {cd[int(0.5*n):].mean():.4f}",
                   xy=(0.52 * n, cd[int(0.5 * n):].mean()), xytext=(4, 6),
                   textcoords="offset points", fontsize=9)
    ax[0].set_ylabel("Cd")
    ax[1].plot(t, cl, color="#d62728", lw=0.8)
    ax[1].axvspan(0, 0.5 * n, color="0.85", alpha=0.6)
    ax[1].set_ylabel("Cl")
    ax[1].set_xlabel("lattice step")
    ax[0].set_title("D=8 force histories (grey = spin-up, excluded from statistics)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_force_traces.png", dpi=150)
    plt.close(fig)


def ref_vs_got():
    j = json.load(open(ART / "judge.json"))
    cases = j["cases"]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.2))
    tags = list(cases)
    # St
    ax[0].bar(tags, [cases[t]["st"] for t in tags], color="#1f77b4", width=0.5, label="got (LBM)")
    ax[0].axhline(0.13, color="k", ls="--", lw=1.4, label="registered referee 0.13 (WITHHELD)")
    ax[0].axhspan(0.13 * 0.92, 0.13 * 1.08, color="k", alpha=0.08, label="referee ±8% band")
    ax[0].axhspan(0.159, 0.17, color="green", alpha=0.12,
                  label="traceable 2D numerics bracket (Re150→250)")
    for i, t in enumerate(tags):
        ax[0].annotate(f"{cases[t]['st_err_pct']:+.1f}%", xy=(i, cases[t]["st"]),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax[0].set_ylabel("St = fD/U")
    ax[0].set_title("St: FAIL vs referee, inside the traceable 2D bracket")
    # Cd
    ax[1].bar(tags, [cases[t]["cd"] for t in tags], color="#2ca02c", width=0.5, label="got (LBM)")
    ax[1].axhline(1.5, color="k", ls="--", lw=1.4, label="registered referee 1.5 (WITHHELD)")
    ax[1].axhspan(1.5 * 0.9, 1.5 * 1.1, color="k", alpha=0.08, label="referee ±10% band")
    for i, t in enumerate(tags):
        ax[1].annotate(f"{cases[t]['cd_err_pct']:+.1f}%", xy=(i, cases[t]["cd"]),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax[1].set_ylim(1.2, 1.75)
    ax[1].set_ylabel("Cd")
    ax[1].set_title("Cd: PASS (7-8% < 10%)")
    for a in ax:
        a.grid(alpha=0.3, axis="y")
        a.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_ref_vs_got.png", dpi=150)
    plt.close(fig)


def vorticity():
    z = np.load(ART / "bytecopies/state_D8.npz")
    f = z["f"].astype(np.float64)          # (9, ny, nx)
    ex = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    ey = [0, 0, 1, 0, -1, 1, 1, -1, -1]
    rho = f.sum(axis=0)
    u = sum(f[i] * ex[i] for i in range(9)) / rho
    v = sum(f[i] * ey[i] for i in range(9)) / rho
    # vorticity; note numpy rows = y, columns = x
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    w = dv_dx - du_dy
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(w, origin="lower", cmap="RdBu_r",
                   vmin=-np.nanpercentile(np.abs(w), 99.5),
                   vmax=np.nanpercentile(np.abs(w), 99.5), extent=[0, 20, 0, 20])
    ax.add_patch(plt.Rectangle((5.5, 9.5), 1.0, 1.0, fill=False, ec="k", lw=1.2))
    ax.set_xlim(0, 20); ax.set_ylim(4, 16)
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title(f"Vorticity ω·D/U (D=8, step {int(z['step'])}) — Kármán street, Re=200")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_vorticity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    peaks = spectra()
    traces()
    ref_vs_got()
    vorticity()
    print(json.dumps({"peaks": peaks, "out": str(OUT)}, indent=2))