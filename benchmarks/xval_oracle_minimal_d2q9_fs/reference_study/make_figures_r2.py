#!/usr/bin/env python3
"""Figures for the R2 resolution study (anchor XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS).

Reads ``artifacts/resolution_ladder.json`` + ``artifacts/resolution_analysis.json``
and writes four PNGs under ``benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/figures/``:

  fig_r2_front_curves.png   Z(T) per grid vs Martin & Moyce / Ritter + err(T)
  fig_r2_error_vs_a.png     max |rel err| vs a (log-log) with E_inf + C a^-p fits
  fig_r2_tau_sweep.png      relaxation sweep at a = 32 (M&M / Ritter bands)
  fig_r2_field_a64.png      final fill fraction + |u| at the finest grid

Every number is read from the artifacts (no hand-copied values).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import resolution_study as R  # noqa: E402  (metric/reference definitions)

MM_T, MM_Z = R.MM_T, R.MM_Z
RIT_T, RIT_Z = R.RIT_T, R.RIT_Z
DESIGN = R.DESIGN


def load():
    payload = json.loads((ART / "resolution_ladder.json").read_text())
    analysis = json.loads((ART / "resolution_analysis.json").read_text())
    return payload, analysis


def ladder_runs(payload):
    return {
        r["a"]: R.eval_run(R.eval_run(r, "mm"), "ritter")
        for r in payload["runs"]
        if r.get("kind") == "arm"
        and r.get("arm") == "tensorlbm_koerner"
        and r.get("tau") == 1.0
        and r.get("status") == "ok"
    }


# ---------------------------------------------------------------------------
def fig_front_curves(payload):
    runs = ladder_runs(payload)
    alist = sorted(runs)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax.plot(RIT_T, RIT_Z, "k:", lw=1.4, label="Ritter 1892 (exact SW, $T<2$)")
    ax.plot(MM_T, MM_Z, "ko", ms=4.5, mfc="none", label="Martin & Moyce 1952 (exp.)")
    cmap = plt.get_cmap("viridis")
    for i, a in enumerate(alist):
        pts = [p for p in runs[a]["vs_mm"]["points"]]
        T = [p["T"] for p in pts]
        Z = [p["Z_sub0.05"] for p in pts]
        ax.plot(
            T,
            Z,
            "-o",
            ms=3,
            lw=1.3,
            color=cmap(i / max(1, len(alist) - 1)),
            label=f"TensorLBM Körner a={a}",
        )
    ax.set_xlabel(r"$T = t\sqrt{g/h_0}$")
    ax.set_ylabel(r"$Z = x_{front}/a$ (sub-cell toe, $f=0.05$)")
    ax.set_title("Dam-break front: grid ladder vs references")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    for i, a in enumerate(alist):
        pts = runs[a]["vs_mm"]["points"]
        T = [p["T"] for p in pts]
        ax2.plot(
            T,
            [p["Z_sub0.05_err_pct"] for p in pts],
            "-o",
            ms=3,
            lw=1.3,
            color=cmap(i / max(1, len(alist) - 1)),
            label=f"a={a}",
        )
    ax2.axhspan(-15, 15, color="0.85", zorder=0, label="owner ±15 % band")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel(r"$T$")
    ax2.set_ylabel("rel. error vs M&M  [%]")
    ax2.set_title("Same data as an error curve")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5)
    fig.tight_layout()
    out = FIG / "fig_r2_front_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_error_vs_a(analysis):
    keys = ["Z_cell", "Z_sub0.02", "Z_sub0.05", "Z_sub0.1"]
    labels = {
        "Z_cell": r"$Z_{cell}$ (R1 metric, O(1/a) quantised)",
        "Z_sub0.02": r"$Z_{sub}$, toe depth $0.02\,h_0$",
        "Z_sub0.05": r"$Z_{sub}$, toe depth $0.05\,h_0$ (primary)",
        "Z_sub0.1": r"$Z_{sub}$, toe depth $0.10\,h_0$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, ref, name in (
        (axes[0], "mm", "Martin & Moyce 1952"),
        (axes[1], "ritter", "Ritter 1892"),
    ):
        a_fit = np.linspace(14, 90, 300)
        for k in keys:
            f = analysis["fits"].get(ref, {}).get(k, {})
            if not f.get("ok"):
                continue
            s = np.array(f["samples"])
            ax.plot(s[:, 0], s[:, 1], "o", ms=6, label=labels[k])
            ax.plot(
                a_fit,
                f["E_inf_pct"] + f["C"] * a_fit ** (-f["p"]),
                "-",
                lw=1.2,
                label=(
                    rf"{labels[k]}: $p$={f['p']:.2f}, "
                    rf"$E_\infty$={f['E_inf_pct']:.2f} %"
                ),
            )
        ax.axhline(15, color="C3", ls="--", lw=1.2, label="15 % acceptance")
        ax.axhline(3, color="C2", ls=":", lw=1.2, label="3 % asymptotic target")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("grid resolution  a  [cells per initial column width]")
        ax.set_ylabel(r"max $|$rel err$|$ over $T\in[0.5,4]$  [%]")
        ax.set_title(f"Convergence vs {name}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    out = FIG / "fig_r2_error_vs_a.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_tau_sweep(analysis):
    ts = analysis.get("tau_sweep", {})
    if not ts:
        return None
    taus = sorted((float(t) for t in ts), key=float)
    mm = [ts[f"{t:g}"]["Z_sub0.05_max_err_pct"] for t in taus]
    rit = [
        ts[f"{t:g}"]["Z_sub0.05_ritter_pct"]
        if f"{t:g}" in ts
        else ts[[k for k in ts if float(k) == t][0]]["Z_sub0.05_ritter_pct"]
        for t in taus
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    w = 0.38
    ax.bar(np.arange(len(taus)) - w / 2, mm, w, label="vs M&M (exp.)", color="C0")
    ax.bar(np.arange(len(taus)) + w / 2, rit, w, label="vs Ritter (exact SW)", color="C2")
    ax.axhline(15, color="C3", ls="--", lw=1.2, label="15 % acceptance")
    ax.set_xticks(np.arange(len(taus)))
    ax.set_xticklabels([f"τ={t:g}\n(ν={(t - 0.5) / 3:.3f})" for t in taus], fontsize=8)
    ax.set_ylabel(r"max $|$rel err$|$ of $Z_{sub}(0.05)$, $T\in[0.5,4]$  [%]")
    ax.set_title("Relaxation sweep at a = 32")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig_r2_tau_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_field(payload):
    r = next(
        (
            r
            for r in payload["runs"]
            if r.get("kind") == "arm"
            and r.get("arm") == "tensorlbm_koerner"
            and r.get("tau") == 1.0
            and "final_fill" in r
        ),
        None,
    )
    if r is None:
        return None
    fill = np.array(r["final_fill"], dtype=float)
    umag = np.array(r["final_umag"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    im0 = axes[0].imshow(fill, origin="lower", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title(f"fill fraction, a={r['a']}, T=end")
    axes[0].set_xlabel("x [cells]")
    axes[0].set_ylabel("y [cells]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(umag, origin="lower", cmap="magma")
    axes[1].set_title(r"$|u|$ [lattice]")
    axes[1].set_xlabel("x [cells]")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    out = FIG / f"fig_r2_field_a{r['a']}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    payload, analysis = load()
    outs = [
        p
        for p in (
            fig_front_curves(payload),
            fig_error_vs_a(analysis),
            fig_tau_sweep(analysis),
            fig_field(payload),
        )
        if p
    ]
    for p in outs:
        print(f"[fig] {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
